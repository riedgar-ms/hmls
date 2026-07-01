# Planning for Squads

The goal of this directory is to provide a computer player which manages a
**squad of multiple tanks**. Ultimately it should be able to connect as a player
to a game hosted by `hmls-server`, but that server integration is a later
milestone (see [Open questions](#open-questions-and-deferred-decisions)). The
first target is a working training pipeline plus a squad player that can be run
through `hmls-testharness`.

This is a design document, intended to be read by humans and refined over time.
It captures the intended shape of the system and the decisions made so far; it
is not a low-level implementation checklist.

## High-level design

In this game a **player** controls *all* of its team's tanks. The squad player
is therefore a single composite player that internally runs two kinds of model:

- **Planner** — a single stateful instance for the whole squad.
- **Executor** — one shared, order-conditioned model, applied per tank.

Each turn, for the currently active tank, the flow is:

1. The **planner** runs on a fresh view and produces (or refreshes) an **order**
   for that tank.
2. An **order-translation layer** converts the order into inputs the executor
   can use in its egocentric frame.
3. The **executor** produces the tank's low-level action.

The executor uses the same architecture and the same weights for every tank;
only its per-tank internal state (GRU hidden state) differs. The planner is
discouraged from changing a tank's order too frequently (see below), so orders
stay reasonably stable across turns.

### Planning cadence

The planner runs **per active tank**: immediately before a tank acts, on the
current view, producing or refreshing only that tank's order. This gives each
tank a fresh observation to plan from, but it means the planner must **remember
what it has ordered its teammates** — so the planner is stateful (see
[Planner](#planner)).

## Orders

The planner can issue three orders. One of them is *parameterised* (it carries a
target location):

| Order | Parameter | Intent |
|-------|-----------|--------|
| `Move-to-location` | absolute target `(x, y)` | Move to a specific cell on the map |
| `Explore` | — | Prioritise covering unseen terrain; keep moving |
| `Hunt` | — | Seek and engage enemies; more willing to hold position |

`Explore` and `Hunt` are related, but trained toward different behaviours:
`Explore` favours movement and map coverage, while `Hunt` is more willing to
stay in one place to fight.

### Discouraging order churn

If orders flip every turn, neither the executor's memory nor the planner has
anything stable to anchor an order's meaning to. The planner is therefore
discouraged from changing a given tank's order too often via a **soft penalty**
in the reward/loss when it switches that tank's order — rather than a hard rule
forcing an order to be held for a fixed number of turns.

## Order-translation layer

A `Move-to-location(x, y)` order carries an **absolute** map coordinate, but the
executor sees a **rotated egocentric patch** (forward is "up"), so a raw `(x, y)`
means nothing in its frame. A small **order-translation layer** sits between the
planner and the executor (inside the squad player) and, each turn, converts the
order into an **egocentric bearing plus normalised distance** to the target,
computed from the tank's current position and heading.

Importantly, this does **not** re-issue the order — the order stays fixed at
`Move-to-location(x, y)`. Only the derived executor input changes as the tank
moves toward the target. `Explore` and `Hunt` carry no target and simply pass a
categorical order identifier through to the executor.

## Planner

The planner is **stateful**. It maintains two things:

- an **allocentric world model** of the map (an absolute-frame grid), and
- an **internal recurrent state vector** (memory across turns, including what it
  has ordered which tanks).

### World model (allocentric map)

As the squad explores, egocentric visibility patches reveal terrain. The planner
projects those patches back into world coordinates to build a persistent,
absolute-frame map — the same idea as the automapper already used by
`hmls-client`. This map is a multi-channel grid, suitable for processing by a
CNN, carrying:

- **Terrain / exploration** — explored vs unknown, and passable vs impassable
  (the base automap layer).
- **Friendly tank positions** — where the squad's own tanks currently are.
- **Last-seen enemy positions** — where enemies were most recently observed,
  with **age-out**: a sighting is cleared immediately if that cell is later
  observed to be empty, and otherwise **fades over a TTL** via a decayed
  recency value (so stale sightings gradually lose influence rather than
  lingering as hard flags).

The planner combines the CNN's reading of this map with its recurrent state to
choose an order (and, for `Move-to-location`, a target) for the active tank.

### AutoMap refactor

The terrain/exploration automap currently lives in `hmls-client`
(`hmls.client.automap`). Because it is really a **core game concept** — and both
the client and the squad planner need it — it should be **refactored into
`hmls-core`**. The squad-specific channels (friendly positions, last-seen
enemies with age-out) are then layered on top as a squad-side extension of the
core automap.

## Executor

The executor follows the same basic structure as the Mk-I single tanks: a CNN
over the egocentric patch, feeding a GRU (temporal memory across turns), feeding
an action head. On top of the patch it also receives:

- the tank's current location and orientation,
- the current order — as a categorical identifier, plus (for
  `Move-to-location`) the egocentric bearing/distance produced by the
  order-translation layer.

There is a **single shared executor model**: the same architecture and weights
serve every tank and every order. Order-conditioning is achieved by feeding the
order in as input, so behaviour differs by order without needing separate
models. Each tank keeps its own GRU hidden state so their temporal contexts stay
independent even though the weights are shared.

## Training

Training proceeds in two broad stages, executor first, then planner (with
continued executor training alongside).

### Executor training (per-order curriculum)

Because the executor is order-conditioned, it can be trained one order at a time
against an order-specific reward:

- **Move-to-location** — reward for reducing distance to / reaching the target.
- **Explore** — reward for covering new terrain and keeping moving.
- **Hunt** — reward for finding and hitting enemies; trained against an
  existing opponent (one of the single-tank players, or the random tank).

These are training *phases* over the **same shared executor weights**, not
separate models. As with the single-tank trainer, each phase runs across
**multiple maps of multiple sizes**.

### Planner training

Once the executor has a solid grounding in each order, the planner is trained
(executor training may continue in parallel). Initially the squad plays against
**groups of single tanks** (random or Mk-n); over time this becomes switchable
to playing against **another squad**. As with executor training, each run uses
multiple maps of differing sizes.

## Package structure

The implementation is split into several packages under
`packages/computerplayers/simplesquad/`:

| Package (role) | Purpose |
|----------------|---------|
| shared base | Order definitions, abstract planner/executor bases, and the squad-side automap extension |
| executor | The shared order-conditioned executor model |
| planner | The stateful, map-building planner model |
| player | The composite `SquadPlayer` that runs the planner, order-translation layer, and executors |
| trainer | REINFORCE-style training for the squad (executor curriculum, then planner) |

The exact package names and boundaries can be refined as implementation begins;
the intent is that models (planner / executor) can be swapped independently of
the player and trainer.

## Open questions and deferred decisions

The following are intentionally left open for now, to be resolved in later
sessions:

- **Planner reward strategy.** How the planner is rewarded (e.g. aggregated
  per-tank executor rewards, team outcome, or a planner-specific signal) is not
  yet decided. This does **not** affect executor training — executors are
  trained on their own order-specific rewards — so the choice can be made later.
- **`hmls-server` / model-registry integration.** The current model registry
  assumes a single single-tank model per registered player, whereas the squad is
  a composite player with two model types. Making the squad player loadable and
  playable via `hmls-server` is a later milestone, deliberately not designed
  here.
- **Concrete map encodings and hyperparameters.** The exact channel encodings
  for the planner's world model, the enemy-sighting TTL, and similar details are
  left to be pinned down during implementation.

## Future directions

Beyond the initial design, natural extensions include richer cross-tank
coordination in the planner (e.g. attention- or graph-based planners instead of
a map-plus-state model), a larger or more expressive order vocabulary, and
inter-executor communication. These are noted only as possibilities; the design
above is the intended starting point.
