# Multi-Tank Squad Architecture: Design Discussion

This document records the design discussion and decisions made for adding multi-tank squad support to the hmls tank game. It traces the reasoning from initial concept through to the planned implementation.

## Starting Point

The current system supports neural network models that each control a **single tank**. Each model sees an egocentric visibility patch (a small NxN window of the map, rotated so forward is "up") and outputs one of 5 low-level actions (move forward, turn left, turn right, fire, pass). A GRU provides temporal memory across turns within a game.

The game engine already supports multiple tanks per team — `place_tanks()` accepts a configurable `tanks_per_player`, and `PlayerView` already provides patches for all alive friendly tanks plus metadata for all friendly tanks (alive or dead). However, no NN model currently uses this multi-tank information.

## Goal

Add a hierarchical multi-tank system where:
- A **planner model** observes all friendly tanks and issues high-level **orders**
- Per-tank **executor models** translate those orders + their local patch into low-level actions
- The number of tanks per team is configurable (e.g. 2–5)

## Design Decisions

### Decision 1: Hierarchical planner + executor (not monolithic)

**Options considered:**
1. **Planner dispatches high-level orders to tank models** — A planner model sees all friendly tank patches + global info, and outputs per-tank "orders". Individual executor models then translate orders into low-level actions.
2. **Single model selects all actions directly** — One model receives all friendly tank views and directly outputs actions for all tanks.

**Chosen: Option 1 (hierarchical)**

**Reasoning:** Option 2 is simpler but less modular. The hierarchical approach allows the planner and executors to be trained and improved independently. It also provides a natural separation of concerns: strategic thinking (which tank should do what?) vs. tactical execution (how do I carry out this order given what I see?). This mirrors real-world command structures.

### Decision 2: Simple discrete orders (not parameterised)

**Options considered:**
1. **~6–8 discrete orders** — Simple categorical output, e.g. ADVANCE, RETREAT, HOLD, ENGAGE, EVADE, SCOUT, FLANK_LEFT, FLANK_RIGHT
2. **Order type + directional parameter** — More expressive (e.g. "move toward bearing NE") but harder to train
3. **Start simple, extend later** — Begin with discrete, design the system so parameterised orders can be added

**Chosen: Option 1 (simple discrete, ~8 orders)**

**Reasoning:** The planner's output is a categorical distribution — one order per alive tank. Parameterised orders would require a more complex action space (mixed discrete/continuous) which significantly complicates the REINFORCE training. Starting with a small, fixed vocabulary lets us get the system working and evaluate whether finer-grained orders are actually needed.

**Planned order vocabulary:**
| Order | Intent |
|-------|--------|
| ADVANCE | Move toward the front / explore forward |
| RETREAT | Fall back / move away from enemies |
| HOLD | Stay in current area, minimal movement |
| ENGAGE | Seek out and fire at enemies aggressively |
| EVADE | Avoid confrontation, escape threats |
| SCOUT | Explore unseen terrain |
| FLANK_LEFT | Move to attack from the left |
| FLANK_RIGHT | Move to attack from the right |

### Decision 3: New executor model (not reusing existing single-tank models)

**Options considered:**
1. **Reuse existing `TankModelBase` models, augment with order input** — The existing CNN→GRU→Linear architecture could be extended with an order embedding concatenated before the GRU. Pre-trained weights could be transferred for the CNN/GRU layers.
2. **New executor architecture, with option to bootstrap from pre-trained weights** — New architecture designed for orders from the start, but structurally similar enough to initialise shared layers from pre-trained single-tank weights.
3. **Fully new executor architecture, trained from scratch** — Clean design, no weight-loading complexity.

**Chosen: Option 3 (fully new, trained from scratch)**

**Reasoning:** The existing single-tank models don't accept orders — their forward signature is `(patch_tensor, hidden) → (logits, new_hidden)`. Adding an order input changes the architecture fundamentally. While transfer learning from pre-trained weights could theoretically speed up initial training, the practical benefit is marginal because:

- The pre-trained CNN/GRU weights encode a "wander around until you see an enemy and home in on it" policy. This only matches the ENGAGE order — the other 7 orders need fundamentally different behaviours.
- The transfer would only help with basic spatial understanding (terrain parsing, wall avoidance), not the order-specific tactical behaviours that are the whole point.
- After the initial weight load, there is no structural difference between Options 2 and 3 — so the transfer benefit is purely a faster initial convergence, not a better final model.

### Decision 4: Separate trainer package

**Options considered:**
1. **Extend `hmls-reinforcetrainer` with a multi-tank mode** — Add configuration options and code paths for squad training alongside single-tank training.
2. **New `hmls-squadtrainer` package** — Independent trainer, purpose-built for the hierarchical architecture.
3. **New trainer initially, refactor to share code later** — Start separate, merge common utilities if patterns emerge.

**Chosen: Option 3 (new trainer, refactor later)**

**Reasoning:** The single-tank trainer is tightly coupled to 1v1 games (two `ModelRef`s, two players, one tank each). The squad trainer needs qualitatively different logic: multiple tanks per team, two models to update per team (planner + executor), order-conditioned reward attribution, and team-level aggregation for planner rewards. Starting separate avoids polluting the working single-tank trainer. If common patterns emerge (e.g. the REINFORCE update itself), they can be extracted into `hmls-nncore` later.

### Decision 5: Configurable tanks per team (2–5)

The number of tanks per team is set in the trainer configuration. The system should handle any count from 2 to 5 (or more, though practical limits exist due to map size and training complexity).

### Decision 6: Asymmetric match support

**Chosen:** The trainer config supports two team types:
- `"squad"` — hierarchical planner + executor
- `"independent"` — N copies of a single-tank model, each acting independently

This allows bootstrapping: train a squad against known-good single-tank opponents before attempting squad-vs-squad training. The config is designed so the team type is the primary discriminator:

```json
{
  "team_a": { "type": "squad", "planner": {...}, "executor": {...}, "tanks_per_team": 3 },
  "team_b": { "type": "independent", "model": {...}, "tanks_per_team": 3 }
}
```

## Key Technical Decisions

### Handling variable tank counts (tank deaths)

Tanks will be destroyed during gameplay, so the planner must handle a shrinking team. This is not an edge case — it is the normal course of a game.

**Planner approach: set-pooling.** Each alive tank's patch is encoded independently, then aggregated via mean-pooling across the set. The planner outputs orders only for alive tanks. This naturally handles any number of alive tanks without padding or masking.

**Why not attention?** Attention/transformer-based cross-tank communication would allow richer coordination (each tank's order could depend on what other tanks see). However, it is significantly more complex to implement and train. The set-pooling approach is the right starting point — the abstract base class design (see below) means an attention-based planner can be added as a drop-in replacement later.

**Executor:** No special handling needed. Each executor processes its own tank independently. When a tank dies, the engine simply stops calling it.

### Reward strategy for the planner

The planner's reward signal is the hardest design question. Four options were identified:

| Option | Signal | Density | Complexity | Verdict |
|--------|--------|---------|------------|---------|
| 1. Shared team reward | Win/loss outcome | Very sparse | Minimal | Too sparse to learn from alone |
| 2. Aggregated executor rewards | Mean of per-step executor rewards | Dense | Low | **Recommended starting point** |
| 3. Order-conditioned rewards | Per-order reward modifiers | Dense | Medium (hyperparameter surface) | Useful but premature to tune |
| 4. Hybrid (2 + 3) | Aggregated + order-conditioned | Dense | Higher | **Recommended eventual approach** |

**Decision:** Start with Option 2. Define the order-conditioned reward config types (Option 3) in the `hmls-squad` package from the start, so they're available when needed, but don't require them initially.

**What are order-conditioned rewards?** They modify the standard reward signal based on the current order. For example:
- **SCOUT** order: amplify rewards for seeing new cells, suppress rewards for hitting enemies
- **HOLD** order: penalise movement, reward staying near the starting position
- **ENGAGE** order: amplify rewards for hits and closing distance to enemies

This incentivises executors to actually follow their orders, and gives the planner credit when its order assignments match the situation.

## Extensibility Design

The architecture is designed in tiers to support future upgrades without breaking existing code:

**Tier 1: Stable contract (`hmls-squad`)**
Defines abstract base classes and shared types that all implementations depend on:
- `Order` enum
- `PlannerModelBase(ABC, nn.Module)` — abstract base for all planner architectures
- Order-conditioned reward configuration types

**Tier 2: Concrete implementations (separate packages)**
Each is a separate package that can be swapped in via `model_config.json`:

*Planner implementations:*
- `hmls-squadplanner` (initial): set-pooling + MLP — simple, handles variable tank counts
- `hmls-squadplanner-attn` (future): transformer/attention for cross-tank communication
- `hmls-squadplanner-gnn` (future): graph neural network treating tanks as nodes

*Executor implementations:*
- `hmls-squadexecutor` (initial): CNN + order embedding → GRU → action logits
- Future variants could add inter-tank communication channels

**Tier 3: Training infrastructure (`hmls-squadtrainer`)**
Depends only on Tier 1 abstractions, not concrete implementations. Any planner/executor pair can be used.

## Planned Package Structure

| Package | Purpose | Dependencies |
|---------|---------|-------------|
| `hmls-squad` | Order enum, abstract bases, reward config | `hmls-core`, `hmls-nncore` |
| `hmls-squadexecutor` | CNN + order embedding → GRU → actions | `hmls-squad`, `hmls-nncore` |
| `hmls-squadplanner` | Set-pooling planner (initial impl) | `hmls-squad`, `hmls-nncore` |
| `hmls-squadplayer` | Composite `SquadPlayer(Player)` | `hmls-squad`, `hmls-squadexecutor`, `hmls-squadplanner` |
| `hmls-squadtrainer` | REINFORCE trainer for squads | `hmls-squad`, `hmls-squadplayer`, `hmls-nncore`, `hmls-mapgenerator` |

## Left for the Future

1. **Attention-based planner** — A new package (`hmls-squadplanner-attn`) implementing `PlannerModelBase` with transformer/attention layers for richer cross-tank coordination. Drop-in replacement via `model_config.json`.

2. **Parameterised orders** — Extending the order vocabulary with directional parameters (e.g. "advance toward bearing NE"). Would require changes to the `Order` type, planner output layer, and executor input encoding.

3. **Order-conditioned rewards (full implementation)** — The config types will be defined from the start, but the initial trainer will use simple aggregated executor rewards. Enabling order-conditioned rewards is a configuration change, not a code change.

4. **Shared training utilities** — If patterns emerge between `hmls-reinforcetrainer` and `hmls-squadtrainer` (e.g. the REINFORCE update, return baseline), common code can be extracted into `hmls-nncore`.

5. **Inter-executor communication** — Future executor architectures could receive information from other executors (e.g. shared hidden states or attention across the team).

6. **Dynamic planning frequency** — The initial implementation calls the planner every turn. Future work could explore planning every N turns, or only on significant events (tank death, enemy spotted).

## How Multi-Tank Training and Inference Work

The squad architecture uses a **single executor model** shared across all tanks on a team, and a **single planner model** per team. This section explains how both inference and training handle the fact that multiple tanks interact with the same model simultaneously.

### Per-tank hidden state during play

The existing single-tank `NNPlayer` holds one GRU hidden-state tensor (`self._hidden`). When the engine calls `choose_action("A1", view)` followed by `choose_action("A2", view)`, the hidden state from A1's forward pass would be used as input for A2 — mixing their temporal contexts.

The `SquadPlayer` solves this by maintaining **per-tank hidden states** in a dictionary:

```python
self._hidden_states: dict[TankId, torch.Tensor]
```

When `choose_action` is called for tank A1, the player looks up `_hidden_states["A1"]`, runs the executor forward pass, and stores the new hidden state back. When called for A2, it uses `_hidden_states["A2"]`. Each tank's GRU memory is independent and continuous across turns, even though the **model weights are shared**.

This is analogous to running the same RNN on multiple input sequences in parallel — same parameters, independent hidden states.

**Initialisation:** On episode reset (`reset_episode`), the dict is cleared. On the first call for a new `tank_id`, the hidden state is initialised via `model.initial_hidden()` (zeros). When a tank dies, the engine simply stops calling `choose_action` for it; the stale hidden state is harmless and gets cleared on the next episode reset.

### Per-tank reward and trajectories during training

REINFORCE needs a trajectory — a sequence of (log-probability, reward) pairs — to compute a policy gradient. With multiple tanks sharing one model, there are multiple trajectories being generated simultaneously, each with its own reward signal.

The `SquadPlayer` maintains **per-tank episodes**:

```python
self._episodes: dict[TankId, Episode]
self._log_prob_tensors: dict[TankId, list[torch.Tensor]]
self._entropy_tensors: dict[TankId, list[torch.Tensor]]
```

During a game:

1. Engine calls `choose_action("A1", view)` → executor forward pass with A1's hidden state → action sampled → log-prob and entropy appended to **A1's** trajectory.
2. Engine calls `choose_action("A2", view)` → same model, A2's hidden state → appended to **A2's** trajectory.
3. When the engine reports a step result for A1, the reward is assigned to A1's episode. Same for A2.

### Accumulated gradient update (executor)

After a game, each tank's trajectory is an independent episode through the **same model**. The recommended approach is to **accumulate the loss across all tanks** and perform a single optimiser step:

```python
total_loss = torch.tensor(0.0)
for tank_id in alive_tanks:
    episode = player.episodes[tank_id]
    returns = compute_returns(episode.rewards(), gamma)
    advantages = baseline.compute_advantages(returns)
    log_probs = torch.stack(player.log_prob_tensors[tank_id])
    total_loss += -(log_probs * advantages.detach()).sum()
    # plus entropy bonus

optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

This is equivalent to treating each tank as a separate rollout from the same policy — a standard approach in multi-agent RL with parameter sharing. One optimiser step per game regardless of tank count, and each tank's trajectory provides independent gradient samples, giving a lower-variance estimate than any single tank alone.

The `ReturnBaseline` (exponential moving average of returns) is **shared across all tanks**: it tracks the running average return for the executor policy regardless of which tank generated it.

### Planner training

The planner has a simpler structure:

- **One trajectory per team** — the planner acts once per planning round (initially every turn), outputting one order per alive tank. This entire decision is a single trajectory step.
- **Team-level reward** — the planner's reward is the mean of per-step executor rewards across all tanks (Option 2 from the reward strategy discussion above).
- **Separate optimiser** — the planner and executor have independent optimisers and independent REINFORCE updates.

### Summary

| Aspect | Single-tank (current) | Squad executor | Squad planner |
|--------|----------------------|----------------|---------------|
| Hidden state | 1 per player | 1 per tank (`dict`) | 1 per team |
| Episode/trajectory | 1 per player | 1 per tank (`dict`) | 1 per team |
| Reward signal | Per-step for this tank | Per-step per tank | Aggregated team reward |
| REINFORCE update | 1 loss → backward → step | Sum per-tank losses → 1 backward → step | 1 loss → backward → step |
| Model weights | Unique per player | Shared across all tanks | Unique per team |
