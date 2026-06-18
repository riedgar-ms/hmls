# hmls-simplesquadplayer

Composite squad player combining a planner and executor model for multi-tank teams.

## How It Works

The `SimpleSquadPlayer` is a `Player` implementation that uses a hierarchical architecture:

1. **Planning phase**: When the first tank on the team acts in a turn, the planner model runs once, observing all alive friendly tanks (patches + positions + directions), and assigns a discrete order to each
2. **Execution phase**: Each tank's executor model translates its assigned order + local egocentric patch into a low-level action

### Per-tank state management

- **Hidden states**: Each tank maintains its own GRU hidden state (`dict[TankId, Tensor]`), independent of other tanks despite sharing model weights
- **Trajectories**: Each tank has its own episode/log-prob/entropy tracking for REINFORCE training
- **Planner trajectory**: A single team-level trajectory tracks the planner's decisions

### Planning frequency

The planner runs once per planning round. Call `begin_round()` before the first `choose_action()` in each turn to trigger fresh planning. If not called, the previous round's orders are reused.

## Squad Directory Layout

```
squad_dir/
├── planner/
│   ├── model_config.json    # SimplePlannerConfig
│   └── model.pt             # Planner weights
└── executor/
    ├── model_config.json    # SimpleExecutorConfig
    └── model.pt             # Executor weights
```

## Loading a Squad for Play

```python
from hmls.nncore.squad import resolve_squad_id

persistence = resolve_squad_id("simplesquad")
player = persistence.create_player(Path("path/to/squad_dir"), team="A", mode="play")
```

Or directly:

```python
from hmls.simplesquadplayer.persistence import PERSISTENCE

player = PERSISTENCE.create_player(Path("path/to/squad_dir"), team="A")
```

## Entry Point Registration

Registered under the `hmls.squads` group:

```toml
[project.entry-points."hmls.squads"]
simplesquad = "hmls.simplesquadplayer.persistence:PERSISTENCE"
```

This allows runtime discovery by the test harness, server, or any other infrastructure without compile-time dependencies on this package.

## Dependencies

- `hmls-core` — Player ABC, game types
- `hmls-nncore` — squad base classes, encoding, trajectory
- `hmls-simplesquadplanner` — concrete planner model
- `hmls-simplesquadexecutor` — concrete executor model
