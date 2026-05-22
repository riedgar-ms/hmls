---
applyTo: "packages/hmls-singlemk*/**,packages/hmls-nncore/**,packages/hmls-randomtank/**,packages/hmls-reinforcetrainer/**"
---

# Neural Network & Training Packages

## Package Roles

| Package | Purpose |
|---------|---------|
| `hmls-nncore` | Shared base classes: `NNPlayerBase`, patch encoding, persistence protocol |
| `hmls-singlemki` | CNN → GRU → Linear policy network |
| `hmls-singlemkii` | CNN → stacked GRU (2 layers) → Linear |
| `hmls-singlemkiii` | Flatten → GRU → Linear (no CNN) |
| `hmls-randomtank` | Rule-based (deterministic fire, random movement) |
| `hmls-reinforcetrainer` | REINFORCE policy-gradient trainer |

## Tank Package Requirements

Every tank package must provide four semantic components:

1. **Model config** — a frozen Pydantic `TankModelConfig` subclass defining architecture hyperparameters, serialised as `model_config.json`.
2. **Model class** — a `TankModelBase` subclass (PyTorch `nn.Module`) implementing the forward pass.
3. **Player** — an `NNPlayerBase` subclass (defined in `player_base.py`) supporting both `"play"` mode (inference) and `"learn"` mode (records log-probs for training). The concrete `NNPlayer` (in `player.py`) from `hmls-nncore` handles this for standard NN tanks; a custom player is only needed for non-standard action logic (e.g. rule-based `hmls-randomtank`).
4. **Persistence** — a `PERSISTENCE` constant (an `NNPlayerModelPersistence` instance) exposing load/save/create operations, registered via an entry point.

The existing packages generally follow this file layout, though it is not prescribed:

```
packages/hmls-singlemkX/
├── pyproject.toml          # Must include entry point
├── src/hmls/singlemkX/
│   ├── __init__.py
│   ├── model.py            # Config class + model class
│   ├── persistence.py      # PERSISTENCE constant
│   └── player.py           # (only if custom NNPlayerBase subclass needed)
└── tests/
```

Note: In `hmls-nncore` itself, `player_base.py` defines the abstract `NNPlayerBase` and `player.py` provides the concrete `NNPlayer`.

## Entry Point Registration

Each tank registers via `pyproject.toml`:

```toml
[project.entry-points."hmls.models"]
singlemkX = "hmls.singlemkX.persistence:PERSISTENCE"
```

The key becomes the `model_id` referenced in `model_config.json` files.

## Dependencies

All NN tanks depend on `hmls-core` and `hmls-nncore`. They use PyTorch (`torch>=2.0`).

## Important Conventions

- Model configs are frozen Pydantic models serialised as `model_config.json`.
- The `Player` class must support both `"play"` mode (inference) and `"learn"` mode (records log-probs for training).
- Patch size must be consistent across all models in a training run.
