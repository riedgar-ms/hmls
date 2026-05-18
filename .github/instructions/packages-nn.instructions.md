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

## Tank Package Structure

Every neural network tank package follows this structure:

```
packages/hmls-singlemkX/
├── pyproject.toml          # Must include entry point
├── src/hmls/singlemkX/
│   ├── __init__.py
│   ├── model.py            # Frozen Pydantic config + PyTorch nn.Module
│   ├── persistence.py      # PERSISTENCE constant
│   └── player.py           # (optional) Custom NNPlayerBase subclass
└── tests/
```

The config class and model class are defined together in `model.py`.
A custom `player.py` is only needed when the generic `NNPlayer` from
`hmls-nncore` is insufficient (e.g. rule-based logic in `hmls-randomtank`).

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
