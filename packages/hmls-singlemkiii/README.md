# hmls-singlemkiii

Neural network player for the HMLS tank game — single-tank Flatten→GRU→policy-head architecture with no CNN.

## Architecture

The `MkIIITankPolicyNetwork` processes an egocentric visibility patch through three stages to produce action logits:

```
Encoded Patch ──► Flatten ──► GRU (temporal memory) ──► Linear Head (action logits)
[5, P, P]         [5·P·P]     hidden state h              → 5 action logits
```

Unlike the Mk-I and Mk-II architectures, the Mk-III **removes all convolutional layers**. The 5-channel encoded patch is flattened directly into a 1-D vector and fed into the GRU. This makes the model simpler and faster at the cost of losing learned spatial feature extraction.

### Input Encoding

Each turn, the tank's egocentric visibility patch is encoded as a `[5, patch_size, patch_size]` float tensor with the following channels:

| Channel | Name | Encoding |
|---------|------|----------|
| 0 | Terrain | passable = 1.0, impassable / boundary = 0.0, fog = −1.0 |
| 1 | Friendly tank | 1.0 if an alive friendly tank occupies the cell |
| 2 | Enemy tank | 1.0 if an alive enemy tank occupies the cell |
| 3 | Wreckage | 1.0 if a dead tank (any team) occupies the cell |
| 4 | Visibility mask | 1.0 if visible or boundary, 0.0 if fog |

### Flatten Stage

The encoded patch tensor `[5, P, P]` is reshaped into a flat vector of length `5 × P × P`. For the default `patch_size = 9`, this gives a 405-element input vector.

### GRU Cell

A single `GRUCell` receives the flattened patch vector and maintains a hidden state across turns within an episode. The hidden state is zero-initialised at the start of each episode.

### Policy Head

A single `Linear` layer maps the GRU hidden state to logits over the 5-action space:

| Index | Action |
|-------|--------|
| 0 | `MOVE_FORWARD` |
| 1 | `TURN_LEFT` |
| 2 | `TURN_RIGHT` |
| 3 | `FIRE` |
| 4 | `PASS` |

## Model Configuration

The `MkIIIModelConfig` (Pydantic model, frozen) controls the network architecture. It is serialised as `model_config.json` in the model directory.

Example:

```json
{
  "patch_size": 9,
  "model_package": "singlemkiii",
  "gru_hidden_size": 128
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patch_size` | int | `9` | Side length of the input visibility patch (must be ≥ 3) |
| `model_package` | str | `"hmls.singlemkiii"` | Entry-point name or package path for persistence registry |
| `gru_hidden_size` | int | `128` | Dimensionality of the GRU hidden state |

**Note:** The GRU input size is computed as `5 × patch_size²`. Larger patch sizes significantly increase the number of GRU parameters. The `patch_size` must match the visibility patch size used by the game engine.

## Comparison with Mk-I and Mk-II

| Feature | Mk-I | Mk-II | **Mk-III** |
|---------|------|-------|------------|
| CNN | Yes (configurable layers) | Yes (configurable layers) | **None** |
| GRU layers | 1 | 2 (stacked) | **1** |
| Spatial feature extraction | Learned (conv) | Learned (conv) | **None (raw pixels)** |
| Parameter count | Medium | Highest | **Lowest** |
| Best for | General use | Complex temporal patterns | **Baseline / small patches** |
