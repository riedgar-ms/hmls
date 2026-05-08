# hmls-singlemki

Neural network player for the HMLS tank game — single-tank CNN→GRU→policy-head architecture.

## Architecture

The `MkITankPolicyNetwork` processes an egocentric visibility patch through three stages to produce action logits:

```
Encoded Patch ──► CNN (spatial features) ──► GRU Cell (temporal memory) ──► Linear Head (action logits)
[5, P, P]         [Conv→ReLU→Pool] × N       hidden state across turns       → 5 action logits
```

### Input Encoding

Each turn, the tank's egocentric visibility patch is encoded as a `[5, patch_size, patch_size]` float tensor with the following channels:

| Channel | Name | Encoding |
|---------|------|----------|
| 0 | Terrain | passable = 1.0, impassable / boundary = 0.0, fog = −1.0 |
| 1 | Friendly tank | 1.0 if an alive friendly tank occupies the cell |
| 2 | Enemy tank | 1.0 if an alive enemy tank occupies the cell |
| 3 | Wreckage | 1.0 if a dead tank (any team) occupies the cell |
| 4 | Visibility mask | 1.0 if visible or boundary, 0.0 if fog |

### CNN Stage

A sequence of convolutional blocks extracts spatial features from the encoded patch. Each block consists of:

1. `Conv2d` with same-padding (kernel size ÷ 2) to preserve spatial dimensions
2. `ReLU` activation
3. `MaxPool2d` for spatial downsampling

The number of blocks equals `len(cnn_channels)`. The first block takes 5 input channels (from the encoder); subsequent blocks chain their output channels. With the default `cnn_channels = (32, 64)`, there are two blocks: 5→32 and 32→64.

### GRU Cell

The flattened CNN output feeds a `GRUCell` that maintains a hidden state across turns within an episode. This gives the network temporal memory — it can learn to integrate information observed over multiple turns. The hidden state is zero-initialised at the start of each episode.

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

The `MkIModelConfig` (Pydantic model, frozen) controls the network architecture. It is serialised as `model_config.json` in the model directory.

Example:

```json
{
  "patch_size": 9,
  "cnn_channels": [32, 64],
  "gru_hidden_size": 128,
  "conv_kernel_size": 3,
  "pool_kernel_size": 2,
  "pool_stride": 2
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patch_size` | int | `9` | Side length of the input visibility patch (must be ≥ 3) |
| `cnn_channels` | list[int] | `[32, 64]` | Output channels for each conv block; length determines number of blocks |
| `gru_hidden_size` | int | `128` | Dimensionality of the GRU hidden state |
| `conv_kernel_size` | int | `3` | Kernel size for each `Conv2d` layer (must be ≥ 1; odd values recommended for symmetric padding) |
| `pool_kernel_size` | int | `2` | Kernel size for each `MaxPool2d` layer (must be ≥ 1) |
| `pool_stride` | int | `2` | Stride for each `MaxPool2d` layer (must be ≥ 1) |

**Note:** Increasing `cnn_channels` depth or `gru_hidden_size` increases model capacity but also parameter count and inference cost. The `patch_size` must match the visibility patch size used by the game engine.
