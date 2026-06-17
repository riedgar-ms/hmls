# hmls-singlemkii

Neural network player for the HMLS tank game — single-tank CNN→GRU₁→GRU₂→policy-head architecture with dual stacked GRU cells.

## Architecture

The `MkIITankPolicyNetwork` processes an egocentric visibility patch through four stages to produce action logits:

```
Encoded Patch ──► CNN (spatial features) ──► GRU₁ (temporal memory) ──► GRU₂ (deep temporal) ──► Linear Head (action logits)
[5, P, P]         [Conv→ReLU→Pool] × N       hidden state h₁              hidden state h₂          → 5 action logits
```

Compared to the Mk-I single-GRU architecture, the Mk-II stacks two GRU cells with independently configurable hidden sizes, allowing the model to learn hierarchical temporal representations.

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

### Stacked GRU Cells

The flattened CNN output feeds two stacked `GRUCell` layers that maintain hidden states across turns within an episode:

- **GRU₁** receives the flattened CNN features and produces hidden state `h₁`.
- **GRU₂** receives the output of GRU₁ and produces hidden state `h₂`.

The two hidden states are stored as a single concatenated tensor of shape `[batch, gru1_hidden_size + gru2_hidden_size]` and split internally during the forward pass. Both hidden states are zero-initialised at the start of each episode.

This stacking gives the network deeper temporal reasoning — the first GRU can learn to compress and filter the spatial features over time, while the second GRU can learn higher-level temporal patterns from that filtered representation.

### Policy Head

A single `Linear` layer maps the second GRU's hidden state (`h₂`) to logits over the 5-action space:

| Index | Action |
|-------|--------|
| 0 | `MOVE_FORWARD` |
| 1 | `TURN_LEFT` |
| 2 | `TURN_RIGHT` |
| 3 | `FIRE` |
| 4 | `PASS` |

## Model Configuration

The `MkIIModelConfig` (Pydantic model, frozen) controls the network architecture. It is serialised as `model_config.json` in the model directory.

Example:

```json
{
  "patch_size": 9,
  "cnn_channels": [32, 64],
  "gru1_hidden_size": 128,
  "gru2_hidden_size": 64,
  "conv_kernel_size": 3,
  "pool_kernel_size": 2,
  "pool_stride": 2
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patch_size` | int | `9` | Side length of the input visibility patch (must be ≥ 3) |
| `cnn_channels` | tuple[int, ...] | `(32, 64)` | Output channels for each conv block; length determines number of blocks |
| `gru1_hidden_size` | int | `128` | Dimensionality of the first GRU hidden state |
| `gru2_hidden_size` | int | `64` | Dimensionality of the second GRU hidden state |
| `conv_kernel_size` | int | `3` | Kernel size for each `Conv2d` layer (must be ≥ 1; odd values recommended for symmetric padding) |
| `pool_kernel_size` | int | `2` | Kernel size for each `MaxPool2d` layer (must be ≥ 1) |
| `pool_stride` | int | `2` | Stride for each `MaxPool2d` layer (must be ≥ 1) |

**Note:** The total hidden state size is `gru1_hidden_size + gru2_hidden_size`. Increasing either GRU size increases model capacity but also parameter count and inference cost. The `patch_size` must match the visibility patch size used by the game engine.
