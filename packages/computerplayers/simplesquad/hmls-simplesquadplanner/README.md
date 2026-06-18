# hmls-simplesquadplanner

Set-pooling planner model for the simple squad architecture.

## Architecture

The planner observes all alive friendly tanks and assigns a discrete tactical order to each:

1. **Per-tank CNN encoder** (shared weights): Extracts spatial features from each tank's encoded 5-channel patch
2. **Metadata encoding**: Normalised position (x/map_width, y/map_height) + one-hot direction → concatenated with CNN features
3. **Tank feature projection**: Linear layer maps [CNN ∥ metadata] → fixed-size feature vector
4. **Set-pooling**: Mean-pool across all alive tank features → global team context vector
5. **Per-tank decision MLP**: [tank_i features ∥ global context] → order logits (one distribution per alive tank)

### Handling variable tank counts

The set-pooling approach naturally handles any number of alive tanks (1 to max_tanks):
- When a tank dies, it is simply excluded from the next planning round
- No padding, masking, or fixed-size input tensors required
- The global context adapts automatically (mean over fewer tanks)

### Inputs

The planner uses information from `PlayerView`:
- **Patch grids** (`TankPatch.grid`): egocentric NxN visibility for each alive tank
- **Absolute positions** (`TankPatch.position`): normalised to [0, 1] for map-size invariance
- **Directions** (`TankPatch.direction`): one-hot encoded (4 cardinal directions)

Absolute positions enable strategic reasoning: tank spacing, flanking geometry, map coverage.

## Configuration

```json
{
    "model_id": "hmls.simplesquadplanner",
    "patch_size": 9,
    "num_orders": 8,
    "max_tanks": 5,
    "cnn_channels": [32, 64],
    "cnn_kernel_size": 3,
    "pool_kernel_size": 2,
    "pool_stride": 2,
    "tank_feature_dim": 64,
    "mlp_hidden_dim": 64
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 9 | Side length of visibility patch |
| `num_orders` | 8 | Number of orders the planner can issue |
| `max_tanks` | 5 | Maximum tanks per team supported |
| `cnn_channels` | [32, 64] | Per-tank CNN encoder channels |
| `cnn_kernel_size` | 3 | Conv2d kernel size |
| `pool_kernel_size` | 2 | MaxPool2d kernel size |
| `pool_stride` | 2 | MaxPool2d stride |
| `tank_feature_dim` | 64 | Per-tank encoded feature dimension |
| `mlp_hidden_dim` | 64 | Decision MLP hidden layer size |

## Usage

This package is not used standalone — it is consumed by `hmls-simplesquadplayer` as part of the composite squad player. See that package for end-to-end usage.
