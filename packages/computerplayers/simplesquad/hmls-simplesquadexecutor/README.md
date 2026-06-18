# hmls-simplesquadexecutor

CNN + order embedding → GRU → action logits executor model for the simple squad architecture.

## Architecture

The executor translates a planner's high-level order into low-level actions using local egocentric information:

1. **CNN encoder**: Extracts spatial features from the encoded 5-channel visibility patch
2. **Order embedding**: Learned vector representation of the discrete order (one of 8 tactical orders)
3. **Concatenation**: CNN features + order embedding joined into a single vector
4. **GRU cell**: Maintains temporal memory across turns within an episode
5. **Policy head**: Linear layer producing logits over 5 actions (move forward, turn left, turn right, fire, pass)

The executor does NOT receive absolute position — it operates purely from local context, translating orders into tactical actions based on what it can see.

## Configuration

```json
{
    "model_id": "hmls.simplesquadexecutor",
    "patch_size": 9,
    "num_orders": 8,
    "cnn_channels": [32, 64],
    "gru_hidden_size": 128,
    "order_embedding_dim": 16,
    "conv_kernel_size": 3,
    "pool_kernel_size": 2,
    "pool_stride": 2
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 9 | Side length of visibility patch (must be odd, ≥ 3) |
| `num_orders` | 8 | Number of discrete orders accepted |
| `cnn_channels` | [32, 64] | Output channels per conv layer |
| `gru_hidden_size` | 128 | GRU hidden state dimensionality |
| `order_embedding_dim` | 16 | Learned embedding size per order |
| `conv_kernel_size` | 3 | Conv2d kernel size |
| `pool_kernel_size` | 2 | MaxPool2d kernel size |
| `pool_stride` | 2 | MaxPool2d stride |

## Usage

This package is not used standalone — it is consumed by `hmls-simplesquadplayer` as part of the composite squad player. See that package for end-to-end usage.
