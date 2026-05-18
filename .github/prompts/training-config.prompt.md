# Training Configuration

Create or modify a reinforcement learning training configuration for the hmls trainer.

## Configuration File Structure

The trainer takes a JSON config file. Key sections:

### Model Selection

```json
{
  "model_a": { "dir": "path/to/weights_a", "train": true },
  "model_b": { "dir": "path/to/weights_b", "train": false }
}
```

- Each model directory must contain `model_config.json` (architecture definition).
- Set `"train": true` to update weights, `false` to freeze as opponent.
- Both models' `patch_size` must match.

### Map Generation

```json
{
  "map": {
    "min_size": 10,
    "max_size": 15,
    "impassable_fraction": 0.1,
    "strategies": ["Blob", "Line"]
  }
}
```

### Game Settings

```json
{
  "total_maps": 100,
  "games_per_map": 10,
  "max_turns": 400
}
```

### Reward Shaping

Categories: `actions`, `firing`, `game_state`, `exploration`, `situational`.
Keep values between −1 and 1. Win/loss should be the largest magnitude signals.

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `learning_rate` | 0.001 | Lower for fine-tuning |
| `gamma` | 0.99 | Discount factor |
| `entropy_coeff` | 0.01 | Higher = more exploration |
| `max_grad_norm` | 0.5 | Gradient clipping |

### Lethargy Policy

Prevents tanks from spinning indefinitely. Set a `consecutive_turn_limit` (default 10).

## Running

```shell
uv run hmls-reinforcetrainer path/to/config.json
```

## Tips

- Start with the sample config: `sample_configs/single_tanks/config.json`
- Use small runs first (`total_maps: 10`, `games_per_map: 5`) to verify setup.
- Review sample game replays with `uv run hmls-replayviewer`.
- See `packages/hmls-reinforcetrainer/README.md` for full reference.
