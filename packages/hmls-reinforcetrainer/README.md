# hmls-reinforcetrainer

REINFORCE policy gradient trainer for HMLS single-tank neural network models.

## Overview

This package trains `singlemki` neural network models by having two networks play against each other on randomly generated maps. It uses the REINFORCE algorithm (policy gradient with baseline normalisation) to improve each network's policy.

## Installation

From the workspace root:

```bash
uv sync
```

## Model Directory Structure

Each model directory **must** contain a JSON configuration file before training starts:

```
models/player_a/
├── model_config.json    # Neural network architecture
└── model.pt             # (created during training)
```

### `model_config.json`

Defines the neural network architecture. Example:

```json
{
  "patch_size": 9,
  "cnn_channels": [32, 64],
  "gru_hidden_size": 128
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patch_size` | int | `9` | Side length of input patch (≥ 3) |
| `cnn_channels` | list[int] | `[32, 64]` | Output channels for each conv layer |
| `gru_hidden_size` | int | `128` | Dimensionality of the GRU hidden state |

The two models may have different `cnn_channels` and `gru_hidden_size`, but **`patch_size` must match** between them.

## Usage

### Basic self-play training (both models learn)

Create a configuration file (e.g. `train_config.json`):

```json
{
  "model_a": { "dir": "models/player_a" },
  "model_b": { "dir": "models/player_b" }
}
```

Then run:

```bash
uv run hmls-reinforcetrainer train_config.json
```

Both model directories must contain `model_config.json`. If no `model.pt` exists, a fresh model is created using the architecture from `model_config.json`. Reward shaping is configured per-model in `train_config.json` (see below).

### Train one model against a frozen opponent

```json
{
  "model_a": { "dir": "models/trainee", "train": true },
  "model_b": { "dir": "models/frozen_opponent", "train": false }
}
```

### Full configuration example

```json
{
  "model_a": {
    "dir": "models/player_a",
    "train": true,
    "reward": {
      "reward_type": "basic",
      "fire_hit_reward": 0.5,
      "exploration_reward": 0.05
    }
  },
  "model_b": {
    "dir": "models/player_b",
    "train": true,
    "reward": {
      "reward_type": "basic",
      "fire_hit_reward": 0.3
    }
  },
  "map": {
    "min_size": 15,
    "max_size": 25,
    "impassable_fraction": 0.25,
    "strategy": "Blob & Line"
  },
  "game": {
    "games_per_map": 20,
    "total_maps": 500,
    "max_turns": 300,
    "patch_size": 9
  },
  "output": {
    "sample_game_dir": "output/samples",
    "sample_game_interval": 100,
    "save_weights_interval": 200
  },
  "hyperparameters": {
    "learning_rate": 0.0005,
    "gamma": 0.995,
    "seed": 42
  }
}
```

```bash
uv run hmls-reinforcetrainer full_config.json
```

### Running as a Python module

```bash
uv run python -m hmls.reinforcetrainer train_config.json
```

## Configuration Reference

The configuration file is a JSON object with the following sections. All sections except `model_a` and `model_b` are optional and use sensible defaults.

**Path convention:** All paths in the JSON file should use unix-style forward slashes (e.g. `"output/samples"`), even on Windows. They are converted to platform-native paths automatically.

### `model_a` / `model_b` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | string | *(required)* | Path to model directory (must contain config files) |
| `train` | bool | `true` | Whether to train this model (false = frozen opponent) |
| `reward` | object | *(basic defaults)* | Reward shaping config (see below) |

#### `reward` (nested in `model_a` / `model_b`)

Each model can have its own reward shaping parameters. The `reward_type` field selects which reward function to use (currently only `"basic"` is supported). All fields have sensible defaults and may be omitted.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reward_type` | string | `"basic"` | Discriminator for the reward function type |
| `fire_hit_reward` | float | `0.5` | Reward for hitting an enemy tank |
| `death_reward` | float | `-1.0` | Reward (negative) when the player's tank dies |
| `win_reward` | float | `1.0` | Reward for winning the game |
| `loss_reward` | float | `-1.0` | Reward (negative) for losing the game |
| `step_reward` | float | `-0.01` | Per-step reward (negative to encourage faster play) |
| `exploration_reward` | float | `0.02` | Reward per newly discovered cell |
| `invalid_move_reward` | float | `-0.1` | Reward (negative) for attempting an invalid action |
| `fire_miss_reward` | float | `-0.05` | Reward (negative) for firing and missing |
| `fire_neglect_reward` | float | `-0.1` | Reward (negative) for not firing when an enemy is directly ahead |
| `consecutive_miss_reward` | float | `0.0` | Escalating reward multiplier for consecutive fire misses (typically negative) |
| `pass_reward` | float | `-0.02` | Reward (negative) for deliberately choosing to pass |
| `enemy_in_cone_reward` | float | `0.01` | Per-enemy reward for visible enemies in the forward cone |
| `turn_left_reward` | float | `0.0` | Reward for choosing to turn left |
| `turn_right_reward` | float | `0.0` | Reward for choosing to turn right |
| `move_forward_reward` | float | `0.0` | Reward for choosing to move forward |
| `consecutive_turn_reward` | float | `0.0` | Escalating reward multiplier for consecutive turns (typically negative) |
| `consecutive_pass_reward` | float | `0.0` | Escalating reward multiplier for consecutive passes (typically negative) |

### `map`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_size` | int | `15` | Minimum width/height of generated maps (≥5) |
| `max_size` | int | `25` | Maximum width/height of generated maps (≥ min_size) |
| `impassable_fraction` | float | `0.3` | Fraction of cells that are impassable (0.0–0.8) |
| `strategy` | string | `"Blob & Line"` | Map generation strategy name |

Each time a new map is generated, the width and height are chosen independently and uniformly at random from the inclusive range [`min_size`, `max_size`].

### `game`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `games_per_map` | int | `10` | Games played on each map before regeneration |
| `total_maps` | int | `100` | Total number of maps to generate |
| `max_turns` | int | `200` | Maximum turns per game before draw |
| `patch_size` | int | `9` | Side length of visibility patches (must be odd, ≥ 3) |

The `patch_size` must match the `patch_size` in both models' `model_config.json`. A mismatch will cause a startup error.

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_game_dir` | string | `"sample_games"` | Directory for sample replay files |
| `sample_game_interval` | int | `50` | Save a sample game every N games |
| `save_weights_interval` | int | `100` | Save model weights every N games |

### `hyperparameters`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | float | `0.001` | Adam optimizer learning rate (>0) |
| `gamma` | float | `0.99` | Discount factor for return computation (0–1] |
| `seed` | int \| null | `null` | Random seed for reproducibility |

## Output

### Model weights

Models are saved as `model.pt` files in the respective model directories. They are saved:
- Periodically (every `save_weights_interval` games)
- At the end of training

If training is interrupted, the most recent periodic save will be available to resume from.

### Sample games

Sample games are saved as JSON files in the `sample_game_dir` directory, named `game_000050.json`, `game_000100.json`, etc. These files are in the standard `GameResult` format and can be viewed using the replay viewer:

```bash
uv run hmls-replayviewer output/samples/game_000050.json
```

### Progress output

The trainer prints progress to stdout after each map is completed, showing:
- Completion percentage
- Games played
- Win/loss/draw counts
- Average policy gradient loss (for models that are training)

## Training tips

- **Start small**: Use `"total_maps": 10, "games_per_map": 5` to verify everything works before long runs.
- **Map variety**: More maps with fewer games each gives broader generalisation. Fewer maps with more games gives deeper exploitation of each map's structure.
- **Self-play vs frozen**: Self-play trains faster initially but can lead to co-adapted strategies. Training against a frozen opponent is more stable but may converge slower.
- **Monitoring**: Check sample games regularly in the replay viewer to see if agents are learning meaningful behaviour (exploration, combat, movement).
