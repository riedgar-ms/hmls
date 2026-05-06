# hmls-reinforcetrainer

REINFORCE policy gradient trainer for HMLS single-tank neural network models.

## Overview

This package trains `singletanknn` neural network models by having two networks play against each other on randomly generated maps. It uses the REINFORCE algorithm (policy gradient with baseline normalisation) to improve each network's policy.

## Installation

From the workspace root:

```bash
uv sync
```

## Model Directory Structure

Each model directory **must** contain two JSON configuration files before training starts:

```
models/player_a/
├── model_config.json    # Neural network architecture
├── reward_config.json   # Reward function parameters
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

### `reward_config.json`

Defines the reward shaping parameters. Example:

```json
{
  "hit_reward": 0.5,
  "death_penalty": -1.0,
  "win_reward": 1.0,
  "loss_penalty": -1.0,
  "step_penalty": -0.01,
  "exploration_bonus": 0.02
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hit_reward` | float | `0.5` | Reward for hitting an enemy tank |
| `death_penalty` | float | `-1.0` | Penalty when the player's tank dies |
| `win_reward` | float | `1.0` | Reward for winning the game |
| `loss_penalty` | float | `-1.0` | Penalty for losing the game |
| `step_penalty` | float | `-0.01` | Per-step penalty (encourages faster play) |
| `exploration_bonus` | float | `0.02` | Reward per newly discovered cell |

Each model uses its own reward configuration, so you can experiment with different reward shaping strategies.

## Usage

### Basic self-play training (both models learn)

```bash
uv run hmls-reinforcetrainer \
    --model-a-dir models/player_a \
    --model-b-dir models/player_b
```

Both model directories must contain `model_config.json` and `reward_config.json`. If no `model.pt` exists, a fresh model is created using the architecture from `model_config.json`.

### Train one model against a frozen opponent

```bash
uv run hmls-reinforcetrainer \
    --model-a-dir models/trainee \
    --model-b-dir models/frozen_opponent \
    --freeze-b
```

### Full configuration example

```bash
uv run hmls-reinforcetrainer \
    --model-a-dir models/player_a \
    --model-b-dir models/player_b \
    --map-width 25 \
    --map-height 25 \
    --impassable-fraction 0.25 \
    --map-strategy "Blob & Line" \
    --games-per-map 20 \
    --total-maps 500 \
    --max-turns 300 \
    --sample-game-dir output/samples \
    --sample-game-interval 100 \
    --save-weights-interval 200 \
    --learning-rate 0.0005 \
    --gamma 0.995 \
    --seed 42
```

### Running as a Python module

```bash
uv run python -m hmls.reinforcetrainer --model-a-dir models/a --model-b-dir models/b
```

## Configuration Reference

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Model A directory | `--model-a-dir` | *(required)* | Path to model A directory (must contain config files) |
| Model B directory | `--model-b-dir` | *(required)* | Path to model B directory (must contain config files) |
| Freeze A | `--freeze-a` | `false` | Don't train model A (use as fixed opponent) |
| Freeze B | `--freeze-b` | `false` | Don't train model B (use as fixed opponent) |
| Map width | `--map-width` | `20` | Width of generated maps (≥5) |
| Map height | `--map-height` | `20` | Height of generated maps (≥5) |
| Impassable fraction | `--impassable-fraction` | `0.3` | Fraction of cells that are impassable (0.0–0.8) |
| Map strategy | `--map-strategy` | `"Blob & Line"` | Map generation strategy name |
| Games per map | `--games-per-map` | `10` | Games played on each map before regeneration |
| Total maps | `--total-maps` | `100` | Total number of maps to generate |
| Max turns | `--max-turns` | `200` | Maximum turns per game before draw |
| Sample game dir | `--sample-game-dir` | `sample_games/` | Directory for sample replay files |
| Sample game interval | `--sample-game-interval` | `50` | Save a sample game every N games |
| Save weights interval | `--save-weights-interval` | `100` | Save model weights every N games |
| Learning rate | `--learning-rate` | `0.001` | Adam optimizer learning rate |
| Gamma (γ) | `--gamma` | `0.99` | Discount factor for return computation |
| Seed | `--seed` | `None` | Random seed for reproducibility |

## Output

### Model weights

Models are saved as `model.pt` files in the respective model directories. They are saved:
- Periodically (every `--save-weights-interval` games)
- At the end of training

If training is interrupted, the most recent periodic save will be available to resume from.

### Sample games

Sample games are saved as JSON files in the `--sample-game-dir` directory, named `game_000050.json`, `game_000100.json`, etc. These files are in the standard `GameResult` format and can be viewed using the replay viewer:

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

- **Start small**: Use `--total-maps 10 --games-per-map 5` to verify everything works before long runs.
- **Map variety**: More maps with fewer games each gives broader generalisation. Fewer maps with more games gives deeper exploitation of each map's structure.
- **Self-play vs frozen**: Self-play trains faster initially but can lead to co-adapted strategies. Training against a frozen opponent is more stable but may converge slower.
- **Monitoring**: Check sample games regularly in the replay viewer to see if agents are learning meaningful behaviour (exploration, combat, movement).
