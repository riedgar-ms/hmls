# hmls-reinforcetrainer

REINFORCE policy gradient trainer for HMLS single-tank neural network models.

## Overview

This package trains `singletanknn` neural network models by having two networks play against each other on randomly generated maps. It uses the REINFORCE algorithm (policy gradient with baseline normalisation) to improve each network's policy.

## Installation

From the workspace root:

```bash
uv sync
```

## Usage

### Basic self-play training (both models learn)

```bash
uv run hmls-reinforcetrainer \
    --model-a-dir models/player_a \
    --model-b-dir models/player_b
```

Both models will be created fresh if the directories are empty, or loaded from existing `model.pt` files if present.

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
| Model A directory | `--model-a-dir` | *(required)* | Path to directory for model A weights |
| Model B directory | `--model-b-dir` | *(required)* | Path to directory for model B weights |
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
