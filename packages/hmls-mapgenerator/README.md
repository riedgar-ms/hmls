# hmls.mapgenerator

A randomised game map generator with pluggable strategies and an interactive
Textual TUI.  Part of the [hmls](../../README.md) workspace.

## Quick start

From the workspace root:

```bash
# Install all workspace packages (only needed once)
uv sync --all-packages

# Launch the TUI
uv run hmls-mapgen
# or equivalently:
uv run python -m hmls.mapgenerator
```

The TUI lets you configure grid size, impassable fraction, seed,
strategy, and strategy-specific parameters, then press **G** to generate.

## Programmatic usage

```python
from hmls.mapgenerator import generate_map, BlobAndLineStrategy, PerlinNoiseStrategy

# Generate a 40×20 map with 30% impassable terrain (default strategy)
game_map = generate_map(40, 20, impassable_fraction=0.3, seed=42)

# Use a specific strategy
game_map = generate_map(60, 30, strategy=PerlinNoiseStrategy(scale=10.0, octaves=4))

# Connected obstacles (best-effort bridging of impassable regions)
game_map = generate_map(40, 20, connected_obstacles=True)

# Print a simple text representation
print(game_map)
```

`generate_map()` always returns a `GameMap` with fully 4-connected passable
terrain, regardless of strategy.

## Strategies

| Strategy | Description | Key parameters |
|---|---|---|
| **Blob & Line** (default) | Places elliptical blobs and thick Bresenham lines | `shape` (0.0 = linear, 1.0 = circular) |
| **Perlin Noise** | Fractal noise thresholding (pure Python) | `scale`, `octaves` |

Both strategies are registered in `STRATEGY_REGISTRY` and appear
automatically in the TUI.

### Adding a custom strategy

Implement the `MapStrategy` protocol and register it:

```python
from hmls.mapgenerator.generators.base import STRATEGY_REGISTRY, StrategyParam

class MyStrategy:
    """My custom obstacle placement strategy."""

    params: tuple[StrategyParam, ...] = (
        StrategyParam("density", "Density", float, 0.5, 0.0, 1.0),
    )

    def __init__(self, density: float = 0.5) -> None:
        self.density = density

    def place_obstacles(self, game_map, fraction, rng):
        ...  # your logic here

STRATEGY_REGISTRY["My Strategy"] = MyStrategy
```

## Coordinate convention

The package uses **(x, y)** coordinates throughout:
- **x** = column (0 = left edge)
- **y** = row (0 = top edge)

This matches `hmls.core.GameMap`.

## Running tests

```bash
uv run pytest packages/hmls-mapgenerator/tests -v
```
