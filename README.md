# hmls

Assorted Experiments in AI

## Packages

| Package | Description |
|---|---|
| `hmls.core` | Core data types (`GameMap`, `CellType`) |
| `hmls.mapgenerator` | Randomised map generation with Textual TUI |
| `hmls.testharness` | Interactive TUI for manually testing tank game behaviour |

## Getting started

```bash
# Clone and install
git clone <repo-url>
cd hmls
uv sync --all-packages
```

### Map Generator TUI

```bash
uv run hmls-mapgen
```

See [packages/hmls-mapgenerator/README.md](packages/hmls-mapgenerator/README.md)
for full documentation.

### Test Harness TUI

The test harness lets you play through a tank game interactively,
controlling every tank yourself. It shows a god-view map alongside
each player's fog-of-war patches.

```bash
# Generate a map first, then run the harness with it
uv run hmls-testharness path/to/map.json 3
```

**Arguments:**

| Argument | Description |
|---|---|
| `map_file` | Path to a JSON map file (as saved by `hmls-mapgen`) |
| `tanks_per_player` | Number of tanks each of the two teams starts with |
| `--seed N` | Random seed for tank placement (optional) |
| `--max-turns N` | Maximum individual turns before the game ends (default 200) |
| `--patch-size N` | Visibility patch size, odd ≥ 3 (default 7) |

**Controls:**

| Key | Action |
|---|---|
| `W` | Move forward |
| `A` | Turn left |
| `D` | Turn right |
| `Space` | Fire |
| `Tab` | Pass (skip turn) |
| `Q` | Quit |

When the game ends, a summary is shown and you are prompted to save the
full game history as JSON.

## Development

```bash
uv run ruff format --check .   # check formatting
uv run ruff check .            # lint
uv run mypy .                  # type check
uv run pytest                  # run all tests
```
