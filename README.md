# hmls

Assorted Experiments in AI

## Packages

| Package | Description |
|---|---|
| `hmls.core` | Core data types (`GameMap`, `CellType`) |
| `hmls.mapgenerator` | Randomised map generation with Textual TUI |

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

## Development

```bash
uv run ruff format --check .   # check formatting
uv run ruff check .            # lint
uv run mypy .                  # type check
uv run pytest                  # run all tests
```
