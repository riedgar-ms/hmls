# hmls

Assorted Experiments in AI

## Packages

| Package | Description |
|---|---|
| `hmls.core` | Core data types (`GameMap`, `CellType`), game engine, and visibility system |
| `hmls.protocol` | Wire protocol models for server/client WebSocket communication |
| `hmls.server` | Headless WebSocket game server (FastAPI + Uvicorn) |
| `hmls.client` | WebSocket game client with Textual TUI and automapper |
| `hmls.observer` | TUI observer client — connects to a running server and displays the full game map and event log in real-time (no fog-of-war) |
| `hmls.testharness` | Interactive TUI for manually testing tank game behaviour |
| `hmls.replayviewer` | TUI replay viewer for game history files |
| `hmls.mapgenerator` | Randomised map generation with Textual TUI |
| `hmls.uxcommon` | Shared TUI widgets and styles |

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
| `--patch-size N` | Visibility patch size, odd ≥ 3 (default 9) |

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

### Replay Viewer

The replay viewer lets you step through a saved game history file,
viewing the full game state at each turn.

```bash
uv run hmls-replayviewer path/to/history.json
```

**Arguments:**

| Argument | Description |
|---|---|
| `history_file` | Path to a JSON game history file (as saved by the test harness or server) |

### Game Server

The game server is a headless WebSocket server (FastAPI + Uvicorn) that
hosts a single game, accepting two player clients (one per team) and any
number of observer clients. It logs events to the console but has no TUI
of its own — use `hmls-observer` to watch the game visually.

```bash
uv run hmls-server path/to/map.json 3
```

**Arguments:**

| Argument | Description |
|---|---|
| `map_file` | Path to a JSON map file (as saved by `hmls-mapgen`) |
| `tanks_per_player` | Number of tanks each team starts with |
| `--port N` | WebSocket server port (default 8765) |
| `--seed N` | Random seed for tank placement (optional) |
| `--max-turns N` | Maximum individual turns before the game ends (default 200) |
| `--patch-size N` | Visibility patch size, odd ≥ 3 (default 9) |

### Game Observer

The observer connects to a running server and displays a full god-view
map alongside a real-time event log. Observers see the complete game
state without fog-of-war restrictions and do not affect gameplay.

```bash
uv run hmls-observer --url ws://localhost:8765/ws --name "Spectator"
```

**Arguments:**

| Argument | Description |
|---|---|
| `--url URL` | WebSocket server URL (default `ws://localhost:8765/ws`) |
| `--name NAME` | Display name for this observer (default "Observer") |

### Game Client

The game client connects to a running server and provides an interactive
TUI with an automapper. As your tanks explore, the automapper reveals
passable/impassable terrain, removing fog-of-war from previously seen areas.

```bash
uv run hmls-client ws://localhost:8765/ws --name "Alice"
```

**Arguments:**

| Argument | Description |
|---|---|
| `server_url` | WebSocket server URL (e.g. `ws://localhost:8765/ws`) |
| `--name NAME` | Player name sent to the server (default "Player") |

**Controls** (same as test harness):

| Key | Action |
|---|---|
| `W` | Move forward |
| `A` | Turn left |
| `D` | Turn right |
| `Space` | Fire |
| `Tab` | Pass (skip turn) |
| `Q` | Quit |

## Development

```bash
uv run ruff format --check .   # check formatting
uv run ruff check .            # lint
uv run mypy .                  # type check
uv run pytest                  # run all tests
```
