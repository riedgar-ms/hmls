# Playing hmls

This guide walks through playing a networked game.
For installation, see the [main README](../README.md#getting-started).

## Map generation

Generate a map using the interactive TUI:

```bash
uv run hmls-mapgenerator
```

This opens a Textual UI where you can configure grid dimensions, impassable fraction, obstacle connectivity, and generation strategy.
Press **G** to generate and **S** to save the map as a JSON file (e.g. `map.json`).

## Starting the server

Launch the game server, providing a map file and the number of tanks per
player:

```bash
uv run hmls-server map.json 3
```

The server waits for two players to connect before starting the game.


## Connecting Players

Each player connects a client to the server's WebSocket endpoint.
You need two clients (in separate terminals) for a full game:

```bash
# Terminal 1 — Player A
uv run hmls-client ws://localhost:8765/ws --name Alice

# Terminal 2 — Player B
uv run hmls-client ws://localhost:8765/ws --name Bob
```

Once both players are connected, the game starts. The client displays an
automapped view of explored terrain and accepts keyboard input:

| Key | Action |
|-----|--------|
| `W` | Move forward |
| `A` | Turn left |
| `D` | Turn right |
| `Space` | Fire |
| `Tab` | Pass (skip turn) |
| `Q` | Quit |

## Connecting Observers

Observers see the full map (no fog-of-war) and can spectate a game in progress.
Connect with:

```bash
uv run hmls-observer ws://localhost:8765/ws --name Spectator
```

Observers can connect at any time — before or after the game starts.