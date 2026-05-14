# hmls-server

Headless WebSocket game server for the HMLS tank game.
Accepts two player connections and any number of read-only observers,
then drives a turn-by-turn game to completion.

## Usage

### Running the server

```bash
# Via the installed entry-point
uv run hmls-server MAP_FILE TANKS_PER_PLAYER [OPTIONS]

# Or as a Python module
uv run python -m hmls.server MAP_FILE TANKS_PER_PLAYER [OPTIONS]
```

### CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `map_file` | positional | *(required)* | Path to a JSON map file |
| `tanks_per_player` | positional | *(required)* | Number of tanks per team |
| `--patch-size` | int | 9 | Fog-of-war visibility patch size |
| `--max-turns` | int | 200 | Maximum turns before the game is declared a draw |
| `--seed` | int | *(random)* | Random seed for tank placement |
| `--port` | int | 8765 | WebSocket server port |
| `--history-file` | path | `history.json` | Where to save game history JSON after the game ends |
| `--no-history` | flag | — | Disable saving game history (mutually exclusive with `--history-file`) |

### Connecting

All clients connect to a single WebSocket endpoint at `/ws`.
The first message a client sends determines its role:

- **Player** — send a `JoinMessage` (`{"type": "join", "player_name": "…"}`).
  Players are assigned to teams in connection order: the first becomes Team A,
  the second Team B.
- **Observer** — send an `ObserveMessage` (`{"type": "observe", "observer_name": "…"}`).
  Observers can connect at any time and immediately receive the current game
  state.

## Architecture

### Component overview

The server is composed of four main components, wired together in `app.py`:

```
┌──────────────┐  events   ┌──────────┐  protocol msgs   ┌─────────────┐
│  Game        │ ────────► │  Event   │ ───────────────► │  Network    │ ◄──► WebSocket
│  Orchestrator│           │  Bus     │                  │  Manager    │      clients
└──────────────┘           └──────────┘                  └─────────────┘
       │                                                        │
       │              ┌───────────────┐                         │
       └─────────────►│ RemotePlayer  │◄────────────────────────┘
                      │ (×2)          │
                      └───────────────┘
```

- **`EventBus`** (`events.py`) — A lightweight async pub/sub bus.  Subscribers
  register by event type; when an event is emitted, all matching callbacks are
  invoked sequentially.  This decouples the game logic from the networking
  layer so neither depends on the other directly.

- **`GameOrchestrator`** (`orchestrator.py`) — Owns the `GameEngine` and drives
  the turn-by-turn game loop.  It waits for both players to connect, then
  iterates: build the fog-of-war view, request an action from the current
  player, execute the engine step, and emit events for every phase
  (`GameStartedEvent`, `YourTurnEvent`, `StateUpdatedEvent`,
  `TurnCompletedEvent`, `GameOverEvent`).  It never touches WebSocket
  connections directly.

- **`NetworkManager`** (`network_manager.py`) — Owns all WebSocket connections
  (players and observers).  It subscribes to `EventBus` events and translates
  them into the appropriate protocol messages for each client.  It also handles
  incoming client messages, routing `ActionMessage`s to the correct
  `RemotePlayer`.

- **`RemotePlayer`** (`remote_player.py`) — Bridges the gap between the async
  WebSocket world and the synchronous `Player` interface expected by the game
  engine.  Each `RemotePlayer` holds an `asyncio.Future` that the orchestrator
  awaits and the `NetworkManager` resolves when an `ActionMessage` arrives from
  the client.

### Data flow for a single turn

1. The **Orchestrator** builds the `PlayerView` and calls
   `RemotePlayer.request_action()` to set up a pending future.
2. The **Orchestrator** emits a `YourTurnEvent`.
3. The **EventBus** delivers this to the **NetworkManager**, which sends a
   `YourTurnMessage` to the acting player's WebSocket.
4. The client responds with an `ActionMessage`.
5. The **NetworkManager** calls `RemotePlayer.submit_action()`, resolving the
   future.
6. The **Orchestrator**'s `await` completes; it calls `engine.step()`.
7. The **Orchestrator** emits `StateUpdatedEvent` and `TurnCompletedEvent`.
8. The **NetworkManager** broadcasts the state to observers and sends the
   `TurnResultMessage` to the acting player.

### Edge cases

- **Player disconnection mid-game** — When a player's WebSocket drops, the
  `NetworkManager` emits a `PlayerDisconnectedEvent`.  The orchestrator handles
  this by ending the game and declaring the remaining player the winner.

- **Action timeout** — If a player does not respond within 300 seconds, the
  orchestrator logs a warning and terminates the game.

- **Late-joining observers** — An observer that connects after the game has
  started immediately receives a `GameStartMessage` (with the full map) and,
  if available, a `StateUpdateMessage` with the latest game state.  This means
  observers can join and leave freely without affecting the game.

- **Game full** — If a third client sends a `JoinMessage` when both player
  slots are taken, it receives an `ErrorMessage` and the connection is closed.

- **Invalid client messages** — Malformed JSON or unexpected message types
  result in an `ErrorMessage` sent back to the client.  The connection is
  *not* dropped, so the client can retry.

- **Observer disconnection** — Observers that disconnect are silently removed
  from the broadcast list.  This has no effect on the game.

- **Event handler errors** — If an event callback raises an exception, it is
  logged and the remaining subscribers for that event still execute.  This
  prevents a bug in one handler from cascading to others.
