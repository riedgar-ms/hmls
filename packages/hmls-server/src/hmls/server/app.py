"""Main server application: FastAPI WebSocket server + Textual TUI.

The server runs Uvicorn in a background thread and uses Textual as the
main application for displaying the full map and game log.
"""

from __future__ import annotations

import asyncio
import random
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, RichLog, Static

from hmls.core.engine import GameEngine
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.core.visibility import TankInfo, build_player_view
from hmls.protocol import (
    ActionMessage,
    AssignMessage,
    ClientMessage,
    ErrorMessage,
    GameOverMessage,
    JoinMessage,
    TurnResultMessage,
    WaitingMessage,
    YourTurnMessage,
)
from hmls.server.cli import parse_args
from hmls.server.remote_player import RemotePlayer
from hmls.uxcommon.widgets.map_view import MapView

# ── Type adapters for protocol parsing ────────────────────────────────

_client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)

# ── Helpers ───────────────────────────────────────────────────────────


def _load_map(path: Path) -> GameMap:
    """Load a GameMap from a JSON file."""
    if not path.exists():
        print(f"Error: map file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        text = path.read_text(encoding="utf-8")
        return GameMap.model_validate_json(text)
    except Exception as exc:
        print(f"Error loading map: {exc}", file=sys.stderr)
        sys.exit(1)


def _place_tanks(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks randomly on passable cells for two teams."""
    total_needed = tanks_per_player * 2
    passable_positions = [
        Position(x, y) for x, y in game_map.all_positions() if game_map[x, y] == CellType.PASSABLE
    ]
    if len(passable_positions) < total_needed:
        print(
            f"Error: need {total_needed} passable cells but map only has {len(passable_positions)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    chosen = rng.sample(passable_positions, total_needed)
    directions = list(Direction)
    tanks: list[Tank] = []

    for team_idx, team_name in enumerate(["A", "B"]):
        for i in range(tanks_per_player):
            pos = chosen[team_idx * tanks_per_player + i]
            tanks.append(
                Tank(
                    id=f"{team_name}{i + 1}",
                    team=team_name,
                    position=pos,
                    direction=rng.choice(directions),
                )
            )

    return tanks


# ── Game session state ────────────────────────────────────────────────


class GameSession:
    """Manages the game state and communication with connected clients.

    Attributes:
        game_map: The map being played on.
        tanks: All tanks in the game.
        engine: The game engine (created once both players connect).
        players: Mapping of team name to RemotePlayer.
        websockets: Mapping of team name to WebSocket connection.
    """

    def __init__(
        self,
        game_map: GameMap,
        tanks: list[Tank],
        max_turns: int,
        patch_size: int,
        log_callback: Callable[[str], None],
    ) -> None:
        self.game_map = game_map
        self.tanks = tanks
        self.max_turns = max_turns
        self.patch_size = patch_size
        self._log = log_callback

        self.players: dict[str, RemotePlayer] = {
            "A": RemotePlayer("A"),
            "B": RemotePlayer("B"),
        }
        self.websockets: dict[str, WebSocket] = {}
        self.player_names: dict[str, str] = {}
        self._both_connected: asyncio.Event = asyncio.Event()
        self._game_over: bool = False
        self.engine: GameEngine | None = None
        self._state_callback: Callable[[GameState], None] | None = None

    def set_state_callback(self, callback: Callable[[GameState], None]) -> None:
        """Set a callback invoked after each state change."""
        self._state_callback = callback

    @property
    def game_over(self) -> bool:
        """Whether the game has ended."""
        return self._game_over

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection from a client."""
        await websocket.accept()

        # Determine which team slot is free.
        if "A" not in self.websockets:
            team = "A"
        elif "B" not in self.websockets:
            team = "B"
        else:
            await websocket.send_text(ErrorMessage(message="Game is full").model_dump_json())
            await websocket.close()
            return

        # Wait for join message.
        try:
            raw = await websocket.receive_text()
            msg = _client_message_adapter.validate_json(raw)
        except (WebSocketDisconnect, Exception) as exc:
            self._log(f"Connection error before join: {exc}")
            return

        if not isinstance(msg, JoinMessage):
            await websocket.send_text(
                ErrorMessage(message="Expected 'join' message").model_dump_json()
            )
            await websocket.close()
            return

        self.websockets[team] = websocket
        self.player_names[team] = msg.player_name
        self._log(f"Player '{msg.player_name}' joined as Team {team}")

        # If first player, tell them to wait.
        if len(self.websockets) == 1:
            await websocket.send_text(
                WaitingMessage(message="Waiting for opponent to connect...").model_dump_json()
            )
        else:
            # Both connected — start the game.
            self._both_connected.set()

        # Keep connection alive until game ends or disconnect.
        try:
            await self._client_loop(team, websocket)
        except WebSocketDisconnect:
            self._log(f"Team {team} ({self.player_names.get(team, '?')}) disconnected")
            if not self._game_over:
                self._game_over = True
                other_team = "B" if team == "A" else "A"
                if other_team in self.websockets:
                    try:
                        await self.websockets[other_team].send_text(
                            GameOverMessage(
                                winner=other_team,
                                reason="Opponent disconnected",
                            ).model_dump_json()
                        )
                    except Exception:
                        pass

    async def _client_loop(self, team: str, websocket: WebSocket) -> None:
        """Listen for action messages from a connected client."""
        while not self._game_over:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                raise

            try:
                msg = _client_message_adapter.validate_json(raw)
            except Exception as exc:
                await websocket.send_text(
                    ErrorMessage(message=f"Invalid message: {exc}").model_dump_json()
                )
                continue

            if isinstance(msg, ActionMessage):
                player = self.players[team]
                try:
                    player.submit_action(msg.action)
                except RuntimeError as exc:
                    await websocket.send_text(ErrorMessage(message=str(exc)).model_dump_json())
            else:
                await websocket.send_text(
                    ErrorMessage(
                        message=f"Unexpected message type: {getattr(msg, 'type', 'unknown')}"
                    ).model_dump_json()
                )

    async def run_game(self) -> None:
        """Run the game loop once both players are connected."""
        await self._both_connected.wait()

        # Send assign messages to both clients.
        for team in ["A", "B"]:
            team_tanks = [t for t in self.tanks if t.team == team]
            tank_infos = [
                TankInfo(
                    tank_id=t.id,
                    position=t.position,
                    direction=t.direction,
                    alive=t.alive,
                )
                for t in team_tanks
            ]
            assign_msg = AssignMessage(
                team=team,
                tanks=tank_infos,
                map_width=self.game_map.width,
                map_height=self.game_map.height,
                patch_size=self.patch_size,
            )
            await self.websockets[team].send_text(assign_msg.model_dump_json())

        self._log("Both players connected. Starting game!")

        # Create the engine.
        players: dict[str, Player] = {
            "A": self.players["A"],
            "B": self.players["B"],
        }
        self.engine = GameEngine(
            self.game_map,
            self.tanks,
            players,
            max_turns=self.max_turns,
            patch_size=self.patch_size,
        )

        if self._state_callback:
            self._state_callback(self.engine.state)

        loop = asyncio.get_event_loop()

        # Game loop.
        while not self.engine.game_over and not self._game_over:
            tank_id = self.engine.current_tank_id
            team = self.engine.current_team
            player = self.players[team]

            # Build the view and request action.
            view = build_player_view(self.engine.state, self.game_map, team, self.patch_size)
            player.request_action(tank_id, view, loop)

            # Send your_turn to the client.
            your_turn_msg = YourTurnMessage(tank_id=tank_id, view=view)
            try:
                await self.websockets[team].send_text(your_turn_msg.model_dump_json())
            except Exception:
                self._log(f"Failed to send your_turn to Team {team}")
                self._game_over = True
                break

            # Wait for the client's response.
            try:
                await asyncio.wait_for(player.wait_for_action(), timeout=300.0)
            except asyncio.TimeoutError:
                self._log(f"Team {team} timed out. Ending game.")
                self._game_over = True
                break
            except RuntimeError:
                self._log(f"Team {team} action error. Ending game.")
                self._game_over = True
                break

            # Execute the step.
            entry = self.engine.step()
            self._log(
                f"Turn {self.engine.turns_taken}: {entry.tank_id} → "
                f"{entry.applied_action.value}"
                f"{'' if entry.valid else f' (INVALID: {entry.reason})'}"
            )

            if self._state_callback:
                self._state_callback(self.engine.state)

            # Send turn_result to both clients.
            result_msg = TurnResultMessage(
                tank_id=entry.tank_id,
                action=entry.applied_action,
                valid=entry.valid,
                reason=entry.reason,
            )
            for t in ["A", "B"]:
                if t in self.websockets:
                    try:
                        await self.websockets[t].send_text(result_msg.model_dump_json())
                    except Exception:
                        pass

        # Game over.
        self._game_over = True
        winner = self.engine.winner if self.engine else None
        reason = "Game complete"
        if self.engine and self.engine.game_over:
            if winner:
                reason = f"Team {winner} wins!"
            else:
                reason = "Draw — turn limit reached"

        self._log(f"Game over: {reason}")

        game_over_msg = GameOverMessage(winner=winner, reason=reason)
        for t in ["A", "B"]:
            if t in self.websockets:
                try:
                    await self.websockets[t].send_text(game_over_msg.model_dump_json())
                except Exception:
                    pass


# ── FastAPI application ───────────────────────────────────────────────


def create_fastapi_app(session: GameSession) -> FastAPI:
    """Create the FastAPI application with WebSocket endpoint.

    Args:
        session: The game session to attach.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="HMLS Game Server")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await session.handle_connection(websocket)

    return app


# ── Textual TUI ───────────────────────────────────────────────────────


class ServerApp(App[None]):
    """Textual TUI for the HMLS game server.

    Displays the full game map and a scrollable log panel.
    """

    CSS = """
    #map-scroll {
        height: 2fr;
        min-height: 10;
    }
    #log-panel {
        height: 1fr;
        min-height: 5;
        border-top: solid $primary;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(
        self,
        game_map: GameMap,
        initial_state: GameState,
        session: GameSession,
        port: int,
    ) -> None:
        super().__init__()
        self._game_map = game_map
        self._state = initial_state
        self._session = session
        self._port = port
        self._log_messages: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the server TUI layout."""
        yield Header()
        with ScrollableContainer(id="map-scroll"):
            yield MapView(self._game_map, self._state, id="map-view")
        yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Static(f"Server listening on ws://0.0.0.0:{self._port}/ws", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Start the game loop after mount."""
        log_panel = self.query_one("#log-panel", RichLog)
        log_panel.write("[bold]HMLS Game Server[/bold]")
        log_panel.write(f"Listening on ws://0.0.0.0:{self._port}/ws")
        log_panel.write("Waiting for players to connect...")

        # Process any log messages that arrived before mount.
        for msg in self._log_messages:
            log_panel.write(msg)
        self._log_messages.clear()

        # Start the game loop as a worker.
        self.run_worker(self._game_loop())

    def log_message(self, message: str) -> None:
        """Add a message to the log panel (thread-safe via call_from_thread)."""
        try:
            log_panel = self.query_one("#log-panel", RichLog)
            log_panel.write(message)
        except Exception:
            self._log_messages.append(message)

    def update_game_state(self, state: GameState) -> None:
        """Update the map view with new game state."""
        try:
            map_view = self.query_one("#map-view", MapView)
            map_view.update_state(state)
            if self._session.engine and not self._session.engine.game_over:
                map_view.active_tank_id = self._session.engine.current_tank_id
            else:
                map_view.active_tank_id = ""
        except Exception:
            pass

    async def _game_loop(self) -> None:
        """Run the game session."""
        await self._session.run_game()

        # Update status bar.
        try:
            status = self.query_one("#status-bar", Static)
            if self._session.engine:
                winner = self._session.engine.winner
                if winner:
                    status.update(f"GAME OVER — Team {winner} wins!")
                else:
                    status.update("GAME OVER — Draw!")
            else:
                status.update("GAME OVER — Disconnection")
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the server application."""
    args = parse_args()
    game_map = _load_map(args.map_file)
    tanks = _place_tanks(game_map, args.tanks_per_player, seed=args.seed)
    initial_state = GameState(tanks=tanks)

    # We need to bridge log messages from the server thread to the TUI.
    # The TUI will be set up after app creation.
    app_ref: list[ServerApp] = []

    def log_callback(message: str) -> None:
        if app_ref:
            app_ref[0].call_from_thread(app_ref[0].log_message, message)
        else:
            print(message)

    def state_callback(state: GameState) -> None:
        if app_ref:
            app_ref[0].call_from_thread(app_ref[0].update_game_state, state)

    session = GameSession(
        game_map=game_map,
        tanks=tanks,
        max_turns=args.max_turns,
        patch_size=args.patch_size,
        log_callback=log_callback,
    )
    session.set_state_callback(state_callback)

    # Create FastAPI app.
    fastapi_app = create_fastapi_app(session)

    # Run uvicorn in a background thread.
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    def run_server() -> None:
        asyncio.run(server.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Run the Textual TUI.
    app = ServerApp(game_map, initial_state, session, args.port)
    app_ref.append(app)
    app.title = "HMLS Game Server"
    app.run()


if __name__ == "__main__":
    main()
