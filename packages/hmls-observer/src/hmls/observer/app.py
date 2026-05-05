"""Main observer application: WebSocket client + Textual TUI.

Connects to the HMLS game server as an observer and displays the full
game map and event log in real-time, without fog-of-war restrictions.
"""

from __future__ import annotations

from typing import Any

import websockets
import websockets.asyncio.client
from pydantic import TypeAdapter
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, RichLog, Static

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.observer.cli import parse_args
from hmls.protocol import (
    ErrorMessage,
    GameOverMessage,
    GameStartMessage,
    ObserveMessage,
    ServerMessage,
    StateUpdateMessage,
    TurnResultMessage,
    WaitingMessage,
)
from hmls.uxcommon.mixins import LogStatusMixin
from hmls.uxcommon.widgets.map_view import MapView

# ── Type adapter for server messages ─────────────────────────────────

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


# ── Observer TUI ──────────────────────────────────────────────────────


class ObserverApp(LogStatusMixin, App[None]):
    """Textual TUI for observing an HMLS game in progress.

    Displays the full game map (no fog-of-war) and a scrollable event log.
    Connects to the server via WebSocket and receives state updates.
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

    def __init__(self, server_url: str, observer_name: str) -> None:
        super().__init__()
        self._server_url = server_url
        self._observer_name = observer_name
        self._game_map: GameMap | None = None
        self._state: GameState | None = None
        self._tanks: list[Tank] = []
        self._player_names: dict[str, str] = {}
        self._game_over: bool = False
        self._ws: Any = None

    def compose(self) -> ComposeResult:
        """Compose the observer TUI layout."""
        yield Header()
        with ScrollableContainer(id="map-scroll"):
            yield Static("Connecting to server...", id="map-placeholder")
        yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Static(f"Connecting to {self._server_url}...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Start the WebSocket connection after mount."""
        log_panel = self.query_one("#log-panel", RichLog)
        log_panel.write("[bold]HMLS Game Observer[/bold]")
        log_panel.write(f"Connecting to {self._server_url}...")
        self.run_worker(self._connection_loop())

    async def _connection_loop(self) -> None:
        """Manage the WebSocket connection and message handling."""
        try:
            async with websockets.asyncio.client.connect(self._server_url) as ws:
                self._ws = ws

                # Send observe message to identify as an observer.
                observe_msg = ObserveMessage(observer_name=self._observer_name)
                await ws.send(observe_msg.model_dump_json())
                self._write_log("Connected as observer.")
                self._update_status("Connected — waiting for game to start...")

                # Message handling loop.
                async for raw in ws:
                    if self._game_over:
                        break
                    await self._handle_message(str(raw))

        except Exception as exc:
            self._write_log(f"[red]Connection error: {exc}[/red]")
            self._update_status("DISCONNECTED")

    async def _handle_message(self, raw: str) -> None:
        """Parse and dispatch a server message."""
        try:
            msg = _server_message_adapter.validate_json(raw)
        except Exception as exc:
            self._write_log(f"[red]Invalid message: {exc}[/red]")
            return

        if isinstance(msg, GameStartMessage):
            await self._handle_game_start(msg)
        elif isinstance(msg, StateUpdateMessage):
            self._handle_state_update(msg)
        elif isinstance(msg, TurnResultMessage):
            self._handle_turn_result(msg)
        elif isinstance(msg, GameOverMessage):
            self._handle_game_over(msg)
        elif isinstance(msg, WaitingMessage):
            self._write_log(msg.message)
        elif isinstance(msg, ErrorMessage):
            self._write_log(f"[red]Server error: {msg.message}[/red]")

    async def _handle_game_start(self, msg: GameStartMessage) -> None:
        """Handle the game_start message: set up or update the map view."""
        self._game_map = msg.game_map
        self._tanks = msg.tanks
        self._player_names = msg.player_names
        self._state = GameState(tanks=msg.tanks)

        self._write_log("Game starting!")
        self._write_log(
            f"Map: {msg.game_map.width}×{msg.game_map.height} | "
            f"Tanks per team: {len(msg.tanks) // 2}"
        )
        for team, name in msg.player_names.items():
            self._write_log(f"  Team {team}: {name}")

        scroll = self.query_one("#map-scroll", ScrollableContainer)

        # If the MapView already exists (e.g. from an earlier pre-game
        # GameStartMessage), update it in place.  Otherwise, replace the
        # placeholder with a fresh MapView.
        existing = self.query("#map-view")
        if existing:
            map_view = existing.first(MapView)
            map_view._game_map = msg.game_map
            map_view.update_state(self._state)
        else:
            placeholder = self.query_one("#map-placeholder", Static)
            await placeholder.remove()
            map_view = MapView(self._game_map, self._state, id="map-view")
            await scroll.mount(map_view)

        names = " vs ".join(f"{name} ({team})" for team, name in self._player_names.items())
        self._update_status(f"Game in progress: {names}")

    def _handle_state_update(self, msg: StateUpdateMessage) -> None:
        """Handle a state update: refresh the map display."""
        self._state = msg.state
        try:
            map_view = self.query_one("#map-view", MapView)
            map_view.update_state(msg.state)
            map_view.active_tank_id = msg.current_tank_id
        except Exception:
            pass

        if msg.current_tank_id:
            self._update_status(
                f"Turn {msg.turns_taken} | Next: {msg.current_tank_id} | "
                + " vs ".join(f"{name} ({team})" for team, name in self._player_names.items())
            )

    def _handle_turn_result(self, msg: TurnResultMessage) -> None:
        """Handle a turn result: log the action."""
        self._log_turn_result(msg.tank_id, msg.action.value, msg.valid, msg.reason, msg.hit)

    def _handle_game_over(self, msg: GameOverMessage) -> None:
        """Handle game over notification."""
        self._game_over = True
        if msg.winner:
            winner_name = self._player_names.get(msg.winner, msg.winner)
            self._write_log(
                f"[bold green]GAME OVER — {winner_name} (Team {msg.winner}) wins![/bold green]"
            )
            self._update_status(f"GAME OVER — {winner_name} (Team {msg.winner}) wins!")
        else:
            self._write_log(f"[bold yellow]GAME OVER — Draw: {msg.reason}[/bold yellow]")
            self._update_status(f"GAME OVER — Draw: {msg.reason}")


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the observer application."""
    args = parse_args()
    app = ObserverApp(args.url, args.name)
    app.title = "HMLS Game Observer"
    app.run()


if __name__ == "__main__":
    main()
