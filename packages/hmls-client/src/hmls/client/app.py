"""Main client application: WebSocket client + Textual TUI with automapper.

The client connects to the game server, displays an automapped view of
explored terrain, and accepts keyboard input for tank control.
"""

from __future__ import annotations

import asyncio
from typing import Any

import websockets
import websockets.asyncio.client
from pydantic import TypeAdapter
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.events import Key
from textual.widgets import Footer, Header, RichLog, Static

from hmls.client.automap import AutoMap, CellState
from hmls.client.cli import parse_args
from hmls.core.tank import TankId
from hmls.core.types import Action, Position
from hmls.core.visibility import TankInfo
from hmls.protocol import (
    ActionMessage,
    AssignMessage,
    ErrorMessage,
    GameOverMessage,
    JoinMessage,
    ServerMessage,
    TurnResultMessage,
    WaitingMessage,
    YourTurnMessage,
)
from hmls.uxcommon.styles import (
    ACTIVE_HIGHLIGHT_STYLE,
    CELL_CHARS,
    CELL_WIDTH,
    DEAD_MARKER,
    DEAD_STYLE,
    DIRECTION_ARROWS,
    FOG_STYLE,
    IMPASSABLE_STYLE,
    PASSABLE_STYLE,
    TEAM_A_STYLE,
    TEAM_B_STYLE,
)
from hmls.uxcommon.widgets import PatchView

# ── Type adapter for server messages ─────────────────────────────────

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)

# ── Automap widget ────────────────────────────────────────────────────


_TEAM_STYLES: dict[str, str] = {
    "A": TEAM_A_STYLE,
    "B": TEAM_B_STYLE,
}


class AutoMapView(Static):
    """Renders the automapped terrain in the TUI.

    Shows explored cells using the shared uxcommon rendering style (2-char
    block cells), unknown/fog cells in fog style, and tank positions with
    directional arrows.
    """

    def __init__(self, width: int, height: int, *, team: str = "A", id: str | None = None) -> None:
        super().__init__(id=id)
        self._map_width = width
        self._map_height = height
        self._team = team
        self._automap: AutoMap | None = None
        self._tank_infos: list[TankInfo] = []
        self._active_tank_id: TankId = ""

    def set_automap(self, automap: AutoMap) -> None:
        """Set the automap data source."""
        self._automap = automap
        self._render_map()

    def update_tanks(self, tanks: list[TankInfo], active_tank_id: TankId = "") -> None:
        """Update tank positions and re-render."""
        self._tank_infos = tanks
        self._active_tank_id = active_tank_id
        self._render_map()

    def refresh_display(self) -> None:
        """Force a re-render of the map."""
        self._render_map()

    def _render_map(self) -> None:
        """Render the automap to a Rich Text object."""
        if self._automap is None:
            self.update("Waiting for game data...")
            return

        # Build position → tank lookup.
        tank_positions: dict[Position, TankInfo] = {}
        for t in self._tank_infos:
            tank_positions[t.position] = t

        text = Text()
        for y in range(self._map_height):
            for x in range(self._map_width):
                pos = Position(x, y)
                if pos in tank_positions:
                    tank = tank_positions[pos]
                    is_active = tank.tank_id == self._active_tank_id

                    if not tank.alive:
                        style = ACTIVE_HIGHLIGHT_STYLE if is_active else DEAD_STYLE
                        text.append(DEAD_MARKER, style=style)
                    else:
                        arrow = DIRECTION_ARROWS.get(int(tank.direction), "? ")
                        if is_active:
                            style = ACTIVE_HIGHLIGHT_STYLE
                        else:
                            style = _TEAM_STYLES.get(self._team, TEAM_A_STYLE)
                        text.append(arrow, style=style)
                else:
                    cell = self._automap[x, y]
                    if cell == CellState.PASSABLE:
                        text.append(CELL_CHARS, style=PASSABLE_STYLE)
                    elif cell == CellState.IMPASSABLE:
                        text.append(CELL_CHARS, style=IMPASSABLE_STYLE)
                    else:
                        text.append(CELL_CHARS, style=FOG_STYLE)
            text.append("\n")

        self.styles.min_width = self._map_width * CELL_WIDTH
        self.update(text)


# ── Main client TUI ──────────────────────────────────────────────────


class ClientApp(App[None]):
    """Textual TUI for the HMLS game client.

    Displays the automapped terrain, tank positions, and a log panel.
    Accepts keyboard input for tank actions using the same keys as the
    test harness.

    Key bindings:
        - ``W``: Move forward
        - ``A``: Turn left
        - ``D``: Turn right
        - ``Space``: Fire
        - ``Tab``: Pass (skip turn)
        - ``Q``: Quit
    """

    CSS = """
    #automap-view {
        height: 2fr;
        min-height: 10;
    }
    #patches-panel {
        height: auto;
        min-height: 5;
        overflow-x: auto;
    }
    #log-panel {
        height: 1fr;
        min-height: 5;
        border-top: solid $primary;
    }
    #status-bar {
        dock: bottom;
        height: 2;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(self, server_url: str, player_name: str) -> None:
        super().__init__()
        self._server_url = server_url
        self._player_name = player_name
        self._automap: AutoMap | None = None
        self._map_view: AutoMapView | None = None
        self._team: str = ""
        self._tanks: list[TankInfo] = []
        self._active_tank_id: TankId = ""
        self._awaiting_action: bool = False
        self._action_queue: asyncio.Queue[Action] = asyncio.Queue()
        self._ws: Any = None
        self._game_over: bool = False

    def compose(self) -> ComposeResult:
        """Compose the client TUI layout."""
        yield Header()
        yield AutoMapView(1, 1, id="automap-view")  # Placeholder until assign.
        yield Horizontal(id="patches-panel")
        yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Static("Connecting...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Start the WebSocket connection after mount."""
        log_panel = self.query_one("#log-panel", RichLog)
        log_panel.write(f"[bold]HMLS Game Client[/bold] — {self._player_name}")
        log_panel.write(f"Connecting to {self._server_url}...")
        self.run_worker(self._connection_loop())

    def _write_log(self, message: str) -> None:
        """Write a message to the log panel."""
        try:
            log_panel = self.query_one("#log-panel", RichLog)
            log_panel.write(message)
        except Exception:
            pass

    def _update_status(self, text: str) -> None:
        """Update the status bar."""
        try:
            status = self.query_one("#status-bar", Static)
            status.update(text)
        except Exception:
            pass

    async def _connection_loop(self) -> None:
        """Manage the WebSocket connection and message handling."""
        try:
            async with websockets.asyncio.client.connect(self._server_url) as ws:
                self._ws = ws

                # Send join message.
                join_msg = JoinMessage(player_name=self._player_name)
                await ws.send(join_msg.model_dump_json())
                self._write_log("Join message sent.")

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

        if isinstance(msg, WaitingMessage):
            self._write_log(msg.message)
            self._update_status("Waiting for opponent...")

        elif isinstance(msg, AssignMessage):
            await self._handle_assign(msg)

        elif isinstance(msg, YourTurnMessage):
            await self._handle_your_turn(msg)

        elif isinstance(msg, TurnResultMessage):
            self._handle_turn_result(msg)

        elif isinstance(msg, GameOverMessage):
            self._handle_game_over(msg)

        elif isinstance(msg, ErrorMessage):
            self._write_log(f"[red]Server error: {msg.message}[/red]")

    async def _handle_assign(self, msg: AssignMessage) -> None:
        """Handle team assignment and initialize the automap."""
        self._team = msg.team
        self._tanks = list(msg.tanks)
        self._automap = AutoMap(msg.map_width, msg.map_height)

        self._write_log(f"Assigned to [bold]Team {msg.team}[/bold]")
        self._write_log(f"Map size: {msg.map_width}×{msg.map_height}")
        self._write_log(f"Tanks: {', '.join(t.tank_id for t in msg.tanks)}")

        # Replace the placeholder AutoMapView with correctly sized one.
        old_view = self.query_one("#automap-view", AutoMapView)
        new_view = AutoMapView(msg.map_width, msg.map_height, team=msg.team, id="automap-view")
        await old_view.remove()
        await self.mount(new_view, before="#log-panel")
        self._map_view = new_view
        new_view.set_automap(self._automap)
        new_view.update_tanks(self._tanks)

        self._update_status(f"Team {msg.team} | Waiting for game to start...")

    async def _handle_your_turn(self, msg: YourTurnMessage) -> None:
        """Handle a your_turn message: update automap and wait for input."""
        if self._automap is None:
            return

        # Update automap with new visibility.
        self._automap.update(msg.view)

        # Update tank info from the view.
        self._tanks = list(msg.view.tanks)
        self._active_tank_id = msg.tank_id

        # Refresh the display.
        if self._map_view is not None:
            self._map_view.refresh_display()
            self._map_view.update_tanks(self._tanks, self._active_tank_id)

        # Rebuild patches panel with current visibility patches.
        try:
            patches_panel = self.query_one("#patches-panel", Horizontal)
            await patches_panel.remove_children()
            patch_widgets: list[PatchView] = [
                PatchView(
                    patch.tank_id,
                    patch,
                    is_active=(patch.tank_id == msg.tank_id),
                )
                for patch in msg.view.patches
            ]
            if patch_widgets:
                await patches_panel.mount_all(patch_widgets)
        except Exception:
            pass

        self._update_status(
            f"Team {self._team} | Tank {msg.tank_id}'s turn\n"
            f"W=Forward  A=Left  D=Right  Space=Fire  Tab=Pass  Q=Quit"
        )

        # Wait for user input.
        self._awaiting_action = True
        action = await self._action_queue.get()
        self._awaiting_action = False

        # Send action to server.
        if self._ws and not self._game_over:
            action_msg = ActionMessage(action=action)
            await self._ws.send(action_msg.model_dump_json())

    def _handle_turn_result(self, msg: TurnResultMessage) -> None:
        """Handle a turn result notification."""
        if not msg.valid:
            status = f"[red]✗ ({msg.reason})[/red]"
        elif msg.hit is True:
            status = "[bold green]HIT![/bold green]"
        elif msg.hit is False:
            status = "[dim]miss[/dim]"
        else:
            status = "✓"
        self._write_log(f"  {msg.tank_id} → {msg.action.value} — {status}")

    def _handle_game_over(self, msg: GameOverMessage) -> None:
        """Handle game over."""
        self._game_over = True
        if msg.winner == self._team:
            self._write_log("[bold green]YOU WIN![/bold green]")
        elif msg.winner is None:
            self._write_log("[bold yellow]DRAW![/bold yellow]")
        else:
            self._write_log("[bold red]YOU LOSE![/bold red]")
        self._write_log(f"Reason: {msg.reason}")
        self._update_status(f"GAME OVER — {msg.reason} | Press Q to quit")

        # Unblock any pending action wait.
        if self._awaiting_action:
            self._action_queue.put_nowait(Action.PASS)

    # ── Key handlers ──────────────────────────────────────────────

    def on_key(self, event: Key) -> None:
        """Handle key presses for tank actions."""
        if event.key == "q":
            self.exit()
            return

        if not self._awaiting_action:
            return

        action_map: dict[str, Action] = {
            "w": Action.MOVE_FORWARD,
            "a": Action.TURN_LEFT,
            "d": Action.TURN_RIGHT,
            "space": Action.FIRE,
            "tab": Action.PASS,
        }
        if event.key in action_map:
            event.prevent_default()
            self._action_queue.put_nowait(action_map[event.key])


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the client application."""
    args = parse_args()
    app = ClientApp(args.server_url, args.name)
    app.title = "HMLS Game Client"
    app.run()


if __name__ == "__main__":
    main()
