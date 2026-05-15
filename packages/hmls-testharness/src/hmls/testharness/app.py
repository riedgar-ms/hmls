"""Main Textual TUI application for the HMLS test harness.

Provides an interactive game viewer where the user controls every tank
via keyboard input, watching the full map and each player's fog-of-war
view update in real time.
"""

from __future__ import annotations

import logging
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static, TabbedContent, TabPane

from hmls.core.engine import GameEngine, HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.player import Player
from hmls.core.types import Action
from hmls.testharness.cli import build_initial_state, load_map, parse_args, place_tanks
from hmls.testharness.interactive_player import InteractivePlayer
from hmls.uxcommon import LogStatusMixin
from hmls.uxcommon.log_tab import LogTabMixin
from hmls.uxcommon.widgets.map_view import MapView
from hmls.uxcommon.widgets.player_view import PlayerViewRegion

logger = logging.getLogger("hmls.testharness")

# ── Save dialog ───────────────────────────────────────────────────────


class SaveDialog(ModalScreen[str | None]):
    """Modal dialog prompting the user to save the game history."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    SaveDialog {
        align: center middle;
    }
    #save-dialog-container {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #save-path-input {
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the save dialog layout."""
        with VerticalScroll(id="save-dialog-container"):
            yield Label("Save game history to file:")
            yield Input(placeholder="path/to/history.json", id="save-path-input")
            yield Label("Press Enter to save, Escape to skip.", classes="hint")

    def on_mount(self) -> None:
        """Focus the input when the dialog opens."""
        self.query_one("#save-path-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        if event.input.id == "save-path-input":
            value = event.value.strip()
            self.dismiss(value if value else None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)


# ── Main application ──────────────────────────────────────────────────


class TestHarnessApp(LogTabMixin, LogStatusMixin, App[None]):
    """Interactive TUI for testing the HMLS tank game.

    The user controls each tank in turn using keyboard keys.

    Key bindings:
        - ``W``: Move forward
        - ``A``: Turn left
        - ``D``: Turn right
        - ``Space``: Fire
        - ``Tab``: Pass (skip turn)
        - ``Q``: Quit
    """

    CSS = """
    #map-scroll {
        height: 1fr;
        min-height: 10;
    }
    #player-a-region {
        height: auto;
    }
    #player-b-region {
        height: auto;
    }
    #log-panel {
        height: auto;
        max-height: 5;
        border-top: solid $primary;
    }
    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    .team-label {
        width: 10;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        game_map: GameMap,
        initial_state: GameState,
        engine: GameEngine,
    ) -> None:
        super().__init__()
        self._game_map = game_map
        self._engine = engine
        self._state = initial_state

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        with TabbedContent(initial="game-tab"):
            with TabPane("Game", id="game-tab"):
                with ScrollableContainer(id="map-scroll"):
                    yield MapView(
                        self._game_map,
                        self._state,
                        id="map-view",
                    )

                yield PlayerViewRegion(
                    "A",
                    self._game_map,
                    self._state,
                    patch_size=self._engine.patch_size,
                    active_tank_id=self._engine.current_tank_id,
                    id="player-a-region",
                )
                yield PlayerViewRegion(
                    "B",
                    self._game_map,
                    self._state,
                    patch_size=self._engine.patch_size,
                    active_tank_id=self._engine.current_tank_id,
                    id="player-b-region",
                )

                yield RichLog(id="log-panel", highlight=True, markup=True, max_lines=50)
                yield Static(self._build_status_text(), id="status-bar")
            yield from self._compose_log_tab()
        yield Footer()

    def on_mount(self) -> None:
        """Set the active tank highlight after widgets are mounted."""
        self._setup_log_tab()
        self._update_active_highlight()

    def _build_status_text(self) -> str:
        """Build the status bar text."""
        eng = self._engine
        if eng.game_over:
            winner = eng.winner
            if winner:
                return f"GAME OVER — Team {winner} wins! | Turns: {eng.turns_taken}"
            return f"GAME OVER — Draw! | Turns: {eng.turns_taken}"

        tank_id = eng.current_tank_id
        team = eng.current_team
        return (
            f"Turn {eng.turns_taken + 1}/{eng.max_turns} | "
            f"Active: {tank_id} (Team {team})\n"
            f"W=Forward  A=Left  D=Right  Space=Fire  Tab=Pass  Q=Quit"
        )

    def _update_active_highlight(self) -> None:
        """Update the active tank highlight on the map view."""
        map_view = self.query_one("#map-view", MapView)
        map_view.active_tank_id = self._engine.current_tank_id

    def _log_action_result(self, tank_id: str, entry: HistoryEntry) -> None:
        """Log an action result to the log panel."""
        self._log_turn_result(
            tank_id, entry.applied_action.value, entry.valid, entry.reason, entry.hit
        )

    def _do_action(self, action: Action) -> None:
        """Execute an action and refresh the UI."""
        if self._engine.game_over:
            return

        # Pre-load the action on the current team's InteractivePlayer.
        team = self._engine.current_team
        tank_id = self._engine.current_tank_id
        player = self._engine.players[team]
        if not isinstance(player, InteractivePlayer):
            return
        player.set_next_action(action)
        entry = self._engine.step()
        self._state = self._engine.state

        # Log the action result.
        self._log_action_result(tank_id, entry)

        # Update the map view.
        map_view = self.query_one("#map-view", MapView)
        map_view.update_state(self._state)

        if not self._engine.game_over:
            map_view.active_tank_id = self._engine.current_tank_id
        else:
            map_view.active_tank_id = ""

        # Refresh player view regions.
        self.run_worker(self._refresh_player_views())

        # Update status bar.
        status = self.query_one("#status-bar", Static)
        status.update(self._build_status_text())

        # Check game over.
        if self._engine.game_over:
            self._show_game_over()

    async def _refresh_player_views(self) -> None:
        """Refresh both player view regions."""
        active_id = self._engine.current_tank_id if not self._engine.game_over else ""
        for region_id in ("#player-a-region", "#player-b-region"):
            region = self.query_one(region_id, PlayerViewRegion)
            await region.refresh_patches(self._state, active_id)

    def _show_game_over(self) -> None:
        """Show the game-over dialog."""
        self.push_screen(SaveDialog(), callback=self._on_save_result)

    def _on_save_result(self, path_str: str | None) -> None:
        """Handle the save dialog result."""
        status = self.query_one("#status-bar", Static)
        if path_str is None:
            status.update(self._build_status_text() + "\n(History not saved)")
            return

        path = Path(path_str)
        try:
            result = self._engine.make_result()
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            status.update(self._build_status_text() + f"\nSaved to {path.resolve()}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to save history to %s", path, exc_info=True)
            status.update(self._build_status_text() + f"\nSave error: {exc}")

    # ── Key handlers ──────────────────────────────────────────────

    def on_key(self, event: Key) -> None:
        """Handle key presses for tank actions."""
        key = event.key
        action_map: dict[str, Action] = {
            "w": Action.MOVE_FORWARD,
            "a": Action.TURN_LEFT,
            "d": Action.TURN_RIGHT,
            "space": Action.FIRE,
            "tab": Action.PASS,
        }
        if key in action_map:
            event.prevent_default()
            self._do_action(action_map[key])


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the TUI application."""
    import sys

    from hmls.core.map import MapLoadError
    from hmls.core.placement import InsufficientPassableCellsError

    args = parse_args()
    try:
        game_map = load_map(args.map_file)
    except (FileNotFoundError, MapLoadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        tanks = place_tanks(game_map, args.tanks_per_player, seed=args.seed)
    except InsufficientPassableCellsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    initial_state = build_initial_state(tanks)

    players: dict[str, Player] = {
        "A": InteractivePlayer("A"),
        "B": InteractivePlayer("B"),
    }
    engine = GameEngine(
        game_map,
        tanks,
        players,
        max_turns=args.max_turns,
        patch_size=args.patch_size,
    )

    app = TestHarnessApp(game_map, initial_state, engine)
    app.title = "HMLS Test Harness"
    app.run()


if __name__ == "__main__":
    main()
