"""Main Textual TUI application for the HMLS test harness.

Provides an interactive game viewer where the user controls every tank
via keyboard input, watching the full map and each player's fog-of-war
view update in real time.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, Static

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.types import Action
from hmls.testharness.cli import build_initial_state, load_map, parse_args, place_tanks
from hmls.testharness.game_loop import GameLoop
from hmls.testharness.widgets.map_view import MapView
from hmls.testharness.widgets.player_view import PlayerViewRegion

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


class TestHarnessApp(App[None]):
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
        max-height: 14;
    }
    #player-b-region {
        height: auto;
        max-height: 14;
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
        game_loop: GameLoop,
    ) -> None:
        super().__init__()
        self._game_map = game_map
        self._game_loop = game_loop
        self._state = initial_state

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

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
            patch_size=self._game_loop.patch_size,
            active_tank_id=self._game_loop.current_tank_id,
            id="player-a-region",
        )
        yield PlayerViewRegion(
            "B",
            self._game_map,
            self._state,
            patch_size=self._game_loop.patch_size,
            active_tank_id=self._game_loop.current_tank_id,
            id="player-b-region",
        )

        yield Static(self._build_status_text(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set the active tank highlight after widgets are mounted."""
        self._update_active_highlight()

    def _build_status_text(self) -> str:
        """Build the status bar text."""
        gl = self._game_loop
        if gl.game_over:
            winner = gl.winner
            if winner:
                return f"GAME OVER — Team {winner} wins! | Turns: {gl.turns_taken}"
            return f"GAME OVER — Draw! | Turns: {gl.turns_taken}"

        tank_id = gl.current_tank_id
        team = gl.current_team
        return (
            f"Turn {gl.turns_taken + 1}/{gl.max_turns} | "
            f"Active: {tank_id} (Team {team})\n"
            f"W=Forward  A=Left  D=Right  Space=Fire  Tab=Pass  Q=Quit"
        )

    def _update_active_highlight(self) -> None:
        """Update the active tank highlight on the map view."""
        map_view = self.query_one("#map-view", MapView)
        map_view.active_tank_id = self._game_loop.current_tank_id

    def _do_action(self, action: Action) -> None:
        """Execute an action and refresh the UI."""
        if self._game_loop.game_over:
            return

        self._game_loop.step(action)
        self._state = self._game_loop.state

        # Update the map view.
        map_view = self.query_one("#map-view", MapView)
        map_view.update_state(self._state)

        if not self._game_loop.game_over:
            map_view.active_tank_id = self._game_loop.current_tank_id
        else:
            map_view.active_tank_id = ""

        # Refresh player view regions.
        self.run_worker(self._refresh_player_views())

        # Update status bar.
        status = self.query_one("#status-bar", Static)
        status.update(self._build_status_text())

        # Check game over.
        if self._game_loop.game_over:
            self._show_game_over()

    async def _refresh_player_views(self) -> None:
        """Refresh both player view regions."""
        active_id = self._game_loop.current_tank_id if not self._game_loop.game_over else ""
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
            from hmls.core.engine import GameResult
            from hmls.core.engine import HistoryEntry as CoreHistoryEntry

            result = GameResult(
                winner=self._game_loop.winner,
                game_map=self._game_map,
                initial_state=self._game_loop.history[0].state_after
                if self._game_loop.history
                else self._state,
                history=[
                    CoreHistoryEntry(
                        tank_id=entry.tank_id,
                        requested_action=entry.requested_action,
                        applied_action=entry.applied_action,
                        valid=entry.valid,
                        reason=entry.reason,
                        state_after=entry.state_after,
                    )
                    for entry in self._game_loop.history
                ],
                turns_played=self._game_loop.turns_taken,
            )
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            status.update(self._build_status_text() + f"\nSaved to {path.resolve()}")
        except Exception as exc:
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
    args = parse_args()
    game_map = load_map(args.map_file)
    tanks = place_tanks(game_map, args.tanks_per_player, seed=args.seed)
    initial_state = build_initial_state(tanks)
    game_loop = GameLoop(
        game_map,
        initial_state,
        max_turns=args.max_turns,
        patch_size=args.patch_size,
    )

    app = TestHarnessApp(game_map, initial_state, game_loop)
    app.title = "HMLS Test Harness"
    app.run()


if __name__ == "__main__":
    main()
