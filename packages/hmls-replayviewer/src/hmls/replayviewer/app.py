"""Main Textual TUI application for the HMLS game replay viewer.

Lets the user step through a recorded game history, watching the map
and fog-of-war patches update at each turn.  Supports both manual
stepping (arrow keys) and automatic playback with adjustable speed.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.timer import Timer
from textual.widgets import Footer, Header, Static

from hmls.core.engine import GameResult, HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.replayviewer.cli import build_state_timeline, load_game_result, parse_args
from hmls.uxcommon.widgets.map_view import MapView

# ── Constants ─────────────────────────────────────────────────────────

_MIN_DELAY: float = 0.1
"""Minimum auto-play delay in seconds."""

_MAX_DELAY: float = 5.0
"""Maximum auto-play delay in seconds."""

_DEFAULT_DELAY: float = 0.5
"""Default auto-play delay in seconds."""

_DELAY_STEP: float = 0.1
"""Amount to change delay per key press."""


class ReplayViewerApp(App[None]):
    """TUI application for replaying HMLS tank game history files.

    Key bindings:
        - ``Left``: Step backward one turn
        - ``Right``: Step forward one turn
        - ``Space``: Toggle auto-play on/off
        - ``Up``: Speed up (decrease delay)
        - ``Down``: Slow down (increase delay)
        - ``Home``: Jump to start
        - ``End``: Jump to end
        - ``Q``: Quit
    """

    CSS = """
    #map-scroll {
        height: 1fr;
        min-height: 10;
    }
    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True, show=True),
        Binding("space", "toggle_play", "Play/Pause", show=True),
        Binding("left", "step_back", "← Back", show=True),
        Binding("right", "step_forward", "→ Forward", show=True),
        Binding("up", "speed_up", "Speed up", show=True),
        Binding("down", "slow_down", "Slow down", show=True),
        Binding("home", "jump_start", "Start", show=True),
        Binding("end", "jump_end", "End", show=True),
    ]

    def __init__(
        self,
        result: GameResult,
    ) -> None:
        super().__init__()
        self._result = result
        self._game_map: GameMap = result.game_map
        self._states: list[GameState] = build_state_timeline(result)
        self._history: list[HistoryEntry] = list(result.history)

        self._current_step: int = 0
        self._playing: bool = False
        self._delay: float = _DEFAULT_DELAY
        self._timer: Timer | None = None

    # ── Layout ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        with ScrollableContainer(id="map-scroll"):
            yield MapView(
                self._game_map,
                self._states[0],
                id="map-view",
            )

        yield Static(self._build_status_text(), id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Set initial active tank highlight."""
        self._update_display()

    # ── Navigation helpers ────────────────────────────────────────

    @property
    def _max_step(self) -> int:
        """Maximum valid step index."""
        return len(self._states) - 1

    def _active_tank_id(self) -> str:
        """Return the ID of the tank that acted at the current step.

        At step 0 (initial state) there is no acting tank, so returns
        the first tank's ID from the initial state for display purposes.
        For steps ≥ 1, returns the tank from the corresponding history entry.
        """
        if self._current_step == 0:
            state = self._states[0]
            return state.current_tank_id if state.current_tank_id else ""
        entry = self._history[self._current_step - 1]
        return entry.tank_id

    def _go_to_step(self, step: int) -> None:
        """Navigate to a specific step and refresh the display.

        Args:
            step: Target step index (clamped to valid range).
        """
        step = max(0, min(step, self._max_step))
        if step == self._current_step:
            return
        self._current_step = step
        self._update_display()

    def _update_display(self) -> None:
        """Refresh all widgets for the current step."""
        state = self._states[self._current_step]
        active_id = self._active_tank_id()

        # Update map.
        map_view = self.query_one("#map-view", MapView)
        map_view.update_state(state)
        map_view.active_tank_id = active_id

        # Update status bar.
        status = self.query_one("#status-bar", Static)
        status.update(self._build_status_text())

    # ── Status bar ────────────────────────────────────────────────

    def _build_status_text(self) -> str:
        """Build the status bar text showing current position and controls."""
        step = self._current_step
        total = self._max_step
        play_state = "▶ Playing" if self._playing else "⏸ Paused"
        delay_str = f"{self._delay:.1f}s"

        # Action info for current step.
        if step == 0:
            action_info = "Initial state"
        else:
            entry = self._history[step - 1]
            action_str = entry.applied_action.value
            validity = "" if entry.valid else f" (INVALID: {entry.reason})"
            action_info = f"Tank {entry.tank_id}: {action_str}{validity}"

        # Winner info.
        winner = self._result.winner
        winner_str = ""
        if step == total:
            if winner:
                winner_str = f" | Winner: Team {winner}"
            else:
                winner_str = " | Draw"

        return f"Step {step}/{total} | {play_state} | Delay: {delay_str}{winner_str}\n{action_info}"

    # ── Auto-play ─────────────────────────────────────────────────

    def _start_timer(self) -> None:
        """Start the auto-play timer."""
        self._stop_timer()
        self._timer = self.set_interval(self._delay, self._auto_step)

    def _stop_timer(self) -> None:
        """Stop the auto-play timer if running."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _auto_step(self) -> None:
        """Advance one step during auto-play.  Stops at the end."""
        if self._current_step >= self._max_step:
            self._playing = False
            self._stop_timer()
            self._update_status_only()
            return
        self._go_to_step(self._current_step + 1)

    def _update_status_only(self) -> None:
        """Refresh only the status bar (used for play state changes)."""
        status = self.query_one("#status-bar", Static)
        status.update(self._build_status_text())

    # ── Actions ───────────────────────────────────────────────────

    def action_toggle_play(self) -> None:
        """Toggle auto-play on/off."""
        if self._playing:
            self._playing = False
            self._stop_timer()
        else:
            if self._current_step >= self._max_step:
                # At the end — restart from beginning.
                self._current_step = 0
                self._update_display()
            self._playing = True
            self._start_timer()
        self._update_status_only()

    def action_step_back(self) -> None:
        """Step backward one turn."""
        self._playing = False
        self._stop_timer()
        self._go_to_step(self._current_step - 1)

    def action_step_forward(self) -> None:
        """Step forward one turn."""
        self._playing = False
        self._stop_timer()
        self._go_to_step(self._current_step + 1)

    def action_speed_up(self) -> None:
        """Decrease the auto-play delay (speed up)."""
        self._delay = max(_MIN_DELAY, round(self._delay - _DELAY_STEP, 1))
        if self._playing:
            self._start_timer()  # Restart with new delay.
        self._update_status_only()

    def action_slow_down(self) -> None:
        """Increase the auto-play delay (slow down)."""
        self._delay = min(_MAX_DELAY, round(self._delay + _DELAY_STEP, 1))
        if self._playing:
            self._start_timer()  # Restart with new delay.
        self._update_status_only()

    def action_jump_start(self) -> None:
        """Jump to the start of the game."""
        self._playing = False
        self._stop_timer()
        self._go_to_step(0)

    def action_jump_end(self) -> None:
        """Jump to the end of the game."""
        self._playing = False
        self._stop_timer()
        self._go_to_step(self._max_step)


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the replay viewer TUI application."""
    args = parse_args()
    result = load_game_result(args.history_file)

    app = ReplayViewerApp(result)
    app.title = "HMLS Replay Viewer"
    app.run()


if __name__ == "__main__":
    main()
