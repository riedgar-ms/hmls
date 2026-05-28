"""Main Textual TUI application for the HMLS game replay viewer.

Lets the user step through a recorded game history, watching the map
and tank positions update at each turn.  Supports both manual
stepping (arrow keys) and automatic playback with adjustable speed.

Action logs are displayed per-tank in a grid (one column per team),
so the user can see each tank's recent actions at a glance.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.timer import Timer
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import GameResult, HistoryEntry
from hmls.core.tank import TankId
from hmls.replayviewer.cli import build_state_timeline, load_game_result, parse_args
from hmls.uxcommon import LogStatusMixin
from hmls.uxcommon.log_tab import LogTabMixin
from hmls.uxcommon.mixins import format_turn_status
from hmls.uxcommon.styles import TEAM_STYLES
from hmls.uxcommon.widgets.map_view import MapView
from hmls.uxcommon.widgets.team_legend import TeamLegend

logger = logging.getLogger("hmls.replayviewer")

# ── Constants ─────────────────────────────────────────────────────────

_MIN_DELAY: float = 0.1
"""Minimum auto-play delay in seconds."""

_MAX_DELAY: float = 5.0
"""Maximum auto-play delay in seconds."""

_DEFAULT_DELAY: float = 0.5
"""Default auto-play delay in seconds."""

_DELAY_STEP: float = 0.1
"""Amount to change delay per key press."""

_MAX_LOG_LINES: int = 10
"""Maximum number of recent entries shown per tank log."""


def _tank_log_id(tank_id: TankId) -> str:
    """Return the DOM id for a tank's per-tank ``RichLog`` widget.

    Args:
        tank_id: The tank identifier (e.g. ``"A1"``).

    Returns:
        A CSS-safe DOM id string (e.g. ``"log-tank-A1"``).
    """
    return f"log-tank-{tank_id}"


def _tank_panel_id(tank_id: TankId) -> str:
    """Return the DOM id for a tank's surrounding panel container.

    Args:
        tank_id: The tank identifier.

    Returns:
        A CSS-safe DOM id string (e.g. ``"panel-tank-A1"``).
    """
    return f"panel-tank-{tank_id}"


def compute_tank_logs(
    history: list[HistoryEntry],
    current_step: int,
    all_tank_ids: list[TankId],
) -> dict[TankId, list[str]]:
    """Compute formatted log lines per tank for entries up to the current step.

    This is the pure-computation core of ``_rebuild_log``, factored out
    for testability.

    Args:
        history: The full game history list.
        current_step: Current replay step (entries ``[:current_step]`` are shown).
        all_tank_ids: Ordered list of all tank IDs in the game.

    Returns:
        Mapping from tank ID to a list of formatted log-line strings
        (most recent ``_MAX_LOG_LINES`` entries only).
    """
    per_tank: dict[TankId, list[HistoryEntry]] = defaultdict(list)
    for entry in history[:current_step]:
        per_tank[entry.tank_id].append(entry)

    result: dict[TankId, list[str]] = {}
    for tid in all_tank_ids:
        entries = per_tank.get(tid, [])
        lines: list[str] = []
        for entry in entries[-_MAX_LOG_LINES:]:
            status = format_turn_status(entry.valid, entry.reason, entry.hit)
            lines.append(f"  {entry.applied_action.value} — {status}")
        result[tid] = lines
    return result


def compute_toggle_play_state(
    playing: bool,
    current_step: int,
    max_step: int,
) -> tuple[bool, int | None]:
    """Compute the new play state after a toggle-play action.

    This is the pure-logic core of ``action_toggle_play``, factored out
    for testability.

    Args:
        playing: Whether playback is currently active.
        current_step: The current step index.
        max_step: The maximum valid step index.

    Returns:
        A tuple of ``(new_playing, new_step)`` where *new_step* is ``None``
        if the step should not change, or an integer if it should be reset.
    """
    if playing:
        return (False, None)
    # Starting playback: if at end, restart from beginning.
    if current_step >= max_step:
        return (True, 0)
    return (True, None)


def compute_clamped_step(step: int, max_step: int) -> int:
    """Clamp a step value to the valid range ``[0, max_step]``.

    Args:
        step: The desired step index (may be out of bounds).
        max_step: The maximum valid step index.

    Returns:
        The clamped step value.
    """
    return max(0, min(step, max_step))


def compute_new_delay(current_delay: float, direction: int) -> float:
    """Compute the new auto-play delay after a speed adjustment.

    Args:
        current_delay: The current delay in seconds.
        direction: ``-1`` to speed up (decrease delay),
            ``+1`` to slow down (increase delay).

    Returns:
        The new delay, clamped to ``[_MIN_DELAY, _MAX_DELAY]``.
    """
    new_delay = round(current_delay + direction * _DELAY_STEP, 1)
    return max(_MIN_DELAY, min(_MAX_DELAY, new_delay))


class ReplayViewerApp(LogTabMixin, LogStatusMixin, App[None]):
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
    #log-scroll {
        height: auto;
        max-height: 12;
        border-top: solid $primary;
    }
    #log-grid {
        height: auto;
    }
    .team-col {
        width: 1fr;
        height: auto;
    }
    .tank-panel {
        height: auto;
        border: solid $secondary;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    .tank-panel.active-tank {
        border: solid $success;
    }
    .tank-label {
        height: 1;
        padding: 0;
    }
    .tank-log {
        height: auto;
        max-height: 6;
    }
    #log-panel {
        display: none;
    }
    #status-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True, show=True),
        Binding("space", "toggle_play", "Play/Pause", priority=True, show=True),
        Binding("left", "step_back", "← Back", priority=True, show=True),
        Binding("right", "step_forward", "→ Forward", priority=True, show=True),
        Binding("up", "speed_up", "Speed up", priority=True, show=True),
        Binding("down", "slow_down", "Slow down", priority=True, show=True),
        Binding("home", "jump_start", "Start", priority=True, show=True),
        Binding("end", "jump_end", "End", priority=True, show=True),
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

        # Build an ordered mapping of team → list of tank IDs.
        teams: dict[str, list[TankId]] = defaultdict(list)
        for tank in result.initial_state.tanks:
            teams[tank.team].append(tank.id)
        self._teams: dict[str, list[TankId]] = dict(sorted(teams.items()))
        self._all_tank_ids: list[TankId] = [tid for tids in self._teams.values() for tid in tids]

        self._current_step: int = 0
        self._playing: bool = False
        self._delay: float = _DEFAULT_DELAY
        self._timer: Timer | None = None

    # ── Layout ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with TabbedContent(initial="game-tab"):
            with TabPane("Game", id="game-tab"):
                yield TeamLegend(TEAM_STYLES, id="team-legend")

                with ScrollableContainer(id="map-scroll"):
                    yield MapView(
                        self._game_map,
                        self._states[0],
                        id="map-view",
                    )

                # Per-tank log grid: one column per team, one panel per tank.
                with ScrollableContainer(id="log-scroll"):
                    with Horizontal(id="log-grid"):
                        for team, tank_ids in self._teams.items():
                            style = TEAM_STYLES.get(team, "")
                            with Vertical(classes="team-col"):
                                for tid in tank_ids:
                                    with Vertical(id=_tank_panel_id(tid), classes="tank-panel"):
                                        yield Static(
                                            f"[{style}]{tid}[/{style}]",
                                            classes="tank-label",
                                        )
                                        yield RichLog(
                                            id=_tank_log_id(tid),
                                            highlight=True,
                                            markup=True,
                                            max_lines=_MAX_LOG_LINES,
                                            classes="tank-log",
                                        )

                # Hidden RichLog keeps LogStatusMixin._write_log working.
                yield RichLog(id="log-panel")
                yield Static(self._build_status_text(), id="status-bar")
            yield from self._compose_log_tab()
        yield Footer()

    def on_mount(self) -> None:
        """Set initial active tank highlight."""
        self._setup_log_tab()
        self._update_display()

    # ── Navigation helpers ────────────────────────────────────────

    @property
    def _max_step(self) -> int:
        """Maximum valid step index."""
        return len(self._states) - 1

    def _active_tank_id(self) -> TankId:
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
        step = compute_clamped_step(step, self._max_step)
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

        # Rebuild the log panel with entries up to current step.
        self._rebuild_log()

        # Update status bar.
        status = self.query_one("#status-bar", Static)
        status.update(self._build_status_text())

    # ── Log & status bar ─────────────────────────────────────────

    def _rebuild_log(self) -> None:
        """Clear and rebuild per-tank log panels with entries up to the current step."""
        tank_logs = compute_tank_logs(self._history, self._current_step, self._all_tank_ids)
        active_id = self._active_tank_id()

        for tid in self._all_tank_ids:
            try:
                log_widget = self.query_one(f"#{_tank_log_id(tid)}", RichLog)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to find log widget for tank %s", tid, exc_info=True)
                continue

            log_widget.clear()
            for line in tank_logs.get(tid, []):
                log_widget.write(line)

            # Highlight the panel of the tank that just acted.
            try:
                panel = self.query_one(f"#{_tank_panel_id(tid)}")
            except Exception:  # noqa: BLE001
                logger.debug("Failed to find panel for tank %s", tid, exc_info=True)
                continue
            if tid == active_id:
                panel.add_class("active-tank")
            else:
                panel.remove_class("active-tank")

    def _build_status_text(self) -> str:
        """Build the status bar text showing current position and controls."""
        step = self._current_step
        total = self._max_step
        play_state = "▶ Playing" if self._playing else "⏸ Paused"
        delay_str = f"{self._delay:.1f}s"

        # Winner info.
        winner = self._result.winner
        winner_str = ""
        if step == total:
            if winner:
                winner_str = f" | Winner: Team {winner}"
            else:
                winner_str = " | Draw"

        return f"Step {step}/{total} | {play_state} | Delay: {delay_str}{winner_str}"

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
        new_playing, new_step = compute_toggle_play_state(
            self._playing, self._current_step, self._max_step
        )
        if new_step is not None:
            self._current_step = new_step
            self._update_display()
        self._playing = new_playing
        if self._playing:
            self._start_timer()
        else:
            self._stop_timer()
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
        self._delay = compute_new_delay(self._delay, -1)
        if self._playing:
            self._start_timer()  # Restart with new delay.
        self._update_status_only()

    def action_slow_down(self) -> None:
        """Increase the auto-play delay (slow down)."""
        self._delay = compute_new_delay(self._delay, +1)
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
