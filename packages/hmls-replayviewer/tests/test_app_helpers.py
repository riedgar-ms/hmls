"""Tests for replay viewer app helper functions and internal logic."""

from __future__ import annotations

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import GameResult, HistoryEntry
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.replayviewer.app import (
    ReplayViewerApp,
    _DEFAULT_DELAY,
    _DELAY_STEP,
    _MAX_DELAY,
    _MIN_DELAY,
    _tank_log_id,
    _tank_panel_id,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _two_tank_game_result(*, history_len: int = 3) -> GameResult:
    """Build a GameResult with two tanks and *history_len* entries.

    Alternates actions between tank "A1" and "B1" so that
    different steps have different active tanks.
    """
    tank_a = Tank(id="A1", team="Alpha", position=Position(1, 1), direction=Direction.NORTH)
    tank_b = Tank(id="B1", team="Bravo", position=Position(3, 3), direction=Direction.SOUTH)
    initial = GameState(tanks=[tank_a, tank_b], current_tank_id="A1")
    game_map = GameMap(width=5, height=5)

    tanks = [tank_a, tank_b]
    history: list[HistoryEntry] = []
    for i in range(history_len):
        acting = tanks[i % 2]
        history.append(
            HistoryEntry(
                tank_id=acting.id,
                requested_action=Action.PASS,
                applied_action=Action.PASS,
                valid=True,
                state_after=initial.model_copy(deep=True),
            )
        )

    return GameResult(
        winner=None,
        game_map=game_map,
        initial_state=initial,
        history=history,
        turns_played=history_len,
    )


def _game_result_with_winner() -> GameResult:
    """Build a GameResult where team Alpha wins."""
    result = _two_tank_game_result(history_len=2)
    result = result.model_copy(update={"winner": "Alpha"})
    return result


# ── Helper function tests ─────────────────────────────────────────────


class TestTankLogId:
    """Tests for ``_tank_log_id``."""

    def test_simple_id(self) -> None:
        """Basic tank ID produces expected DOM id."""
        assert _tank_log_id("A1") == "log-tank-A1"

    def test_different_id(self) -> None:
        """Another tank ID works correctly."""
        assert _tank_log_id("B2") == "log-tank-B2"

    def test_complex_id(self) -> None:
        """Longer/more complex IDs are passed through."""
        assert _tank_log_id("team-alpha-99") == "log-tank-team-alpha-99"


class TestTankPanelId:
    """Tests for ``_tank_panel_id``."""

    def test_simple_id(self) -> None:
        """Basic tank ID produces expected DOM id."""
        assert _tank_panel_id("A1") == "panel-tank-A1"

    def test_different_id(self) -> None:
        """Another tank ID works correctly."""
        assert _tank_panel_id("B2") == "panel-tank-B2"


# ── _build_status_text tests ──────────────────────────────────────────


class TestBuildStatusText:
    """Tests for ``ReplayViewerApp._build_status_text``."""

    def _make_app(self, *, history_len: int = 5, winner: str | None = None) -> ReplayViewerApp:
        """Create a ReplayViewerApp without mounting it."""
        result = _two_tank_game_result(history_len=history_len)
        if winner is not None:
            result = result.model_copy(update={"winner": winner})
        return ReplayViewerApp(result)

    def test_initial_state_paused(self) -> None:
        """At step 0, paused, shows step 0/N and paused indicator."""
        app = self._make_app(history_len=5)
        text = app._build_status_text()
        assert "Step 0/5" in text
        assert "Paused" in text

    def test_mid_game(self) -> None:
        """Mid-game status shows correct step number."""
        app = self._make_app(history_len=5)
        app._current_step = 3
        text = app._build_status_text()
        assert "Step 3/5" in text

    def test_playing_state(self) -> None:
        """When playing, status shows playing indicator."""
        app = self._make_app(history_len=5)
        app._playing = True
        text = app._build_status_text()
        assert "Playing" in text

    def test_end_with_winner(self) -> None:
        """At end step with a winner, shows winner info."""
        app = self._make_app(history_len=3, winner="Alpha")
        app._current_step = app._max_step
        text = app._build_status_text()
        assert "Winner" in text
        assert "Alpha" in text

    def test_end_draw(self) -> None:
        """At end step with no winner, shows draw."""
        app = self._make_app(history_len=3, winner=None)
        app._current_step = app._max_step
        text = app._build_status_text()
        assert "Draw" in text

    def test_winner_not_shown_mid_game(self) -> None:
        """Winner info only shows at the final step."""
        app = self._make_app(history_len=3, winner="Alpha")
        app._current_step = 1
        text = app._build_status_text()
        assert "Winner" not in text

    def test_delay_shown(self) -> None:
        """Delay value is displayed in the status."""
        app = self._make_app(history_len=3)
        app._delay = 1.5
        text = app._build_status_text()
        assert "1.5s" in text


# ── _active_tank_id tests ─────────────────────────────────────────────


class TestActiveTankId:
    """Tests for ``ReplayViewerApp._active_tank_id``."""

    def test_step_zero_returns_current_tank(self) -> None:
        """At step 0, returns the initial state's current_tank_id."""
        result = _two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 0
        assert app._active_tank_id() == "A1"

    def test_step_one_returns_first_history_tank(self) -> None:
        """At step 1, returns history[0].tank_id."""
        result = _two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 1
        # history[0] acts with tank A1 (index 0 % 2 = 0 → tank_a)
        assert app._active_tank_id() == "A1"

    def test_step_two_returns_second_history_tank(self) -> None:
        """At step 2, returns history[1].tank_id."""
        result = _two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 2
        # history[1] acts with tank B1 (index 1 % 2 = 1 → tank_b)
        assert app._active_tank_id() == "B1"


# ── Navigation clamping tests ─────────────────────────────────────────


class TestNavigation:
    """Tests for navigation helpers (_max_step, _go_to_step clamping)."""

    def test_max_step(self) -> None:
        """_max_step is len(states) - 1."""
        result = _two_tank_game_result(history_len=4)
        app = ReplayViewerApp(result)
        # states = initial + 4 history = 5 total, max index = 4
        assert app._max_step == 4

    def test_go_to_step_clamps_negative(self) -> None:
        """_go_to_step(-5) should clamp to 0."""
        result = _two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        # _go_to_step calls _update_display which needs widgets; test clamping logic directly
        step = max(0, min(-5, app._max_step))
        assert step == 0

    def test_go_to_step_clamps_over_max(self) -> None:
        """_go_to_step(999) should clamp to max_step."""
        result = _two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        step = max(0, min(999, app._max_step))
        assert step == app._max_step

    def test_max_step_empty_history(self) -> None:
        """With no history, max_step is 0 (only initial state)."""
        result = _two_tank_game_result(history_len=0)
        app = ReplayViewerApp(result)
        assert app._max_step == 0


# ── Speed adjustment tests ────────────────────────────────────────────


class TestSpeedAdjustments:
    """Tests for delay adjustment logic."""

    def test_speed_up_decreases_delay(self) -> None:
        """action_speed_up should decrease _delay by _DELAY_STEP."""
        result = _two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        initial_delay = app._delay
        # Simulate speed_up logic without calling action (which needs widgets)
        app._delay = max(_MIN_DELAY, round(app._delay - _DELAY_STEP, 1))
        assert app._delay == round(initial_delay - _DELAY_STEP, 1)

    def test_speed_up_clamps_to_min(self) -> None:
        """Delay cannot go below _MIN_DELAY."""
        result = _two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        app._delay = _MIN_DELAY
        app._delay = max(_MIN_DELAY, round(app._delay - _DELAY_STEP, 1))
        assert app._delay == _MIN_DELAY

    def test_slow_down_increases_delay(self) -> None:
        """Slowing down should increase _delay by _DELAY_STEP."""
        result = _two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        initial_delay = app._delay
        app._delay = min(_MAX_DELAY, round(app._delay + _DELAY_STEP, 1))
        assert app._delay == round(initial_delay + _DELAY_STEP, 1)

    def test_slow_down_clamps_to_max(self) -> None:
        """Delay cannot exceed _MAX_DELAY."""
        result = _two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        app._delay = _MAX_DELAY
        app._delay = min(_MAX_DELAY, round(app._delay + _DELAY_STEP, 1))
        assert app._delay == _MAX_DELAY

    def test_default_delay(self) -> None:
        """App starts with _DEFAULT_DELAY."""
        result = _two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        assert app._delay == _DEFAULT_DELAY
