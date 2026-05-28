"""Tests for replay viewer app helper functions and internal logic."""

from __future__ import annotations

import pytest

from hmls.core.results import GameResult
from hmls.replayviewer.app import (
    _DEFAULT_DELAY,
    _DELAY_STEP,
    _MAX_DELAY,
    _MIN_DELAY,
    ReplayViewerApp,
    _tank_log_id,
    _tank_panel_id,
    compute_clamped_step,
    compute_new_delay,
)

from ._fixtures import make_two_tank_game_result


def _game_result_with_winner() -> GameResult:
    """Build a GameResult where team Alpha wins."""
    return make_two_tank_game_result(history_len=2, winner="Alpha")


# ── Helper function tests ─────────────────────────────────────────────


class TestTankLogId:
    """Tests for ``_tank_log_id``."""

    @pytest.mark.parametrize(
        ("tank_id", "expected"),
        [
            ("A1", "log-tank-A1"),
            ("B2", "log-tank-B2"),
            ("team-alpha-99", "log-tank-team-alpha-99"),
        ],
    )
    def test_produces_expected_dom_id(self, tank_id: str, expected: str) -> None:
        """Tank IDs produce correct DOM ids."""
        assert _tank_log_id(tank_id) == expected


class TestTankPanelId:
    """Tests for ``_tank_panel_id``."""

    @pytest.mark.parametrize(
        ("tank_id", "expected"),
        [
            ("A1", "panel-tank-A1"),
            ("B2", "panel-tank-B2"),
        ],
    )
    def test_produces_expected_dom_id(self, tank_id: str, expected: str) -> None:
        """Tank IDs produce correct DOM ids."""
        assert _tank_panel_id(tank_id) == expected


# ── _build_status_text tests ──────────────────────────────────────────


class TestBuildStatusText:
    """Tests for ``ReplayViewerApp._build_status_text``."""

    def _make_app(self, *, history_len: int = 5, winner: str | None = None) -> ReplayViewerApp:
        """Create a ReplayViewerApp without mounting it."""
        result = make_two_tank_game_result(history_len=history_len)
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
        result = make_two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 0
        assert app._active_tank_id() == "A1"

    def test_step_one_returns_first_history_tank(self) -> None:
        """At step 1, returns history[0].tank_id."""
        result = make_two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 1
        # history[0] acts with tank A1 (index 0 % 2 = 0 → tank_a)
        assert app._active_tank_id() == "A1"

    def test_step_two_returns_second_history_tank(self) -> None:
        """At step 2, returns history[1].tank_id."""
        result = make_two_tank_game_result(history_len=3)
        app = ReplayViewerApp(result)
        app._current_step = 2
        # history[1] acts with tank B1 (index 1 % 2 = 1 → tank_b)
        assert app._active_tank_id() == "B1"


# ── Navigation clamping tests ─────────────────────────────────────────


class TestNavigation:
    """Tests for navigation helpers (_max_step, compute_clamped_step)."""

    def test_max_step(self) -> None:
        """_max_step is len(states) - 1."""
        result = make_two_tank_game_result(history_len=4)
        app = ReplayViewerApp(result)
        # states = initial + 4 history = 5 total, max index = 4
        assert app._max_step == 4

    def test_clamp_negative_to_zero(self) -> None:
        """Negative step values clamp to 0."""
        assert compute_clamped_step(-5, max_step=3) == 0

    def test_clamp_over_max(self) -> None:
        """Step values exceeding max_step clamp to max_step."""
        assert compute_clamped_step(999, max_step=3) == 3

    def test_clamp_within_range(self) -> None:
        """Valid step values are returned unchanged."""
        assert compute_clamped_step(2, max_step=5) == 2

    def test_clamp_at_boundaries(self) -> None:
        """Boundary values (0 and max_step) are valid."""
        assert compute_clamped_step(0, max_step=5) == 0
        assert compute_clamped_step(5, max_step=5) == 5

    def test_max_step_empty_history(self) -> None:
        """With no history, max_step is 0 (only initial state)."""
        result = make_two_tank_game_result(history_len=0)
        app = ReplayViewerApp(result)
        assert app._max_step == 0


# ── Speed adjustment tests ────────────────────────────────────────────


class TestSpeedAdjustments:
    """Tests for delay adjustment logic via compute_new_delay."""

    def test_speed_up_decreases_delay(self) -> None:
        """Speeding up decreases delay by _DELAY_STEP."""
        new_delay = compute_new_delay(_DEFAULT_DELAY, -1)
        assert new_delay == round(_DEFAULT_DELAY - _DELAY_STEP, 1)

    def test_speed_up_clamps_to_min(self) -> None:
        """Delay cannot go below _MIN_DELAY."""
        new_delay = compute_new_delay(_MIN_DELAY, -1)
        assert new_delay == _MIN_DELAY

    def test_slow_down_increases_delay(self) -> None:
        """Slowing down increases delay by _DELAY_STEP."""
        new_delay = compute_new_delay(_DEFAULT_DELAY, +1)
        assert new_delay == round(_DEFAULT_DELAY + _DELAY_STEP, 1)

    def test_slow_down_clamps_to_max(self) -> None:
        """Delay cannot exceed _MAX_DELAY."""
        new_delay = compute_new_delay(_MAX_DELAY, +1)
        assert new_delay == _MAX_DELAY

    def test_default_delay(self) -> None:
        """App starts with _DEFAULT_DELAY."""
        result = make_two_tank_game_result(history_len=1)
        app = ReplayViewerApp(result)
        assert app._delay == _DEFAULT_DELAY
