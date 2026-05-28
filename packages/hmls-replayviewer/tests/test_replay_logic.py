"""Unit tests for extracted pure functions in app.py."""

from __future__ import annotations

import pytest

from hmls.core.results import HistoryEntry
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.replayviewer.app import (
    _MAX_LOG_LINES,
    compute_tank_logs,
    compute_toggle_play_state,
)

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_entry(tank_id: str, action: Action = Action.PASS, valid: bool = True) -> HistoryEntry:
    """Create a minimal HistoryEntry for testing."""
    from hmls.core.game_state import GameState

    tank = Tank(id=tank_id, team="A", position=Position(1, 1), direction=Direction.NORTH)
    state = GameState(tanks=[tank], current_tank_id=tank_id)
    return HistoryEntry(
        tank_id=tank_id,
        requested_action=action,
        applied_action=action,
        valid=valid,
        state_after=state,
    )


# ── compute_tank_logs tests ───────────────────────────────────────────


class TestComputeTankLogs:
    """Tests for the ``compute_tank_logs`` pure function."""

    def test_empty_history(self) -> None:
        """No history entries produces empty log lists."""
        result = compute_tank_logs([], current_step=0, all_tank_ids=["A1", "B1"])
        assert result == {"A1": [], "B1": []}

    def test_step_zero_shows_nothing(self) -> None:
        """At step 0, no entries are shown even if history exists."""
        history = [_make_entry("A1")]
        result = compute_tank_logs(history, current_step=0, all_tank_ids=["A1"])
        assert result == {"A1": []}

    def test_step_one_shows_first_entry(self) -> None:
        """At step 1, the first entry is visible."""
        history = [_make_entry("A1", Action.MOVE_FORWARD)]
        result = compute_tank_logs(history, current_step=1, all_tank_ids=["A1"])
        assert len(result["A1"]) == 1
        assert "move_forward" in result["A1"][0].lower() or "Move" in result["A1"][0]

    def test_entries_bucketed_by_tank(self) -> None:
        """Entries are correctly bucketed per tank."""
        history = [
            _make_entry("A1", Action.MOVE_FORWARD),
            _make_entry("B1", Action.FIRE),
            _make_entry("A1", Action.PASS),
        ]
        result = compute_tank_logs(history, current_step=3, all_tank_ids=["A1", "B1"])
        assert len(result["A1"]) == 2
        assert len(result["B1"]) == 1

    def test_max_log_lines_limit(self) -> None:
        """Only the most recent _MAX_LOG_LINES entries are returned."""
        history = [_make_entry("A1", Action.PASS) for _ in range(_MAX_LOG_LINES + 5)]
        result = compute_tank_logs(history, current_step=len(history), all_tank_ids=["A1"])
        assert len(result["A1"]) == _MAX_LOG_LINES

    def test_tank_with_no_entries_gets_empty_list(self) -> None:
        """Tanks that haven't acted still appear with empty list."""
        history = [_make_entry("A1")]
        result = compute_tank_logs(history, current_step=1, all_tank_ids=["A1", "B1"])
        assert result["B1"] == []

    def test_invalid_entry_formats_status(self) -> None:
        """Invalid entries still produce formatted output."""
        entry = _make_entry("A1", Action.MOVE_FORWARD, valid=False)
        entry = entry.model_copy(update={"reason": "blocked"})
        result = compute_tank_logs([entry], current_step=1, all_tank_ids=["A1"])
        assert len(result["A1"]) == 1
        # The line should contain some indication of the action
        assert "move_forward" in result["A1"][0].lower() or "Move" in result["A1"][0]


# ── compute_toggle_play_state tests ───────────────────────────────────


class TestComputeTogglePlayState:
    """Tests for the ``compute_toggle_play_state`` pure function."""

    @pytest.mark.parametrize(
        ("playing", "current_step", "max_step", "expected_playing", "expected_step"),
        [
            pytest.param(True, 3, 5, False, None, id="pause_when_playing"),
            pytest.param(False, 3, 5, True, None, id="start_from_middle"),
            pytest.param(False, 5, 5, True, 0, id="restart_from_end"),
            pytest.param(False, 0, 5, True, None, id="start_at_zero"),
            pytest.param(False, 0, 0, True, 0, id="single_step_game_at_end"),
        ],
    )
    def test_toggle_play_state(
        self,
        playing: bool,
        current_step: int,
        max_step: int,
        expected_playing: bool,
        expected_step: int | None,
    ) -> None:
        """Verify toggle play state transitions."""
        new_playing, new_step = compute_toggle_play_state(
            playing=playing, current_step=current_step, max_step=max_step
        )
        assert new_playing is expected_playing
        assert new_step == expected_step
