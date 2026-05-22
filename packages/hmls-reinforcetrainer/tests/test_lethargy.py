"""Tests for lethargy detection policies."""

from __future__ import annotations

import pytest

from hmls.core.game_state import GameState
from hmls.core.results import HistoryEntry
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.reinforcetrainer.lethargy import (
    ConsecutiveTurnLimit,
    NoLethargyCheck,
)


def _make_entry(
    tank_id: str = "A1",
    action: Action = Action.MOVE_FORWARD,
    valid: bool = True,
    hit: bool | None = None,
) -> HistoryEntry:
    """Create a minimal HistoryEntry for lethargy testing.

    Args:
        tank_id: The tank performing the action.
        action: The requested action.
        valid: Whether the action is valid.
        hit: Whether a fire action hit.
    """
    team = tank_id[0]
    tank = Tank(
        id=tank_id,
        team=team,
        position=Position(1, 1),
        direction=Direction.NORTH,
    )
    state = GameState(tanks=[tank], current_tank_id=tank_id)
    return HistoryEntry(
        tank_id=tank_id,
        requested_action=action,
        applied_action=action if valid else Action.PASS,
        valid=valid,
        reason="" if valid else "test reason",
        hit=hit,
        state_after=state,
    )


class TestNoLethargyCheck:
    """Tests for NoLethargyCheck policy."""

    def test_always_returns_none(self) -> None:
        """No lethargy is ever detected."""
        policy = NoLethargyCheck()
        policy.reset()
        for _ in range(100):
            assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None

    def test_reset_is_noop(self) -> None:
        """Reset does not raise or change behaviour."""
        policy = NoLethargyCheck()
        policy.reset()
        policy.reset()
        assert policy.observe_action(_make_entry(action=Action.TURN_RIGHT)) is None


class TestConsecutiveTurnLimit:
    """Tests for ConsecutiveTurnLimit policy."""

    def test_triggers_at_threshold(self) -> None:
        """Lethargy triggers exactly at max_consecutive_turns."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_RIGHT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_does_not_trigger_below_threshold(self) -> None:
        """No lethargy when turns are below the threshold."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=5)
        policy.reset()
        for _ in range(4):
            assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_LEFT)) is None

    def test_resets_on_valid_move_forward(self) -> None:
        """A valid move forward resets the consecutive counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_RIGHT)) is None
        # Valid move forward breaks the streak
        assert policy.observe_action(_make_entry(action=Action.MOVE_FORWARD)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        # Now it triggers (3 consecutive)
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_tracks_tanks_independently(self) -> None:
        """Each tank has its own counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        # Interleave actions from two tanks
        assert policy.observe_action(_make_entry(tank_id="A1", action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_RIGHT)) is None
        assert policy.observe_action(_make_entry(tank_id="A1", action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_RIGHT)) is None
        assert policy.observe_action(_make_entry(tank_id="A1", action=Action.TURN_LEFT)) == "A"
        # B1 hasn't reached threshold yet
        assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_RIGHT)) == "B"

    def test_returns_correct_team_for_tank_b(self) -> None:
        """Team B tanks trigger with team 'B'."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=2)
        policy.reset()
        assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_RIGHT)) is None
        assert policy.observe_action(_make_entry(tank_id="B1", action=Action.TURN_RIGHT)) == "B"

    def test_reset_clears_counters(self) -> None:
        """Reset clears all accumulated turn counts."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.reset()
        # Counter should be back to 0
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None

    def test_fire_hit_resets_streak(self) -> None:
        """Fire-and-hit resets the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        assert policy.observe_action(_make_entry(action=Action.FIRE, hit=True)) is None
        # Streak reset: need 3 more turns to trigger
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_fire_miss_does_not_reset_streak(self) -> None:
        """Fire-and-miss does NOT reset the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        # Fire miss: streak stays at 2
        assert policy.observe_action(_make_entry(action=Action.FIRE, hit=False)) is None
        # One more turn triggers (streak was 2, now 3)
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_pass_does_not_reset_streak(self) -> None:
        """Pass action does NOT reset the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        # Pass: streak stays at 2
        assert policy.observe_action(_make_entry(action=Action.PASS)) is None
        # One more turn triggers (streak was 2, now 3)
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_invalid_move_does_not_reset_streak(self) -> None:
        """Invalid move does NOT reset the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        policy.observe_action(_make_entry(action=Action.TURN_LEFT))
        # Invalid move forward (applied_action becomes PASS): streak stays at 2
        assert policy.observe_action(_make_entry(action=Action.MOVE_FORWARD, valid=False)) is None
        # One more turn triggers (streak was 2, now 3)
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_min_threshold_is_two(self) -> None:
        """max_consecutive_turns must be at least 2."""
        with pytest.raises(ValueError, match="must be >= 2"):
            ConsecutiveTurnLimit(max_consecutive_turns=1)

    def test_default_threshold_is_five(self) -> None:
        """Default threshold is 5 consecutive turns."""
        policy = ConsecutiveTurnLimit()
        policy.reset()
        for _ in range(4):
            assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"

    def test_mixed_turn_directions_still_count(self) -> None:
        """Both TURN_LEFT and TURN_RIGHT count toward the same streak."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_RIGHT)) is None
        assert policy.observe_action(_make_entry(action=Action.TURN_LEFT)) == "A"
