"""Tests for lethargy detection policies."""

from __future__ import annotations

import pytest

from hmls.core.types import Action
from hmls.reinforcetrainer.lethargy import (
    ConsecutiveTurnLimit,
    NoLethargyCheck,
)


class TestNoLethargyCheck:
    """Tests for NoLethargyCheck policy."""

    def test_always_returns_none(self) -> None:
        """No lethargy is ever detected."""
        policy = NoLethargyCheck()
        policy.reset()
        for _ in range(100):
            assert policy.observe_action("A1", Action.TURN_LEFT) is None

    def test_reset_is_noop(self) -> None:
        """Reset does not raise or change behaviour."""
        policy = NoLethargyCheck()
        policy.reset()
        policy.reset()
        assert policy.observe_action("A1", Action.TURN_RIGHT) is None


class TestConsecutiveTurnLimit:
    """Tests for ConsecutiveTurnLimit policy."""

    def test_triggers_at_threshold(self) -> None:
        """Lethargy triggers exactly at max_consecutive_turns."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_RIGHT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) == "A"

    def test_does_not_trigger_below_threshold(self) -> None:
        """No lethargy when turns are below the threshold."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=5)
        policy.reset()
        for _ in range(4):
            assert policy.observe_action("B1", Action.TURN_LEFT) is None

    def test_resets_on_non_turn_action(self) -> None:
        """A non-turn action resets the consecutive counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_RIGHT) is None
        # Non-turn action breaks the streak
        assert policy.observe_action("A1", Action.MOVE_FORWARD) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        # Now it triggers (3 consecutive)
        assert policy.observe_action("A1", Action.TURN_LEFT) == "A"

    def test_tracks_tanks_independently(self) -> None:
        """Each tank has its own counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        # Interleave actions from two tanks
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("B1", Action.TURN_RIGHT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("B1", Action.TURN_RIGHT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) == "A"
        # B1 hasn't reached threshold yet
        assert policy.observe_action("B1", Action.TURN_RIGHT) == "B"

    def test_returns_correct_team_for_tank_b(self) -> None:
        """Team B tanks trigger with team 'B'."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=2)
        policy.reset()
        assert policy.observe_action("B1", Action.TURN_RIGHT) is None
        assert policy.observe_action("B1", Action.TURN_RIGHT) == "B"

    def test_reset_clears_counters(self) -> None:
        """Reset clears all accumulated turn counts."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action("A1", Action.TURN_LEFT)
        policy.observe_action("A1", Action.TURN_LEFT)
        policy.reset()
        # Counter should be back to 0
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None

    def test_fire_resets_streak(self) -> None:
        """Fire action resets the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action("A1", Action.TURN_LEFT)
        policy.observe_action("A1", Action.TURN_LEFT)
        assert policy.observe_action("A1", Action.FIRE) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None

    def test_pass_resets_streak(self) -> None:
        """Pass action resets the consecutive turn counter."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        policy.observe_action("A1", Action.TURN_LEFT)
        policy.observe_action("A1", Action.TURN_LEFT)
        assert policy.observe_action("A1", Action.PASS) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) is None

    def test_min_threshold_is_two(self) -> None:
        """max_consecutive_turns must be at least 2."""
        with pytest.raises(ValueError, match="must be >= 2"):
            ConsecutiveTurnLimit(max_consecutive_turns=1)

    def test_default_threshold_is_five(self) -> None:
        """Default threshold is 5 consecutive turns."""
        policy = ConsecutiveTurnLimit()
        policy.reset()
        for _ in range(4):
            assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) == "A"

    def test_mixed_turn_directions_still_count(self) -> None:
        """Both TURN_LEFT and TURN_RIGHT count toward the same streak."""
        policy = ConsecutiveTurnLimit(max_consecutive_turns=3)
        policy.reset()
        assert policy.observe_action("A1", Action.TURN_LEFT) is None
        assert policy.observe_action("A1", Action.TURN_RIGHT) is None
        assert policy.observe_action("A1", Action.TURN_LEFT) == "A"
