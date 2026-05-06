"""Tests for the reward module."""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.singletanknn.reward import DefaultReward


def _make_entry(
    action: Action = Action.MOVE_FORWARD,
    valid: bool = True,
    hit: bool | None = None,
) -> HistoryEntry:
    """Create a minimal HistoryEntry for testing."""
    tank = Tank(id="t1", team="alpha", position=Position(1, 1), direction=Direction.NORTH)
    state = GameState(tanks=[tank], current_tank_id="t1")
    return HistoryEntry(
        tank_id="t1",
        requested_action=action,
        applied_action=action if valid else Action.PASS,
        valid=valid,
        reason="" if valid else "test reason",
        hit=hit,
        state_after=state,
    )


def test_default_reward_step_penalty() -> None:
    """A plain step incurs the step penalty."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    assert abs(reward - (-0.01)) < 1e-7


def test_default_reward_hit() -> None:
    """A successful hit adds the hit reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_exploration() -> None:
    """New positions add exploration bonus."""
    reward_fn = DefaultReward()
    entry = _make_entry()
    reward = reward_fn.compute_step_reward(
        entry, explored_positions={Position(1, 1)}, new_positions_this_step=3
    )
    expected = -0.01 + 0.02 * 3
    assert abs(reward - expected) < 1e-7


def test_default_reward_invalid_action() -> None:
    """Invalid action incurs extra penalty."""
    reward_fn = DefaultReward()
    entry = _make_entry(valid=False)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    # Double step penalty
    assert abs(reward - (-0.01 + -0.01)) < 1e-7


def test_default_reward_episode_win() -> None:
    """Winning gives positive terminal reward."""
    reward_fn = DefaultReward()
    assert reward_fn.compute_episode_end_reward(won=True, total_explored=10) == 1.0


def test_default_reward_episode_loss() -> None:
    """Losing gives negative terminal reward."""
    reward_fn = DefaultReward()
    assert reward_fn.compute_episode_end_reward(won=False, total_explored=10) == -1.0


def test_default_reward_episode_draw() -> None:
    """Draw gives zero terminal reward."""
    reward_fn = DefaultReward()
    assert reward_fn.compute_episode_end_reward(won=None, total_explored=10) == 0.0
