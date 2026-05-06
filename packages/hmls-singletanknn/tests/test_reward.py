"""Tests for the reward module."""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.singletanknn.reward import DefaultReward, DefaultRewardConfig


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
    """Invalid action incurs the dedicated invalid_move_penalty."""
    reward_fn = DefaultReward()
    entry = _make_entry(valid=False)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    # step_penalty + invalid_move_penalty
    assert abs(reward - (-0.01 + -0.1)) < 1e-7


def test_default_reward_fire_miss() -> None:
    """Firing and missing incurs the fire_miss_penalty."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=False)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    # step_penalty + fire_miss_penalty
    assert abs(reward - (-0.01 + -0.05)) < 1e-7


def test_default_reward_hit_no_miss_penalty() -> None:
    """A successful hit does NOT incur the fire miss penalty."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    reward = reward_fn.compute_step_reward(
        entry, explored_positions=set(), new_positions_this_step=0
    )
    # step_penalty + hit_reward only (no fire_miss_penalty)
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


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


def test_default_reward_from_config() -> None:
    """Construction from an explicit config works correctly."""
    config = DefaultRewardConfig(hit_reward=1.0, step_penalty=-0.05)
    reward_fn = DefaultReward(config=config)
    assert reward_fn.hit_reward == 1.0
    assert reward_fn.step_penalty == -0.05
    # Other defaults are preserved
    assert reward_fn.win_reward == 1.0
    assert reward_fn.exploration_bonus == 0.02


def test_default_reward_config_round_trip() -> None:
    """Config survives serialisation round-trip via model_dump/model_validate."""
    config = DefaultRewardConfig(
        hit_reward=0.8,
        death_penalty=-0.5,
        win_reward=2.0,
        loss_penalty=-2.0,
        step_penalty=-0.02,
        exploration_bonus=0.05,
        invalid_move_penalty=-0.2,
        fire_miss_penalty=-0.08,
    )
    dumped = config.model_dump()
    restored = DefaultRewardConfig.model_validate(dumped)
    assert restored == config


def test_default_reward_config_json_round_trip() -> None:
    """Config survives JSON serialisation round-trip."""
    config = DefaultRewardConfig(hit_reward=0.75)
    json_str = config.model_dump_json()
    restored = DefaultRewardConfig.model_validate_json(json_str)
    assert restored == config


def test_default_reward_config_is_frozen() -> None:
    """Config should be immutable."""
    config = DefaultRewardConfig()
    try:
        config.hit_reward = 999.0  # type: ignore[misc]
        assert False, "Should have raised"
    except TypeError, ValueError, AttributeError:
        pass
