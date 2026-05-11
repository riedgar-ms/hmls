"""Tests for the reward module."""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, TankPatch, VisibleCell
from hmls.nncore.reward import BasicReward, BasicRewardConfig, FocusedReward, FocusedRewardConfig


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


def _make_empty_patch(size: int = 9) -> TankPatch:
    """Create a patch with all passable visible cells and no tanks."""
    grid: list[list[VisibleCell | FogCell]] = []
    for _row in range(size):
        row_cells: list[VisibleCell | FogCell] = []
        for _col in range(size):
            row_cells.append(VisibleCell(cell_type=CellType.PASSABLE))
        grid.append(row_cells)
    return TankPatch(
        tank_id="t1",
        position=Position(1, 1),
        direction=Direction.NORTH,
        grid=grid,
    )


def _make_patch_with_enemy_ahead(size: int = 9) -> TankPatch:
    """Create a patch with an alive enemy directly ahead (one cell forward)."""
    patch = _make_empty_patch(size)
    half = size // 2
    enemy = Tank(id="e1", team="bravo", position=Position(1, 0), direction=Direction.SOUTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy)
    return patch


def _make_patch_with_enemy_in_cone(size: int = 9) -> TankPatch:
    """Create a patch with an alive enemy in the forward cone but not directly ahead."""
    patch = _make_empty_patch(size)
    half = size // 2
    enemy = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    # Place enemy two rows ahead, one column right (still in 45° cone)
    patch.grid[half - 2][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy)
    return patch


def _make_patch_with_fogged_enemy(size: int = 9) -> TankPatch:
    """Create a patch with a fog cell where an enemy would be in the cone."""
    patch = _make_empty_patch(size)
    half = size // 2
    # Put fog in the forward cone area
    patch.grid[half - 2][half] = FogCell()
    return patch


# ── Existing reward tests (updated for new signature) ────────────────


def test_default_reward_step_reward() -> None:
    """A plain step incurs the step reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    assert abs(reward - (-0.01)) < 1e-7


def test_default_reward_hit() -> None:
    """A successful hit adds the hit reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_exploration() -> None:
    """New positions add exploration bonus."""
    reward_fn = BasicReward()
    reward_fn.reset()
    entry = _make_entry()
    patch = _make_empty_patch(size=3)
    # First observation discovers all cells; a 3x3 fully visible patch = 9 cells
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + exploration_reward * 9 (all cells new)
    expected = -0.01 + 0.02 * 9
    assert abs(reward - expected) < 1e-7

    # Second observation of same patch discovers nothing new
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_default_reward_invalid_action() -> None:
    """Invalid action incurs the dedicated invalid_move_reward."""
    reward_fn = BasicReward()
    entry = _make_entry(valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + invalid_move_reward
    assert abs(reward - (-0.01 + -0.1)) < 1e-7


def test_default_reward_fire_miss() -> None:
    """Firing and missing incurs the fire_miss_reward (penalty)."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.FIRE, hit=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + fire_miss_reward
    assert abs(reward - (-0.01 + -0.05)) < 1e-7


def test_default_reward_hit_no_miss_penalty() -> None:
    """A successful hit does NOT incur the fire miss reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + fire_hit_reward only (no fire_miss_reward)
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_episode_win() -> None:
    """Winning gives positive terminal reward."""
    reward_fn = BasicReward()
    assert reward_fn.compute_episode_end_reward(won=True) == 1.0


def test_default_reward_episode_loss() -> None:
    """Losing gives negative terminal reward."""
    reward_fn = BasicReward()
    assert reward_fn.compute_episode_end_reward(won=False) == -1.0


def test_default_reward_episode_draw() -> None:
    """Draw gives zero terminal reward."""
    reward_fn = BasicReward()
    assert reward_fn.compute_episode_end_reward(won=None) == 0.0


def test_default_reward_from_config() -> None:
    """Construction from an explicit config works correctly."""
    config = BasicRewardConfig(fire_hit_reward=1.0, step_reward=-0.05)
    reward_fn = BasicReward(config=config)
    assert reward_fn.fire_hit_reward == 1.0
    assert reward_fn.step_reward == -0.05
    # Other defaults are preserved
    assert reward_fn.win_reward == 1.0
    assert reward_fn.exploration_reward == 0.02


def test_default_reward_config_round_trip() -> None:
    """Config survives serialisation round-trip via model_dump/model_validate."""
    config = BasicRewardConfig(
        fire_hit_reward=0.8,
        death_reward=-0.5,
        win_reward=2.0,
        loss_reward=-2.0,
        step_reward=-0.02,
        exploration_reward=0.05,
        invalid_move_reward=-0.2,
        fire_miss_reward=-0.08,
        fire_neglect_reward=-0.15,
        pass_reward=-0.03,
        enemy_in_cone_reward=0.05,
    )
    dumped = config.model_dump()
    restored = BasicRewardConfig.model_validate(dumped)
    assert restored == config


def test_default_reward_config_json_round_trip() -> None:
    """Config survives JSON serialisation round-trip."""
    config = BasicRewardConfig(fire_hit_reward=0.75)
    json_str = config.model_dump_json()
    restored = BasicRewardConfig.model_validate_json(json_str)
    assert restored == config


def test_default_reward_config_is_frozen() -> None:
    """Config should be immutable."""
    config = BasicRewardConfig()
    try:
        config.fire_hit_reward = 999.0  # type: ignore[misc]
        assert False, "Should have raised"
    except TypeError, ValueError, AttributeError:
        pass


# ── New reward signal tests ───────────────────────────────────────────


def test_fire_neglect_reward_enemy_ahead() -> None:
    """Non-fire action when enemy is directly ahead incurs fire_neglect_reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + fire_neglect_reward + enemy_in_cone_reward (enemy is also in cone)
    expected = -0.01 + -0.1 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_on_fire() -> None:
    """Fire neglect reward is NOT applied when the action was FIRE (hit)."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + fire_hit_reward + enemy_in_cone_reward (enemy visible)
    expected = -0.01 + 0.5 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_no_enemy() -> None:
    """No fire neglect reward when there is no enemy directly ahead."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # Just step_reward
    assert abs(reward - (-0.01)) < 1e-7


def test_pass_reward_deliberate() -> None:
    """A deliberate PASS incurs the pass_reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.PASS, valid=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + pass_reward
    expected = -0.01 + -0.02
    assert abs(reward - expected) < 1e-7


def test_pass_reward_not_applied_on_invalid() -> None:
    """Invalid action converted to PASS does NOT incur pass_reward."""
    reward_fn = BasicReward()
    # requested MOVE_FORWARD but invalid -> applied PASS
    entry = _make_entry(action=Action.MOVE_FORWARD, valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + invalid_move_reward (no pass_reward)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_single_enemy() -> None:
    """A single enemy in the forward cone gives one unit of reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_in_cone()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + enemy_in_cone_reward * 1
    expected = -0.01 + 0.01
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_multiple_enemies() -> None:
    """Multiple enemies in the forward cone scale the reward linearly."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    enemy1 = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    enemy2 = Tank(id="e2", team="bravo", position=Position(2, 0), direction=Direction.SOUTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy1)
    patch.grid[half - 3][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy2)
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # step_reward + enemy_in_cone_reward * 2
    expected = -0.01 + 0.01 * 2
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_not_applied_when_fogged() -> None:
    """Enemies behind fog are not counted for the cone reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_fogged_enemy()
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (no cone reward since enemy is fogged)
    assert abs(reward - (-0.01)) < 1e-7


def test_enemy_in_cone_friendly_not_counted() -> None:
    """Friendly tanks in the forward cone do NOT trigger the reward."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (friendly not counted)
    assert abs(reward - (-0.01)) < 1e-7


def test_turn_left_reward_applied() -> None:
    """Turn left reward is applied when the action is TURN_LEFT and valid."""
    config = BasicRewardConfig(turn_left_reward=0.05)
    reward_fn = BasicReward(config=config)
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.05
    assert abs(reward - expected) < 1e-7


def test_turn_right_reward_applied() -> None:
    """Turn right reward is applied when the action is TURN_RIGHT and valid."""
    config = BasicRewardConfig(turn_right_reward=0.03)
    reward_fn = BasicReward(config=config)
    entry = _make_entry(action=Action.TURN_RIGHT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.03
    assert abs(reward - expected) < 1e-7


def test_move_forward_reward_applied() -> None:
    """Move forward reward is applied when the action is MOVE_FORWARD and valid."""
    config = BasicRewardConfig(move_forward_reward=0.04)
    reward_fn = BasicReward(config=config)
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.04
    assert abs(reward - expected) < 1e-7


def test_turn_left_reward_not_applied_when_invalid() -> None:
    """Turn left reward is NOT applied when the action is invalid."""
    config = BasicRewardConfig(turn_left_reward=0.05)
    reward_fn = BasicReward(config=config)
    entry = _make_entry(action=Action.TURN_LEFT, valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step_reward + invalid_move_reward (no turn_left_reward)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_movement_rewards_default_to_zero() -> None:
    """Movement rewards default to zero and don't affect existing behaviour."""
    reward_fn = BasicReward()
    assert reward_fn.turn_left_reward == 0.0
    assert reward_fn.turn_right_reward == 0.0
    assert reward_fn.move_forward_reward == 0.0

    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step_reward, no movement bonus
    assert abs(reward - (-0.01)) < 1e-7


def test_missed_fire_friendly_ahead_no_penalty() -> None:
    """No missed fire reward when a friendly tank is directly ahead."""
    reward_fn = BasicReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(
        entry,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (no fire_neglect_reward for friendly)
    assert abs(reward - (-0.01)) < 1e-7


# ── Consecutive turn penalty tests ───────────────────────────────────


def test_consecutive_turn_reward_disabled_by_default() -> None:
    """Default config has consecutive_turn_reward=0.0, so no extra penalty."""
    reward_fn = BasicReward()
    assert reward_fn.consecutive_turn_reward == 0.0

    reward_fn.reset()
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step_reward, no escalating penalty
    assert abs(reward - (-0.01)) < 1e-7


def test_consecutive_turn_reward_escalates() -> None:
    """Consecutive turns incur escalating reward: reward × streak_count."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # First turn: streak=1, penalty = -0.02 × 1 = -0.02
    entry1 = _make_entry(action=Action.TURN_LEFT)
    r1 = reward_fn.compute_step_reward(entry1, patch=patch, team="alpha")
    assert abs(r1 - (-0.02)) < 1e-7

    # Second consecutive turn: streak=2, penalty = -0.02 × 2 = -0.04
    entry2 = _make_entry(action=Action.TURN_RIGHT)
    r2 = reward_fn.compute_step_reward(entry2, patch=patch, team="alpha")
    assert abs(r2 - (-0.04)) < 1e-7

    # Third consecutive turn: streak=3, penalty = -0.02 × 3 = -0.06
    entry3 = _make_entry(action=Action.TURN_LEFT)
    r3 = reward_fn.compute_step_reward(entry3, patch=patch, team="alpha")
    assert abs(r3 - (-0.06)) < 1e-7


def test_consecutive_turn_reward_resets_on_valid_move_forward() -> None:
    """Valid move forward resets the streak to zero."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Two consecutive turns
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Move forward resets streak
    reward_fn.compute_step_reward(
        _make_entry(action=Action.MOVE_FORWARD), patch=patch, team="alpha"
    )

    # Next turn starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_reward_resets_on_fire_hit() -> None:
    """Fire-and-hit resets the streak to zero."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0, fire_hit_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Two consecutive turns
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Fire hit resets streak
    reward_fn.compute_step_reward(
        _make_entry(action=Action.FIRE, hit=True), patch=patch, team="alpha"
    )

    # Next turn starts fresh: streak=1, penalty = -0.02 × 1 = -0.02
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_fire_miss() -> None:
    """Fire-and-miss does NOT reset the streak."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0, fire_miss_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Fire miss: streak stays at 2 (no penalty applied for the fire step)
    reward_fn.compute_step_reward(
        _make_entry(action=Action.FIRE, hit=False), patch=patch, team="alpha"
    )

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_pass() -> None:
    """Pass action does NOT reset the streak."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0, pass_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Pass: streak stays at 2
    reward_fn.compute_step_reward(_make_entry(action=Action.PASS), patch=patch, team="alpha")

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_not_reset_by_invalid_move() -> None:
    """Invalid move does NOT reset the streak."""
    config = BasicRewardConfig(
        consecutive_turn_reward=-0.02, step_reward=0.0, invalid_move_reward=0.0
    )
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Two consecutive turns: streak reaches 2
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_RIGHT), patch=patch, team="alpha")

    # Invalid move forward: streak stays at 2
    reward_fn.compute_step_reward(
        _make_entry(action=Action.MOVE_FORWARD, valid=False), patch=patch, team="alpha"
    )

    # Next turn: streak=3, penalty = -0.02 × 3 = -0.06
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.06)) < 1e-7


def test_consecutive_turn_reward_resets_on_episode() -> None:
    """reset() clears streak tracking between episodes."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.02, step_reward=0.0)
    reward_fn = BasicReward(config=config)
    reward_fn.reset()
    patch = _make_empty_patch()

    # Build up a streak
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")
    reward_fn.compute_step_reward(_make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha")

    # Reset for new episode
    reward_fn.reset()

    # First turn of new episode starts fresh: streak=1
    r = reward_fn.compute_step_reward(
        _make_entry(action=Action.TURN_LEFT), patch=patch, team="alpha"
    )
    assert abs(r - (-0.02)) < 1e-7


def test_consecutive_turn_reward_config_round_trip() -> None:
    """consecutive_turn_reward survives config serialisation round-trip."""
    config = BasicRewardConfig(consecutive_turn_reward=-0.03)
    dumped = config.model_dump()
    restored = BasicRewardConfig.model_validate(dumped)
    assert restored.consecutive_turn_reward == -0.03


# ── Discriminated union & factory tests ───────────────────────────────


def test_reward_config_discriminated_union_basic() -> None:
    """RewardConfig union correctly deserialises a 'basic' reward config."""
    from pydantic import TypeAdapter

    from hmls.nncore.reward import RewardConfig

    adapter: TypeAdapter[RewardConfig] = TypeAdapter(RewardConfig)
    raw_json = '{"reward_type": "basic", "fire_hit_reward": 0.9}'
    config = adapter.validate_json(raw_json)
    assert isinstance(config, BasicRewardConfig)
    assert config.reward_type == "basic"
    assert config.fire_hit_reward == 0.9


def test_reward_config_union_rejects_unknown_type() -> None:
    """RewardConfig union rejects an unknown reward_type value."""
    import pytest
    from pydantic import TypeAdapter, ValidationError

    from hmls.nncore.reward import RewardConfig

    adapter: TypeAdapter[RewardConfig] = TypeAdapter(RewardConfig)
    raw_json = '{"reward_type": "unknown", "fire_hit_reward": 0.9}'
    with pytest.raises(ValidationError):
        adapter.validate_json(raw_json)


def test_create_reward_returns_basic_reward() -> None:
    """create_reward produces a BasicReward from a BasicRewardConfig."""
    from hmls.nncore.reward import create_reward

    config = BasicRewardConfig(fire_hit_reward=0.7)
    reward_fn = create_reward(config)
    assert isinstance(reward_fn, BasicReward)
    assert reward_fn.fire_hit_reward == 0.7


def test_create_reward_rejects_unknown_config() -> None:
    """create_reward raises TypeError for unrecognised config types."""
    import pytest

    from hmls.nncore.reward import create_reward

    class FakeConfig:
        reward_type = "fake"

    with pytest.raises(TypeError, match="Unknown reward config type"):
        create_reward(FakeConfig())  # type: ignore[arg-type]


def test_basic_reward_config_includes_reward_type_in_json() -> None:
    """Serialised JSON includes the reward_type discriminator field."""
    import json

    config = BasicRewardConfig(fire_hit_reward=0.5)
    data = json.loads(config.model_dump_json())
    assert data["reward_type"] == "basic"


# ── FocusedReward tests ──────────────────────────────────────────────


def _make_patch_at(pos: Position, size: int = 9) -> TankPatch:
    """Create an empty patch centred on *pos*."""
    grid: list[list[VisibleCell | FogCell]] = []
    for _row in range(size):
        row_cells: list[VisibleCell | FogCell] = []
        for _col in range(size):
            row_cells.append(VisibleCell(cell_type=CellType.PASSABLE))
        grid.append(row_cells)
    return TankPatch(
        tank_id="t1",
        position=pos,
        direction=Direction.NORTH,
        grid=grid,
    )


def test_focused_step_reward_basic() -> None:
    """A plain valid move incurs step reward only (no action-specific bonuses)."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    patch = _make_patch_at(Position(0, 0))
    rf.observe_patch(patch)
    entry = _make_entry(action=Action.MOVE_FORWARD)
    # First step: exploration bonus for new cell, window not full
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01 + 0.05  # step + exploration
    assert abs(reward - expected) < 1e-7


def test_focused_no_exploration_on_revisit() -> None:
    """Revisiting a cell gives no exploration bonus."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    pos = Position(3, 3)
    patch = _make_patch_at(pos)
    entry = _make_entry(action=Action.MOVE_FORWARD)

    rf.observe_patch(patch)
    rf.compute_step_reward(entry, patch, "alpha")

    # Second visit to same position
    rf.observe_patch(patch)
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01  # step only, no exploration
    assert abs(reward - expected) < 1e-7


def test_focused_exploration_only_on_tank_position() -> None:
    """Exploration only counts the tank's position, not all visible cells."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    # Large patch sees many cells but tank is at (5,5)
    patch = _make_patch_at(Position(5, 5))
    rf.observe_patch(patch)
    # Only 1 new position (the tank's), not the whole grid
    assert rf._last_new_positions == 1
    assert rf._explored_positions == {Position(5, 5)}


def test_focused_fire_hit() -> None:
    """Hitting an enemy gives the fire hit reward."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    patch = _make_patch_at(Position(0, 0))
    rf.observe_patch(patch)
    entry = _make_entry(action=Action.FIRE, hit=True)
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01 + 0.05 + 0.5  # step + exploration + hit
    assert abs(reward - expected) < 1e-7


def test_focused_fire_miss() -> None:
    """Missing a fire gives the fire miss penalty."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    patch = _make_patch_at(Position(0, 0))
    rf.observe_patch(patch)
    entry = _make_entry(action=Action.FIRE, hit=False)
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01 + 0.05 + (-0.1)  # step + exploration + miss
    assert abs(reward - expected) < 1e-7


def test_focused_fire_neglect() -> None:
    """Not firing when enemy is directly ahead gives neglect penalty."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    patch = _make_patch_with_enemy_ahead()
    rf.observe_patch(patch)
    entry = _make_entry(action=Action.MOVE_FORWARD)
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01 + 0.05 + (-0.7)  # step + exploration + neglect
    assert abs(reward - expected) < 1e-7


def test_focused_invalid_move() -> None:
    """An invalid action gives the invalid move penalty."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=100))
    patch = _make_patch_at(Position(0, 0))
    rf.observe_patch(patch)
    entry = _make_entry(action=Action.MOVE_FORWARD, valid=False)
    reward = rf.compute_step_reward(entry, patch, "alpha")
    expected = -0.01 + 0.05 + (-0.05)  # step + exploration + invalid
    assert abs(reward - expected) < 1e-7


def test_focused_no_action_specific_rewards() -> None:
    """Turning and passing do not produce action-specific rewards."""
    rf = FocusedReward(
        FocusedRewardConfig(
            liveliness_window=100,
            exploration_reward=0.0,
        )
    )
    for action in [Action.TURN_LEFT, Action.TURN_RIGHT, Action.PASS]:
        patch = _make_patch_at(Position(0, 0))
        rf.observe_patch(patch)
        entry = _make_entry(action=action)
        reward = rf.compute_step_reward(entry, patch, "alpha")
        assert abs(reward - (-0.01)) < 1e-7, f"Unexpected reward for {action}"


def test_focused_liveliness_diverse() -> None:
    """Liveliness reward is positive when unique positions exceed target."""
    cfg = FocusedRewardConfig(
        liveliness_window=5,
        liveliness_target=3,
        liveliness_reward=0.1,
        exploration_reward=0.0,
    )
    rf = FocusedReward(cfg)
    entry = _make_entry(action=Action.MOVE_FORWARD)

    # Fill window with 5 distinct positions
    for i in range(5):
        patch = _make_patch_at(Position(i, 0))
        rf.observe_patch(patch)

    patch = _make_patch_at(Position(4, 0))
    reward = rf.compute_step_reward(entry, patch, "alpha")
    # 5 unique in window, target 3 → liveliness = 0.1 * (5 - 3) = 0.2
    expected = -0.01 + 0.2
    assert abs(reward - expected) < 1e-7


def test_focused_liveliness_stuck() -> None:
    """Liveliness reward is negative when stuck in one place."""
    cfg = FocusedRewardConfig(
        liveliness_window=5,
        liveliness_target=3,
        liveliness_reward=0.1,
        exploration_reward=0.0,
    )
    rf = FocusedReward(cfg)
    entry = _make_entry(action=Action.TURN_LEFT)

    # Fill window with same position
    for _ in range(5):
        patch = _make_patch_at(Position(2, 2))
        rf.observe_patch(patch)

    patch = _make_patch_at(Position(2, 2))
    reward = rf.compute_step_reward(entry, patch, "alpha")
    # 1 unique in window, target 3 → liveliness = 0.1 * (1 - 3) = -0.2
    expected = -0.01 + (-0.2)
    assert abs(reward - expected) < 1e-7


def test_focused_liveliness_at_target() -> None:
    """Liveliness reward is zero when unique positions equal target."""
    cfg = FocusedRewardConfig(
        liveliness_window=5,
        liveliness_target=3,
        liveliness_reward=0.1,
        exploration_reward=0.0,
    )
    rf = FocusedReward(cfg)
    entry = _make_entry(action=Action.MOVE_FORWARD)

    # 3 unique positions in 5 steps
    positions = [Position(0, 0), Position(1, 0), Position(2, 0), Position(2, 0), Position(2, 0)]
    for pos in positions:
        rf.observe_patch(_make_patch_at(pos))

    patch = _make_patch_at(Position(2, 0))
    reward = rf.compute_step_reward(entry, patch, "alpha")
    # 3 unique, target 3 → liveliness = 0.0
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_focused_liveliness_not_applied_before_window_full() -> None:
    """Liveliness reward is not applied until the window is full."""
    cfg = FocusedRewardConfig(
        liveliness_window=5,
        liveliness_target=3,
        liveliness_reward=0.1,
        exploration_reward=0.0,
    )
    rf = FocusedReward(cfg)
    entry = _make_entry(action=Action.TURN_LEFT)

    # Only 3 steps — window not full
    for _ in range(3):
        rf.observe_patch(_make_patch_at(Position(0, 0)))

    patch = _make_patch_at(Position(0, 0))
    reward = rf.compute_step_reward(entry, patch, "alpha")
    # No liveliness component
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_focused_liveliness_window_slides() -> None:
    """The liveliness window drops old positions as new ones are added."""
    cfg = FocusedRewardConfig(
        liveliness_window=3,
        liveliness_target=2,
        liveliness_reward=0.1,
        exploration_reward=0.0,
    )
    rf = FocusedReward(cfg)
    entry = _make_entry(action=Action.MOVE_FORWARD)

    # Fill with 3 distinct positions
    for i in range(3):
        rf.observe_patch(_make_patch_at(Position(i, 0)))

    # Now add a duplicate — oldest (Position(0,0)) falls off
    rf.observe_patch(_make_patch_at(Position(1, 0)))
    # Window is [Pos(1,0), Pos(2,0), Pos(1,0)] → 2 unique
    patch = _make_patch_at(Position(1, 0))
    reward = rf.compute_step_reward(entry, patch, "alpha")
    # 2 unique, target 2 → liveliness = 0.0
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_focused_episode_end_win() -> None:
    """Win gives positive terminal reward."""
    rf = FocusedReward()
    assert abs(rf.compute_episode_end_reward(True) - 1.0) < 1e-7


def test_focused_episode_end_loss() -> None:
    """Loss gives negative terminal reward."""
    rf = FocusedReward()
    assert abs(rf.compute_episode_end_reward(False) - (-1.0)) < 1e-7


def test_focused_episode_end_draw() -> None:
    """Draw gives zero terminal reward."""
    rf = FocusedReward()
    assert abs(rf.compute_episode_end_reward(None)) < 1e-7


def test_focused_reset_clears_state() -> None:
    """Reset clears exploration and liveliness state."""
    rf = FocusedReward(FocusedRewardConfig(liveliness_window=3))
    for i in range(3):
        rf.observe_patch(_make_patch_at(Position(i, 0)))
    assert len(rf._explored_positions) == 3
    assert len(rf._position_window) == 3

    rf.reset()
    assert len(rf._explored_positions) == 0
    assert len(rf._position_window) == 0


def test_focused_config_serialisation() -> None:
    """FocusedRewardConfig round-trips through JSON."""
    import json

    config = FocusedRewardConfig(fire_hit_reward=0.7, liveliness_target=8)
    data = json.loads(config.model_dump_json())
    assert data["reward_type"] == "focused"
    assert data["fire_hit_reward"] == 0.7
    assert data["liveliness_target"] == 8
