"""Tests for the reward module."""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, TankPatch, VisibleCell
from hmls.nncore.reward import DefaultReward, DefaultRewardConfig


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
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    assert abs(reward - (-0.01)) < 1e-7


def test_default_reward_hit() -> None:
    """A successful hit adds the hit reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_exploration() -> None:
    """New positions add exploration bonus."""
    reward_fn = DefaultReward()
    entry = _make_entry()
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions={Position(1, 1)},
        new_positions_this_step=3,
        patch=patch,
        team="alpha",
    )
    expected = -0.01 + 0.02 * 3
    assert abs(reward - expected) < 1e-7


def test_default_reward_invalid_action() -> None:
    """Invalid action incurs the dedicated invalid_move_reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + invalid_move_reward
    assert abs(reward - (-0.01 + -0.1)) < 1e-7


def test_default_reward_fire_miss() -> None:
    """Firing and missing incurs the fire_miss_reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + fire_miss_reward
    assert abs(reward - (-0.01 + -0.05)) < 1e-7


def test_default_reward_hit_no_miss_penalty() -> None:
    """A successful hit does NOT incur the fire miss reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + hit_reward only (no fire_miss_reward)
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
    config = DefaultRewardConfig(hit_reward=1.0, step_reward=-0.05)
    reward_fn = DefaultReward(config=config)
    assert reward_fn.hit_reward == 1.0
    assert reward_fn.step_reward == -0.05
    # Other defaults are preserved
    assert reward_fn.win_reward == 1.0
    assert reward_fn.exploration_reward == 0.02


def test_default_reward_config_round_trip() -> None:
    """Config survives serialisation round-trip via model_dump/model_validate."""
    config = DefaultRewardConfig(
        hit_reward=0.8,
        death_reward=-0.5,
        win_reward=2.0,
        loss_reward=-2.0,
        step_reward=-0.02,
        exploration_reward=0.05,
        invalid_move_reward=-0.2,
        fire_miss_reward=-0.08,
        missed_fire_reward=-0.15,
        pass_reward=-0.03,
        enemy_in_cone_reward=0.05,
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


# ── New reward signal tests ───────────────────────────────────────────


def test_missed_fire_reward_enemy_ahead() -> None:
    """Non-fire action when enemy is directly ahead incurs missed_fire_reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + missed_fire_reward + enemy_in_cone_reward (enemy is also in cone)
    expected = -0.01 + -0.1 + 0.01
    assert abs(reward - expected) < 1e-7


def test_missed_fire_reward_not_applied_on_fire() -> None:
    """Missed fire reward is NOT applied when the action was FIRE (hit)."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + hit_reward + enemy_in_cone_reward (enemy visible)
    expected = -0.01 + 0.5 + 0.01
    assert abs(reward - expected) < 1e-7


def test_missed_fire_reward_not_applied_no_enemy() -> None:
    """No missed fire reward when there is no enemy directly ahead."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # Just step_reward
    assert abs(reward - (-0.01)) < 1e-7


def test_pass_reward_deliberate() -> None:
    """A deliberate PASS incurs the pass_reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.PASS, valid=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + pass_reward
    expected = -0.01 + -0.02
    assert abs(reward - expected) < 1e-7


def test_pass_reward_not_applied_on_invalid() -> None:
    """Invalid action converted to PASS does NOT incur pass_reward."""
    reward_fn = DefaultReward()
    # requested MOVE_FORWARD but invalid -> applied PASS
    entry = _make_entry(action=Action.MOVE_FORWARD, valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + invalid_move_reward (no pass_reward)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_single_enemy() -> None:
    """A single enemy in the forward cone gives one unit of reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_in_cone()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + enemy_in_cone_reward * 1
    expected = -0.01 + 0.01
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_multiple_enemies() -> None:
    """Multiple enemies in the forward cone scale the reward linearly."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    enemy1 = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    enemy2 = Tank(id="e2", team="bravo", position=Position(2, 0), direction=Direction.SOUTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy1)
    patch.grid[half - 3][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy2)
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # step_reward + enemy_in_cone_reward * 2
    expected = -0.01 + 0.01 * 2
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_not_applied_when_fogged() -> None:
    """Enemies behind fog are not counted for the cone reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_fogged_enemy()
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (no cone reward since enemy is fogged)
    assert abs(reward - (-0.01)) < 1e-7


def test_enemy_in_cone_friendly_not_counted() -> None:
    """Friendly tanks in the forward cone do NOT trigger the reward."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (friendly not counted)
    assert abs(reward - (-0.01)) < 1e-7


def test_missed_fire_friendly_ahead_no_penalty() -> None:
    """No missed fire reward when a friendly tank is directly ahead."""
    reward_fn = DefaultReward()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(
        entry,
        explored_positions=set(),
        new_positions_this_step=0,
        patch=patch,
        team="alpha",
    )
    # Just step_reward (no missed_fire_reward for friendly)
    assert abs(reward - (-0.01)) < 1e-7
