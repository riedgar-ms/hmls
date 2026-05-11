"""Tests for the reward module."""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, TankPatch, VisibleCell
from hmls.nncore.reward import (
    ActionsRewardConfig,
    ExplorationRewardConfig,
    FiringRewardConfig,
    GameStateRewardConfig,
    RewardConfig,
    RewardFunction,
    SituationalRewardConfig,
)


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


# ── Basic step / hit / miss / win / loss tests ──────────────────────


def test_default_reward_step_reward() -> None:
    """A plain step incurs the step reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - (-0.01)) < 1e-7


def test_default_reward_hit() -> None:
    """A successful hit adds the hit reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_exploration_see_cell() -> None:
    """New positions add see_cell exploration bonus."""
    reward_fn = RewardFunction()
    reward_fn.reset()
    entry = _make_entry()
    patch = _make_empty_patch(size=3)
    # First observation discovers all cells; a 3x3 fully visible patch = 9 cells
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + see_cell * 9
    expected = -0.01 + 0.02 * 9
    assert abs(reward - expected) < 1e-7

    # Second observation of same patch discovers nothing new
    reward_fn.observe_patch(patch)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01
    assert abs(reward - expected) < 1e-7


def test_occupy_cell_reward() -> None:
    """Moving to a new cell gives occupy_cell reward."""
    config = RewardConfig(
        exploration=ExplorationRewardConfig(see_cell=0.0, occupy_cell=0.05),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config)
    reward_fn.reset()

    # First position — new cell
    patch1 = _make_patch_at(Position(0, 0), size=3)
    reward_fn.observe_patch(patch1)
    entry = _make_entry(action=Action.MOVE_FORWARD)
    r1 = reward_fn.compute_step_reward(entry, patch=patch1, team="alpha")
    assert abs(r1 - 0.05) < 1e-7

    # Same position — no reward
    reward_fn.observe_patch(patch1)
    r2 = reward_fn.compute_step_reward(entry, patch=patch1, team="alpha")
    assert abs(r2) < 1e-7

    # New position — reward again
    patch2 = _make_patch_at(Position(1, 0), size=3)
    reward_fn.observe_patch(patch2)
    r3 = reward_fn.compute_step_reward(entry, patch=patch2, team="alpha")
    assert abs(r3 - 0.05) < 1e-7


def test_default_reward_invalid_action() -> None:
    """Invalid action incurs the dedicated invalid_move reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move
    assert abs(reward - (-0.01 + -0.1)) < 1e-7


def test_default_reward_fire_miss() -> None:
    """Firing and missing incurs the miss reward (penalty)."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.FIRE, hit=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + miss
    assert abs(reward - (-0.01 + -0.05)) < 1e-7


def test_default_reward_hit_no_miss_penalty() -> None:
    """A successful hit does NOT incur the fire miss reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + hit only (no miss)
    assert abs(reward - (-0.01 + 0.5)) < 1e-7


def test_default_reward_episode_win() -> None:
    """Winning gives positive terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=True) == 1.0


def test_default_reward_episode_loss() -> None:
    """Losing gives negative terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=False) == -1.0


def test_default_reward_episode_draw() -> None:
    """Draw gives zero terminal reward."""
    reward_fn = RewardFunction()
    assert reward_fn.compute_episode_end_reward(won=None) == 0.0


def test_reward_from_config() -> None:
    """Construction from an explicit config works correctly."""
    config = RewardConfig(
        firing=FiringRewardConfig(hit=1.0),
        game_state=GameStateRewardConfig(step=-0.05),
    )
    reward_fn = RewardFunction(config=config)
    assert reward_fn.config.firing.hit == 1.0
    assert reward_fn.config.game_state.step == -0.05
    # Other defaults are preserved
    assert reward_fn.config.game_state.win == 1.0
    assert reward_fn.config.exploration.see_cell == 0.02


def test_reward_config_round_trip() -> None:
    """Config survives serialisation round-trip via model_dump/model_validate."""
    config = RewardConfig(
        firing=FiringRewardConfig(hit=0.8, miss=-0.08, neglect=-0.15),
        game_state=GameStateRewardConfig(
            win=2.0, loss=-2.0, step=-0.02, invalid_move=-0.2, death=-0.5
        ),
        actions=ActionsRewardConfig(pass_action=-0.03),
        exploration=ExplorationRewardConfig(see_cell=0.05),
        situational=SituationalRewardConfig(enemy_in_cone=0.05),
    )
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored == config


def test_reward_config_json_round_trip() -> None:
    """Config survives JSON serialisation round-trip."""
    config = RewardConfig(firing=FiringRewardConfig(hit=0.75))
    json_str = config.model_dump_json()
    restored = RewardConfig.model_validate_json(json_str)
    assert restored == config


def test_reward_config_is_frozen() -> None:
    """Config should be immutable."""
    config = RewardConfig()
    try:
        config.actions = ActionsRewardConfig()  # type: ignore[misc]
        assert False, "Should have raised"
    except TypeError, ValueError, AttributeError:
        pass


# ── Firing outcome tests ─────────────────────────────────────────────


def test_fire_neglect_reward_enemy_ahead() -> None:
    """Non-fire action when enemy is directly ahead incurs neglect reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + neglect + enemy_in_cone (enemy is also in cone)
    expected = -0.01 + -0.1 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_on_fire() -> None:
    """Fire neglect reward is NOT applied when the action was FIRE (hit)."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_patch_with_enemy_ahead()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + hit + enemy_in_cone (enemy visible)
    expected = -0.01 + 0.5 + 0.01
    assert abs(reward - expected) < 1e-7


def test_fire_neglect_reward_not_applied_no_enemy() -> None:
    """No fire neglect reward when there is no enemy directly ahead."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step
    assert abs(reward - (-0.01)) < 1e-7


# ── Action reward tests ──────────────────────────────────────────────


def test_pass_reward_deliberate() -> None:
    """A deliberate PASS incurs the pass_action reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.PASS, valid=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + pass_action
    expected = -0.01 + -0.02
    assert abs(reward - expected) < 1e-7


def test_pass_reward_not_applied_on_invalid() -> None:
    """Invalid action converted to PASS does NOT incur pass_action reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD, valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move (no pass_action)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_turn_left_reward_applied() -> None:
    """Turn left reward is applied when the action is TURN_LEFT and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_left=0.05))
    reward_fn = RewardFunction(config=config)
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.05
    assert abs(reward - expected) < 1e-7


def test_turn_right_reward_applied() -> None:
    """Turn right reward is applied when the action is TURN_RIGHT and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_right=0.03))
    reward_fn = RewardFunction(config=config)
    entry = _make_entry(action=Action.TURN_RIGHT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.03
    assert abs(reward - expected) < 1e-7


def test_move_forward_reward_applied() -> None:
    """Move forward reward is applied when the action is MOVE_FORWARD and valid."""
    config = RewardConfig(actions=ActionsRewardConfig(move_forward=0.04))
    reward_fn = RewardFunction(config=config)
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    expected = -0.01 + 0.04
    assert abs(reward - expected) < 1e-7


def test_fire_action_reward_applied() -> None:
    """Fire action reward is applied when FIRE is valid (in addition to hit/miss)."""
    config = RewardConfig(
        actions=ActionsRewardConfig(fire=0.03),
        firing=FiringRewardConfig(hit=0.0, miss=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
    entry = _make_entry(action=Action.FIRE, hit=True)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    assert abs(reward - 0.03) < 1e-7


def test_turn_left_reward_not_applied_when_invalid() -> None:
    """Turn left reward is NOT applied when the action is invalid."""
    config = RewardConfig(actions=ActionsRewardConfig(turn_left=0.05))
    reward_fn = RewardFunction(config=config)
    entry = _make_entry(action=Action.TURN_LEFT, valid=False)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + invalid_move (no turn_left)
    expected = -0.01 + -0.1
    assert abs(reward - expected) < 1e-7


def test_action_rewards_default_to_zero() -> None:
    """Movement rewards default to zero and don't affect existing behaviour."""
    reward_fn = RewardFunction()
    assert reward_fn.config.actions.turn_left == 0.0
    assert reward_fn.config.actions.turn_right == 0.0
    assert reward_fn.config.actions.move_forward == 0.0
    assert reward_fn.config.actions.fire == 0.0

    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step, no movement bonus
    assert abs(reward - (-0.01)) < 1e-7


# ── Situational reward tests ────────────────────────────────────────


def test_enemy_in_cone_reward_single_enemy() -> None:
    """A single enemy in the forward cone gives one unit of reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_enemy_in_cone()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + enemy_in_cone * 1
    expected = -0.01 + 0.01
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_multiple_enemies() -> None:
    """Multiple enemies in the forward cone scale the reward linearly."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    enemy1 = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    enemy2 = Tank(id="e2", team="bravo", position=Position(2, 0), direction=Direction.SOUTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy1)
    patch.grid[half - 3][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy2)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + enemy_in_cone * 2
    expected = -0.01 + 0.01 * 2
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_not_applied_when_fogged() -> None:
    """Enemies behind fog are not counted for the cone reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_patch_with_fogged_enemy()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (no cone reward since enemy is fogged)
    assert abs(reward - (-0.01)) < 1e-7


def test_enemy_in_cone_friendly_not_counted() -> None:
    """Friendly tanks in the forward cone do NOT trigger the reward."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (friendly not counted)
    assert abs(reward - (-0.01)) < 1e-7


def test_missed_fire_friendly_ahead_no_penalty() -> None:
    """No missed fire reward when a friendly tank is directly ahead."""
    reward_fn = RewardFunction()
    entry = _make_entry(action=Action.MOVE_FORWARD)
    patch = _make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (no neglect for friendly)
    assert abs(reward - (-0.01)) < 1e-7


# ── Consecutive turn penalty tests ──────────────────────────────────


def test_consecutive_turn_reward_disabled_by_default() -> None:
    """Default config has consecutive_turn=0.0, so no extra penalty."""
    reward_fn = RewardFunction()
    assert reward_fn.config.actions.consecutive_turn == 0.0

    reward_fn.reset()
    entry = _make_entry(action=Action.TURN_LEFT)
    patch = _make_empty_patch()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Only step, no escalating penalty
    assert abs(reward - (-0.01)) < 1e-7


def test_consecutive_turn_reward_escalates() -> None:
    """Consecutive turns incur escalating reward: reward × streak_count."""
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        firing=FiringRewardConfig(hit=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        firing=FiringRewardConfig(miss=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02, pass_action=0.0),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0, invalid_move=0.0),
    )
    reward_fn = RewardFunction(config=config)
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
    config = RewardConfig(
        actions=ActionsRewardConfig(consecutive_turn=-0.02),
        game_state=GameStateRewardConfig(step=0.0),
    )
    reward_fn = RewardFunction(config=config)
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


def test_consecutive_turn_config_round_trip() -> None:
    """consecutive_turn survives config serialisation round-trip."""
    config = RewardConfig(actions=ActionsRewardConfig(consecutive_turn=-0.03))
    dumped = config.model_dump()
    restored = RewardConfig.model_validate(dumped)
    assert restored.actions.consecutive_turn == -0.03


# ── Reset / state tests ─────────────────────────────────────────────


def test_reset_clears_all_state() -> None:
    """Reset clears seen, occupied, and streak state."""
    reward_fn = RewardFunction()
    for i in range(3):
        reward_fn.observe_patch(_make_patch_at(Position(i, 0)))
    assert len(reward_fn._seen_positions) > 0
    assert len(reward_fn._occupied_positions) > 0

    reward_fn.reset()
    assert len(reward_fn._seen_positions) == 0
    assert len(reward_fn._occupied_positions) == 0
    assert len(reward_fn._turn_streaks) == 0


# ── Nested config JSON tests ────────────────────────────────────────


def test_reward_config_nested_json() -> None:
    """Nested config round-trips through JSON correctly."""
    import json

    config = RewardConfig(
        actions=ActionsRewardConfig(move_forward=0.04, consecutive_turn=-0.03),
        firing=FiringRewardConfig(hit=0.7),
        game_state=GameStateRewardConfig(win=2.0),
        exploration=ExplorationRewardConfig(see_cell=0.05, occupy_cell=0.1),
        situational=SituationalRewardConfig(enemy_in_cone=0.02),
    )
    data = json.loads(config.model_dump_json())
    assert data["actions"]["move_forward"] == 0.04
    assert data["actions"]["consecutive_turn"] == -0.03
    assert data["firing"]["hit"] == 0.7
    assert data["game_state"]["win"] == 2.0
    assert data["exploration"]["see_cell"] == 0.05
    assert data["exploration"]["occupy_cell"] == 0.1
    assert data["situational"]["enemy_in_cone"] == 0.02

    restored = RewardConfig.model_validate(data)
    assert restored == config
