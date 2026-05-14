"""Situational reward tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import VisibleCell
from hmls.nncore.reward import (
    RewardConfig,
    RewardFunction,
    SituationalRewardConfig,
)

if TYPE_CHECKING:
    from tests.rewards.conftest import (
        MakeEntryFactory,
        MakePatchFactory,
    )


def test_enemy_in_cone_reward_single_enemy(
    make_entry: MakeEntryFactory, make_patch_with_enemy_in_cone: MakePatchFactory
) -> None:
    """A single enemy in the forward cone gives one unit of reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_patch_with_enemy_in_cone()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + enemy_in_cone * 1
    expected = -0.01 + 0.01
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_multiple_enemies(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Multiple enemies in the forward cone scale the reward linearly."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    half = 9 // 2
    enemy1 = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    enemy2 = Tank(id="e2", team="bravo", position=Position(2, 0), direction=Direction.SOUTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy1)
    patch.grid[half - 3][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy2)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + enemy_in_cone * 2
    expected = -0.01 + 0.01 * 2
    assert abs(reward - expected) < 1e-7


def test_enemy_in_cone_reward_not_applied_when_fogged(
    make_entry: MakeEntryFactory, make_patch_with_fogged_enemy: MakePatchFactory
) -> None:
    """Enemies behind fog are not counted for the cone reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_patch_with_fogged_enemy()
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (no cone reward since enemy is fogged)
    assert abs(reward - (-0.01)) < 1e-7


def test_enemy_in_cone_friendly_not_counted(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Friendly tanks in the forward cone do NOT trigger the reward."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 2][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (friendly not counted)
    assert abs(reward - (-0.01)) < 1e-7


def test_enemy_in_cone_distance_discount(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """Distance discount reduces contribution of far-away enemies."""
    discount = 0.5
    cfg = RewardConfig(
        situational=SituationalRewardConfig(
            enemy_in_cone=1.0,
            enemy_in_cone_distance_discount=discount,
        ),
    )
    reward_fn = RewardFunction(config=cfg)
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    half = 9 // 2
    # Enemy at (half-2, half+1): Manhattan distance = 2 + 1 = 3
    enemy1 = Tank(id="e1", team="bravo", position=Position(0, 0), direction=Direction.SOUTH)
    patch.grid[half - 2][half + 1] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy1)
    # Enemy at (half-1, half): Manhattan distance = 1 + 0 = 1
    enemy2 = Tank(id="e2", team="bravo", position=Position(2, 0), direction=Direction.SOUTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy2)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # step + enemy_in_cone * (discount^3 + discount^1)
    # Also neglect because enemy2 is directly ahead
    expected = cfg.game_state.step + cfg.firing.neglect + 1.0 * (0.5**3 + 0.5**1)
    assert abs(reward - expected) < 1e-7


def test_missed_fire_friendly_ahead_no_penalty(
    make_entry: MakeEntryFactory, make_empty_patch: MakePatchFactory
) -> None:
    """No missed fire reward when a friendly tank is directly ahead."""
    reward_fn = RewardFunction()
    entry = make_entry(action=Action.MOVE_FORWARD)
    patch = make_empty_patch()
    half = 9 // 2
    friendly = Tank(id="f1", team="alpha", position=Position(1, 0), direction=Direction.NORTH)
    patch.grid[half - 1][half] = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
    reward = reward_fn.compute_step_reward(entry, patch=patch, team="alpha")
    # Just step (no neglect for friendly)
    assert abs(reward - (-0.01)) < 1e-7
