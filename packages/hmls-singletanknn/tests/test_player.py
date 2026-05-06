"""Tests for the NNPlayer."""

from __future__ import annotations

import pytest

from hmls.core.map import CellType
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, PlayerView, TankInfo, TankPatch, VisibleCell
from hmls.singletanknn.constants import ACTION_INDEX_TO_ACTION
from hmls.singletanknn.model import ModelConfig, TankPolicyNetwork
from hmls.singletanknn.player import NNPlayer


def _make_view(patch_size: int = 9, team: str = "alpha") -> PlayerView:
    """Create a minimal PlayerView with a single all-passable visible patch."""
    grid: list[list[VisibleCell | FogCell]] = []
    for _row in range(patch_size):
        row_cells: list[VisibleCell | FogCell] = []
        for _col in range(patch_size):
            row_cells.append(VisibleCell(cell_type=CellType.PASSABLE))
        grid.append(row_cells)

    patch = TankPatch(
        tank_id="t1",
        position=Position(5, 5),
        direction=Direction.NORTH,
        grid=grid,
    )
    tank_info = TankInfo(
        tank_id="t1", position=Position(5, 5), direction=Direction.NORTH, alive=True
    )
    return PlayerView(patches=[patch], tanks=[tank_info])


def test_player_choose_action_play_mode() -> None:
    """Player returns a valid action in play mode."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    model.eval()
    player = NNPlayer(team="alpha", model=model, mode="play", patch_size=9)

    view = _make_view(patch_size=9)
    action = player.choose_action("t1", view)
    assert action in ACTION_INDEX_TO_ACTION


def test_player_choose_action_learn_mode_records_trajectory() -> None:
    """In learn mode, trajectory steps are recorded."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    player = NNPlayer(team="alpha", model=model, mode="learn", patch_size=9)

    view = _make_view(patch_size=9)
    player.choose_action("t1", view)
    assert len(player.episode) == 1
    step = player.episode.steps[0]
    assert 0 <= step.action_index < 5
    assert step.log_prob < 0  # log-probability is negative


def test_player_patch_size_mismatch_raises() -> None:
    """Player raises ValueError if patch size doesn't match."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    player = NNPlayer(team="alpha", model=model, mode="play", patch_size=9)

    # Create a view with wrong patch size (7x7)
    view = _make_view(patch_size=7)
    with pytest.raises(ValueError, match="Patch size mismatch"):
        player.choose_action("t1", view)


def test_player_reset_episode() -> None:
    """reset_episode clears trajectory and exploration."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    player = NNPlayer(team="alpha", model=model, mode="learn", patch_size=9)

    view = _make_view(patch_size=9)
    player.choose_action("t1", view)
    assert len(player.episode) == 1
    assert len(player.explored_positions) > 0

    player.reset_episode()
    assert len(player.episode) == 0
    assert len(player.explored_positions) == 0


def test_player_exploration_tracking() -> None:
    """Player tracks explored positions across turns."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    player = NNPlayer(team="alpha", model=model, mode="play", patch_size=9)

    view = _make_view(patch_size=9)
    player.choose_action("t1", view)

    # All visible cells should be in explored_positions
    assert len(player.explored_positions) > 0


def test_player_no_patch_returns_pass() -> None:
    """If no patch found for tank_id, returns PASS."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    player = NNPlayer(team="alpha", model=model, mode="play", patch_size=9)

    # Create view with patch for different tank
    view = _make_view(patch_size=9)
    action = player.choose_action("nonexistent_tank", view)
    assert action == Action.PASS


def test_action_index_mapping_covers_all_actions() -> None:
    """ACTION_INDEX_TO_ACTION covers all Action enum members."""
    assert set(ACTION_INDEX_TO_ACTION) == set(Action)
