"""Tests for the encoding module."""

from __future__ import annotations

import torch

from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.core.visibility import FogCell, TankPatch, VisibleCell
from hmls.singlemki.encoding import NUM_CHANNELS, encode_patch


def _make_patch(grid_size: int = 3) -> TankPatch:
    """Create a simple 3x3 patch with known contents for testing."""
    # Centre cell: visible, passable, with the observing tank (friendly)
    # Top-centre: visible, passable, with an enemy tank
    # Other cells: fog
    friendly_tank = Tank(id="t1", team="alpha", position=Position(5, 5), direction=Direction.NORTH)
    enemy_tank = Tank(id="t2", team="beta", position=Position(5, 4), direction=Direction.SOUTH)
    dead_tank = Tank(
        id="t3", team="beta", position=Position(4, 5), direction=Direction.EAST, alive=False
    )

    fog = FogCell()
    grid: list[list[VisibleCell | FogCell]] = [
        [fog, VisibleCell(cell_type=CellType.PASSABLE, tank=enemy_tank), fog],
        [
            VisibleCell(cell_type=CellType.PASSABLE, tank=dead_tank),
            VisibleCell(cell_type=CellType.PASSABLE, tank=friendly_tank),
            VisibleCell(cell_type=CellType.IMPASSABLE),
        ],
        [fog, VisibleCell(cell_type=CellType.PASSABLE), fog],
    ]

    return TankPatch(
        tank_id="t1",
        position=Position(5, 5),
        direction=Direction.NORTH,
        grid=grid,
    )


def test_encode_patch_shape() -> None:
    """Encoded tensor has correct shape."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    assert tensor.shape == (NUM_CHANNELS, 3, 3)
    assert tensor.dtype == torch.float32


def test_encode_patch_fog_channel() -> None:
    """Fog cells get -1 terrain and 0 visibility."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Top-left is fog
    assert tensor[0, 0, 0].item() == -1.0  # terrain = -1 (fog)
    assert tensor[4, 0, 0].item() == 0.0  # visibility = 0


def test_encode_patch_terrain_channel() -> None:
    """Visible cells encode terrain correctly."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Centre (1,1): passable
    assert tensor[0, 1, 1].item() == 1.0
    # Right of centre (1,2): impassable
    assert tensor[0, 1, 2].item() == 0.0


def test_encode_patch_friendly_channel() -> None:
    """Friendly alive tank is encoded in channel 1."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Centre (1,1) has friendly tank
    assert tensor[1, 1, 1].item() == 1.0
    # No friendly tank elsewhere
    assert tensor[1, 0, 1].item() == 0.0


def test_encode_patch_enemy_channel() -> None:
    """Enemy alive tank is encoded in channel 2."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Top-centre (0,1) has enemy tank
    assert tensor[2, 0, 1].item() == 1.0


def test_encode_patch_wreckage_channel() -> None:
    """Dead tank is encoded in channel 3."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Left of centre (1,0) has dead tank
    assert tensor[3, 1, 0].item() == 1.0


def test_encode_patch_visibility_channel() -> None:
    """Visibility mask channel is correct."""
    patch = _make_patch(3)
    tensor = encode_patch(patch, team="alpha")
    # Visible cells should be 1.0
    assert tensor[4, 1, 1].item() == 1.0
    assert tensor[4, 0, 1].item() == 1.0
    # Fog cells should be 0.0
    assert tensor[4, 0, 0].item() == 0.0
