"""Tests for the AutoMap class."""

from __future__ import annotations

from hmls.client.automap import AutoMap, CellState
from hmls.core.map import CellType
from hmls.core.types import Direction, Position
from hmls.core.visibility import (
    FogCell,
    PlayerView,
    TankInfo,
    TankPatch,
    VisibleCell,
)


def test_automap_initial_state() -> None:
    """All cells should start as UNKNOWN."""
    am = AutoMap(5, 5)
    for y in range(5):
        for x in range(5):
            assert am[x, y] == CellState.UNKNOWN


def test_automap_update_marks_visible_cells() -> None:
    """Visible cells from a patch should be marked in the automap."""
    am = AutoMap(10, 10)

    # Create a simple 3x3 patch centred on (5, 5) facing NORTH.
    # The centre cell (1,1) is the tank itself at (5,5).
    # (0,0) = forward-left = (4,4), (0,1) = forward = (5,4), (0,2) = forward-right = (6,4)
    # (1,0) = left = (4,5), (1,1) = centre = (5,5), (1,2) = right = (6,5)
    # (2,0) = back-left = (4,6), (2,1) = back = (5,6), (2,2) = back-right = (6,6)
    grid: list[list[VisibleCell | FogCell]] = [
        [
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.IMPASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
        ],
        [
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
        ],
        [
            FogCell(),
            FogCell(),
            FogCell(),
        ],
    ]

    patch = TankPatch(
        tank_id="A1",
        position=Position(5, 5),
        direction=Direction.NORTH,
        grid=grid,
    )

    view = PlayerView(
        patches=[patch],
        tanks=[
            TankInfo(
                tank_id="A1",
                position=Position(5, 5),
                direction=Direction.NORTH,
                alive=True,
            )
        ],
    )

    am.update(view)

    # Forward row should be marked.
    assert am[4, 4] == CellState.PASSABLE  # forward-left
    assert am[5, 4] == CellState.IMPASSABLE  # forward
    assert am[6, 4] == CellState.PASSABLE  # forward-right

    # Middle row should be marked.
    assert am[4, 5] == CellState.PASSABLE  # left
    assert am[5, 5] == CellState.PASSABLE  # centre (tank position)
    assert am[6, 5] == CellState.PASSABLE  # right

    # Back row is fog — should remain UNKNOWN.
    assert am[4, 6] == CellState.UNKNOWN
    assert am[5, 6] == CellState.UNKNOWN
    assert am[6, 6] == CellState.UNKNOWN


def test_automap_accumulates_across_updates() -> None:
    """Multiple updates should accumulate knowledge, not overwrite."""
    am = AutoMap(10, 10)

    # First observation at (2, 2) facing EAST.
    grid1: list[list[VisibleCell | FogCell]] = [
        [
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
        ],
        [
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
        ],
        [FogCell(), FogCell(), FogCell()],
    ]
    patch1 = TankPatch(
        tank_id="A1",
        position=Position(2, 2),
        direction=Direction.EAST,
        grid=grid1,
    )
    view1 = PlayerView(
        patches=[patch1],
        tanks=[
            TankInfo(
                tank_id="A1",
                position=Position(2, 2),
                direction=Direction.EAST,
                alive=True,
            )
        ],
    )
    am.update(view1)

    # Second observation at (4, 4) facing SOUTH.
    grid2: list[list[VisibleCell | FogCell]] = [
        [
            VisibleCell(cell_type=CellType.IMPASSABLE),
            VisibleCell(cell_type=CellType.IMPASSABLE),
            VisibleCell(cell_type=CellType.IMPASSABLE),
        ],
        [
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
            VisibleCell(cell_type=CellType.PASSABLE),
        ],
        [FogCell(), FogCell(), FogCell()],
    ]
    patch2 = TankPatch(
        tank_id="A1",
        position=Position(4, 4),
        direction=Direction.SOUTH,
        grid=grid2,
    )
    view2 = PlayerView(
        patches=[patch2],
        tanks=[
            TankInfo(
                tank_id="A1",
                position=Position(4, 4),
                direction=Direction.SOUTH,
                alive=True,
            )
        ],
    )
    am.update(view2)

    # First observation cells should still be known.
    assert am[2, 2] == CellState.PASSABLE  # from first update

    # Second observation cells should also be known.
    assert am[4, 4] == CellState.PASSABLE  # from second update
