"""Shared fixtures for reward tests."""

from __future__ import annotations

from typing import Protocol

import pytest

from hmls.core.engine import HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import FogCell, TankPatch, VisibleCell


class MakeEntryFactory(Protocol):
    """Protocol for the make_entry factory callable."""

    def __call__(
        self,
        action: Action = ...,
        valid: bool = ...,
        hit: bool | None = ...,
    ) -> HistoryEntry: ...


class MakePatchFactory(Protocol):
    """Protocol for patch factory callables that accept a size."""

    def __call__(self, size: int = ...) -> TankPatch: ...


class MakePatchAtFactory(Protocol):
    """Protocol for the make_patch_at factory callable."""

    def __call__(self, pos: Position, size: int = ...) -> TankPatch: ...


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


@pytest.fixture
def make_entry() -> MakeEntryFactory:
    """Fixture returning a factory for HistoryEntry objects."""
    return _make_entry


@pytest.fixture
def make_empty_patch() -> MakePatchFactory:
    """Fixture returning a factory for empty TankPatch objects."""
    return _make_empty_patch


@pytest.fixture
def make_patch_with_enemy_ahead() -> MakePatchFactory:
    """Fixture returning a factory for patches with enemy directly ahead."""
    return _make_patch_with_enemy_ahead


@pytest.fixture
def make_patch_with_enemy_in_cone() -> MakePatchFactory:
    """Fixture returning a factory for patches with enemy in cone."""
    return _make_patch_with_enemy_in_cone


@pytest.fixture
def make_patch_with_fogged_enemy() -> MakePatchFactory:
    """Fixture returning a factory for patches with fogged enemy."""
    return _make_patch_with_fogged_enemy


@pytest.fixture
def make_patch_at() -> MakePatchAtFactory:
    """Fixture returning a factory for patches at a specific position."""
    return _make_patch_at
