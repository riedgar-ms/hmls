"""Tests for hmls.core.placement."""

from __future__ import annotations

import pytest

from hmls.core.map import CellType, GameMap
from hmls.core.placement import InsufficientPassableCellsError, place_tanks
from hmls.core.types import Direction


def _make_map(width: int, height: int, passable: bool = True) -> GameMap:
    """Create a simple map for testing."""
    cell = CellType.PASSABLE if passable else CellType.IMPASSABLE
    return GameMap(width=width, height=height, cells=[cell] * (width * height))


class TestPlaceTanks:
    """Tests for the place_tanks function."""

    def test_correct_number_of_tanks(self) -> None:
        """Should create 2 * tanks_per_player tanks."""
        game_map = _make_map(10, 10)
        tanks = place_tanks(game_map, 3, seed=42)
        assert len(tanks) == 6

    def test_team_assignment(self) -> None:
        """Tanks should be split evenly between teams A and B."""
        game_map = _make_map(10, 10)
        tanks = place_tanks(game_map, 2, seed=0)
        team_a = [t for t in tanks if t.team == "A"]
        team_b = [t for t in tanks if t.team == "B"]
        assert len(team_a) == 2
        assert len(team_b) == 2

    def test_tank_ids(self) -> None:
        """Tank IDs should follow the pattern team + 1-based index."""
        game_map = _make_map(10, 10)
        tanks = place_tanks(game_map, 2, seed=0)
        ids = [t.id for t in tanks]
        assert ids == ["A1", "A2", "B1", "B2"]

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical placements."""
        game_map = _make_map(10, 10)
        tanks1 = place_tanks(game_map, 3, seed=123)
        tanks2 = place_tanks(game_map, 3, seed=123)
        assert tanks1 == tanks2

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (almost certainly) produce different placements."""
        game_map = _make_map(10, 10)
        tanks1 = place_tanks(game_map, 3, seed=1)
        tanks2 = place_tanks(game_map, 3, seed=2)
        positions1 = [t.position for t in tanks1]
        positions2 = [t.position for t in tanks2]
        assert positions1 != positions2

    def test_directions_are_valid(self) -> None:
        """All tank directions should be valid Direction enum members."""
        game_map = _make_map(10, 10)
        tanks = place_tanks(game_map, 4, seed=99)
        for tank in tanks:
            assert tank.direction in list(Direction)

    def test_positions_are_unique(self) -> None:
        """No two tanks should occupy the same position."""
        game_map = _make_map(10, 10)
        tanks = place_tanks(game_map, 4, seed=7)
        positions = [t.position for t in tanks]
        assert len(positions) == len(set(positions))

    def test_insufficient_cells_raises(self) -> None:
        """Should raise InsufficientPassableCellsError when not enough cells."""
        game_map = _make_map(2, 2)  # only 4 passable cells
        with pytest.raises(InsufficientPassableCellsError) as exc_info:
            place_tanks(game_map, 3)  # needs 6
        assert exc_info.value.needed == 6
        assert exc_info.value.available == 4

    def test_all_impassable_raises(self) -> None:
        """Should raise when map has zero passable cells."""
        game_map = _make_map(5, 5, passable=False)
        with pytest.raises(InsufficientPassableCellsError):
            place_tanks(game_map, 1)

    def test_positions_on_passable_cells(self) -> None:
        """Tanks should only be placed on passable cells."""
        # Create a map with some impassable cells
        game_map = GameMap(width=5, height=5)
        game_map[0, 0] = CellType.IMPASSABLE
        game_map[1, 1] = CellType.IMPASSABLE
        tanks = place_tanks(game_map, 2, seed=42)
        for tank in tanks:
            assert game_map[tank.position.x, tank.position.y] == CellType.PASSABLE
