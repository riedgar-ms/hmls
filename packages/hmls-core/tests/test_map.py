"""Tests for hmls.core.map – GameMap and CellType."""

import json

import pytest

from hmls.core.map import CellType, GameMap


class TestGameMapConstruction:
    """Tests for creating GameMap instances."""

    def test_default_cells_are_passable(self) -> None:
        """A map without explicit cells should be all-passable."""
        m = GameMap(width=3, height=2)
        assert len(m.cells) == 6
        assert all(c == CellType.PASSABLE for c in m.cells)

    def test_explicit_cells(self) -> None:
        """A map can be created with an explicit flat cell list."""
        cells = [CellType.PASSABLE, CellType.IMPASSABLE, CellType.PASSABLE]
        m = GameMap(width=3, height=1, cells=cells)
        assert m.cells == cells

    def test_wrong_cell_count_raises(self) -> None:
        """Providing a cell list whose length != width*height should fail."""
        with pytest.raises(ValueError, match="does not match"):
            GameMap(width=2, height=2, cells=[CellType.PASSABLE])

    def test_zero_width_raises(self) -> None:
        """Width must be >= 1."""
        with pytest.raises(ValueError, match="width must be >= 1"):
            GameMap(width=0, height=1)

    def test_zero_height_raises(self) -> None:
        """Height must be >= 1."""
        with pytest.raises(ValueError, match="height must be >= 1"):
            GameMap(width=1, height=0)

    def test_negative_dimensions_raise(self) -> None:
        """Negative dimensions should be rejected."""
        with pytest.raises(ValueError, match=r"must be >= 1"):
            GameMap(width=-1, height=3)

    def test_total_cells(self) -> None:
        """total_cells property returns width * height."""
        m = GameMap(width=7, height=3)
        assert m.total_cells == 21


class TestGameMapAccess:
    """Tests for reading and writing cells via (x, y) indexing."""

    def test_getitem(self) -> None:
        """Cells can be read via (x, y) indexing."""
        m = GameMap(width=3, height=2)
        assert m[0, 0] == CellType.PASSABLE
        assert m[2, 1] == CellType.PASSABLE

    def test_setitem(self) -> None:
        """Cells can be written via (x, y) indexing."""
        m = GameMap(width=3, height=2)
        m[1, 1] = CellType.IMPASSABLE
        assert m[1, 1] == CellType.IMPASSABLE
        # Neighbours should be unaffected.
        assert m[1, 0] == CellType.PASSABLE
        assert m[0, 1] == CellType.PASSABLE

    def test_getitem_out_of_bounds(self) -> None:
        """Accessing an out-of-bounds cell should raise IndexError."""
        m = GameMap(width=2, height=2)
        with pytest.raises(IndexError):
            _ = m[2, 0]
        with pytest.raises(IndexError):
            _ = m[0, 2]
        with pytest.raises(IndexError):
            _ = m[-1, 0]

    def test_setitem_out_of_bounds(self) -> None:
        """Setting an out-of-bounds cell should raise IndexError."""
        m = GameMap(width=2, height=2)
        with pytest.raises(IndexError):
            m[2, 0] = CellType.IMPASSABLE


class TestBoundsAndNeighbours:
    """Tests for in_bounds, neighbours, and all_positions."""

    def test_in_bounds(self) -> None:
        """in_bounds correctly identifies valid and invalid coordinates."""
        m = GameMap(width=5, height=5)
        assert m.in_bounds(0, 0) is True
        assert m.in_bounds(4, 4) is True
        assert m.in_bounds(5, 0) is False
        assert m.in_bounds(-1, 0) is False

    def test_corner_neighbours(self) -> None:
        """Top-left corner should have only right and down neighbours."""
        m = GameMap(width=5, height=5)
        n = set(m.neighbours(0, 0))
        assert n == {(1, 0), (0, 1)}

    def test_centre_neighbours(self) -> None:
        """Centre cell should have four neighbours."""
        m = GameMap(width=5, height=5)
        n = set(m.neighbours(2, 2))
        assert n == {(2, 1), (3, 2), (2, 3), (1, 2)}

    def test_all_positions_count(self) -> None:
        """all_positions yields every cell exactly once."""
        m = GameMap(width=4, height=3)
        positions = list(m.all_positions())
        assert len(positions) == 12


class TestCounting:
    """Tests for count_passable, count_impassable, total_cells."""

    def test_default_all_passable(self) -> None:
        """A fresh map should have all cells passable."""
        m = GameMap(width=4, height=4)
        assert m.count_passable() == 16
        assert m.count_impassable() == 0

    def test_mixed(self) -> None:
        """Counting after setting some impassable cells."""
        m = GameMap(width=3, height=3)
        m[0, 0] = CellType.IMPASSABLE
        m[1, 1] = CellType.IMPASSABLE
        assert m.count_passable() == 7
        assert m.count_impassable() == 2


class TestStringRepresentation:
    """Tests for __str__ and __repr__."""

    def test_str_passable(self) -> None:
        """An all-passable 3×2 map renders as dots."""
        m = GameMap(width=3, height=2)
        assert str(m) == "...\n..."

    def test_str_mixed(self) -> None:
        """Impassable cells render as '#'."""
        m = GameMap(width=3, height=2)
        m[1, 0] = CellType.IMPASSABLE
        assert str(m) == ".#.\n..."

    def test_repr(self) -> None:
        """repr includes dimensions."""
        m = GameMap(width=3, height=2)
        r = repr(m)
        assert "3" in r
        assert "2" in r


class TestGameMapSerialisation:
    """Tests for JSON round-trip via Pydantic."""

    def test_json_round_trip(self) -> None:
        """A GameMap should survive a JSON serialise/deserialise cycle."""
        original = GameMap(width=3, height=2)
        original[1, 0] = CellType.IMPASSABLE

        json_str = original.model_dump_json()
        restored = GameMap.model_validate_json(json_str)

        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.cells == original.cells

    def test_json_structure(self) -> None:
        """The JSON output should have the expected keys."""
        m = GameMap(width=2, height=1)
        data = json.loads(m.model_dump_json())
        assert set(data.keys()) == {"width", "height", "cells"}
        assert data["width"] == 2
        assert data["height"] == 1
        assert len(data["cells"]) == 2
