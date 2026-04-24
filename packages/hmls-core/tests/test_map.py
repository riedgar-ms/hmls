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
        with pytest.raises(ValueError):
            GameMap(width=-1, height=3)


class TestGameMapAccess:
    """Tests for reading and writing cells."""

    def test_getitem(self) -> None:
        """Cells can be read via (row, col) indexing."""
        m = GameMap(width=3, height=2)
        assert m[0, 0] == CellType.PASSABLE
        assert m[1, 2] == CellType.PASSABLE

    def test_set_cell(self) -> None:
        """set_cell changes the value at the given position."""
        m = GameMap(width=3, height=2)
        m.set_cell(1, 1, CellType.IMPASSABLE)
        assert m[1, 1] == CellType.IMPASSABLE
        # Neighbours should be unaffected.
        assert m[0, 1] == CellType.PASSABLE
        assert m[1, 0] == CellType.PASSABLE

    def test_getitem_out_of_bounds(self) -> None:
        """Accessing an out-of-bounds cell should raise IndexError."""
        m = GameMap(width=2, height=2)
        with pytest.raises(IndexError):
            _ = m[2, 0]
        with pytest.raises(IndexError):
            _ = m[0, 2]
        with pytest.raises(IndexError):
            _ = m[-1, 0]

    def test_set_cell_out_of_bounds(self) -> None:
        """Setting an out-of-bounds cell should raise IndexError."""
        m = GameMap(width=2, height=2)
        with pytest.raises(IndexError):
            m.set_cell(2, 0, CellType.IMPASSABLE)


class TestGameMapSerialisation:
    """Tests for JSON round-trip via Pydantic."""

    def test_json_round_trip(self) -> None:
        """A GameMap should survive a JSON serialise/deserialise cycle."""
        original = GameMap(width=3, height=2)
        original.set_cell(0, 1, CellType.IMPASSABLE)

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
