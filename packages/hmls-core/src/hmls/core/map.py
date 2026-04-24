"""Game map definition: a cartesian grid of passable/impassable cells."""

from __future__ import annotations

from collections.abc import Iterator
from enum import IntEnum
from typing import Self

from pydantic import BaseModel, model_validator


class CellType(IntEnum):
    """Type of a single map cell.

    Values are integers for compact JSON serialisation:
    ``0`` for impassable, ``1`` for passable.
    """

    IMPASSABLE = 0
    PASSABLE = 1


class GameMap(BaseModel):
    """A rectangular grid map for the tank game.

    Each cell is either passable or impassable.  Cells are stored in
    row-major order (row 0 first, then row 1, …).  The coordinate system
    uses ``(x, y)`` where *x* is the column (0 = left) and *y* is the
    row (0 = top), with ``(0, 0)`` at the top-left corner.

    Attributes:
        width: Number of columns (must be >= 1).
        height: Number of rows (must be >= 1).
        cells: Flat list of cell types in row-major order.
            Defaults to all-passable when not supplied explicitly.
    """

    width: int
    height: int
    cells: list[CellType] = []

    @model_validator(mode="after")
    def _validate_cells(self) -> Self:
        """Fill default cells or validate that an explicit list matches dimensions."""
        expected = self.width * self.height
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        if self.height < 1:
            raise ValueError(f"height must be >= 1, got {self.height}")
        if not self.cells:
            self.cells = [CellType.PASSABLE] * expected
        elif len(self.cells) != expected:
            raise ValueError(
                f"cells length {len(self.cells)} does not match width*height ({expected})"
            )
        return self

    # ── Cell access ───────────────────────────────────────────────────

    def in_bounds(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is within the map."""
        return 0 <= x < self.width and 0 <= y < self.height

    def __getitem__(self, pos: tuple[int, int]) -> CellType:
        """Return the cell type at ``(x, y)``.

        Raises:
            IndexError: If the position is out of bounds.
        """
        x, y = pos
        if not self.in_bounds(x, y):
            raise IndexError(
                f"Position ({x}, {y}) is out of bounds for a {self.width}×{self.height} map"
            )
        return self.cells[y * self.width + x]

    def __setitem__(self, pos: tuple[int, int], cell_type: CellType) -> None:
        """Set the cell type at ``(x, y)``.

        Raises:
            IndexError: If the position is out of bounds.
        """
        x, y = pos
        if not self.in_bounds(x, y):
            raise IndexError(
                f"Position ({x}, {y}) is out of bounds for a {self.width}×{self.height} map"
            )
        self.cells[y * self.width + x] = cell_type

    def set_cell(self, x: int, y: int, cell_type: CellType) -> None:
        """Set the cell type at ``(x, y)``.

        Convenience wrapper around :meth:`__setitem__` for callers that
        prefer positional arguments.

        Raises:
            IndexError: If the position is out of bounds.
        """
        self[x, y] = cell_type

    # ── Iteration helpers ─────────────────────────────────────────────

    def all_positions(self) -> Iterator[tuple[int, int]]:
        """Yield every ``(x, y)`` position in the map, row by row."""
        for y in range(self.height):
            for x in range(self.width):
                yield x, y

    def neighbours(self, x: int, y: int) -> Iterator[tuple[int, int]]:
        """Yield the 4-connected neighbours of ``(x, y)`` that are in bounds.

        Order: up, right, down, left.
        """
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield nx, ny

    # ── Counting ──────────────────────────────────────────────────────

    @property
    def total_cells(self) -> int:
        """Total number of cells in the map."""
        return self.width * self.height

    def count_passable(self) -> int:
        """Return the number of passable cells."""
        return sum(1 for c in self.cells if c == CellType.PASSABLE)

    def count_impassable(self) -> int:
        """Return the number of impassable cells."""
        return sum(1 for c in self.cells if c == CellType.IMPASSABLE)

    # ── Text representation ───────────────────────────────────────────

    def __str__(self) -> str:
        """Simple text representation: ``'.'`` for passable, ``'#'`` for impassable."""
        lines: list[str] = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                line += "." if self[x, y] == CellType.PASSABLE else "#"
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GameMap({self.width}, {self.height}, "
            f"passable={self.count_passable()}/{self.total_cells})"
        )
