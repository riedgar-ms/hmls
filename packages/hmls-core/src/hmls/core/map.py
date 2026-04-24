"""Game map definition: a cartesian grid of passable/impassable cells."""

from enum import Enum
from typing import Self

from pydantic import BaseModel, model_validator


class CellType(str, Enum):
    """Type of a single map cell."""

    PASSABLE = "passable"
    IMPASSABLE = "impassable"


class GameMap(BaseModel):
    """A rectangular grid map for the tank game.

    Each cell is either passable or impassable.  Cells are stored in
    row-major order (row 0 first, then row 1, …).  The coordinate system
    uses ``(row, col)`` with ``(0, 0)`` at the top-left corner.

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

    def __getitem__(self, pos: tuple[int, int]) -> CellType:
        """Return the cell type at ``(row, col)``.

        Raises:
            IndexError: If the position is out of bounds.
        """
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"Position ({row}, {col}) is out of bounds for a {self.height}×{self.width} map"
            )
        return self.cells[row * self.width + col]

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        """Set the cell type at ``(row, col)``.

        Raises:
            IndexError: If the position is out of bounds.
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError(
                f"Position ({row}, {col}) is out of bounds for a {self.height}×{self.width} map"
            )
        self.cells[row * self.width + col] = cell_type
