"""Automapper: accumulates terrain knowledge from fog-of-war observations.

As the player's tanks explore the map, visibility patches reveal terrain.
The automapper projects egocentric patch cells back to world coordinates
and maintains a persistent map of known terrain, removing fog-of-war for
previously explored areas.
"""

from __future__ import annotations

from enum import IntEnum

from hmls.core.types import Direction
from hmls.core.visibility import PlayerView, TankPatch, VisibleCell


class CellState(IntEnum):
    """Knowledge state of a cell in the automapper.

    Values:
        UNKNOWN: Not yet observed.
        PASSABLE: Observed as passable terrain.
        IMPASSABLE: Observed as impassable terrain (wall).
    """

    UNKNOWN = 0
    PASSABLE = 1
    IMPASSABLE = 2


class AutoMap:
    """Accumulates terrain knowledge from PlayerView observations.

    Maintains a grid of :class:`CellState` values, initially all
    ``UNKNOWN``. Each time :meth:`update` is called with a new
    :class:`PlayerView`, visible cells are projected from egocentric
    coordinates to world coordinates and marked accordingly.

    Args:
        width: Map width in cells.
        height: Map height in cells.
    """

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._grid: list[list[CellState]] = [
            [CellState.UNKNOWN for _ in range(width)] for _ in range(height)
        ]

    @property
    def width(self) -> int:
        """Map width in cells."""
        return self._width

    @property
    def height(self) -> int:
        """Map height in cells."""
        return self._height

    def __getitem__(self, pos: tuple[int, int]) -> CellState:
        """Get the cell state at (x, y).

        Args:
            pos: ``(x, y)`` coordinates.

        Returns:
            The current knowledge state of the cell.
        """
        x, y = pos
        return self._grid[y][x]

    def update(self, view: PlayerView) -> None:
        """Update the automap with observations from a PlayerView.

        Projects each visible cell in every tank patch back to world
        coordinates and marks the corresponding cell in the automap.

        Args:
            view: The fog-of-war view received from the server.
        """
        for patch in view.patches:
            self._apply_patch(patch)

    def _apply_patch(self, patch: TankPatch) -> None:
        """Project a single tank patch onto the automap.

        Args:
            patch: An egocentric visibility patch.
        """
        grid = patch.grid
        patch_size = len(grid)
        half = patch_size // 2

        # Compute direction vectors for this tank's orientation.
        forward = patch.direction.forward_delta()
        right = Direction((patch.direction + 1) % 4).forward_delta()

        for ego_row in range(patch_size):
            for ego_col in range(patch_size):
                cell = grid[ego_row][ego_col]
                if not isinstance(cell, VisibleCell):
                    continue

                # Convert egocentric (row, col) to world offset.
                fwd_steps = half - ego_row
                rgt_steps = ego_col - half
                fx, fy = forward
                rx, ry = right

                world_x = patch.position.x + fwd_steps * fx + rgt_steps * rx
                world_y = patch.position.y + fwd_steps * fy + rgt_steps * ry

                # Check bounds.
                if not (0 <= world_x < self._width and 0 <= world_y < self._height):
                    continue

                # Mark the cell.
                from hmls.core.map import CellType

                if cell.cell_type == CellType.PASSABLE:
                    self._grid[world_y][world_x] = CellState.PASSABLE
                else:
                    self._grid[world_y][world_x] = CellState.IMPASSABLE
