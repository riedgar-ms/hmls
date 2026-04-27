"""Foundational types for the tank game: directions, positions, and actions."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import NamedTuple


class Direction(IntEnum):
    """Cardinal direction a tank can face.

    Values are ordered clockwise starting from north, so turning
    is simple modular arithmetic: ``(d + 1) % 4`` turns right,
    ``(d - 1) % 4`` turns left.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def turn_left(self) -> Direction:
        """Return the direction 90° counter-clockwise from this one."""
        return Direction((self - 1) % 4)

    def turn_right(self) -> Direction:
        """Return the direction 90° clockwise from this one."""
        return Direction((self + 1) % 4)

    def forward_delta(self) -> tuple[int, int]:
        """Return the ``(dx, dy)`` offset for one step in this direction.

        The coordinate system matches :class:`~hmls.core.map.GameMap`:
        *x* increases rightward, *y* increases downward.
        """
        return _DIRECTION_DELTAS[self]


_DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}


class Position(NamedTuple):
    """An ``(x, y)`` coordinate on the game grid.

    Immutable and hashable, suitable for use as dictionary keys
    (e.g. in occupancy maps).
    """

    x: int
    y: int


class Action(Enum):
    """An action a tank can take on its turn."""

    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    FIRE = "fire"
    PASS = "pass"
