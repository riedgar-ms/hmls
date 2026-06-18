"""Discrete order vocabulary for squad planner models.

Defines the :class:`Order` enum representing the high-level tactical
orders a planner can issue to individual tanks.  These are simple
categorical labels with no parameters — parameterised orders (e.g.
directional movement targets) are planned for a future extension.
"""

from __future__ import annotations

from enum import IntEnum


class Order(IntEnum):
    """High-level tactical orders issued by a planner to individual tanks.

    Each order communicates a strategic *intent* that the executor model
    translates into low-level actions (move, turn, fire, pass) based on
    its local egocentric patch.

    Attributes:
        ADVANCE: Move toward the front / explore forward.
        RETREAT: Fall back / move away from enemies.
        HOLD: Stay in current area, minimal movement.
        ENGAGE: Seek out and fire at enemies aggressively.
        EVADE: Avoid confrontation, escape threats.
        SCOUT: Explore unseen terrain.
        FLANK_LEFT: Move to attack from the left.
        FLANK_RIGHT: Move to attack from the right.
    """

    ADVANCE = 0
    RETREAT = 1
    HOLD = 2
    ENGAGE = 3
    EVADE = 4
    SCOUT = 5
    FLANK_LEFT = 6
    FLANK_RIGHT = 7


NUM_ORDERS: int = len(Order)
"""Total number of discrete orders in the vocabulary."""
