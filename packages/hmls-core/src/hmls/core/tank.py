"""Tank model: a single tank on the game map."""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.types import Direction, Position

TankId = str
"""Unique identifier for a tank (type alias)."""


class Tank(BaseModel, extra="forbid"):
    """A single tank in the game.

    Attributes:
        id: Unique identifier for this tank.
        team: The team this tank belongs to.
        position: Current ``(x, y)`` grid position.
        direction: The cardinal direction the tank is currently facing.
        alive: Whether the tank is still in play.
    """

    id: TankId
    team: str
    position: Position
    direction: Direction
    alive: bool = True
