"""HMLS Core – foundational types for the tank game."""

from hmls.core.actions import ActionResult, apply_action, validate_action
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action, Direction, Position

__all__ = [
    "Action",
    "ActionResult",
    "CellType",
    "Direction",
    "GameMap",
    "GameState",
    "Position",
    "Tank",
    "TankId",
    "apply_action",
    "validate_action",
]
