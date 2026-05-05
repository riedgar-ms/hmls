"""HMLS Core – foundational types for the tank game."""

from hmls.core.actions import ActionResult, ApplyResult, apply_action, validate_action
from hmls.core.engine import GameEngine, GameResult, HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import (
    FogCell,
    PatchCell,
    PlayerView,
    TankInfo,
    TankPatch,
    VisibleCell,
    build_player_view,
    compute_visibility_mask,
    extract_patch,
)

__all__ = [
    "Action",
    "ActionResult",
    "ApplyResult",
    "CellType",
    "Direction",
    "FogCell",
    "GameEngine",
    "GameMap",
    "GameResult",
    "GameState",
    "HistoryEntry",
    "PatchCell",
    "Player",
    "PlayerView",
    "Position",
    "Tank",
    "TankId",
    "TankInfo",
    "TankPatch",
    "VisibleCell",
    "apply_action",
    "build_player_view",
    "compute_visibility_mask",
    "extract_patch",
    "validate_action",
]
