"""HMLS Core – foundational types for the tank game."""

from hmls.core.actions import ActionResult, ApplyResult, apply_action, validate_action
from hmls.core.engine import GameEngine, GameResult, HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap, MapLoadError, load_map
from hmls.core.placement import InsufficientPassableCellsError, place_tanks
from hmls.core.player import PendingActionPlayer, Player
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import (
    BoundaryCell,
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
    "BoundaryCell",
    "CellType",
    "Direction",
    "FogCell",
    "GameEngine",
    "GameMap",
    "GameResult",
    "GameState",
    "HistoryEntry",
    "InsufficientPassableCellsError",
    "MapLoadError",
    "PatchCell",
    "PendingActionPlayer",
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
    "load_map",
    "place_tanks",
    "validate_action",
]
