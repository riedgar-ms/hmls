"""Typed game event dataclasses for decoupled server components.

Events are simple dataclasses exchanged between the
:class:`~hmls.server.orchestrator.GameOrchestrator` and the
:class:`~hmls.server.network_manager.NetworkManager` via the
:class:`~hmls.server.event_bus.EventBus`.  Most events flow from
orchestrator to network manager, but :class:`PlayerDisconnectedEvent`
flows in the opposite direction.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import HistoryEntry
from hmls.core.tank import Tank, TankId
from hmls.core.visibility import PlayerView

EventCallback = Callable[..., Coroutine[Any, Any, None]]


@dataclass
class GameStartedEvent:
    """Emitted once both players have connected and the game begins."""

    game_map: GameMap
    tanks: list[Tank]
    player_names: dict[str, str]
    patch_size: int
    max_turns: int


@dataclass
class YourTurnEvent:
    """Emitted when a player's tank needs to act."""

    tank_id: TankId
    team: str
    view: PlayerView


@dataclass
class StateUpdatedEvent:
    """Emitted after each step to broadcast the new game state."""

    state: GameState
    current_tank_id: TankId | None
    turns_taken: int


@dataclass
class TurnCompletedEvent:
    """Emitted after a step completes, carrying the history entry."""

    entry: HistoryEntry
    acting_team: str


@dataclass
class GameOverEvent:
    """Emitted when the game ends (normal completion or disconnection)."""

    winner: str | None
    reason: str


@dataclass
class PlayerDisconnectedEvent:
    """Emitted when a player disconnects mid-game."""

    team: str
