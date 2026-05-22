"""Typed game events and async event bus for decoupled server components.

Events are simple dataclasses exchanged between the
:class:`~hmls.server.orchestrator.GameOrchestrator` and the
:class:`~hmls.server.network_manager.NetworkManager` via the
:class:`EventBus`.  Most events flow from orchestrator to network
manager, but :class:`PlayerDisconnectedEvent` flows in the opposite
direction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import HistoryEntry
from hmls.core.tank import Tank, TankId
from hmls.core.visibility import PlayerView

logger = logging.getLogger("hmls.server.events")

# ── Event types ───────────────────────────────────────────────────────

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


# ── Event bus ─────────────────────────────────────────────────────────


class EventBus:
    """Simple async event bus: subscribe by event type, emit to all subscribers.

    Usage::

        bus = EventBus()
        bus.subscribe(GameOverEvent, my_handler)
        await bus.emit(GameOverEvent(winner="A", reason="Victory"))
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: type, callback: EventCallback) -> None:
        """Register *callback* to be called whenever *event_type* is emitted.

        Args:
            event_type: The event class to listen for.
            callback: An async callable that accepts the event as its sole
                positional argument.
        """
        self._subscribers[event_type].append(callback)

    async def emit(self, event: object) -> None:
        """Emit *event* to all subscribers registered for its type.

        Callbacks are invoked sequentially in registration order.  If a
        callback raises, the exception is logged and remaining callbacks
        still execute.

        Args:
            event: The event instance to dispatch.
        """
        for callback in self._subscribers.get(type(event), []):
            try:
                await callback(event)
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    callback.__qualname__,
                    type(event).__name__,
                )
