"""Async event bus for decoupled server component communication.

The :class:`EventBus` dispatches typed events to registered async
callbacks, allowing the orchestrator and network manager to communicate
without direct coupling.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from hmls.server.event_types import EventCallback

logger = logging.getLogger("hmls.server.event_bus")


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
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error in event handler %s for %s",
                    callback.__qualname__,
                    type(event).__name__,
                )
