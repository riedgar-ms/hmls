"""Shared WebSocket networking for HMLS game clients.

Provides :class:`GameWebSocketSession`, a reusable async context manager
that handles connecting to the game server, sending an identity message,
and yielding parsed :class:`~hmls.protocol.ServerMessage` objects.
"""

from hmls.networking.session import GameWebSocketSession

__all__ = ["GameWebSocketSession"]
