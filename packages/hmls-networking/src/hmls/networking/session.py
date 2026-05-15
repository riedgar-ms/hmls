"""WebSocket session management for HMLS game clients.

Provides :class:`GameWebSocketSession`, a reusable async context manager
encapsulating the connect → identify → receive loop pattern shared by
the game client and observer applications.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import websockets
import websockets.asyncio.client
from pydantic import BaseModel, TypeAdapter
from websockets.asyncio.client import ClientConnection

from hmls.protocol import ServerMessage

logger = logging.getLogger("hmls.networking")

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


class GameWebSocketSession:
    """Manages a WebSocket connection to the HMLS game server.

    Handles connecting, sending an identity message, parsing incoming
    server messages, and sending outbound messages. Designed to be used
    via the :meth:`connect` async context manager::

        async with GameWebSocketSession.connect(url, identity_msg) as session:
            async for msg in session.receive_messages():
                ...
    """

    def __init__(self, ws: ClientConnection, url: str) -> None:
        """Initialise with an already-connected WebSocket.

        Users should not call this directly; use :meth:`connect` instead.

        Args:
            ws: The connected WebSocket instance.
            url: The server URL (for logging/diagnostics).
        """
        self._ws = ws
        self._url = url

    @staticmethod
    @asynccontextmanager
    async def connect(
        url: str,
        identity_message: BaseModel,
    ) -> AsyncIterator[GameWebSocketSession]:
        """Connect to the server and send the identity message.

        This is an async context manager that yields a
        :class:`GameWebSocketSession` after the connection is established
        and the identity message has been sent.

        Args:
            url: WebSocket server URL (e.g. ``ws://localhost:8765/ws``).
            identity_message: The message to send immediately after
                connecting (typically a :class:`~hmls.protocol.JoinMessage`
                or :class:`~hmls.protocol.ObserveMessage`).

        Yields:
            A connected session ready to receive/send messages.

        Raises:
            websockets.exceptions.WebSocketException: On connection failure.
        """
        logger.debug("Connecting to %s", url)
        async with websockets.asyncio.client.connect(url) as ws:
            session = GameWebSocketSession(ws, url)
            await ws.send(identity_message.model_dump_json())
            logger.debug("Identity message sent: %s", type(identity_message).__name__)
            yield session

    async def receive_messages(self) -> AsyncIterator[ServerMessage]:
        """Yield parsed server messages from the WebSocket connection.

        Iterates over incoming frames, parsing each as a
        :class:`~hmls.protocol.ServerMessage`. Invalid messages are
        logged and skipped rather than raising.

        Yields:
            Parsed :class:`~hmls.protocol.ServerMessage` instances.
        """
        async for raw in self._ws:
            try:
                msg = _server_message_adapter.validate_json(str(raw))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse server message: %s", exc)
                continue
            yield msg

    async def send(self, message: BaseModel) -> None:
        """Send a message to the server.

        Args:
            message: A Pydantic model to serialise and send as JSON.

        Raises:
            websockets.exceptions.WebSocketException: If the connection
                is closed or broken.
        """
        await self._ws.send(message.model_dump_json())

    @property
    def closed(self) -> bool:
        """Whether the underlying WebSocket connection is closed."""
        return self._ws.close_code is not None
