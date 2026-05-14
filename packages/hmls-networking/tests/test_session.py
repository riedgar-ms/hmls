"""Tests for hmls.networking.session module."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hmls.core.types import Action
from hmls.networking.session import GameWebSocketSession
from hmls.protocol import (
    ActionMessage,
    ErrorMessage,
    JoinMessage,
    ObserveMessage,
    ServerMessage,
    WaitingMessage,
)


class FakeWebSocket:
    """Fake WebSocket that yields pre-configured messages."""

    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self._sent: list[str] = []
        self.close_code: int | None = None

    async def send(self, data: str) -> None:
        """Record sent messages."""
        self._sent.append(data)

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


class FakeWebSocketCM:
    """Fake async context manager mimicking websockets.connect."""

    def __init__(self, ws: FakeWebSocket) -> None:
        self._ws = ws

    async def __aenter__(self) -> FakeWebSocket:
        return self._ws

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.mark.anyio
async def test_connect_sends_identity_message() -> None:
    """The session sends the identity message JSON on connect."""
    identity = JoinMessage(player_name="TestPlayer")
    ws = FakeWebSocket([])

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            assert session is not None

    assert len(ws._sent) == 1
    sent_data = json.loads(ws._sent[0])
    assert sent_data["type"] == "join"
    assert sent_data["player_name"] == "TestPlayer"


@pytest.mark.anyio
async def test_connect_with_observe_message() -> None:
    """The session works with ObserveMessage identity."""
    identity = ObserveMessage(observer_name="TestObserver")
    ws = FakeWebSocket([])

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            assert session is not None

    sent_data = json.loads(ws._sent[0])
    assert sent_data["type"] == "observe"
    assert sent_data["observer_name"] == "TestObserver"


@pytest.mark.anyio
async def test_receive_messages_parses_valid_messages() -> None:
    """receive_messages yields parsed ServerMessage objects."""
    waiting_msg = WaitingMessage(message="Waiting for opponent")
    error_msg = ErrorMessage(message="Something went wrong")
    raw_messages = [
        waiting_msg.model_dump_json(),
        error_msg.model_dump_json(),
    ]
    ws = FakeWebSocket(raw_messages)
    identity = JoinMessage(player_name="Test")

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            received: list[ServerMessage] = []
            async for msg in session.receive_messages():
                received.append(msg)

    assert len(received) == 2
    assert isinstance(received[0], WaitingMessage)
    assert received[0].message == "Waiting for opponent"
    assert isinstance(received[1], ErrorMessage)
    assert received[1].message == "Something went wrong"


@pytest.mark.anyio
async def test_receive_messages_skips_invalid_json() -> None:
    """Invalid messages are skipped, not raised."""
    valid_msg = WaitingMessage(message="Hello")
    raw_messages = [
        "not valid json {{{",
        valid_msg.model_dump_json(),
    ]
    ws = FakeWebSocket(raw_messages)
    identity = JoinMessage(player_name="Test")

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            received: list[ServerMessage] = []
            async for msg in session.receive_messages():
                received.append(msg)

    assert len(received) == 1
    assert isinstance(received[0], WaitingMessage)


@pytest.mark.anyio
async def test_send_serializes_message() -> None:
    """send() serializes the message as JSON and sends it."""
    ws = FakeWebSocket([])
    identity = JoinMessage(player_name="Test")

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            action_msg = ActionMessage(action=Action.FIRE)
            await session.send(action_msg)

    # First sent is identity, second is our action
    assert len(ws._sent) == 2
    action_data = json.loads(ws._sent[1])
    assert action_data["type"] == "action"
    assert action_data["action"] == "fire"


@pytest.mark.anyio
async def test_closed_property() -> None:
    """closed property reflects WebSocket state."""
    ws = FakeWebSocket([])
    identity = JoinMessage(player_name="Test")

    with patch(
        "hmls.networking.session.websockets.asyncio.client.connect",
        return_value=FakeWebSocketCM(ws),
    ):
        async with GameWebSocketSession.connect("ws://localhost:8765/ws", identity) as session:
            assert session.closed is False
            ws.close_code = 1000
            assert session.closed is True
