"""Tests for server observer support: connect, broadcast, disconnect resilience."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.protocol import (
    StateUpdateMessage,
)
from hmls.server.app import GameSession, create_fastapi_app


def _make_simple_map() -> GameMap:
    """Create a small 5x5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_tanks() -> list[Tank]:
    """Create a minimal set of tanks for two teams."""
    return [
        Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
        Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
    ]


def _make_session() -> GameSession:
    """Create a GameSession for testing."""
    game_map = _make_simple_map()
    tanks = _make_tanks()
    return GameSession(
        game_map=game_map,
        tanks=tanks,
        max_turns=10,
        patch_size=5,
    )


class TestObserverConnection:
    """Tests for observer WebSocket connections."""

    def test_observer_connect_before_game_start(self) -> None:
        """An observer connecting before the game starts should be accepted."""
        session = _make_session()
        app = create_fastapi_app(session)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Send observe message.
                ws.send_json({"type": "observe", "observer_name": "TestObserver"})
                # Observer should stay connected (no immediate response expected
                # until game starts). We just verify no error is raised.

    def test_observer_receives_game_start_when_connected_after_start(self) -> None:
        """Observer connecting after game start should get GameStartMessage."""
        session = _make_session()
        # Simulate game having started.
        session._game_started = True
        session.player_names = {"A": "Alice", "B": "Bob"}

        app = create_fastapi_app(session)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "observe", "observer_name": "TestObserver"})
                # Should receive GameStartMessage.
                data = ws.receive_json()
                assert data["type"] == "game_start"
                assert data["game_map"]["width"] == 5
                assert data["game_map"]["height"] == 5
                assert data["player_names"] == {"A": "Alice", "B": "Bob"}
                assert data["patch_size"] == 5
                assert data["max_turns"] == 10

    def test_player_join_still_works(self) -> None:
        """Players can still join with JoinMessage on the same endpoint."""
        session = _make_session()
        app = create_fastapi_app(session)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "join", "player_name": "Alice"})
                # Should receive waiting message.
                data = ws.receive_json()
                assert data["type"] == "waiting"

    def test_game_full_rejects_third_player(self) -> None:
        """A third player should be rejected when game is full."""
        session = _make_session()
        app = create_fastapi_app(session)

        with TestClient(app) as client:
            # First player joins.
            with client.websocket_connect("/ws") as ws1:
                ws1.send_json({"type": "join", "player_name": "Alice"})
                ws1.receive_json()  # waiting

                # Second player joins.
                with client.websocket_connect("/ws") as ws2:
                    ws2.send_json({"type": "join", "player_name": "Bob"})
                    # Both connected triggers the game start event.

                    # Third player should be rejected.
                    with client.websocket_connect("/ws") as ws3:
                        ws3.send_json({"type": "join", "player_name": "Charlie"})
                        data = ws3.receive_json()
                        assert data["type"] == "error"
                        assert "full" in data["message"].lower()

    def test_invalid_first_message_rejected(self) -> None:
        """Sending an action as first message should be rejected."""
        session = _make_session()
        app = create_fastapi_app(session)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "action", "action": "move_forward"})
                data = ws.receive_json()
                assert data["type"] == "error"


class TestObserverBroadcast:
    """Tests for state broadcasting to observers."""

    @pytest.mark.anyio
    async def test_broadcast_to_observers(self) -> None:
        """StateUpdateMessage should be sent to all connected observers."""
        session = _make_session()
        state = GameState(tanks=_make_tanks())

        # Simulate broadcast with a mock observer list.
        # We test the _broadcast_to_observers method directly.
        sent_messages: list[str] = []

        class FakeWebSocket:
            """Fake WebSocket that records sent messages."""

            async def send_text(self, msg: str) -> None:
                sent_messages.append(msg)

        session._observers = [FakeWebSocket(), FakeWebSocket()]  # type: ignore[list-item]

        state_msg = StateUpdateMessage(
            state=state,
            current_tank_id="A1",
            turns_taken=1,
        )
        await session._broadcast_to_observers(state_msg.model_dump_json())

        assert len(sent_messages) == 2
        for msg_json in sent_messages:
            assert '"type":"state_update"' in msg_json or '"type": "state_update"' in msg_json

    @pytest.mark.anyio
    async def test_broadcast_removes_disconnected_observers(self) -> None:
        """Disconnected observers should be removed from the list."""
        session = _make_session()

        class GoodWebSocket:
            """Fake WebSocket that works."""

            sent: list[str] = []

            async def send_text(self, msg: str) -> None:
                self.sent.append(msg)

        class BadWebSocket:
            """Fake WebSocket that raises on send."""

            async def send_text(self, msg: str) -> None:
                raise ConnectionError("Gone")

        good = GoodWebSocket()
        bad = BadWebSocket()
        session._observers = [good, bad]  # type: ignore[list-item]

        await session._broadcast_to_observers('{"type":"state_update"}')

        # Bad observer should have been removed.
        assert len(session._observers) == 1
        assert session._observers[0] is good  # type: ignore[comparison-overlap]
