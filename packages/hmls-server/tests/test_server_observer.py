"""Tests for observer support: connect, broadcast, disconnect resilience."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.protocol import StateUpdateMessage
from hmls.server.app import create_fastapi_app
from hmls.server.event_bus import EventBus
from hmls.server.network_manager import NetworkManager
from hmls.server.remote_player import RemotePlayer


def _make_simple_map() -> GameMap:
    """Create a small 5x5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_tanks() -> list[Tank]:
    """Create a minimal set of tanks for two teams."""
    return [
        Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
        Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
    ]


def _make_network_manager() -> NetworkManager:
    """Create a NetworkManager for testing."""
    game_map = _make_simple_map()
    tanks = _make_tanks()
    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }
    return NetworkManager(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        patch_size=5,
        max_turns=10,
    )


class TestObserverConnection:
    """Tests for observer WebSocket connections."""

    def test_observer_connect_before_game_start(self) -> None:
        """An observer connecting before the game starts should be accepted."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "observe", "observer_name": "TestObserver"})
                # Observer should stay connected (no immediate response expected
                # until game starts). We just verify no error is raised.

    def test_observer_receives_game_start_when_connected_after_start(self) -> None:
        """Observer connecting after game start should get GameStartMessage."""
        nm = _make_network_manager()
        nm._game_started = True
        nm.player_names = {"A": "Alice", "B": "Bob"}

        app = create_fastapi_app(nm)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "observe", "observer_name": "TestObserver"})
                data = ws.receive_json()
                assert data["type"] == "game_start"
                assert data["game_map"]["width"] == 5
                assert data["game_map"]["height"] == 5
                assert data["player_names"] == {"A": "Alice", "B": "Bob"}
                assert data["patch_size"] == 5
                assert data["max_turns"] == 10

    def test_player_join_still_works(self) -> None:
        """Players can still join with JoinMessage on the same endpoint."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "join", "player_name": "Alice"})
                data = ws.receive_json()
                assert data["type"] == "waiting"

    def test_game_full_rejects_third_player(self) -> None:
        """A third player should be rejected when game is full."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws1:
                ws1.send_json({"type": "join", "player_name": "Alice"})
                ws1.receive_json()  # waiting

                with client.websocket_connect("/ws") as ws2:
                    ws2.send_json({"type": "join", "player_name": "Bob"})

                    with client.websocket_connect("/ws") as ws3:
                        ws3.send_json({"type": "join", "player_name": "Charlie"})
                        data = ws3.receive_json()
                        assert data["type"] == "error"
                        assert "full" in data["message"].lower()

    def test_invalid_first_message_rejected(self) -> None:
        """Sending an action as first message should be rejected."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm)

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
        nm = _make_network_manager()
        state = GameState(tanks=_make_tanks())

        sent_messages: list[str] = []

        class FakeWebSocket:
            """Fake WebSocket that records sent messages."""

            async def send_text(self, msg: str) -> None:
                sent_messages.append(msg)

        nm.observers = [FakeWebSocket(), FakeWebSocket()]  # type: ignore[list-item]

        state_msg = StateUpdateMessage(
            state=state,
            current_tank_id="A1",
            turns_taken=1,
        )
        await nm.broadcast_to_observers(state_msg.model_dump_json())

        assert len(sent_messages) == 2
        for msg_json in sent_messages:
            assert '"type":"state_update"' in msg_json or '"type": "state_update"' in msg_json

    @pytest.mark.anyio
    async def test_broadcast_removes_disconnected_observers(self) -> None:
        """Disconnected observers should be removed from the list."""
        nm = _make_network_manager()

        class GoodWebSocket:
            """Fake WebSocket that works."""

            sent: list[str] = []

            async def send_text(self, msg: str) -> None:
                self.sent.append(msg)

        class BadWebSocket:
            """Fake WebSocket that raises on send."""

            async def send_text(self, msg: str) -> None:
                raise ConnectionError("Gone")  # noqa: EM101

        good = GoodWebSocket()
        bad = BadWebSocket()
        nm.observers = [good, bad]  # type: ignore[list-item]

        await nm.broadcast_to_observers('{"type":"state_update"}')

        assert len(nm.observers) == 1
        assert nm.observers[0] is good  # type: ignore[comparison-overlap]
