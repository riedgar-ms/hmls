"""Unit tests for NetworkManager event handlers and edge cases."""

from __future__ import annotations

import json
from typing import Any

import pytest

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.core.visibility import build_player_view
from hmls.server.event_bus import EventBus
from hmls.server.event_types import (
    GameOverEvent,
    GameStartedEvent,
    StateUpdatedEvent,
    YourTurnEvent,
)
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


def _make_network_manager() -> tuple[NetworkManager, EventBus]:
    """Create a NetworkManager with its EventBus for testing."""
    game_map = _make_simple_map()
    tanks = _make_tanks()
    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }
    nm = NetworkManager(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        patch_size=5,
        max_turns=10,
    )
    return nm, event_bus


class _RecordingWebSocket:
    """Fake WebSocket that records all sent messages."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, msg: str) -> None:
        """Record the message as parsed JSON."""
        self.sent.append(json.loads(msg))


class _FailingWebSocket:
    """Fake WebSocket that raises on send."""

    async def send_text(self, msg: str) -> None:
        """Simulate a broken connection."""
        raise ConnectionError("Connection lost")  # noqa: EM101


class TestOnGameStarted:
    """Tests for the _on_game_started event handler."""

    @pytest.mark.anyio
    async def test_assign_messages_sent_to_both_players(self) -> None:
        """Both players should receive AssignMessage with correct team info."""
        nm, event_bus = _make_network_manager()
        ws_a = _RecordingWebSocket()
        ws_b = _RecordingWebSocket()
        nm.websockets["A"] = ws_a  # type: ignore[assignment]
        nm.websockets["B"] = ws_b  # type: ignore[assignment]

        event = GameStartedEvent(
            game_map=_make_simple_map(),
            tanks=_make_tanks(),
            player_names={"A": "Alice", "B": "Bob"},
            patch_size=5,
            max_turns=10,
        )
        await event_bus.emit(event)

        # Player A should get assign for team A.
        assert len(ws_a.sent) == 1
        assert ws_a.sent[0]["type"] == "assign"
        assert ws_a.sent[0]["team"] == "A"

        # Player B should get assign for team B.
        assert len(ws_b.sent) == 1
        assert ws_b.sent[0]["type"] == "assign"
        assert ws_b.sent[0]["team"] == "B"

    @pytest.mark.anyio
    async def test_observers_receive_game_start(self) -> None:
        """Observers should receive GameStartMessage on game started."""
        nm, event_bus = _make_network_manager()
        ws_a = _RecordingWebSocket()
        ws_b = _RecordingWebSocket()
        obs = _RecordingWebSocket()
        nm.websockets["A"] = ws_a  # type: ignore[assignment]
        nm.websockets["B"] = ws_b  # type: ignore[assignment]
        nm.observers = [obs]  # type: ignore[list-item]

        event = GameStartedEvent(
            game_map=_make_simple_map(),
            tanks=_make_tanks(),
            player_names={"A": "Alice", "B": "Bob"},
            patch_size=5,
            max_turns=10,
        )
        await event_bus.emit(event)

        assert len(obs.sent) == 1
        assert obs.sent[0]["type"] == "game_start"


class TestOnYourTurn:
    """Tests for the _on_your_turn event handler."""

    @pytest.mark.anyio
    async def test_your_turn_sent_to_acting_player(self) -> None:
        """Only the acting player should receive YourTurnMessage."""
        nm, event_bus = _make_network_manager()
        ws_a = _RecordingWebSocket()
        ws_b = _RecordingWebSocket()
        nm.websockets["A"] = ws_a  # type: ignore[assignment]
        nm.websockets["B"] = ws_b  # type: ignore[assignment]

        tanks = _make_tanks()
        state = GameState(tanks=tanks, current_tank_id="A1")
        view = build_player_view(state, _make_simple_map(), "A", patch_size=5)

        event = YourTurnEvent(tank_id="A1", team="A", view=view)
        await event_bus.emit(event)

        assert len(ws_a.sent) == 1
        assert ws_a.sent[0]["type"] == "your_turn"
        assert ws_a.sent[0]["tank_id"] == "A1"
        assert len(ws_b.sent) == 0

    @pytest.mark.anyio
    async def test_your_turn_no_crash_when_player_missing(self) -> None:
        """Should not crash when the acting player's WebSocket isn't registered."""
        nm, event_bus = _make_network_manager()
        # Don't register any WebSockets.

        tanks = _make_tanks()
        state = GameState(tanks=tanks, current_tank_id="A1")
        view = build_player_view(state, _make_simple_map(), "A", patch_size=5)

        event = YourTurnEvent(tank_id="A1", team="A", view=view)
        # Should not raise.
        await event_bus.emit(event)


class TestOnGameOver:
    """Tests for the _on_game_over event handler."""

    @pytest.mark.anyio
    async def test_game_over_sent_to_all(self) -> None:
        """GameOverMessage should be sent to both players and all observers."""
        nm, event_bus = _make_network_manager()
        ws_a = _RecordingWebSocket()
        ws_b = _RecordingWebSocket()
        obs = _RecordingWebSocket()
        nm.websockets["A"] = ws_a  # type: ignore[assignment]
        nm.websockets["B"] = ws_b  # type: ignore[assignment]
        nm.observers = [obs]  # type: ignore[list-item]

        event = GameOverEvent(winner="A", reason="Team A wins!")
        await event_bus.emit(event)

        for ws in [ws_a, ws_b, obs]:
            assert len(ws.sent) == 1
            assert ws.sent[0]["type"] == "game_over"
            assert ws.sent[0]["winner"] == "A"
            assert ws.sent[0]["reason"] == "Team A wins!"

    @pytest.mark.anyio
    async def test_game_over_sets_flag(self) -> None:
        """game_over flag should be set after event."""
        nm, event_bus = _make_network_manager()
        nm.websockets["A"] = _RecordingWebSocket()  # type: ignore[assignment]
        nm.websockets["B"] = _RecordingWebSocket()  # type: ignore[assignment]

        assert not nm.game_over
        await event_bus.emit(GameOverEvent(winner=None, reason="Draw"))
        assert nm.game_over


class TestOnStateUpdated:
    """Tests for the _on_state_updated event handler."""

    @pytest.mark.anyio
    async def test_state_cached_and_broadcast(self) -> None:
        """State should be cached and broadcast to observers."""
        nm, event_bus = _make_network_manager()
        obs = _RecordingWebSocket()
        nm.observers = [obs]  # type: ignore[list-item]

        tanks = _make_tanks()
        state = GameState(tanks=tanks, current_tank_id="A1")
        event = StateUpdatedEvent(state=state, current_tank_id="A1", turns_taken=1)
        await event_bus.emit(event)

        # Cached.
        assert nm._last_state is event

        # Broadcast to observer.
        assert len(obs.sent) == 1
        assert obs.sent[0]["type"] == "state_update"
        assert obs.sent[0]["turns_taken"] == 1


class TestSendToPlayer:
    """Tests for send_to_player edge cases."""

    @pytest.mark.anyio
    async def test_send_to_missing_player_returns_false(self) -> None:
        """Sending to a team with no WebSocket should return False."""
        nm, _ = _make_network_manager()
        result = await nm.send_to_player("A", '{"type":"test"}')
        assert result is False

    @pytest.mark.anyio
    async def test_send_to_failing_player_returns_false(self) -> None:
        """Sending to a player whose WebSocket raises should return False."""
        nm, _ = _make_network_manager()
        nm.websockets["A"] = _FailingWebSocket()  # type: ignore[assignment]
        result = await nm.send_to_player("A", '{"type":"test"}')
        assert result is False

    @pytest.mark.anyio
    async def test_send_to_connected_player_returns_true(self) -> None:
        """Sending to a connected player should return True."""
        nm, _ = _make_network_manager()
        nm.websockets["A"] = _RecordingWebSocket()  # type: ignore[assignment]
        result = await nm.send_to_player("A", '{"type":"test"}')
        assert result is True
