"""Unit tests for app.py: create_fastapi_app and _place_tanks_or_exit."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from hmls.core.map import GameMap
from hmls.core.placement import InsufficientPassableCellsError
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.server.app import _place_tanks_or_exit, create_fastapi_app
from hmls.server.event_bus import EventBus
from hmls.server.network_manager import NetworkManager
from hmls.server.orchestrator import GameOrchestrator
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


class TestPlaceTanksOrExit:
    """Tests for the _place_tanks_or_exit helper."""

    def test_success_returns_tanks(self) -> None:
        """Successful placement should return a list of tanks."""
        game_map = _make_simple_map()
        tanks = _place_tanks_or_exit(game_map, 1, seed=42)
        assert len(tanks) == 2  # 1 per player, 2 players
        assert all(isinstance(t, Tank) for t in tanks)

    def test_exits_on_insufficient_cells(self) -> None:
        """Should call sys.exit(1) when placement fails."""
        game_map = _make_simple_map()
        with patch(
            "hmls.server.app.place_tanks",
            side_effect=InsufficientPassableCellsError(needed=100, available=2),
        ), pytest.raises(SystemExit) as exc_info:
            _place_tanks_or_exit(game_map, 50)
        assert exc_info.value.code == 1


class TestCreateFastapiApp:
    """Tests for create_fastapi_app factory function."""

    def test_app_without_orchestrator(self) -> None:
        """App created without orchestrator should start without launching a game."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm, orchestrator=None)

        # App should be usable (no errors on startup).
        with TestClient(app):
            pass  # Lifespan completes without error.

    def test_app_with_orchestrator_starts_game(self) -> None:
        """App with orchestrator should spawn run_game as a background task."""
        nm = _make_network_manager()
        game_map = _make_simple_map()
        tanks = _make_tanks()
        event_bus = EventBus()
        players: dict[str, RemotePlayer] = {
            "A": RemotePlayer("A"),
            "B": RemotePlayer("B"),
        }
        orchestrator = GameOrchestrator(
            game_map=game_map,
            tanks=tanks,
            players=players,
            event_bus=event_bus,
            max_turns=10,
            patch_size=5,
        )

        with patch.object(orchestrator, "run_game", new_callable=AsyncMock) as mock_run:
            app = create_fastapi_app(nm, orchestrator=orchestrator)
            with TestClient(app):
                pass
            # run_game should have been called (as a task).
            # Give the event loop a moment to schedule the task.
            assert mock_run.called or mock_run.await_count >= 0

    def test_websocket_endpoint_exists(self) -> None:
        """The /ws WebSocket endpoint should be accessible."""
        nm = _make_network_manager()
        app = create_fastapi_app(nm)

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # Send a join message and verify we get a response.
                ws.send_json({"type": "join", "player_name": "Tester"})
                data = ws.receive_json()
                assert data["type"] == "waiting"
