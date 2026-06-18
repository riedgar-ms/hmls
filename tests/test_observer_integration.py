"""End-to-end observer integration tests.

These tests exercise the observer WebSocket protocol during a full game:
connect as observer → receive game_start → state_update → turn_result → game_over.

Uses FastAPI's TestClient WebSocket support with threading to simulate
two concurrent player connections alongside an observer.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi.testclient import TestClient

from hmls.core.map import GameMap
from hmls.core.placement import place_tanks
from hmls.core.tank import Tank
from hmls.server.app import create_fastapi_app
from hmls.server.event_bus import EventBus
from hmls.server.network_manager import NetworkManager
from hmls.server.orchestrator import GameOrchestrator
from hmls.server.remote_player import RemotePlayer

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_game_components(
    *,
    map_width: int = 5,
    map_height: int = 5,
    tanks_per_player: int = 1,
    max_turns: int = 50,
    patch_size: int = 7,
    seed: int = 42,
) -> tuple[
    GameMap, list[Tank], dict[str, RemotePlayer], EventBus, NetworkManager, GameOrchestrator
]:
    """Wire up all game server components for testing.

    Returns:
        Tuple of (game_map, tanks, players, event_bus, network_manager, orchestrator).
    """
    game_map = GameMap(width=map_width, height=map_height)
    tanks = place_tanks(game_map, tanks_per_player, seed=seed)

    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }

    network_manager = NetworkManager(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        patch_size=patch_size,
        max_turns=max_turns,
    )

    orchestrator = GameOrchestrator(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        max_turns=max_turns,
        patch_size=patch_size,
    )

    orchestrator.both_connected = network_manager.both_connected
    orchestrator.player_names = network_manager.player_names

    return game_map, tanks, players, event_bus, network_manager, orchestrator


def _create_test_app(
    network_manager: NetworkManager,
    orchestrator: GameOrchestrator,
) -> TestClient:
    """Create a TestClient with the game app and orchestrator wired up."""
    app = create_fastapi_app(network_manager, orchestrator=orchestrator)
    return TestClient(app)


# ── Helper: threaded player ───────────────────────────────────────────


def _run_player_in_thread(
    client: TestClient,
    player_name: str,
    strategy: str = "pass",
    max_actions: int = 200,
) -> dict[str, Any]:
    """Run a player connection in a separate thread.

    Returns:
        Dict with keys: "messages", "team", "game_over_msg".
    """
    result: dict[str, Any] = {"messages": [], "team": None, "game_over_msg": None}

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "join", "player_name": player_name})

        actions_taken = 0
        while actions_taken < max_actions:
            try:
                data = ws.receive_json()
            except Exception:  # noqa: BLE001
                break

            result["messages"].append(data)

            if data["type"] == "waiting":
                continue
            elif data["type"] == "assign":
                result["team"] = data["team"]
            elif data["type"] == "your_turn":
                ws.send_json({"type": "action", "action": strategy})
                actions_taken += 1
            elif data["type"] == "turn_result":
                continue
            elif data["type"] == "game_over":
                result["game_over_msg"] = data
                break
            elif data["type"] == "error":
                break

    return result


# ── Helper: threaded observer ─────────────────────────────────────────


def _run_observer_in_thread(
    client: TestClient,
    observer_name: str = "TestObserver",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run an observer connection in a separate thread.

    Collects all messages until game_over or timeout.

    Returns:
        Dict with keys: "messages" (all received), "game_start_msg",
        "state_updates", "turn_results", "game_over_msg".
    """
    result: dict[str, Any] = {
        "messages": [],
        "game_start_msg": None,
        "state_updates": [],
        "turn_results": [],
        "game_over_msg": None,
    }

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "observe", "observer_name": observer_name})

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data = ws.receive_json()
            except Exception:  # noqa: BLE001
                break

            result["messages"].append(data)

            if data["type"] == "game_start":
                result["game_start_msg"] = data
            elif data["type"] == "state_update":
                result["state_updates"].append(data)
            elif data["type"] == "turn_result":
                result["turn_results"].append(data)
            elif data["type"] == "game_over":
                result["game_over_msg"] = data
                break

    return result


# ── Tests ─────────────────────────────────────────────────────────────


class TestObserverIntegration:
    """End-to-end tests for the observer during a full game."""

    def test_observer_receives_full_game_lifecycle(self) -> None:
        """Observer receives game_start, state_update(s), turn_result(s), game_over."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=10)
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}
            player_results: list[dict[str, Any]] = [{}] * 2

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            def run_player(idx: int, name: str) -> None:
                player_results[idx] = _run_player_in_thread(client, name, strategy="pass")

            t_obs = threading.Thread(target=run_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        # Observer should have received game_start.
        assert observer_result["game_start_msg"] is not None
        # Observer should have received at least one state_update.
        assert len(observer_result["state_updates"]) > 0
        # Observer should have received at least one turn_result.
        assert len(observer_result["turn_results"]) > 0
        # Observer should have received game_over.
        assert observer_result["game_over_msg"] is not None

        # Message ordering: game_start first, game_over last.
        msg_types = [m["type"] for m in observer_result["messages"]]
        assert msg_types[0] == "game_start"
        assert msg_types[-1] == "game_over"

    def test_observer_game_start_fields(self) -> None:
        """Observer's game_start message contains correct map and game config."""
        _, _, _, _, nm, orch = _make_game_components(
            map_width=5, map_height=5, tanks_per_player=1, max_turns=6, patch_size=7
        )
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            def run_player(idx: int, name: str) -> None:
                _run_player_in_thread(client, name, strategy="pass")

            t_obs = threading.Thread(target=run_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        gs = observer_result["game_start_msg"]
        assert gs is not None
        assert gs["game_map"]["width"] == 5
        assert gs["game_map"]["height"] == 5
        assert gs["patch_size"] == 7
        assert gs["max_turns"] == 6
        assert len(gs["tanks"]) == 2
        # Player names should be present (Alice and Bob assigned to teams).
        assert "Alice" in gs["player_names"].values()
        assert "Bob" in gs["player_names"].values()

    def test_observer_state_update_content(self) -> None:
        """State updates contain incrementing turns_taken and tank state."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=8)
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            def run_player(idx: int, name: str) -> None:
                _run_player_in_thread(client, name, strategy="pass")

            t_obs = threading.Thread(target=run_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        state_updates = observer_result["state_updates"]
        assert len(state_updates) > 0

        # Each state_update should have required fields.
        for su in state_updates:
            assert "state" in su
            assert "turns_taken" in su
            assert "current_tank_id" in su
            assert "tanks" in su["state"]
            assert len(su["state"]["tanks"]) == 2

        # turns_taken should be non-decreasing.
        turns = [su["turns_taken"] for su in state_updates]
        assert turns == sorted(turns)

    def test_observer_turn_result_content(self) -> None:
        """Turn results contain tank_id, action, and valid flag."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=8)
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            def run_player(idx: int, name: str) -> None:
                _run_player_in_thread(client, name, strategy="pass")

            t_obs = threading.Thread(target=run_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        turn_results = observer_result["turn_results"]
        assert len(turn_results) > 0

        for tr in turn_results:
            assert "tank_id" in tr
            assert "action" in tr
            assert "valid" in tr
            assert tr["tank_id"] in ("A1", "B1")
            assert tr["action"] == "pass"
            assert tr["valid"] is True

    def test_observer_game_over_matches_players(self) -> None:
        """Observer's game_over has same winner as both players."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=20)
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}
            player_results: list[dict[str, Any]] = [{}] * 2

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            def run_player(idx: int, name: str) -> None:
                player_results[idx] = _run_player_in_thread(client, name, strategy="fire")

            t_obs = threading.Thread(target=run_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        obs_go = observer_result["game_over_msg"]
        assert obs_go is not None

        p0_go = player_results[0]["game_over_msg"]
        p1_go = player_results[1]["game_over_msg"]
        assert p0_go is not None
        assert p1_go is not None

        # All three should agree on the winner.
        assert obs_go["winner"] == p0_go["winner"]
        assert obs_go["winner"] == p1_go["winner"]

    def test_observer_mid_game_connect_receives_catchup(self) -> None:
        """Observer connecting mid-game gets game_start + state_update catchup."""
        # Use a long game so it's still running when the observer connects.
        _, _, _, _, nm, orch = _make_game_components(max_turns=100)
        client = _create_test_app(nm, orch)

        with client:
            observer_result: dict[str, Any] = {}
            player_results: list[dict[str, Any]] = [{}] * 2

            def run_player(idx: int, name: str) -> None:
                player_results[idx] = _run_player_in_thread(client, name, strategy="pass")

            # Start players first.
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))
            t_p1.start()
            t_p2.start()

            # Wait for the game to be underway before connecting observer.
            # Poll _game_started which is set in the GameStartedEvent handler.
            deadline = time.monotonic() + 10
            while not nm._game_started and time.monotonic() < deadline:
                time.sleep(0.05)
            assert nm._game_started, "Game did not start in time"

            def run_observer() -> None:
                nonlocal observer_result
                observer_result = _run_observer_in_thread(client)

            t_obs = threading.Thread(target=run_observer)
            t_obs.start()

            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            t_obs.join(timeout=30)
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"
            assert not t_obs.is_alive(), "Observer thread did not finish in time"

        # Observer should still get game_start (sent on connect for late joiners).
        assert observer_result["game_start_msg"] is not None
        # Observer should get game_over eventually.
        assert observer_result["game_over_msg"] is not None
        # Player names should be populated in the game_start (since game already started).
        gs = observer_result["game_start_msg"]
        assert "Alice" in gs["player_names"].values()
        assert "Bob" in gs["player_names"].values()

    def test_observer_disconnect_does_not_disrupt_game(self) -> None:
        """Disconnecting an observer mid-game doesn't break the player game."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=20)
        client = _create_test_app(nm, orch)

        with client:
            player_results: list[dict[str, Any]] = [{}] * 2

            def disconnecting_observer() -> None:
                """Connect as observer, receive a few messages, then disconnect."""
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"type": "observe", "observer_name": "EarlyLeaver"})
                    # Read game_start then disconnect.
                    try:
                        ws.receive_json()
                    except Exception:  # noqa: BLE001
                        pass

            def run_player(idx: int, name: str) -> None:
                player_results[idx] = _run_player_in_thread(client, name, strategy="pass")

            t_obs = threading.Thread(target=disconnecting_observer)
            t_p1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t_p2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t_obs.start()
            t_p1.start()
            t_p2.start()

            t_obs.join(timeout=30)
            t_p1.join(timeout=30)
            t_p2.join(timeout=30)
            assert not t_obs.is_alive(), "Observer thread did not finish in time"
            assert not t_p1.is_alive(), "Player 1 thread did not finish in time"
            assert not t_p2.is_alive(), "Player 2 thread did not finish in time"

        # Both players should still complete the game.
        assert player_results[0]["game_over_msg"] is not None
        assert player_results[1]["game_over_msg"] is not None
        assert player_results[0]["game_over_msg"]["type"] == "game_over"
        assert player_results[1]["game_over_msg"]["type"] == "game_over"
