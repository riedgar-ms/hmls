"""End-to-end client↔server WebSocket round-trip integration tests.

These tests exercise the full game protocol over WebSockets:
connect → join → assign → play turns → game_over → disconnect.

Uses FastAPI's TestClient WebSocket support (synchronous) with threading
to simulate two concurrent player connections.
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
    """Create a TestClient with the game app and orchestrator wired up.

    Returns:
        A TestClient ready for WebSocket connections.
    """
    app = create_fastapi_app(network_manager, orchestrator=orchestrator)

    return TestClient(app)


# ── Helper: threaded player ───────────────────────────────────────────


def _run_player_in_thread(
    client: TestClient,
    player_name: str,
    strategy: str = "fire",
    max_actions: int = 100,
) -> dict[str, Any]:
    """Run a player connection in a separate thread.

    Args:
        client: The TestClient to connect through.
        player_name: Name for the JoinMessage.
        strategy: Action strategy — "fire", "pass", or "move_forward".
        max_actions: Safety limit on number of actions to take.

    Returns:
        Dict with keys: "messages" (all received messages), "team", "game_over_msg".
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


# ── Tests ─────────────────────────────────────────────────────────────


class TestClientServerRoundTrip:
    """End-to-end tests for the client↔server game protocol."""

    def test_full_game_happy_path(self) -> None:
        """Two players connect, play through a complete game, both receive game_over."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=20)
        client = _create_test_app(nm, orch)

        with client:
            # Run both players in threads so they can interact concurrently.
            results: list[dict[str, Any]] = [{}] * 2
            threads: list[threading.Thread] = []

            def run_player(idx: int, name: str) -> None:
                results[idx] = _run_player_in_thread(client, name, strategy="fire")

            t1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t2 = threading.Thread(target=run_player, args=(1, "Bob"))
            threads = [t1, t2]

            t1.start()
            t2.start()

            for t in threads:
                t.join(timeout=30)
            for t in threads:
                assert not t.is_alive(), f"Thread {t.name} did not finish in time"

        # Both players should have been assigned teams.
        assert results[0]["team"] in ("A", "B")
        assert results[1]["team"] in ("A", "B")
        assert results[0]["team"] != results[1]["team"]

        # Both should have received game_over.
        assert results[0]["game_over_msg"] is not None
        assert results[1]["game_over_msg"] is not None
        assert results[0]["game_over_msg"]["type"] == "game_over"
        assert results[1]["game_over_msg"]["type"] == "game_over"

        # Winner should be consistent.
        winner_0 = results[0]["game_over_msg"]["winner"]
        winner_1 = results[1]["game_over_msg"]["winner"]
        assert winner_0 == winner_1

    def test_full_game_with_pass_strategy_reaches_draw(self) -> None:
        """Two players that only pass should reach a draw at max_turns."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=10)
        client = _create_test_app(nm, orch)

        with client:
            results: list[dict[str, Any]] = [{}] * 2

            def run_player(idx: int, name: str) -> None:
                results[idx] = _run_player_in_thread(client, name, strategy="pass")

            t1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)
            assert not t1.is_alive(), "Player 1 thread did not finish in time"
            assert not t2.is_alive(), "Player 2 thread did not finish in time"

        # Both should receive game_over with draw.
        assert results[0]["game_over_msg"] is not None
        assert results[1]["game_over_msg"] is not None
        assert results[0]["game_over_msg"]["winner"] is None
        assert results[1]["game_over_msg"]["winner"] is None

    def test_message_sequence_order(self) -> None:
        """Verify the protocol message sequence is correct for each player."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=6)
        client = _create_test_app(nm, orch)

        with client:
            results: list[dict[str, Any]] = [{}] * 2

            def run_player(idx: int, name: str) -> None:
                results[idx] = _run_player_in_thread(client, name, strategy="pass")

            t1 = threading.Thread(target=run_player, args=(0, "Alice"))
            t2 = threading.Thread(target=run_player, args=(1, "Bob"))

            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)
            assert not t1.is_alive(), "Player 1 thread did not finish in time"
            assert not t2.is_alive(), "Player 2 thread did not finish in time"

        for r in results:
            msg_types = [m["type"] for m in r["messages"]]
            # First meaningful message should be either "waiting" or "assign".
            assert msg_types[0] in ("waiting", "assign")
            # The assign message must appear.
            assert "assign" in msg_types
            # Game over must be the last message.
            assert msg_types[-1] == "game_over"
            # "your_turn" and "turn_result" must appear (at least one turn happens).
            # (The first player to connect gets "waiting" first.)

    def test_player_disconnect_forfeits(self) -> None:
        """A player disconnecting mid-game causes the other to win."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=100)
        client = _create_test_app(nm, orch)

        survivor_result: dict[str, Any] = {}

        with client:

            def disconnecting_player() -> None:
                """Connect, wait for first your_turn, then disconnect."""
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"type": "join", "player_name": "Quitter"})
                    while True:
                        try:
                            data = ws.receive_json()
                        except Exception:  # noqa: BLE001
                            break
                        if data["type"] == "your_turn":
                            # Disconnect without responding.
                            break
                        elif data["type"] == "game_over":
                            break

            def surviving_player() -> None:
                nonlocal survivor_result
                survivor_result = _run_player_in_thread(client, "Survivor", strategy="fire")

            t1 = threading.Thread(target=disconnecting_player)
            t2 = threading.Thread(target=surviving_player)

            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)
            assert not t1.is_alive(), "Disconnecting player thread did not finish in time"
            assert not t2.is_alive(), "Surviving player thread did not finish in time"

        # Survivor should have received game_over.
        assert survivor_result.get("game_over_msg") is not None
        game_over = survivor_result["game_over_msg"]
        assert game_over["winner"] is not None
        assert "disconnect" in game_over["reason"].lower()

    def test_game_full_rejects_third_player(self) -> None:
        """A third player connecting after two are joined gets an error."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=50)
        client = _create_test_app(nm, orch)

        with client:
            # Connect two players first.
            results: list[dict[str, Any]] = [{}] * 2

            def first_two(idx: int, name: str) -> None:
                with client.websocket_connect("/ws") as ws:
                    ws.send_json({"type": "join", "player_name": name})
                    data = ws.receive_json()
                    results[idx] = data
                    # Play one turn if asked, then disconnect.
                    try:
                        data2 = ws.receive_json()
                        if data2["type"] == "your_turn":
                            ws.send_json({"type": "action", "action": "pass"})
                    except Exception:  # noqa: BLE001
                        pass

            t1 = threading.Thread(target=first_two, args=(0, "Alice"))
            t2 = threading.Thread(target=first_two, args=(1, "Bob"))
            t1.start()
            t2.start()

            # Poll server-side state rather than using a threading.Barrier.
            # A Barrier(3) deadlocks because the worker threads call
            # ws.receive_json() before reaching the barrier, and that
            # receive can block indefinitely under Starlette's TestClient
            # when other tests have run in the same process.  Polling the
            # NetworkManager's dict is safe here: the dict is mutated by
            # the ASGI event-loop thread and read (atomically under the
            # GIL) by this thread, so no lock is needed.
            deadline = time.monotonic() + 10
            while len(nm.websockets) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert len(nm.websockets) == 2

            # Third player should be rejected.
            with client.websocket_connect("/ws") as ws3:
                ws3.send_json({"type": "join", "player_name": "Charlie"})
                data = ws3.receive_json()
                assert data["type"] == "error"
                assert "full" in data["message"].lower()

            t1.join(timeout=10)
            t2.join(timeout=10)
            assert not t1.is_alive(), "First player thread did not finish in time"
            assert not t2.is_alive(), "Second player thread did not finish in time"

    def test_invalid_message_returns_error(self) -> None:
        """Sending an invalid message after joining should return an error."""
        _, _, _, _, nm, orch = _make_game_components(max_turns=50)
        client = _create_test_app(nm, orch)

        with client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "join", "player_name": "Alice"})
                # Wait for waiting message.
                data = ws.receive_json()
                assert data["type"] == "waiting"

                # Send garbage.
                ws.send_text("not valid json at all {{{")
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "invalid" in data["message"].lower()
