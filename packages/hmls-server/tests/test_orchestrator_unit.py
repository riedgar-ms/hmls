"""Unit tests for the GameOrchestrator: edge cases and error paths."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.server.event_bus import EventBus
from hmls.server.event_types import (
    GameOverEvent,
    PlayerDisconnectedEvent,
)
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


def _make_orchestrator(
    max_turns: int = 10,
    history_file: Path | None = None,
) -> tuple[GameOrchestrator, EventBus, dict[str, RemotePlayer]]:
    """Create an orchestrator with its dependencies for testing."""
    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }
    orchestrator = GameOrchestrator(
        game_map=_make_simple_map(),
        tanks=_make_tanks(),
        players=players,
        event_bus=event_bus,
        max_turns=max_turns,
        patch_size=5,
        history_file=history_file,
    )
    orchestrator.player_names = {"A": "Alice", "B": "Bob"}
    return orchestrator, event_bus, players


class _RecordingWebSocket:
    """Fake WebSocket that records all sent messages."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, msg: str) -> None:
        """Record the message as parsed JSON."""
        self.sent.append(json.loads(msg))


class TestOrchestratorDisconnect:
    """Tests for player disconnection handling."""

    @pytest.mark.anyio
    async def test_disconnect_mid_game_emits_game_over(self) -> None:
        """Disconnecting mid-game should emit GameOverEvent for the other team."""
        orchestrator, event_bus, _ = _make_orchestrator()
        received_events: list[GameOverEvent] = []

        async def capture_game_over(event: GameOverEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(GameOverEvent, capture_game_over)

        # Simulate player A disconnecting.
        await event_bus.emit(PlayerDisconnectedEvent(team="A"))

        assert len(received_events) == 1
        assert received_events[0].winner == "B"
        assert "disconnected" in received_events[0].reason.lower()

    @pytest.mark.anyio
    async def test_disconnect_after_game_over_no_duplicate(self) -> None:
        """Disconnecting after game is already over should not emit a second GameOverEvent."""
        orchestrator, event_bus, _ = _make_orchestrator()
        received_events: list[GameOverEvent] = []

        async def capture_game_over(event: GameOverEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(GameOverEvent, capture_game_over)

        # Mark game as already over.
        orchestrator.game_over = True

        # Simulate player disconnect — should be ignored.
        await event_bus.emit(PlayerDisconnectedEvent(team="B"))

        assert len(received_events) == 0


class TestOrchestratorRunGame:
    """Tests for the run_game loop."""

    @pytest.mark.anyio
    async def test_draw_on_turn_limit(self) -> None:
        """Game should end in a draw when max_turns is reached."""
        orchestrator, event_bus, players = _make_orchestrator(max_turns=2)
        received_events: list[GameOverEvent] = []

        async def capture_game_over(event: GameOverEvent) -> None:
            received_events.append(event)

        event_bus.subscribe(GameOverEvent, capture_game_over)

        # Signal both connected.
        orchestrator.both_connected.set()

        # Auto-submit PASS for every action request.
        async def auto_submit() -> None:
            while not orchestrator.game_over:
                for player in players.values():
                    try:
                        player.submit_action(Action.PASS)
                    except RuntimeError:
                        pass
                await asyncio.sleep(0.05)

        submit_task = asyncio.create_task(auto_submit())
        await orchestrator.run_game()
        submit_task.cancel()
        try:
            await submit_task
        except asyncio.CancelledError:
            pass

        assert orchestrator.game_over
        assert len(received_events) == 1
        assert received_events[0].winner is None
        assert "draw" in received_events[0].reason.lower()

    @pytest.mark.anyio
    async def test_timeout_ends_game(self) -> None:
        """Player timeout should end the game."""
        orchestrator, event_bus, players = _make_orchestrator(max_turns=10)

        # Signal both connected.
        orchestrator.both_connected.set()

        # Patch wait_for to raise TimeoutError immediately.
        with patch("asyncio.wait_for", side_effect=TimeoutError):
            await orchestrator.run_game()

        assert orchestrator.game_over

    @pytest.mark.anyio
    async def test_runtime_error_during_wait_ends_game(self) -> None:
        """RuntimeError from wait_for_action should end the game."""
        orchestrator, event_bus, players = _make_orchestrator(max_turns=10)

        # Signal both connected.
        orchestrator.both_connected.set()

        # Patch wait_for to raise RuntimeError.
        with patch("asyncio.wait_for", side_effect=RuntimeError("action error")):
            await orchestrator.run_game()

        assert orchestrator.game_over


class TestOrchestratorSaveHistory:
    """Tests for history saving edge cases."""

    def test_save_history_noop_when_no_engine(self) -> None:
        """_save_history should do nothing when engine is None."""
        orchestrator, _, _ = _make_orchestrator(history_file=Path("/tmp/should_not_exist.json"))
        orchestrator.engine = None
        # Should not raise.
        orchestrator._save_history()

    def test_save_history_noop_when_no_file(self) -> None:
        """_save_history should do nothing when history_file is None."""
        orchestrator, _, _ = _make_orchestrator(history_file=None)
        # Should not raise.
        orchestrator._save_history()
