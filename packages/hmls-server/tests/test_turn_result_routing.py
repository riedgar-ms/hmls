"""Tests that turn_result messages are only sent to the acting player's team.

Verifies the fix for the information-leak bug where both players received
every turn_result, revealing the opponent's moves.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.server.app import GameSession


def _make_simple_map() -> GameMap:
    """Create a small 5×5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_tanks() -> list[Tank]:
    """Create a minimal set of tanks for two teams."""
    return [
        Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
        Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
    ]


def _make_session(max_turns: int = 10) -> GameSession:
    """Create a GameSession for testing."""
    game_map = _make_simple_map()
    tanks = _make_tanks()
    return GameSession(
        game_map=game_map,
        tanks=tanks,
        max_turns=max_turns,
        patch_size=5,
    )


class _RecordingWebSocket:
    """Fake WebSocket that records all sent messages."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, msg: str) -> None:
        """Record the message as parsed JSON."""
        self.sent.append(json.loads(msg))


class TestTurnResultRouting:
    """Tests that turn_result messages respect fog-of-war."""

    @pytest.mark.anyio
    async def test_turn_result_sent_only_to_acting_team(self) -> None:
        """After a turn, only the acting team should receive turn_result."""
        session = _make_session(max_turns=4)

        ws_a = _RecordingWebSocket()
        ws_b = _RecordingWebSocket()
        obs = _RecordingWebSocket()

        session.websockets["A"] = ws_a  # type: ignore[assignment]
        session.websockets["B"] = ws_b  # type: ignore[assignment]
        session._observers = [obs]  # type: ignore[list-item]

        # Simulate the game loop having started: set up the engine.
        session._both_connected.set()
        session.player_names = {"A": "Alice", "B": "Bob"}

        # Run the game loop in the background; it will wait for actions.
        game_task = asyncio.create_task(session.run_game())

        # Allow time for the game loop to initialise and send your_turn.
        await asyncio.sleep(0.1)

        # Turn 1: Team A acts (engine alternates A, B, A, B, ...).
        # The game loop should have sent your_turn to A.
        # Submit an action for team A.
        session.players["A"].submit_action(Action.PASS)
        await asyncio.sleep(0.1)

        # Check: A should have received turn_result, B should not.
        a_turn_results = [m for m in ws_a.sent if m.get("type") == "turn_result"]
        b_turn_results = [m for m in ws_b.sent if m.get("type") == "turn_result"]
        assert len(a_turn_results) == 1, f"Expected 1 turn_result for A, got {len(a_turn_results)}"
        assert len(b_turn_results) == 0, (
            f"Expected 0 turn_results for B, got {len(b_turn_results)}: {b_turn_results}"
        )
        assert a_turn_results[0]["tank_id"] == "A1"

        # Turn 2: Team B acts.
        session.players["B"].submit_action(Action.PASS)
        await asyncio.sleep(0.1)

        # Check: B should now have 1 turn_result; A should still have 1.
        a_turn_results = [m for m in ws_a.sent if m.get("type") == "turn_result"]
        b_turn_results = [m for m in ws_b.sent if m.get("type") == "turn_result"]
        assert len(a_turn_results) == 1, (
            f"Expected 1 turn_result for A after B's turn, got {len(a_turn_results)}"
        )
        assert len(b_turn_results) == 1, f"Expected 1 turn_result for B, got {len(b_turn_results)}"
        assert b_turn_results[0]["tank_id"] == "B1"

        # Observers should have received ALL turn_results.
        obs_turn_results = [m for m in obs.sent if m.get("type") == "turn_result"]
        assert len(obs_turn_results) == 2, (
            f"Observer should have 2 turn_results, got {len(obs_turn_results)}"
        )

        # Clean up: end the game.
        session._game_over = True
        # Submit remaining actions to unblock any pending waits.
        try:
            session.players["A"].submit_action(Action.PASS)
        except RuntimeError:
            pass
        try:
            session.players["B"].submit_action(Action.PASS)
        except RuntimeError:
            pass
        # Cancel the task if still running.
        game_task.cancel()
        try:
            await game_task
        except asyncio.CancelledError:
            pass
