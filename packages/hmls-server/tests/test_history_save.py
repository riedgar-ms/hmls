"""Tests for game history saving and CLI argument parsing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hmls.core.engine import GameEngine, GameResult
from hmls.core.map import GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import build_player_view
from hmls.server.cli import parse_args
from hmls.server.events import EventBus
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


def _make_orchestrator(history_file: Path | None = None) -> GameOrchestrator:
    """Create a GameOrchestrator for testing."""
    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }
    return GameOrchestrator(
        game_map=_make_simple_map(),
        tanks=_make_tanks(),
        players=players,
        event_bus=event_bus,
        max_turns=4,
        patch_size=5,
        history_file=history_file,
    )


class TestCliHistoryArgs:
    """Tests for --history-file and --no-history CLI arguments."""

    def test_default_history_file(self) -> None:
        """Default history file should be history.json."""
        ns = parse_args(["map.json", "1"])
        assert ns.history_file == Path("history.json")

    def test_custom_history_file(self) -> None:
        """--history-file should override the default."""
        ns = parse_args(["map.json", "1", "--history-file", "out/game.json"])
        assert ns.history_file == Path("out/game.json")

    def test_no_history_flag(self) -> None:
        """--no-history should set history_file to None."""
        ns = parse_args(["map.json", "1", "--no-history"])
        assert ns.history_file is None

    def test_history_file_and_no_history_mutually_exclusive(self) -> None:
        """--history-file and --no-history cannot be used together."""
        with pytest.raises(SystemExit):
            parse_args(["map.json", "1", "--history-file", "out.json", "--no-history"])


class TestHistorySave:
    """Tests for automatic history saving after a game."""

    def test_history_saved_after_game(self, tmp_path: Path) -> None:
        """GameOrchestrator should write a valid GameResult JSON when game ends."""
        history_file = tmp_path / "history.json"
        orchestrator = _make_orchestrator(history_file=history_file)

        # Manually wire up the engine and simulate a short game.

        players: dict[str, Player] = {
            "A": orchestrator.players["A"],
            "B": orchestrator.players["B"],
        }
        orchestrator.engine = GameEngine(
            orchestrator.game_map,
            orchestrator.tanks,
            players,
            max_turns=4,
            patch_size=5,
        )

        loop = asyncio.new_event_loop()
        try:
            for _ in range(4):
                tank_id = orchestrator.engine.current_tank_id
                team = orchestrator.engine.current_team
                player = orchestrator.players[team]

                view = build_player_view(orchestrator.engine.state, orchestrator.game_map, team, 5)
                player.request_action(tank_id, view, loop)
                player.submit_action(Action.PASS)
                loop.run_until_complete(player.wait_for_action())
                orchestrator.engine.step()
        finally:
            loop.close()

        assert orchestrator.engine.game_over
        orchestrator.game_over = True

        # Call the save method directly.
        orchestrator._save_history()

        assert history_file.exists()
        data = json.loads(history_file.read_text(encoding="utf-8"))
        parsed = GameResult.model_validate(data)
        assert parsed.turns_played == 4

    def test_no_history_file_when_disabled(self, tmp_path: Path) -> None:
        """No file should be written when history_file is None."""
        orchestrator = _make_orchestrator(history_file=None)
        orchestrator.game_over = True

        assert orchestrator.history_file is None
        # The save method should be a no-op.
        orchestrator._save_history()
        assert list(tmp_path.iterdir()) == []
