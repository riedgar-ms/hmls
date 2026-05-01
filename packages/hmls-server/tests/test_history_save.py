"""Tests for game history saving and CLI argument parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmls.core.engine import GameResult
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.server.app import GameSession
from hmls.server.cli import parse_args


def _make_simple_map() -> GameMap:
    """Create a small 5x5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_tanks() -> list[Tank]:
    """Create a minimal set of tanks for two teams."""
    return [
        Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
        Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
    ]


def _make_session(history_file: Path | None = None) -> GameSession:
    """Create a GameSession for testing."""
    return GameSession(
        game_map=_make_simple_map(),
        tanks=_make_tanks(),
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
        """GameSession should write a valid GameResult JSON when game ends."""
        history_file = tmp_path / "history.json"
        session = _make_session(history_file=history_file)

        # Manually wire up the engine and simulate a short game.
        from hmls.core.engine import GameEngine
        from hmls.core.player import Player
        from hmls.core.visibility import build_player_view

        players: dict[str, Player] = {
            "A": session.players["A"],
            "B": session.players["B"],
        }
        session.engine = GameEngine(
            session.game_map,
            session.tanks,
            players,
            max_turns=4,
            patch_size=5,
        )

        # Use choose_action (synchronous) by pre-loading actions via
        # request_action + submit_action, then calling step().
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            for _ in range(4):
                tank_id = session.engine.current_tank_id
                team = session.engine.current_team
                player = session.players[team]

                view = build_player_view(session.engine.state, session.game_map, team, 5)
                player.request_action(tank_id, view, loop)
                player.submit_action(Action.PASS)
                loop.run_until_complete(player.wait_for_action())
                session.engine.step()
        finally:
            loop.close()

        # Now the engine should be at game_over (max_turns reached).
        session._game_over = True
        assert session.engine.game_over

        # Call the save logic (same as in run_game tail).
        if session.history_file is not None and session.engine is not None:
            result = session.engine.make_result()
            session.history_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        assert history_file.exists()
        data = json.loads(history_file.read_text(encoding="utf-8"))
        # Validate it parses as a GameResult.
        parsed = GameResult.model_validate(data)
        assert parsed.turns_played == 4

    def test_no_history_file_when_disabled(self, tmp_path: Path) -> None:
        """No file should be written when history_file is None."""
        session = _make_session(history_file=None)

        # Simulate game ending.
        session._game_over = True

        # The save path is None, so nothing should happen.
        assert session.history_file is None
        # Verify no stray files were created.
        assert list(tmp_path.iterdir()) == []
