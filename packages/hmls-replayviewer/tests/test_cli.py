"""Tests for the CLI module: argument parsing, file loading, and state timeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmls.core.engine import GameResult, HistoryEntry
from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.replayviewer.cli import build_state_timeline, load_game_result, parse_args

# ── Fixtures ──────────────────────────────────────────────────────────


def _minimal_game_result(*, history_len: int = 0) -> GameResult:
    """Build a minimal ``GameResult`` with *history_len* history entries.

    Each history entry simply records a PASS action, and the state after
    is a copy of the initial state (good enough for timeline tests).
    """
    tank = Tank(id="t1", team="A", position=Position(1, 1), direction=Direction.NORTH)
    initial = GameState(tanks=[tank], current_tank_id="t1")
    game_map = GameMap(width=3, height=3)

    history: list[HistoryEntry] = []
    for _ in range(history_len):
        history.append(
            HistoryEntry(
                tank_id="t1",
                requested_action=Action.PASS,
                applied_action=Action.PASS,
                valid=True,
                state_after=initial.model_copy(deep=True),
            )
        )

    return GameResult(
        winner=None,
        game_map=game_map,
        initial_state=initial,
        history=history,
        turns_played=history_len,
    )


# ── parse_args ────────────────────────────────────────────────────────


class TestParseArgs:
    """Tests for ``parse_args``."""

    def test_valid_path(self) -> None:
        """A single positional argument is stored as a Path."""
        ns = parse_args(["some/file.json"])
        assert ns.history_file == Path("some/file.json")

    def test_missing_argument_exits(self) -> None:
        """No arguments should cause SystemExit (argparse error)."""
        with pytest.raises(SystemExit):
            parse_args([])


# ── load_game_result ──────────────────────────────────────────────────


class TestLoadGameResult:
    """Tests for ``load_game_result``."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """A well-formed GameResult JSON file loads successfully."""
        result = _minimal_game_result()
        file = tmp_path / "game.json"
        file.write_text(result.model_dump_json(), encoding="utf-8")

        loaded = load_game_result(file)
        assert loaded.turns_played == result.turns_played
        assert loaded.winner == result.winner

    def test_missing_file_exits(self, tmp_path: Path) -> None:
        """A non-existent file should cause SystemExit(1)."""
        with pytest.raises(SystemExit, match="1"):
            load_game_result(tmp_path / "no-such-file.json")

    def test_malformed_json_exits(self, tmp_path: Path) -> None:
        """Invalid JSON content should cause SystemExit(1)."""
        file = tmp_path / "bad.json"
        file.write_text("{not valid", encoding="utf-8")

        with pytest.raises(SystemExit, match="1"):
            load_game_result(file)

    def test_valid_json_wrong_schema_exits(self, tmp_path: Path) -> None:
        """Valid JSON that doesn't match GameResult schema should cause SystemExit(1)."""
        file = tmp_path / "wrong.json"
        file.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")

        with pytest.raises(SystemExit, match="1"):
            load_game_result(file)


# ── build_state_timeline ──────────────────────────────────────────────


class TestBuildStateTimeline:
    """Tests for ``build_state_timeline``."""

    def test_empty_history_returns_single_state(self) -> None:
        """With no history, the timeline is just the initial state."""
        result = _minimal_game_result(history_len=0)
        timeline = build_state_timeline(result)

        assert len(timeline) == 1
        assert timeline[0] is result.initial_state

    def test_timeline_length(self) -> None:
        """Timeline length should be len(history) + 1."""
        result = _minimal_game_result(history_len=5)
        timeline = build_state_timeline(result)

        assert len(timeline) == 6

    def test_first_element_is_initial_state(self) -> None:
        """Index 0 should be the initial state."""
        result = _minimal_game_result(history_len=3)
        timeline = build_state_timeline(result)

        assert timeline[0] is result.initial_state

    def test_subsequent_elements_match_history(self) -> None:
        """Index i (for i >= 1) should be history[i-1].state_after."""
        result = _minimal_game_result(history_len=3)
        timeline = build_state_timeline(result)

        for i, entry in enumerate(result.history, start=1):
            assert timeline[i] is entry.state_after
