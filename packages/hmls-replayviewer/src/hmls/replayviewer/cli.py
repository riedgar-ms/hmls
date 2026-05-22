"""CLI argument parsing and history file loading for the replay viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hmls.core.game_state import GameState
from hmls.core.results import GameResult


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``history_file`` attribute.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-replayviewer",
        description="Replay an HMLS tank game from a history file.",
    )
    parser.add_argument(
        "history_file",
        type=Path,
        help="Path to a JSON game history file (GameResult format).",
    )
    return parser.parse_args(argv)


def load_game_result(path: Path) -> GameResult:
    """Load and validate a game history file.

    Args:
        path: Path to the JSON file.

    Returns:
        Validated ``GameResult`` instance.

    Raises:
        SystemExit: If the file cannot be read or parsed.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        return GameResult.model_validate_json(raw)
    except Exception as exc:
        print(f"Error parsing {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def build_state_timeline(result: GameResult) -> list[GameState]:
    """Build an ordered list of game states from a game result.

    Index 0 is the initial state (before any actions).
    Index *i* (for *i* ≥ 1) is the state after the *i*-th history entry.

    Args:
        result: A validated game result.

    Returns:
        List of game states, length ``len(result.history) + 1``.
    """
    states: list[GameState] = [result.initial_state]
    for entry in result.history:
        states.append(entry.state_after)
    return states
