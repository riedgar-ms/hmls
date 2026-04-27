"""CLI argument parsing and game initialisation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the test harness.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``map_file`` and ``tanks_per_player``.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-testharness",
        description="Interactive TUI test harness for the HMLS tank game",
    )
    parser.add_argument(
        "map_file",
        type=Path,
        help="Path to a JSON map file (GameMap)",
    )
    parser.add_argument(
        "tanks_per_player",
        type=int,
        help="Number of tanks each player starts with",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=7,
        help="Visibility patch size (odd, >= 3; default 7)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum number of individual turns (default 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for tank placement (default: random)",
    )
    return parser.parse_args(argv)


def load_map(path: Path) -> GameMap:
    """Load a :class:`GameMap` from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialised game map.

    Raises:
        SystemExit: If the file does not exist or cannot be parsed.
    """
    if not path.exists():
        print(f"Error: map file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        text = path.read_text(encoding="utf-8")
        return GameMap.model_validate_json(text)
    except Exception as exc:
        print(f"Error loading map: {exc}", file=sys.stderr)
        sys.exit(1)


def place_tanks(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks randomly on passable cells for two teams.

    Teams are named ``"A"`` and ``"B"``.  Tanks are assigned IDs like
    ``"A1"``, ``"A2"``, ``"B1"``, ``"B2"``, etc.  Each tank gets a
    random direction.

    Args:
        game_map: The map to place tanks on.
        tanks_per_player: Number of tanks per team.
        seed: Optional random seed for reproducibility.

    Returns:
        List of all tanks for both teams.

    Raises:
        SystemExit: If there are not enough passable cells.
    """
    total_needed = tanks_per_player * 2
    passable_positions = [
        Position(x, y) for x, y in game_map.all_positions() if game_map[x, y] == CellType.PASSABLE
    ]
    if len(passable_positions) < total_needed:
        print(
            f"Error: need {total_needed} passable cells but map only has {len(passable_positions)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    chosen = rng.sample(passable_positions, total_needed)
    directions = list(Direction)
    tanks: list[Tank] = []

    for team_idx, team_name in enumerate(["A", "B"]):
        for i in range(tanks_per_player):
            pos = chosen[team_idx * tanks_per_player + i]
            tanks.append(
                Tank(
                    id=f"{team_name}{i + 1}",
                    team=team_name,
                    position=pos,
                    direction=rng.choice(directions),
                )
            )

    return tanks


def build_initial_state(tanks: list[Tank]) -> GameState:
    """Build the initial :class:`GameState` from a tank list.

    Args:
        tanks: All tanks for the game.

    Returns:
        A fresh game state with turn order matching the tank list.
    """
    return GameState(
        tanks=tanks,
        current_turn_index=0,
    )
