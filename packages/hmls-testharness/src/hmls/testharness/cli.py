"""CLI argument parsing and game initialisation."""

from __future__ import annotations

import argparse

from hmls.core.cli_args import add_game_setup_args
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap  # noqa: F401
from hmls.core.map import load_map as load_map
from hmls.core.placement import place_tanks as _place_tanks_core
from hmls.core.tank import Tank


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
    add_game_setup_args(parser)
    return parser.parse_args(argv)


def place_tanks(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks randomly on passable cells for two teams.

    Thin wrapper around :func:`hmls.core.placement.place_tanks`.

    Args:
        game_map: The map to place tanks on.
        tanks_per_player: Number of tanks per team.
        seed: Optional random seed for reproducibility.

    Returns:
        List of all tanks for both teams.

    Raises:
        InsufficientPassableCellsError: If there are not enough passable cells.
    """
    return _place_tanks_core(game_map, tanks_per_player, seed=seed)


def build_initial_state(tanks: list[Tank]) -> GameState:
    """Build the initial :class:`GameState` from a tank list.

    Args:
        tanks: All tanks for the game.

    Returns:
        A fresh game state with turn order matching the tank list.
    """
    return GameState(
        tanks=tanks,
        current_tank_id=tanks[0].id if tanks else None,
    )
