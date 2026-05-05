"""Reusable argparse argument groups for game-setup CLI options."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_game_setup_args(
    parser: argparse.ArgumentParser,
) -> argparse._ArgumentGroup:
    """Add a 'Game setup' argument group to *parser*.

    The group contains the standard arguments shared by packages that
    configure and start a new game (e.g. server, test harness):

    - ``map_file`` (positional) — path to a JSON map file.
    - ``tanks_per_player`` (positional) — tanks per team.
    - ``--patch-size`` — visibility patch size (default 9).
    - ``--max-turns`` — turn limit (default 200).
    - ``--seed`` — random seed for placement (default: random).

    Args:
        parser: The argument parser to extend.

    Returns:
        The newly created argument group (for further customisation).
    """
    group = parser.add_argument_group("Game setup")
    group.add_argument(
        "map_file",
        type=Path,
        help="Path to a JSON map file (GameMap)",
    )
    group.add_argument(
        "tanks_per_player",
        type=int,
        help="Number of tanks each player starts with",
    )
    group.add_argument(
        "--patch-size",
        type=int,
        default=9,
        help="Visibility patch size (odd, >= 3; default 9)",
    )
    group.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum number of individual turns (default 200)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for tank placement (default: random)",
    )
    return group
