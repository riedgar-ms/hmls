"""CLI argument parsing for the HMLS game server."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the game server.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with server configuration.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-server",
        description="WebSocket game server for the HMLS tank game",
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
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=9,
        help="Visibility patch size (odd, >= 3; default 9)",
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

    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--history-file",
        type=Path,
        default=Path("history.json"),
        help="Path to save game history JSON after the game ends (default: history.json)",
    )
    history_group.add_argument(
        "--no-history",
        action="store_true",
        default=False,
        help="Disable saving game history after the game ends",
    )

    ns = parser.parse_args(argv)

    # Normalise: if --no-history was passed, set history_file to None.
    if ns.no_history:
        ns.history_file = None

    return ns
