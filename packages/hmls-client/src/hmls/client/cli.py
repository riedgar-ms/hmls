"""CLI argument parsing for the HMLS game client."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the game client.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with client configuration.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-client",
        description="WebSocket game client for the HMLS tank game",
    )
    parser.add_argument(
        "server_url",
        type=str,
        help="WebSocket server URL (e.g. ws://localhost:8765/ws)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Player",
        help="Player name sent to the server (default: 'Player')",
    )
    return parser.parse_args(argv)
