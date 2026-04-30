"""CLI argument parsing for the HMLS game observer."""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the observer client.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with observer configuration.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-observer",
        description="TUI observer client for the HMLS tank game server",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:8765/ws",
        help="WebSocket URL of the game server (default: ws://localhost:8765/ws)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Observer",
        help="Display name for this observer (default: Observer)",
    )
    return parser.parse_args(argv)
