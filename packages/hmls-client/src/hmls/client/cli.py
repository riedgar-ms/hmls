"""CLI argument parsing for the HMLS game client."""

from __future__ import annotations

import argparse

from hmls.protocol.cli_args import add_connection_args


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
    add_connection_args(
        parser,
        name_default="Player",
        name_help="Player name sent to the server",
    )
    return parser.parse_args(argv)
