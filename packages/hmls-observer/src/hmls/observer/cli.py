"""CLI argument parsing for the HMLS game observer."""

from __future__ import annotations

import argparse

from hmls.protocol.cli_args import add_connection_args


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
    add_connection_args(
        parser,
        name_default="Observer",
        name_help="Display name for this observer",
    )
    return parser.parse_args(argv)
