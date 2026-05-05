"""Reusable argparse argument groups for server-connection CLI options."""

from __future__ import annotations

import argparse


def add_connection_args(
    parser: argparse.ArgumentParser,
    *,
    name_default: str = "Player",
    name_help: str = "Player name sent to the server",
) -> argparse._ArgumentGroup:
    """Add a 'Server connection' argument group to *parser*.

    The group contains standard arguments for connecting to a game server:

    - ``server_url`` (positional) — WebSocket server URL.
    - ``--name`` — display name with configurable default.

    Args:
        parser: The argument parser to extend.
        name_default: Default value for the ``--name`` argument.
        name_help: Help text for the ``--name`` argument.

    Returns:
        The newly created argument group.
    """
    group = parser.add_argument_group("Server connection")
    group.add_argument(
        "server_url",
        type=str,
        help="WebSocket server URL (e.g. ws://localhost:8765/ws)",
    )
    group.add_argument(
        "--name",
        type=str,
        default=name_default,
        help=f"{name_help} (default: '{name_default}')",
    )
    return group
