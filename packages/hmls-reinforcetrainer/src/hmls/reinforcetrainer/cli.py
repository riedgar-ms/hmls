"""CLI argument parsing for the REINFORCE trainer.

Accepts a single positional argument — the path to a JSON configuration
file — and produces a :class:`TrainerConfig`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hmls.reinforcetrainer.config import TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the trainer CLI.

    Returns:
        A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-reinforcetrainer",
        description="Train singletanknn models using REINFORCE policy gradient.",
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the JSON configuration file.",
    )
    return parser


def load_config(config_path: Path) -> TrainerConfig:
    """Load and validate a TrainerConfig from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A validated TrainerConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the JSON content is invalid.
    """
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)

    json_bytes = config_path.read_bytes()
    return TrainerConfig.model_validate_json(json_bytes)


def parse_args(argv: list[str] | None = None) -> TrainerConfig:
    """Parse command-line arguments and load the config file.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        A validated TrainerConfig instance.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return load_config(args.config_file)
