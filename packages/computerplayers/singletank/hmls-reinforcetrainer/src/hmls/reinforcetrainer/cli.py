"""CLI argument parsing for the REINFORCE trainer.

Accepts a single positional argument — the path to a JSON configuration
file — and produces a :class:`CLIResult` containing a
:class:`TrainerConfig` and the requested log level.

Relative paths in the configuration file are resolved relative to the
directory containing the config file itself, making configurations
portable regardless of the working directory.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from hmls.reinforcetrainer.config import ModelRef, OutputConfig, TrainerConfig

logger = logging.getLogger(__name__)

#: Valid choices for the ``--log-level`` CLI argument.
_LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class CLIResult:
    """Result of parsing CLI arguments.

    Attributes:
        config: The validated training configuration.
        log_level: The logging level string (e.g. ``"INFO"``).
    """

    config: TrainerConfig
    log_level: str


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the trainer CLI.

    Returns:
        A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="hmls-reinforcetrainer",
        description="Train tank models using REINFORCE policy gradient.",
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--log-level",
        choices=_LOG_LEVEL_CHOICES,
        default="INFO",
        help="Set the logging verbosity (default: INFO).",
    )
    return parser


def _resolve_path(base_dir: Path, p: Path) -> Path:
    """Resolve a path relative to a base directory.

    Absolute paths are returned unchanged; relative paths are resolved
    against *base_dir*.

    Args:
        base_dir: The directory to resolve relative paths against.
        p: The path to resolve.

    Returns:
        The resolved absolute path.
    """
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_config_paths(config: TrainerConfig, config_dir: Path) -> TrainerConfig:
    """Return a copy of *config* with relative paths resolved against *config_dir*.

    All :class:`~pathlib.Path` fields in the configuration are resolved
    relative to the directory containing the config file.  Absolute
    paths are left unchanged.

    Args:
        config: The parsed (but unresolved) configuration.
        config_dir: The directory containing the JSON config file.

    Returns:
        A new :class:`TrainerConfig` with resolved paths.
    """
    resolved_dir = config_dir.resolve()

    resolved_model_a = ModelRef(
        dir=_resolve_path(resolved_dir, config.model_a.dir),
        train=config.model_a.train,
        reward=config.model_a.reward,
    )
    resolved_model_b = ModelRef(
        dir=_resolve_path(resolved_dir, config.model_b.dir),
        train=config.model_b.train,
        reward=config.model_b.reward,
    )
    resolved_output = OutputConfig(
        sample_game_dir=_resolve_path(resolved_dir, config.output.sample_game_dir),
        sample_game_interval=config.output.sample_game_interval,
        save_weights_interval=config.output.save_weights_interval,
    )

    return config.model_copy(
        update={
            "model_a": resolved_model_a,
            "model_b": resolved_model_b,
            "output": resolved_output,
        }
    )


def load_config(config_path: Path) -> TrainerConfig:
    """Load and validate a TrainerConfig from a JSON file.

    Relative paths in the JSON are resolved relative to the directory
    containing the config file, so configurations are portable
    regardless of the current working directory.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A validated TrainerConfig instance with resolved paths.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the JSON content is invalid.
    """
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    json_bytes = config_path.read_bytes()
    config = TrainerConfig.model_validate_json(json_bytes)
    return _resolve_config_paths(config, config_path.parent)


def main() -> None:
    """Parse CLI arguments, configure logging, and run the training loop."""
    # Lazy import: defer heavy torch/training imports for fast --help
    from hmls.reinforcetrainer.training_loop import train

    result = parse_args()
    logging.basicConfig(
        level=result.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train(result.config)


def parse_args(argv: list[str] | None = None) -> CLIResult:
    """Parse command-line arguments and load the config file.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        A :class:`CLIResult` with the validated config and log level.

    Raises:
        SystemExit: If the config file cannot be loaded.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_file)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
    return CLIResult(config=config, log_level=args.log_level)
