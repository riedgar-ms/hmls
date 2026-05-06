"""CLI argument parsing for the REINFORCE trainer.

Parses command-line arguments and produces a :class:`TrainerConfig`.
"""

from __future__ import annotations

import argparse
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

    # Model directories
    parser.add_argument(
        "--model-a-dir",
        type=Path,
        required=True,
        help="Directory for model A weights (loaded if existing, created if empty).",
    )
    parser.add_argument(
        "--model-b-dir",
        type=Path,
        required=True,
        help="Directory for model B weights (loaded if existing, created if empty).",
    )

    # Training mode
    parser.add_argument(
        "--freeze-a",
        action="store_true",
        default=False,
        help="Freeze model A (use as fixed opponent).",
    )
    parser.add_argument(
        "--freeze-b",
        action="store_true",
        default=False,
        help="Freeze model B (use as fixed opponent).",
    )

    # Map configuration
    parser.add_argument(
        "--map-width", type=int, default=20, help="Width of generated maps (default: 20)."
    )
    parser.add_argument(
        "--map-height", type=int, default=20, help="Height of generated maps (default: 20)."
    )
    parser.add_argument(
        "--impassable-fraction",
        type=float,
        default=0.3,
        help="Fraction of impassable cells (default: 0.3).",
    )
    parser.add_argument(
        "--map-strategy",
        type=str,
        default="Blob & Line",
        help="Map generation strategy name (default: 'Blob & Line').",
    )

    # Game configuration
    parser.add_argument(
        "--games-per-map",
        type=int,
        default=10,
        help="Games to play per map before regeneration (default: 10).",
    )
    parser.add_argument(
        "--total-maps",
        type=int,
        default=100,
        help="Total number of maps to generate (default: 100).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum turns per game (default: 200).",
    )

    # Output configuration
    parser.add_argument(
        "--sample-game-dir",
        type=Path,
        default=Path("sample_games"),
        help="Directory to save sample game replays (default: sample_games/).",
    )
    parser.add_argument(
        "--sample-game-interval",
        type=int,
        default=50,
        help="Save a sample game every N games (default: 50).",
    )
    parser.add_argument(
        "--save-weights-interval",
        type=int,
        default=100,
        help="Save model weights every N games (default: 100).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for returns (default: 0.99).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> TrainerConfig:
    """Parse command-line arguments into a TrainerConfig.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        A validated TrainerConfig instance.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    return TrainerConfig(
        model_a_dir=args.model_a_dir,
        model_b_dir=args.model_b_dir,
        train_a=not args.freeze_a,
        train_b=not args.freeze_b,
        map_width=args.map_width,
        map_height=args.map_height,
        impassable_fraction=args.impassable_fraction,
        map_strategy=args.map_strategy,
        games_per_map=args.games_per_map,
        total_maps=args.total_maps,
        max_turns=args.max_turns,
        sample_game_dir=args.sample_game_dir,
        sample_game_interval=args.sample_game_interval,
        save_weights_interval=args.save_weights_interval,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
    )
