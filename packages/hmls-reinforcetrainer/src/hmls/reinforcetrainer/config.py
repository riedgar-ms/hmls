"""Training configuration model.

Defines :class:`TrainerConfig`, a frozen Pydantic model holding all
parameters needed to configure a REINFORCE training run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TrainerConfig(BaseModel, frozen=True):
    """Configuration for a REINFORCE training run.

    All fields have sensible defaults.  The two model directories are
    required — everything else is optional.

    Attributes:
        model_a_dir: Directory for model A weights (loaded or created).
        model_b_dir: Directory for model B weights (loaded or created).
        train_a: Whether model A is updated during training.
        train_b: Whether model B is updated during training.
        map_width: Width of randomly generated maps.
        map_height: Height of randomly generated maps.
        impassable_fraction: Fraction of cells that are impassable.
        map_strategy: Name of the map generation strategy to use.
        games_per_map: Number of games played on each map before
            regenerating.
        total_maps: Total number of maps to generate over the run.
        max_turns: Maximum turns per game before declaring a draw.
        sample_game_dir: Directory to save sample game replays.
        sample_game_interval: Save a sample game every N games.
        save_weights_interval: Save model weights every N games.
        learning_rate: Adam optimizer learning rate.
        gamma: Discount factor for computing returns.
        seed: Optional random seed for reproducibility.
    """

    model_a_dir: Path
    model_b_dir: Path
    train_a: bool = True
    train_b: bool = True
    map_width: int = Field(default=20, ge=5)
    map_height: int = Field(default=20, ge=5)
    impassable_fraction: float = Field(default=0.3, ge=0.0, le=0.8)
    map_strategy: str = "Blob & Line"
    games_per_map: int = Field(default=10, ge=1)
    total_maps: int = Field(default=100, ge=1)
    max_turns: int = Field(default=200, ge=1)
    sample_game_dir: Path = Path("sample_games")
    sample_game_interval: int = Field(default=50, ge=1)
    save_weights_interval: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int | None = None
