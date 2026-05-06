"""Training configuration model.

Defines :class:`TrainerConfig`, a frozen Pydantic model holding all
parameters needed to configure a REINFORCE training run.  Configuration
is loaded from a JSON file; paths in the JSON should use unix-style
forward slashes (they are converted to platform-native paths automatically).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class MapConfig(BaseModel, frozen=True):
    """Configuration for map generation.

    Map dimensions are randomized on each generation.  The user specifies
    inclusive bounds (``min_size``, ``max_size``) and each new map picks a
    random width and height independently from that range.

    Attributes:
        min_size: Minimum width/height of randomly generated maps (must be >= 5).
        max_size: Maximum width/height of randomly generated maps (must be >= min_size).
        impassable_fraction: Fraction of cells that are impassable.
        strategy: Name of the map generation strategy to use.
    """

    min_size: int = Field(default=15, ge=5)
    max_size: int = Field(default=25, ge=5)
    impassable_fraction: float = Field(default=0.3, ge=0.0, le=0.8)
    strategy: str = "Blob & Line"

    @model_validator(mode="after")
    def _check_size_bounds(self) -> "MapConfig":
        """Ensure max_size >= min_size."""
        if self.max_size < self.min_size:
            raise ValueError(
                f"max_size ({self.max_size}) must be greater than or equal to "
                f"min_size ({self.min_size})"
            )
        return self


class GameConfig(BaseModel, frozen=True):
    """Configuration for game execution.

    Attributes:
        games_per_map: Number of games played on each map before regenerating.
        total_maps: Total number of maps to generate over the run.
        max_turns: Maximum turns per game before declaring a draw.
    """

    games_per_map: int = Field(default=10, ge=1)
    total_maps: int = Field(default=100, ge=1)
    max_turns: int = Field(default=200, ge=1)


class ModelRef(BaseModel, frozen=True):
    """Reference to a model directory and its training state.

    Attributes:
        dir: Directory for model weights (loaded if existing, created if empty).
            Specified as a unix-style path in JSON.
        train: Whether this model is updated during training.
    """

    dir: Path
    train: bool = True


class OutputConfig(BaseModel, frozen=True):
    """Configuration for training output and checkpoints.

    Attributes:
        sample_game_dir: Directory to save sample game replays.
            Specified as a unix-style path in JSON.
        sample_game_interval: Save a sample game every N games.
        save_weights_interval: Save model weights every N games.
    """

    sample_game_dir: Path = Path("sample_games")
    sample_game_interval: int = Field(default=50, ge=1)
    save_weights_interval: int = Field(default=100, ge=1)


class HyperparameterConfig(BaseModel, frozen=True):
    """Training hyperparameters.

    Attributes:
        learning_rate: Adam optimizer learning rate.
        gamma: Discount factor for computing returns.
        seed: Optional random seed for reproducibility.
    """

    learning_rate: float = Field(default=1e-3, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int | None = None


class TrainerConfig(BaseModel, frozen=True):
    """Top-level configuration for a REINFORCE training run.

    Loaded from a JSON file via :meth:`model_validate_json`.  All
    sections have sensible defaults except the two model references
    which are required.

    Attributes:
        model_a: Configuration for model A.
        model_b: Configuration for model B.
        map: Map generation settings.
        game: Game execution settings.
        output: Output and checkpoint settings.
        hyperparameters: Training hyperparameters.
    """

    model_a: ModelRef
    model_b: ModelRef
    map: MapConfig = Field(default_factory=MapConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hyperparameters: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
