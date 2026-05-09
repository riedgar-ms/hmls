"""Training configuration model.

Defines :class:`TrainerConfig`, a frozen Pydantic model holding all
parameters needed to configure a REINFORCE training run.  Configuration
is loaded from a JSON file; paths in the JSON should use unix-style
forward slashes (they are converted to platform-native paths automatically).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MapConfig(BaseModel, frozen=True, extra="forbid"):
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


class GameConfig(BaseModel, frozen=True, extra="forbid"):
    """Configuration for game execution.

    Attributes:
        games_per_map: Number of games played on each map before regenerating.
        total_maps: Total number of maps to generate over the run.
        max_turns: Maximum turns per game before declaring a draw.
        patch_size: Side length of visibility patches (must be odd, ≥ 3).
    """

    games_per_map: int = Field(default=10, ge=1)
    total_maps: int = Field(default=100, ge=1)
    max_turns: int = Field(default=200, ge=1)
    patch_size: int = Field(default=9, ge=3)

    @model_validator(mode="after")
    def _check_patch_size_odd(self) -> "GameConfig":
        """Ensure patch_size is odd."""
        if self.patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {self.patch_size}")
        return self


class ModelRef(BaseModel, frozen=True, extra="forbid"):
    """Reference to a model directory and its training state.

    Attributes:
        dir: Directory for model weights (loaded if existing, created if empty).
            Specified as a unix-style path in JSON.
        train: Whether this model is updated during training.
    """

    dir: Path
    train: bool = True


class OutputConfig(BaseModel, frozen=True, extra="forbid"):
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


class LethargyConfig(BaseModel, frozen=True, extra="forbid"):
    """Configuration for lethargy (degenerate-play) detection.

    Controls whether and how the trainer detects tanks that are
    behaving lethargically (e.g. spinning in place) and forces
    an early loss.

    Attributes:
        policy: Which lethargy policy to use.
            ``"none"`` disables detection entirely.
            ``"consecutive_turn_limit"`` ends the game when a tank
            makes too many consecutive turn actions.
        max_consecutive_turns: Threshold for the
            ``"consecutive_turn_limit"`` policy.  Ignored when
            ``policy`` is ``"none"``.
    """

    policy: Literal["none", "consecutive_turn_limit"] = "consecutive_turn_limit"
    max_consecutive_turns: int = Field(default=5, ge=2)


class HyperparameterConfig(BaseModel, frozen=True, extra="forbid"):
    """Training hyperparameters.

    Attributes:
        learning_rate: Adam optimizer learning rate.
        gamma: Discount factor for computing returns.
        seed: Optional random seed for reproducibility.
        baseline_alpha: EMA decay for the cross-episode return baseline
            (see :class:`~hmls.reinforcetrainer.updater.ReturnBaseline`).
            Higher values make the baseline adapt more slowly; ``0.99``
            averages over roughly the last 100 episodes.
        entropy_coeff: Weight of the entropy bonus in the policy
            gradient loss.  Encourages the policy to maintain
            exploration across all actions, preventing collapse onto a
            narrow subset (e.g. always turning).  ``0.0`` disables the
            bonus; ``0.01`` is a reasonable starting point.
    """

    learning_rate: float = Field(default=1e-3, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int | None = None
    baseline_alpha: float = Field(default=0.99, gt=0.0, lt=1.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)


class TrainerConfig(BaseModel, frozen=True, extra="forbid"):
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
        lethargy: Lethargy (degenerate-play) detection settings.
    """

    model_a: ModelRef
    model_b: ModelRef
    map: MapConfig = Field(default_factory=MapConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hyperparameters: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
    lethargy: LethargyConfig = Field(default_factory=LethargyConfig)
