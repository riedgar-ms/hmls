"""Training configuration for the simple squad trainer.

Defines :class:`SquadTrainerConfig`, supporting asymmetric matches
(squad vs squad, squad vs independent single-tank models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from hmls.mapgenerator import BlobAndLineConfig, PerlinNoiseConfig
from hmls.nncore.reward_config import RewardConfig

StrategyConfigField = Annotated[
    BlobAndLineConfig | PerlinNoiseConfig,
    Field(discriminator="type"),
]


class MapConfig(BaseModel, frozen=True, extra="forbid"):
    """Map generation configuration.

    Attributes:
        min_size: Minimum map width/height (≥ 5).
        max_size: Maximum map width/height (≥ min_size).
        impassable_fraction: Fraction of impassable cells.
        strategies: Ordered list of generation strategies (round-robin).
    """

    min_size: int = Field(default=15, ge=5)
    max_size: int = Field(default=25, ge=5)
    impassable_fraction: float = Field(default=0.3, ge=0.0, le=0.8)
    strategies: list[StrategyConfigField] = Field(
        default_factory=lambda: [BlobAndLineConfig()],  # type: ignore[arg-type]
    )

    @model_validator(mode="after")
    def _check_size_bounds(self) -> MapConfig:
        """Ensure max_size >= min_size."""
        if self.max_size < self.min_size:
            msg = f"max_size ({self.max_size}) must be >= min_size ({self.min_size})"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_strategies_nonempty(self) -> MapConfig:
        """Ensure at least one strategy is provided."""
        if len(self.strategies) == 0:
            msg = "strategies must contain at least one entry"
            raise ValueError(msg)
        return self


class GameConfig(BaseModel, frozen=True, extra="forbid"):
    """Game execution configuration.

    Attributes:
        games_per_map: Games played per map before regenerating.
        total_maps: Total maps to generate over the run.
        max_turns: Maximum turns per game before draw.
        patch_size: Visibility patch side length (odd, ≥ 3).
        tanks_per_team: Number of tanks per team.
    """

    games_per_map: int = Field(default=10, ge=1)
    total_maps: int = Field(default=100, ge=1)
    max_turns: int = Field(default=200, ge=1)
    patch_size: int = Field(default=9, ge=3)
    tanks_per_team: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _check_patch_size_odd(self) -> GameConfig:
        """Ensure patch_size is odd."""
        if self.patch_size % 2 == 0:
            msg = f"patch_size must be odd, got {self.patch_size}"
            raise ValueError(msg)
        return self


class SquadTeamRef(BaseModel, frozen=True, extra="forbid"):
    """Reference to a squad model directory.

    Attributes:
        type: Discriminator — always ``"squad"``.
        dir: Squad directory containing planner/ and executor/ subdirs.
        train: Whether this squad is updated during training.
        reward: Reward configuration for the executor.
    """

    type: Literal["squad"] = "squad"
    dir: Path
    train: bool = True
    reward: RewardConfig = Field(default_factory=RewardConfig)


class IndependentTeamRef(BaseModel, frozen=True, extra="forbid"):
    """Reference to independent single-tank models (N copies).

    Used for asymmetric training: squad vs known-good single-tank opponents.

    Attributes:
        type: Discriminator — always ``"independent"``.
        dir: Model directory (same format as hmls-reinforcetrainer).
        train: Whether this model is updated during training.
        reward: Reward configuration for training.
    """

    type: Literal["independent"] = "independent"
    dir: Path
    train: bool = False
    reward: RewardConfig = Field(default_factory=RewardConfig)


TeamRef = Annotated[SquadTeamRef | IndependentTeamRef, Field(discriminator="type")]


class HyperparameterConfig(BaseModel, frozen=True, extra="forbid"):
    """Training hyperparameters.

    Attributes:
        executor_learning_rate: Adam LR for the executor model.
        planner_learning_rate: Adam LR for the planner model.
        gamma: Discount factor for computing returns.
        seed: Optional random seed.
        baseline_alpha: EMA decay for cross-episode return baselines.
        entropy_coeff: Entropy bonus weight for executor.
        planner_entropy_coeff: Entropy bonus weight for planner.
        loss_reduction: How to aggregate per-step loss (``"sum"`` or
            ``"mean"``).
        max_grad_norm: Maximum gradient norm for clipping (None disables).
    """

    executor_learning_rate: float = Field(default=1e-3, gt=0.0)
    planner_learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    seed: int | None = Field(default=None)
    baseline_alpha: float = Field(default=0.99, gt=0.0, lt=1.0)
    entropy_coeff: float = Field(default=0.01, ge=0.0)
    planner_entropy_coeff: float = Field(default=0.01, ge=0.0)
    loss_reduction: Literal["sum", "mean"] = "sum"
    max_grad_norm: float | None = Field(default=None, gt=0.0)


class OutputConfig(BaseModel, frozen=True, extra="forbid"):
    """Output and checkpoint settings.

    Attributes:
        sample_game_dir: Directory for sample game replays.
        sample_game_interval: Save sample game every N games.
        save_weights_interval: Save weights every N games.
    """

    sample_game_dir: Path = Field(default=Path("sample_games"))
    sample_game_interval: int = Field(default=50, ge=1)
    save_weights_interval: int = Field(default=100, ge=1)


class SquadTrainerConfig(BaseModel, frozen=True, extra="forbid"):
    """Top-level configuration for a squad training run.

    Attributes:
        team_a: Configuration for team A.
        team_b: Configuration for team B.
        map: Map generation settings.
        game: Game execution settings.
        output: Output and checkpoint settings.
        hyperparameters: Training hyperparameters.
    """

    team_a: TeamRef
    team_b: TeamRef
    map: MapConfig = Field(default_factory=MapConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    hyperparameters: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
