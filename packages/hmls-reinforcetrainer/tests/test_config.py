"""Tests for TrainerConfig validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hmls.reinforcetrainer.config import (
    GameConfig,
    HyperparameterConfig,
    LethargyConfig,
    MapConfig,
    ModelRef,
    OutputConfig,
    TrainerConfig,
)


class TestTrainerConfig:
    """Tests for TrainerConfig Pydantic model."""

    def test_minimal_valid_config(self, tmp_path: Path) -> None:
        """Config with only required fields uses sensible defaults."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
        )
        assert config.model_a.train is True
        assert config.model_b.train is True
        assert config.map.min_size == 15
        assert config.map.max_size == 25
        assert config.map.impassable_fraction == 0.3
        assert config.game.games_per_map == 10
        assert config.game.total_maps == 100
        assert config.game.max_turns == 200
        assert config.game.patch_size == 9
        assert config.hyperparameters.gamma == 0.99
        assert config.hyperparameters.baseline_alpha == 0.99
        assert config.hyperparameters.entropy_coeff == 0.01

    def test_full_config(self, tmp_path: Path) -> None:
        """Config with all fields specified."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a", train=False),
            model_b=ModelRef(dir=tmp_path / "b", train=True),
            map=MapConfig(
                min_size=10, max_size=30, impassable_fraction=0.4, strategy="Perlin Noise"
            ),
            game=GameConfig(games_per_map=5, total_maps=50, max_turns=150),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=20,
                save_weights_interval=50,
            ),
            hyperparameters=HyperparameterConfig(learning_rate=0.0005, gamma=0.95, seed=42),
        )
        assert config.model_a.train is False
        assert config.map.min_size == 10
        assert config.hyperparameters.seed == 42

    def test_invalid_map_min_size_too_small(self, tmp_path: Path) -> None:
        """Map min_size below 5 raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                map=MapConfig(min_size=4, max_size=20),
            )

    def test_invalid_map_max_size_less_than_min(self, tmp_path: Path) -> None:
        """max_size must be >= min_size."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                map=MapConfig(min_size=10, max_size=9),
            )

    def test_map_equal_min_max_size(self, tmp_path: Path) -> None:
        """max_size == min_size is valid (fixed-size maps)."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            map=MapConfig(min_size=10, max_size=10),
        )
        assert config.map.min_size == 10
        assert config.map.max_size == 10

    def test_invalid_impassable_fraction(self, tmp_path: Path) -> None:
        """Impassable fraction above 0.8 raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                map=MapConfig(impassable_fraction=0.9),
            )

    def test_invalid_learning_rate(self, tmp_path: Path) -> None:
        """Non-positive learning rate raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                hyperparameters=HyperparameterConfig(learning_rate=0.0),
            )

    def test_invalid_patch_size_even(self, tmp_path: Path) -> None:
        """Even patch_size raises validation error."""
        with pytest.raises(ValidationError, match="patch_size must be odd"):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                game=GameConfig(patch_size=8),
            )

    def test_invalid_patch_size_too_small(self, tmp_path: Path) -> None:
        """patch_size below 3 raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                game=GameConfig(patch_size=1),
            )

    def test_valid_patch_size(self, tmp_path: Path) -> None:
        """Valid odd patch_size >= 3 is accepted."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            game=GameConfig(patch_size=7),
        )
        assert config.game.patch_size == 7

    def test_config_is_frozen(self, tmp_path: Path) -> None:
        """Config is immutable after creation."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
        )
        with pytest.raises(ValidationError):
            config.model_a = ModelRef(dir=tmp_path / "c")  # type: ignore[misc]

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """Config can be serialised to JSON and loaded back."""
        config_data = {
            "model_a": {"dir": "models/a", "train": True},
            "model_b": {"dir": "models/b", "train": False},
            "map": {"min_size": 10, "max_size": 25, "impassable_fraction": 0.2},
            "game": {"games_per_map": 5, "total_maps": 10, "max_turns": 100},
            "hyperparameters": {"learning_rate": 0.01, "gamma": 0.9, "seed": 7},
        }
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps(config_data))

        loaded = TrainerConfig.model_validate_json(json_path.read_bytes())
        assert loaded.model_a.dir == Path("models/a")
        assert loaded.model_b.train is False
        assert loaded.map.min_size == 10
        assert loaded.game.max_turns == 100
        assert loaded.hyperparameters.seed == 7

    def test_unix_paths_in_json(self, tmp_path: Path) -> None:
        """Unix-style paths in JSON are converted to platform Path objects."""
        config_data = {
            "model_a": {"dir": "path/to/model_a"},
            "model_b": {"dir": "another/path/model_b"},
            "output": {"sample_game_dir": "output/samples"},
        }
        loaded = TrainerConfig.model_validate_json(json.dumps(config_data).encode())
        assert loaded.model_a.dir == Path("path/to/model_a")
        assert loaded.output.sample_game_dir == Path("output/samples")

    def test_default_lethargy_config(self, tmp_path: Path) -> None:
        """Default lethargy config uses consecutive_turn_limit with threshold 5."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
        )
        assert config.lethargy.policy == "consecutive_turn_limit"
        assert config.lethargy.max_consecutive_turns == 5

    def test_lethargy_config_none_policy(self, tmp_path: Path) -> None:
        """Lethargy can be disabled with policy='none'."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            lethargy=LethargyConfig(policy="none"),
        )
        assert config.lethargy.policy == "none"

    def test_lethargy_config_custom_threshold(self, tmp_path: Path) -> None:
        """Custom max_consecutive_turns is accepted."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            lethargy=LethargyConfig(max_consecutive_turns=10),
        )
        assert config.lethargy.max_consecutive_turns == 10

    def test_lethargy_config_threshold_too_small(self) -> None:
        """max_consecutive_turns below 2 raises validation error."""
        with pytest.raises(ValidationError):
            LethargyConfig(max_consecutive_turns=1)

    def test_lethargy_config_in_json(self, tmp_path: Path) -> None:
        """Lethargy config round-trips through JSON."""
        config_data = {
            "model_a": {"dir": "models/a"},
            "model_b": {"dir": "models/b"},
            "lethargy": {
                "policy": "consecutive_turn_limit",
                "max_consecutive_turns": 8,
            },
        }
        loaded = TrainerConfig.model_validate_json(json.dumps(config_data).encode())
        assert loaded.lethargy.policy == "consecutive_turn_limit"
        assert loaded.lethargy.max_consecutive_turns == 8

    def test_default_loss_reduction(self, tmp_path: Path) -> None:
        """Default loss_reduction is 'sum' (preserves original behaviour)."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
        )
        assert config.hyperparameters.loss_reduction == "sum"

    def test_loss_reduction_mean(self, tmp_path: Path) -> None:
        """loss_reduction='mean' is accepted."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            hyperparameters=HyperparameterConfig(loss_reduction="mean"),
        )
        assert config.hyperparameters.loss_reduction == "mean"

    def test_invalid_loss_reduction_raises(self) -> None:
        """Invalid loss_reduction value raises ValidationError."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(loss_reduction="max")  # type: ignore[arg-type]

    def test_default_max_grad_norm_is_none(self, tmp_path: Path) -> None:
        """Default max_grad_norm is None (no clipping)."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
        )
        assert config.hyperparameters.max_grad_norm is None

    def test_max_grad_norm_positive(self, tmp_path: Path) -> None:
        """Positive max_grad_norm is accepted."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a"),
            model_b=ModelRef(dir=tmp_path / "b"),
            hyperparameters=HyperparameterConfig(max_grad_norm=1.0),
        )
        assert config.hyperparameters.max_grad_norm == 1.0

    def test_max_grad_norm_zero_raises(self) -> None:
        """max_grad_norm=0 raises ValidationError (must be positive)."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(max_grad_norm=0.0)

    def test_max_grad_norm_negative_raises(self) -> None:
        """Negative max_grad_norm raises ValidationError."""
        with pytest.raises(ValidationError):
            HyperparameterConfig(max_grad_norm=-1.0)

    def test_loss_reduction_in_json(self, tmp_path: Path) -> None:
        """loss_reduction round-trips through JSON."""
        config_data = {
            "model_a": {"dir": "models/a"},
            "model_b": {"dir": "models/b"},
            "hyperparameters": {"loss_reduction": "mean", "max_grad_norm": 5.0},
        }
        loaded = TrainerConfig.model_validate_json(json.dumps(config_data).encode())
        assert loaded.hyperparameters.loss_reduction == "mean"
        assert loaded.hyperparameters.max_grad_norm == 5.0
