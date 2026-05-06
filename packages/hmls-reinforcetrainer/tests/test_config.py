"""Tests for TrainerConfig validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hmls.reinforcetrainer.config import (
    GameConfig,
    HyperparameterConfig,
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
        assert config.map.width == 20
        assert config.map.height == 20
        assert config.map.impassable_fraction == 0.3
        assert config.game.games_per_map == 10
        assert config.game.total_maps == 100
        assert config.game.max_turns == 200
        assert config.hyperparameters.gamma == 0.99

    def test_full_config(self, tmp_path: Path) -> None:
        """Config with all fields specified."""
        config = TrainerConfig(
            model_a=ModelRef(dir=tmp_path / "a", train=False),
            model_b=ModelRef(dir=tmp_path / "b", train=True),
            map=MapConfig(width=30, height=25, impassable_fraction=0.4, strategy="Perlin Noise"),
            game=GameConfig(games_per_map=5, total_maps=50, max_turns=150),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=20,
                save_weights_interval=50,
            ),
            hyperparameters=HyperparameterConfig(learning_rate=0.0005, gamma=0.95, seed=42),
        )
        assert config.model_a.train is False
        assert config.map.width == 30
        assert config.hyperparameters.seed == 42

    def test_invalid_map_width_too_small(self, tmp_path: Path) -> None:
        """Map width below minimum raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a=ModelRef(dir=tmp_path / "a"),
                model_b=ModelRef(dir=tmp_path / "b"),
                map=MapConfig(width=3),
            )

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
            "map": {"width": 25, "height": 15, "impassable_fraction": 0.2},
            "game": {"games_per_map": 5, "total_maps": 10, "max_turns": 100},
            "hyperparameters": {"learning_rate": 0.01, "gamma": 0.9, "seed": 7},
        }
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps(config_data))

        loaded = TrainerConfig.model_validate_json(json_path.read_bytes())
        assert loaded.model_a.dir == Path("models/a")
        assert loaded.model_b.train is False
        assert loaded.map.width == 25
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
