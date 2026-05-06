"""Tests for TrainerConfig validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hmls.reinforcetrainer.config import TrainerConfig


class TestTrainerConfig:
    """Tests for TrainerConfig Pydantic model."""

    def test_minimal_valid_config(self, tmp_path: Path) -> None:
        """Config with only required fields uses sensible defaults."""
        config = TrainerConfig(
            model_a_dir=tmp_path / "a",
            model_b_dir=tmp_path / "b",
        )
        assert config.train_a is True
        assert config.train_b is True
        assert config.map_width == 20
        assert config.map_height == 20
        assert config.impassable_fraction == 0.3
        assert config.games_per_map == 10
        assert config.total_maps == 100
        assert config.max_turns == 200
        assert config.gamma == 0.99

    def test_full_config(self, tmp_path: Path) -> None:
        """Config with all fields specified."""
        config = TrainerConfig(
            model_a_dir=tmp_path / "a",
            model_b_dir=tmp_path / "b",
            train_a=False,
            train_b=True,
            map_width=30,
            map_height=25,
            impassable_fraction=0.4,
            map_strategy="Perlin Noise",
            games_per_map=5,
            total_maps=50,
            max_turns=150,
            sample_game_dir=tmp_path / "samples",
            sample_game_interval=20,
            save_weights_interval=50,
            learning_rate=0.0005,
            gamma=0.95,
            seed=42,
        )
        assert config.train_a is False
        assert config.map_width == 30
        assert config.seed == 42

    def test_invalid_map_width_too_small(self, tmp_path: Path) -> None:
        """Map width below minimum raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a_dir=tmp_path / "a",
                model_b_dir=tmp_path / "b",
                map_width=3,
            )

    def test_invalid_impassable_fraction(self, tmp_path: Path) -> None:
        """Impassable fraction above 0.8 raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a_dir=tmp_path / "a",
                model_b_dir=tmp_path / "b",
                impassable_fraction=0.9,
            )

    def test_invalid_learning_rate(self, tmp_path: Path) -> None:
        """Non-positive learning rate raises validation error."""
        with pytest.raises(ValidationError):
            TrainerConfig(
                model_a_dir=tmp_path / "a",
                model_b_dir=tmp_path / "b",
                learning_rate=0.0,
            )

    def test_config_is_frozen(self, tmp_path: Path) -> None:
        """Config is immutable after creation."""
        config = TrainerConfig(
            model_a_dir=tmp_path / "a",
            model_b_dir=tmp_path / "b",
        )
        with pytest.raises(ValidationError):
            config.train_a = False  # type: ignore[misc]
