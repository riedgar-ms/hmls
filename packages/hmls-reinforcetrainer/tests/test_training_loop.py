"""Integration tests for the training loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from hmls.nncore.persistence import load_or_create_model
from hmls.nncore.reward import DefaultRewardConfig
from hmls.reinforcetrainer.config import (
    GameConfig,
    HyperparameterConfig,
    MapConfig,
    ModelRef,
    OutputConfig,
    TrainerConfig,
)
from hmls.reinforcetrainer.training_loop import (
    _validate_game_patch_size,
    _validate_model_configs,
    train,
)
from hmls.singlemki.model import ModelConfig
from hmls.singlemki.persistence import save_model_config, save_reward_config


def _setup_model_dir(
    directory: Path,
    model_config: ModelConfig | None = None,
    reward_config: DefaultRewardConfig | None = None,
) -> None:
    """Helper to create a model directory with required config files."""
    directory.mkdir(parents=True, exist_ok=True)
    save_model_config(model_config or ModelConfig(), directory)
    save_reward_config(reward_config or DefaultRewardConfig(), directory)


class TestLoadOrCreateModel:
    """Tests for load_or_create_model."""

    def test_creates_fresh_model_from_config(self, tmp_path: Path) -> None:
        """A directory with config but no model.pt produces a fresh model."""
        model_dir = tmp_path / "model"
        _setup_model_dir(model_dir)
        model = load_or_create_model(model_dir)
        assert model is not None
        assert model.config.patch_size == 9

    def test_creates_model_with_custom_config(self, tmp_path: Path) -> None:
        """A directory with custom config creates the correct architecture."""
        model_dir = tmp_path / "model"
        _setup_model_dir(model_dir, model_config=ModelConfig(gru_hidden_size=256))
        model = load_or_create_model(model_dir)
        assert model.config.gru_hidden_size == 256

    def test_loads_existing_model(self, tmp_path: Path) -> None:
        """A directory with model.pt loads the saved model."""
        from hmls.singlemki.model import TankPolicyNetwork
        from hmls.singlemki.persistence import save_model

        model_dir = tmp_path / "model"
        config = ModelConfig(gru_hidden_size=64)
        _setup_model_dir(model_dir, model_config=config)

        # Save a model with the same config
        original = TankPolicyNetwork(config)
        save_model(original, model_dir / "model.pt")

        loaded = load_or_create_model(model_dir)
        assert loaded.config.gru_hidden_size == 64

    def test_missing_model_config_raises(self, tmp_path: Path) -> None:
        """A directory without model_config.json raises FileNotFoundError."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            load_or_create_model(model_dir)


class TestValidateModelConfigs:
    """Tests for _validate_model_configs."""

    def test_matching_patch_size_passes(self) -> None:
        """Configs with same patch_size are valid."""
        config_a = ModelConfig(patch_size=9, cnn_channels=(32, 64))
        config_b = ModelConfig(patch_size=9, cnn_channels=(16, 32, 64))
        _validate_model_configs(config_a, config_b)  # Should not raise

    def test_different_patch_size_raises(self) -> None:
        """Configs with different patch_size raise ValueError."""
        config_a = ModelConfig(patch_size=9)
        config_b = ModelConfig(patch_size=7)
        with pytest.raises(ValueError, match="patch_size"):
            _validate_model_configs(config_a, config_b)

    def test_different_gru_hidden_size_allowed(self) -> None:
        """Configs with different gru_hidden_size are valid."""
        config_a = ModelConfig(gru_hidden_size=128)
        config_b = ModelConfig(gru_hidden_size=256)
        _validate_model_configs(config_a, config_b)  # Should not raise

    def test_different_cnn_channels_allowed(self) -> None:
        """Configs with different cnn_channels are valid."""
        config_a = ModelConfig(cnn_channels=(32, 64))
        config_b = ModelConfig(cnn_channels=(16, 32, 64, 128))
        _validate_model_configs(config_a, config_b)  # Should not raise


class TestValidateGamePatchSize:
    """Tests for _validate_game_patch_size."""

    def test_matching_patch_size_passes(self) -> None:
        """Game patch_size matching both model configs is valid."""
        config_a = ModelConfig(patch_size=7)
        config_b = ModelConfig(patch_size=7)
        _validate_game_patch_size(7, config_a, config_b)  # Should not raise

    def test_game_differs_from_model_a_raises(self) -> None:
        """Game patch_size != model A patch_size raises ValueError."""
        config_a = ModelConfig(patch_size=9)
        config_b = ModelConfig(patch_size=9)
        with pytest.raises(ValueError, match="model A"):
            _validate_game_patch_size(7, config_a, config_b)

    def test_game_differs_from_model_b_raises(self) -> None:
        """Game patch_size != model B patch_size raises ValueError."""
        config_a = ModelConfig(patch_size=7)
        config_b = ModelConfig(patch_size=9)
        with pytest.raises(ValueError, match="model B"):
            _validate_game_patch_size(7, config_a, config_b)


class TestTrainIntegration:
    """Integration test for the full training loop."""

    def test_short_training_run(self, tmp_path: Path) -> None:
        """A minimal training run completes without error."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(width=8, height=8, impassable_fraction=0.2),
            game=GameConfig(games_per_map=2, total_maps=2, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=2,
                save_weights_interval=2,
            ),
            hyperparameters=HyperparameterConfig(learning_rate=1e-3, gamma=0.99, seed=42),
        )

        train(config)

        # Verify weights were saved
        assert (model_a_dir / "model.pt").exists()
        assert (model_b_dir / "model.pt").exists()

        # Verify sample games were saved
        sample_files = list((tmp_path / "samples").glob("*.json"))
        assert len(sample_files) > 0

    def test_frozen_opponent_training(self, tmp_path: Path) -> None:
        """Training with one frozen model completes without error."""
        trainee_dir = tmp_path / "trainee"
        frozen_dir = tmp_path / "frozen"
        _setup_model_dir(trainee_dir)
        _setup_model_dir(frozen_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=trainee_dir, train=True),
            model_b=ModelRef(dir=frozen_dir, train=False),
            map=MapConfig(width=8, height=8, impassable_fraction=0.2),
            game=GameConfig(games_per_map=2, total_maps=2, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=2,
                save_weights_interval=2,
            ),
            hyperparameters=HyperparameterConfig(seed=123),
        )

        train(config)

        # Only trainee should have saved weights
        assert (trainee_dir / "model.pt").exists()

    def test_missing_config_files_raises(self, tmp_path: Path) -> None:
        """Training fails if model directories lack config files."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        model_a_dir.mkdir()
        model_b_dir.mkdir()

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(width=8, height=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=10),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        with pytest.raises(FileNotFoundError):
            train(config)

    def test_incompatible_patch_size_raises(self, tmp_path: Path) -> None:
        """Training fails if models have different patch_size."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir, model_config=ModelConfig(patch_size=9))
        _setup_model_dir(model_b_dir, model_config=ModelConfig(patch_size=7))

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(width=8, height=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=10),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        with pytest.raises(ValueError, match="patch_size"):
            train(config)

    def test_game_patch_size_mismatch_raises(self, tmp_path: Path) -> None:
        """Training fails if game patch_size differs from model patch_size."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir, model_config=ModelConfig(patch_size=9))
        _setup_model_dir(model_b_dir, model_config=ModelConfig(patch_size=9))

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=10, patch_size=7),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        with pytest.raises(ValueError, match="GameConfig patch_size"):
            train(config)

    def test_different_reward_configs(self, tmp_path: Path) -> None:
        """Models with different reward configs train successfully."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(
            model_a_dir,
            reward_config=DefaultRewardConfig(fire_hit_reward=1.0),
        )
        _setup_model_dir(
            model_b_dir,
            reward_config=DefaultRewardConfig(fire_hit_reward=0.1, exploration_reward=0.1),
        )

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(width=8, height=8),
            game=GameConfig(games_per_map=2, total_maps=1, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=2,
                save_weights_interval=2,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        train(config)
        assert (model_a_dir / "model.pt").exists()
        assert (model_b_dir / "model.pt").exists()
