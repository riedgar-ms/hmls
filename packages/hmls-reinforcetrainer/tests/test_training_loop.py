"""Integration tests for the training loop."""

from __future__ import annotations

from pathlib import Path

from hmls.reinforcetrainer.config import TrainerConfig
from hmls.reinforcetrainer.training_loop import load_or_create_model, train


class TestLoadOrCreateModel:
    """Tests for load_or_create_model."""

    def test_creates_fresh_model_from_empty_dir(self, tmp_path: Path) -> None:
        """An empty directory produces a fresh model."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model = load_or_create_model(model_dir)
        assert model is not None
        assert model.config.patch_size == 9

    def test_loads_existing_model(self, tmp_path: Path) -> None:
        """A directory with model.pt loads the saved model."""
        from hmls.singletanknn.model import ModelConfig, TankPolicyNetwork
        from hmls.singletanknn.persistence import save_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Save a model with non-default config
        config = ModelConfig(gru_hidden_size=64)
        original = TankPolicyNetwork(config)
        save_model(original, model_dir / "model.pt")

        loaded = load_or_create_model(model_dir)
        assert loaded.config.gru_hidden_size == 64


class TestTrainIntegration:
    """Integration test for the full training loop."""

    def test_short_training_run(self, tmp_path: Path) -> None:
        """A minimal training run completes without error."""
        config = TrainerConfig(
            model_a_dir=tmp_path / "model_a",
            model_b_dir=tmp_path / "model_b",
            train_a=True,
            train_b=True,
            map_width=8,
            map_height=8,
            impassable_fraction=0.2,
            games_per_map=2,
            total_maps=2,
            max_turns=20,
            sample_game_dir=tmp_path / "samples",
            sample_game_interval=2,
            save_weights_interval=2,
            learning_rate=1e-3,
            gamma=0.99,
            seed=42,
        )

        train(config)

        # Verify weights were saved
        assert (tmp_path / "model_a" / "model.pt").exists()
        assert (tmp_path / "model_b" / "model.pt").exists()

        # Verify sample games were saved
        sample_files = list((tmp_path / "samples").glob("*.json"))
        assert len(sample_files) > 0

    def test_frozen_opponent_training(self, tmp_path: Path) -> None:
        """Training with one frozen model completes without error."""
        config = TrainerConfig(
            model_a_dir=tmp_path / "trainee",
            model_b_dir=tmp_path / "frozen",
            train_a=True,
            train_b=False,
            map_width=8,
            map_height=8,
            impassable_fraction=0.2,
            games_per_map=2,
            total_maps=2,
            max_turns=20,
            sample_game_dir=tmp_path / "samples",
            sample_game_interval=2,
            save_weights_interval=2,
            seed=123,
        )

        train(config)

        # Only trainee should have saved weights
        assert (tmp_path / "trainee" / "model.pt").exists()
        # Frozen model dir may not have model.pt if it was empty
