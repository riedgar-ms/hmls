"""Integration tests for the training loop."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest
import torch

from hmls.mapgenerator import BlobAndLineConfig, PerlinNoiseConfig
from hmls.nncore.persistence import load_or_create_model
from hmls.nncore.reward import ExplorationRewardConfig, FiringRewardConfig, RewardConfig
from hmls.reinforcetrainer._testing.persistence import PERSISTENCE as STUB_PERSISTENCE
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer.config import (
    GameConfig,
    HyperparameterConfig,
    MapConfig,
    ModelRef,
    OutputConfig,
    TrainerConfig,
)
from hmls.reinforcetrainer.game_runner import GameOutcome, create_map
from hmls.reinforcetrainer.training_loop import (
    TrainingSession,
    _validate_game_patch_size,
    _validate_model_configs,
    train,
)


def _setup_model_dir(
    directory: Path,
    model_config: StubModelConfig | None = None,
) -> None:
    """Helper to create a model directory with required config files."""
    directory.mkdir(parents=True, exist_ok=True)
    STUB_PERSISTENCE.save_model_config(model_config or StubModelConfig(), directory)


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
        _setup_model_dir(model_dir, model_config=StubModelConfig(hidden_size=32))
        model = load_or_create_model(model_dir)
        assert model.config.hidden_size == 32  # type: ignore[attr-defined]

    def test_loads_existing_model(self, tmp_path: Path) -> None:
        """A directory with model.pt loads the saved model."""
        model_dir = tmp_path / "model"

        config = StubModelConfig(hidden_size=8)
        _setup_model_dir(model_dir, model_config=config)

        # Save a model with the same config
        original = StubTankModel(config)
        STUB_PERSISTENCE.save_model(original, model_dir / "model.pt")

        loaded = load_or_create_model(model_dir)
        assert loaded.config.hidden_size == 8  # type: ignore[attr-defined]

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
        config_a = StubModelConfig(patch_size=9, hidden_size=16)
        config_b = StubModelConfig(patch_size=9, hidden_size=32)
        _validate_model_configs(config_a, config_b)  # Should not raise

    def test_different_patch_size_raises(self) -> None:
        """Configs with different patch_size raise ValueError."""
        config_a = StubModelConfig(patch_size=9)
        config_b = StubModelConfig(patch_size=7)
        with pytest.raises(ValueError, match="patch_size"):
            _validate_model_configs(config_a, config_b)

    def test_different_hidden_size_allowed(self) -> None:
        """Configs with different hidden_size are valid (only patch_size must match)."""
        config_a = StubModelConfig(hidden_size=16)
        config_b = StubModelConfig(hidden_size=64)
        _validate_model_configs(config_a, config_b)  # Should not raise


class TestValidateGamePatchSize:
    """Tests for _validate_game_patch_size."""

    def test_matching_patch_size_passes(self) -> None:
        """Game patch_size matching both model configs is valid."""
        config_a = StubModelConfig(patch_size=7)
        config_b = StubModelConfig(patch_size=7)
        _validate_game_patch_size(7, config_a, config_b)  # Should not raise

    def test_game_differs_from_model_a_raises(self) -> None:
        """Game patch_size != model A patch_size raises ValueError."""
        config_a = StubModelConfig(patch_size=9)
        config_b = StubModelConfig(patch_size=9)
        with pytest.raises(ValueError, match="model A"):
            _validate_game_patch_size(7, config_a, config_b)

    def test_game_differs_from_model_b_raises(self) -> None:
        """Game patch_size != model B patch_size raises ValueError."""
        config_a = StubModelConfig(patch_size=7)
        config_b = StubModelConfig(patch_size=9)
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
            map=MapConfig(min_size=8, max_size=8, impassable_fraction=0.2),
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
            map=MapConfig(min_size=8, max_size=8, impassable_fraction=0.2),
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
            map=MapConfig(min_size=8, max_size=8),
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
        _setup_model_dir(model_a_dir, model_config=StubModelConfig(patch_size=9))
        _setup_model_dir(model_b_dir, model_config=StubModelConfig(patch_size=7))

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(min_size=8, max_size=8),
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
        _setup_model_dir(model_a_dir, model_config=StubModelConfig(patch_size=9))
        _setup_model_dir(model_b_dir, model_config=StubModelConfig(patch_size=9))

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
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(
                dir=model_a_dir,
                reward=RewardConfig(firing=FiringRewardConfig(hit=1.0)),
            ),
            model_b=ModelRef(
                dir=model_b_dir,
                reward=RewardConfig(
                    firing=FiringRewardConfig(hit=0.1),
                    exploration=ExplorationRewardConfig(see_cell=0.1),
                ),
            ),
            map=MapConfig(min_size=8, max_size=8),
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

    def test_frozen_model_weights_unchanged(self, tmp_path: Path) -> None:
        """A frozen model's parameters should not change during training."""
        trainee_dir = tmp_path / "trainee"

        frozen_dir = tmp_path / "frozen"
        _setup_model_dir(trainee_dir)
        _setup_model_dir(frozen_dir)

        # Create and save a frozen model so we can compare weights
        frozen_model = StubTankModel(StubModelConfig())
        initial_state = {k: v.clone() for k, v in frozen_model.state_dict().items()}
        STUB_PERSISTENCE.save_model(frozen_model, frozen_dir / "model.pt")

        config = TrainerConfig(
            model_a=ModelRef(dir=trainee_dir, train=True),
            model_b=ModelRef(dir=frozen_dir, train=False),
            map=MapConfig(min_size=8, max_size=8, impassable_fraction=0.2),
            game=GameConfig(games_per_map=3, total_maps=2, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=10,
                save_weights_interval=10,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        train(config)

        # Load the frozen model and verify weights are unchanged
        loaded_frozen, _ = STUB_PERSISTENCE.load_model(frozen_dir / "model.pt")
        for name, param in loaded_frozen.state_dict().items():
            assert torch.allclose(initial_state[name], param), (
                f"Frozen model param '{name}' changed during training"
            )

    def test_sample_game_saved_at_correct_intervals(self, tmp_path: Path) -> None:
        """Sample games are saved at the configured interval."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        # 3 maps × 2 games = 6 total, interval = 3 → expect 2 sample files
        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8, impassable_fraction=0.2),
            game=GameConfig(games_per_map=2, total_maps=3, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=3,
                save_weights_interval=100,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        train(config)

        sample_files = sorted((tmp_path / "samples").glob("*.json"))
        assert len(sample_files) == 2
        # Files should be game_000003.json and game_000006.json
        assert sample_files[0].name == "game_000003.json"
        assert sample_files[1].name == "game_000006.json"


class TestTrainingSession:
    """Unit tests for the TrainingSession class."""

    def test_construction_with_valid_config(self, tmp_path: Path) -> None:
        """TrainingSession initializes correctly from a valid config."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=2, total_maps=3, max_turns=20),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)

        assert session.total_games == 0
        assert session.wins_a == 0
        assert session.wins_b == 0
        assert session.draws == 0
        assert session.optimizer_a is not None
        assert session.optimizer_b is not None
        assert session.baseline_a is not None
        assert session.baseline_b is not None
        assert session.total_games_planned == 6

    def test_frozen_model_has_no_optimizer(self, tmp_path: Path) -> None:
        """A frozen model should not have an optimizer or baseline."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=False),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=10),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)

        assert session.optimizer_a is not None
        assert session.optimizer_b is None
        assert session.baseline_a is not None
        assert session.baseline_b is None

    def test_train_one_game_increments_total_games(self, tmp_path: Path) -> None:
        """train_one_game increments total_games and returns a GameOutcome."""
        model_a_dir = tmp_path / "model_a"

        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=20),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)
        game_map = create_map(8, 8, 0.2, BlobAndLineConfig(), seed=123)

        outcome = session.train_one_game(game_map)

        assert isinstance(outcome, GameOutcome)
        assert session.total_games == 1

    def test_stats_tracking_across_multiple_games(self, tmp_path: Path) -> None:
        """Stats are accumulated correctly across multiple games."""
        model_a_dir = tmp_path / "model_a"

        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=100,
                save_weights_interval=100,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)
        game_map = create_map(8, 8, 0.2, BlobAndLineConfig(), seed=123)

        for _ in range(5):
            session.train_one_game(game_map)

        assert session.total_games == 5
        # All games must have some result
        assert (
            session.wins_a
            + session.wins_b
            + session.draws
            + session.lethargy_a
            + session.lethargy_b
        ) == 5

    def test_save_weights_if_due_respects_interval(self, tmp_path: Path) -> None:
        """Weights are only saved at the configured interval."""
        model_a_dir = tmp_path / "model_a"

        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=100,
                save_weights_interval=3,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)
        game_map = create_map(8, 8, 0.2, BlobAndLineConfig(), seed=123)

        # Play 2 games — should NOT save yet
        for _ in range(2):
            session.train_one_game(game_map)
            session.save_weights_if_due()

        assert not (model_a_dir / "model.pt").exists()

        # Play 1 more game (total=3) — should save
        session.train_one_game(game_map)
        session.save_weights_if_due()

        assert (model_a_dir / "model.pt").exists()
        assert (model_b_dir / "model.pt").exists()

    def test_save_sample_if_due_respects_interval(self, tmp_path: Path) -> None:
        """Sample games are only saved at the configured interval."""
        model_a_dir = tmp_path / "model_a"

        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir)
        _setup_model_dir(model_b_dir)

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir, train=True),
            model_b=ModelRef(dir=model_b_dir, train=True),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=20),
            output=OutputConfig(
                sample_game_dir=tmp_path / "samples",
                sample_game_interval=2,
                save_weights_interval=100,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        session = TrainingSession(config)
        game_map = create_map(8, 8, 0.2, BlobAndLineConfig(), seed=123)

        # Game 1 — no sample saved
        outcome1 = session.train_one_game(game_map)
        session.save_sample_if_due(outcome1)
        assert not (tmp_path / "samples").exists() or not list(
            (tmp_path / "samples").glob("*.json")
        )

        # Game 2 — sample saved
        outcome2 = session.train_one_game(game_map)
        session.save_sample_if_due(outcome2)
        sample_files = list((tmp_path / "samples").glob("*.json"))
        assert len(sample_files) == 1

    def test_incompatible_configs_raises(self, tmp_path: Path) -> None:
        """TrainingSession raises ValueError for incompatible model configs."""
        model_a_dir = tmp_path / "model_a"
        model_b_dir = tmp_path / "model_b"
        _setup_model_dir(model_a_dir, model_config=StubModelConfig(patch_size=9))
        _setup_model_dir(model_b_dir, model_config=StubModelConfig(patch_size=7))

        config = TrainerConfig(
            model_a=ModelRef(dir=model_a_dir),
            model_b=ModelRef(dir=model_b_dir),
            map=MapConfig(min_size=8, max_size=8),
            game=GameConfig(games_per_map=1, total_maps=1, max_turns=10),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        with pytest.raises(ValueError, match="patch_size"):
            TrainingSession(config)


class TestStrategyCycling:
    """Tests for round-robin strategy cycling in the training loop."""

    def test_strategies_cycle_round_robin(self, tmp_path: Path) -> None:
        """Maps cycle through the strategies list using modulo indexing."""
        model_dir_a = tmp_path / "model_a"

        model_dir_b = tmp_path / "model_b"
        _setup_model_dir(model_dir_a)
        _setup_model_dir(model_dir_b)

        strategies = [
            BlobAndLineConfig(shape=0.3),
            PerlinNoiseConfig(scale=0.05),
        ]
        config = TrainerConfig(
            model_a=ModelRef(dir=model_dir_a),
            model_b=ModelRef(dir=model_dir_b, train=False),
            map=MapConfig(min_size=8, max_size=8, strategies=strategies),
            game=GameConfig(games_per_map=1, total_maps=4, max_turns=30),
            output=OutputConfig(sample_game_dir=tmp_path / "samples"),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        observed_configs: list[object] = []
        original_generate = __import__(
            "hmls.reinforcetrainer.training_loop", fromlist=["_generate_map"]
        )._generate_map

        def tracking_generate(
            cfg: TrainerConfig,
            rng: object,
            map_idx: int,
            strategy_config: object,
        ) -> object:
            observed_configs.append(strategy_config)
            return original_generate(cfg, rng, map_idx, strategy_config)

        with mock_patch(
            "hmls.reinforcetrainer.training_loop._generate_map",
            side_effect=tracking_generate,
        ):
            train(config)

        assert len(observed_configs) == 4
        assert isinstance(observed_configs[0], BlobAndLineConfig)
        assert isinstance(observed_configs[1], PerlinNoiseConfig)
        assert isinstance(observed_configs[2], BlobAndLineConfig)
        assert isinstance(observed_configs[3], PerlinNoiseConfig)
