"""Tests for the simple squad trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hmls.simplesquadtrainer.config import (
    GameConfig,
    HyperparameterConfig,
    IndependentTeamRef,
    MapConfig,
    SquadTeamRef,
    SquadTrainerConfig,
)


class TestSquadTrainerConfig:
    """Tests for trainer configuration."""

    def test_squad_vs_squad_config(self, tmp_path: Path) -> None:
        """A squad-vs-squad config should parse correctly."""
        config = SquadTrainerConfig(
            team_a=SquadTeamRef(dir=tmp_path / "squad_a"),
            team_b=SquadTeamRef(dir=tmp_path / "squad_b"),
        )
        assert config.team_a.type == "squad"
        assert config.team_b.type == "squad"
        assert config.game.tanks_per_team == 3

    def test_squad_vs_independent_config(self, tmp_path: Path) -> None:
        """Asymmetric config (squad vs independent) should parse."""
        config = SquadTrainerConfig(
            team_a=SquadTeamRef(dir=tmp_path / "squad_a"),
            team_b=IndependentTeamRef(dir=tmp_path / "model_b", train=False),
        )
        assert config.team_a.type == "squad"
        assert config.team_b.type == "independent"

    def test_config_json_round_trip(self, tmp_path: Path) -> None:
        """Config should survive JSON serialisation."""
        original = SquadTrainerConfig(
            team_a=SquadTeamRef(dir=tmp_path / "a"),
            team_b=SquadTeamRef(dir=tmp_path / "b"),
            game=GameConfig(tanks_per_team=2, max_turns=50),
            hyperparameters=HyperparameterConfig(
                executor_learning_rate=5e-4,
                planner_learning_rate=1e-4,
            ),
        )
        json_str = original.model_dump_json()
        restored = SquadTrainerConfig.model_validate_json(json_str)
        assert restored.game.tanks_per_team == 2
        assert restored.hyperparameters.executor_learning_rate == 5e-4

    def test_invalid_patch_size_rejected(self, tmp_path: Path) -> None:
        """Even patch size should be rejected."""
        with pytest.raises(ValueError, match="patch_size must be odd"):
            SquadTrainerConfig(
                team_a=SquadTeamRef(dir=tmp_path / "a"),
                team_b=SquadTeamRef(dir=tmp_path / "b"),
                game=GameConfig(patch_size=8),
            )

    def test_map_config_validation(self) -> None:
        """max_size < min_size should be rejected."""
        with pytest.raises(ValueError, match="max_size"):
            MapConfig(min_size=20, max_size=10)

    def test_hyperparameter_defaults(self) -> None:
        """Default hyperparameters should be sensible."""
        hp = HyperparameterConfig()
        assert hp.executor_learning_rate == 1e-3
        assert hp.planner_learning_rate == 3e-4
        assert hp.gamma == 0.99
        assert hp.entropy_coeff == 0.01
        assert hp.planner_entropy_coeff == 0.01


class TestIntegration:
    """Integration test: run a tiny training session end-to-end."""

    def test_one_game_squad_vs_squad(self, tmp_path: Path) -> None:
        """A single squad-vs-squad game should run without error."""
        # Create squad directories with configs
        squad_a = tmp_path / "squad_a"
        squad_b = tmp_path / "squad_b"
        for squad_dir in [squad_a, squad_b]:
            planner_dir = squad_dir / "planner"
            executor_dir = squad_dir / "executor"
            planner_dir.mkdir(parents=True)
            executor_dir.mkdir(parents=True)

            planner_config = {
                "model_id": "hmls.simplesquadplanner",
                "patch_size": 9,
                "cnn_channels": [8],
                "tank_feature_dim": 8,
                "mlp_hidden_dim": 8,
            }
            executor_config = {
                "model_id": "hmls.simplesquadexecutor",
                "patch_size": 9,
                "cnn_channels": [8],
                "gru_hidden_size": 8,
                "order_embedding_dim": 4,
            }
            (planner_dir / "model_config.json").write_text(json.dumps(planner_config))
            (executor_dir / "model_config.json").write_text(json.dumps(executor_config))

        config = SquadTrainerConfig(
            team_a=SquadTeamRef(dir=squad_a, train=True),
            team_b=SquadTeamRef(dir=squad_b, train=True),
            game=GameConfig(
                tanks_per_team=2,
                max_turns=20,
                total_maps=1,
                games_per_map=1,
                patch_size=9,
            ),
            hyperparameters=HyperparameterConfig(seed=42),
        )

        from hmls.simplesquadtrainer.training_loop import train

        # This should complete without error
        train(config)

        # Weights should have been saved
        assert (squad_a / "planner" / "model.pt").exists()
        assert (squad_a / "executor" / "model.pt").exists()
        assert (squad_b / "planner" / "model.pt").exists()
        assert (squad_b / "executor" / "model.pt").exists()
