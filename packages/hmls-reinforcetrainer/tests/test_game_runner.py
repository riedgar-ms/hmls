"""Tests for the game runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from hmls.reinforcetrainer.game_runner import (
    GameOutcome,
    create_map,
    run_game,
    save_sample_game,
)
from hmls.singlemki.model import ModelConfig, TankPolicyNetwork


class TestCreateMap:
    """Tests for create_map function."""

    def test_creates_map_with_correct_dimensions(self) -> None:
        """Generated map has the requested dimensions."""
        game_map = create_map(15, 10, 0.3, "Blob & Line", seed=42)
        assert game_map.width == 15
        assert game_map.height == 10

    def test_unknown_strategy_raises_key_error(self) -> None:
        """An unknown strategy name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown map strategy"):
            create_map(10, 10, 0.3, "NonExistent Strategy")

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same map."""
        map1 = create_map(10, 10, 0.3, "Blob & Line", seed=123)
        map2 = create_map(10, 10, 0.3, "Blob & Line", seed=123)
        assert map1.cells == map2.cells


class TestRunGame:
    """Tests for run_game function."""

    def test_runs_game_to_completion(self) -> None:
        """A game runs and produces a valid outcome."""
        import random

        model_a = TankPolicyNetwork(ModelConfig())
        model_b = TankPolicyNetwork(ModelConfig())
        game_map = create_map(10, 10, 0.2, "Blob & Line", seed=1)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=50,
            rng=random.Random(42),
        )

        assert isinstance(outcome, GameOutcome)
        assert outcome.result.turns_played <= 50
        assert outcome.result.winner in ("A", "B", None)

    def test_frozen_player_has_empty_episode(self) -> None:
        """A frozen player does not record trajectory steps."""
        import random

        model_a = TankPolicyNetwork(ModelConfig())
        model_b = TankPolicyNetwork(ModelConfig())
        game_map = create_map(10, 10, 0.2, "Blob & Line", seed=2)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=False,
            max_turns=50,
            rng=random.Random(42),
        )

        # Player B should have an empty episode (play mode)
        assert len(outcome.player_b.episode) == 0

    def test_learning_player_has_trajectory(self) -> None:
        """A learning player accumulates trajectory steps."""
        import random

        model_a = TankPolicyNetwork(ModelConfig())
        model_b = TankPolicyNetwork(ModelConfig())
        game_map = create_map(10, 10, 0.2, "Blob & Line", seed=3)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=50,
            rng=random.Random(42),
        )

        # At least one player should have steps (unless game ended immediately)
        total_steps = len(outcome.player_a.episode) + len(outcome.player_b.episode)
        assert total_steps > 0


class TestSaveSampleGame:
    """Tests for save_sample_game function."""

    def test_saves_json_file(self, tmp_path: Path) -> None:
        """Sample game is saved as a valid JSON file."""
        import random

        from hmls.core.engine import GameResult

        model_a = TankPolicyNetwork(ModelConfig())
        model_b = TankPolicyNetwork(ModelConfig())
        game_map = create_map(10, 10, 0.2, "Blob & Line", seed=5)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=False,
            train_b=False,
            max_turns=30,
            rng=random.Random(42),
        )

        filepath = save_sample_game(outcome.result, tmp_path / "samples", 42)
        assert filepath.exists()
        assert filepath.name == "game_000042.json"

        # Verify it can be loaded back
        loaded = GameResult.model_validate_json(filepath.read_text())
        assert loaded.winner == outcome.result.winner
