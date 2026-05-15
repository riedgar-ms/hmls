"""Tests for the game runner."""

from __future__ import annotations

from pathlib import Path

from hmls.mapgenerator import BlobAndLineConfig
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer.game_runner import (
    GameOutcome,
    create_map,
    run_game,
    save_sample_game,
)


class TestCreateMap:
    """Tests for create_map function."""

    def test_creates_map_with_correct_dimensions(self) -> None:
        """Generated map has the requested dimensions."""
        game_map = create_map(15, 10, 0.3, BlobAndLineConfig(), seed=42)
        assert game_map.width == 15
        assert game_map.height == 10

    def test_perlin_strategy_config(self) -> None:
        """A PerlinNoiseConfig strategy produces a valid map."""
        from hmls.mapgenerator import PerlinNoiseConfig

        game_map = create_map(10, 10, 0.3, PerlinNoiseConfig(), seed=42)
        assert game_map.width == 10
        assert game_map.height == 10

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same map."""
        map1 = create_map(10, 10, 0.3, BlobAndLineConfig(), seed=123)
        map2 = create_map(10, 10, 0.3, BlobAndLineConfig(), seed=123)
        assert map1.cells == map2.cells


class TestRunGame:
    """Tests for run_game function."""

    def test_runs_game_to_completion(self) -> None:
        """A game runs and produces a valid outcome."""
        import random

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=1)

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

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=2)

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

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=3)

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

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=5)

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


class TestRunGameWithStubRecording:
    """Tests that use the stub player's recording capabilities."""

    def test_learning_player_receives_correct_patch_size(self) -> None:
        """All patches received by a learning player have the configured patch_size."""
        import random

        from hmls.reinforcetrainer._testing.stub_player import StubNNPlayer

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=10)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=30,
            rng=random.Random(42),
        )

        # Both players are StubNNPlayers due to dynamic dispatch
        assert isinstance(outcome.player_a, StubNNPlayer)
        assert isinstance(outcome.player_b, StubNNPlayer)

        expected_size = StubModelConfig().patch_size
        for record in outcome.player_a.action_records:
            assert len(record.patch.grid) == expected_size
        for record in outcome.player_b.action_records:
            assert len(record.patch.grid) == expected_size

    def test_learning_player_records_in_learn_mode(self) -> None:
        """Action records from a learning player are all in 'learn' mode."""
        import random

        from hmls.reinforcetrainer._testing.stub_player import StubNNPlayer

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=11)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=False,
            max_turns=30,
            rng=random.Random(42),
        )

        player_a = outcome.player_a
        player_b = outcome.player_b
        assert isinstance(player_a, StubNNPlayer)
        assert isinstance(player_b, StubNNPlayer)

        for record in player_a.action_records:
            assert record.mode == "learn"
        for record in player_b.action_records:
            assert record.mode == "play"

    def test_episode_length_matches_action_records(self) -> None:
        """Episode length matches the number of action records for learning player."""
        import random

        from hmls.reinforcetrainer._testing.stub_player import StubNNPlayer

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=12)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=30,
            rng=random.Random(42),
        )

        player_a = outcome.player_a
        player_b = outcome.player_b
        assert isinstance(player_a, StubNNPlayer)
        assert isinstance(player_b, StubNNPlayer)

        # Episode steps == number of learn-mode action records
        learn_records_a = [r for r in player_a.action_records if r.mode == "learn"]
        learn_records_b = [r for r in player_b.action_records if r.mode == "learn"]
        assert len(player_a.episode) == len(learn_records_a)
        assert len(player_b.episode) == len(learn_records_b)

    def test_rewards_assigned_to_correct_steps(self) -> None:
        """Each episode step has a reward assigned (non-None after game)."""
        import random

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=13)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=30,
            rng=random.Random(42),
        )

        # All steps in learning episodes should have rewards assigned
        for step in outcome.player_a.episode.steps:
            assert step.reward is not None
        for step in outcome.player_b.episode.steps:
            assert step.reward is not None

    def test_log_prob_tensors_match_episode_length(self) -> None:
        """Number of log_prob_tensors matches the episode length."""
        import random

        model_a = StubTankModel(StubModelConfig())
        model_b = StubTankModel(StubModelConfig())
        game_map = create_map(10, 10, 0.2, BlobAndLineConfig(), seed=14)

        outcome = run_game(
            game_map,
            model_a,
            model_b,
            train_a=True,
            train_b=True,
            max_turns=30,
            rng=random.Random(42),
        )

        assert len(outcome.player_a.log_prob_tensors) == len(outcome.player_a.episode)
        assert len(outcome.player_b.log_prob_tensors) == len(outcome.player_b.episode)
