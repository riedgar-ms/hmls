"""Tests for RandomTankModelConfig and RandomTankModel."""

from __future__ import annotations

import pytest
import torch

from hmls.randomtank.model import RandomTankModel, RandomTankModelConfig


class TestRandomTankModelConfig:
    """Tests for config validation and defaults."""

    def test_default_config(self) -> None:
        """Default config should have sensible probability values."""
        config = RandomTankModelConfig()
        assert config.model_id == "hmls.randomtank"
        assert config.patch_size == 9
        assert config.prob_forward_on_passable == 0.7
        assert config.prob_turn_left_on_passable == 0.15
        assert config.prob_turn_left_on_blocked == 0.5

    def test_custom_probabilities(self) -> None:
        """Custom probability values should be accepted."""
        config = RandomTankModelConfig(
            prob_forward_on_passable=0.5,
            prob_turn_left_on_passable=0.3,
            prob_turn_left_on_blocked=0.8,
        )
        assert config.prob_forward_on_passable == 0.5
        assert config.prob_turn_left_on_passable == 0.3
        assert config.prob_turn_left_on_blocked == 0.8

    def test_passable_probs_sum_exactly_one(self) -> None:
        """Passable probabilities summing to exactly 1.0 should be valid."""
        config = RandomTankModelConfig(
            prob_forward_on_passable=0.6,
            prob_turn_left_on_passable=0.4,
        )
        assert config.prob_forward_on_passable + config.prob_turn_left_on_passable == 1.0

    def test_passable_probs_exceed_one_rejected(self) -> None:
        """Passable probabilities summing to > 1.0 should be rejected."""
        with pytest.raises(ValueError, match="exceeds 1.0"):
            RandomTankModelConfig(
                prob_forward_on_passable=0.8,
                prob_turn_left_on_passable=0.3,
            )

    def test_negative_probability_rejected(self) -> None:
        """Negative probabilities should be rejected."""
        with pytest.raises(ValueError):
            RandomTankModelConfig(prob_forward_on_passable=-0.1)

    def test_probability_above_one_rejected(self) -> None:
        """Probabilities above 1.0 should be rejected."""
        with pytest.raises(ValueError):
            RandomTankModelConfig(prob_turn_left_on_blocked=1.5)

    def test_json_round_trip(self) -> None:
        """Config should serialise to JSON and deserialise losslessly."""
        config = RandomTankModelConfig(
            prob_forward_on_passable=0.6,
            prob_turn_left_on_passable=0.2,
            prob_turn_left_on_blocked=0.7,
        )
        json_str = config.model_dump_json()
        restored = RandomTankModelConfig.model_validate_json(json_str)
        assert restored == config


class TestRandomTankModel:
    """Tests for the stub model."""

    def test_model_has_config(self) -> None:
        """Model should store its config."""
        config = RandomTankModelConfig()
        model = RandomTankModel(config)
        assert model.config is config

    def test_default_config(self) -> None:
        """Model with no config arg should use defaults."""
        model = RandomTankModel()
        assert model.config.model_id == "hmls.randomtank"

    def test_initial_hidden(self) -> None:
        """initial_hidden should return zeros of the right shape."""
        model = RandomTankModel()
        hidden = model.initial_hidden(batch_size=3)
        assert hidden.shape == (3, 1)
        assert torch.all(hidden == 0)

    def test_total_hidden_size(self) -> None:
        """total_hidden_size should be 1."""
        model = RandomTankModel()
        assert model.total_hidden_size == 1

    def test_forward_returns_correct_shapes(self) -> None:
        """forward should return uniform logits and unchanged hidden."""
        model = RandomTankModel()
        hidden = model.initial_hidden(batch_size=1).squeeze(0)
        patch = torch.randn(5, 9, 9)
        logits, new_hidden = model(patch, hidden)
        assert logits.shape == (5,)  # NUM_ACTIONS
        assert new_hidden.shape == hidden.shape

    def test_has_parameters(self) -> None:
        """Model should have at least one parameter (the placeholder)."""
        model = RandomTankModel()
        params = list(model.parameters())
        assert len(params) >= 1
