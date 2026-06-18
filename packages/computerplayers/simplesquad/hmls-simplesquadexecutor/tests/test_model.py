"""Tests for the simple squad executor model."""

from __future__ import annotations

import pytest
import torch

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.squad.orders import NUM_ORDERS, Order
from hmls.simplesquadexecutor.model import SimpleExecutorConfig, SimpleExecutorModel


class TestSimpleExecutorConfig:
    """Tests for executor configuration."""

    def test_defaults(self) -> None:
        """Default config should produce valid settings."""
        config = SimpleExecutorConfig()
        assert config.patch_size == 9
        assert config.num_orders == NUM_ORDERS
        assert config.gru_hidden_size == 128
        assert config.order_embedding_dim == 16
        assert config.model_id == "hmls.simplesquadexecutor"

    def test_serialisation_round_trip(self) -> None:
        """Config should survive JSON serialisation."""
        config = SimpleExecutorConfig(gru_hidden_size=64, order_embedding_dim=8)
        json_str = config.model_dump_json()
        restored = SimpleExecutorConfig.model_validate_json(json_str)
        assert restored.gru_hidden_size == 64
        assert restored.order_embedding_dim == 8

    def test_invalid_patch_size_rejected(self) -> None:
        """Patch size < 3 should be rejected."""
        with pytest.raises(ValueError, match="greater than or equal to 3"):
            SimpleExecutorConfig(patch_size=1)

    def test_custom_channels(self) -> None:
        """Custom CNN channel configuration should be accepted."""
        config = SimpleExecutorConfig(cnn_channels=(16, 32, 64))
        assert config.cnn_channels == (16, 32, 64)


class TestSimpleExecutorModel:
    """Tests for the executor neural network."""

    @pytest.fixture
    def model(self) -> SimpleExecutorModel:
        """Create a small model for testing."""
        config = SimpleExecutorConfig(
            patch_size=9,
            cnn_channels=(16, 32),
            gru_hidden_size=32,
            order_embedding_dim=8,
        )
        return SimpleExecutorModel(config)

    def test_forward_unbatched_shape(self, model: SimpleExecutorModel) -> None:
        """Unbatched forward pass should produce correct output shapes."""
        patch = torch.randn(5, 9, 9)
        order = torch.tensor(Order.ADVANCE)
        hidden = model.initial_hidden(batch_size=1).squeeze(0)

        logits, new_hidden = model(patch, order, hidden)
        assert logits.shape == (NUM_ACTIONS,)
        assert new_hidden.shape == (model.config.gru_hidden_size,)

    def test_forward_batched_shape(self, model: SimpleExecutorModel) -> None:
        """Batched forward pass should produce correct output shapes."""
        batch_size = 4
        patch = torch.randn(batch_size, 5, 9, 9)
        order = torch.tensor([0, 1, 2, 3])
        hidden = model.initial_hidden(batch_size=batch_size)

        logits, new_hidden = model(patch, order, hidden)
        assert logits.shape == (batch_size, NUM_ACTIONS)
        assert new_hidden.shape == (batch_size, model.config.gru_hidden_size)

    def test_different_orders_produce_different_outputs(self, model: SimpleExecutorModel) -> None:
        """Different orders should produce different logits (most of the time)."""
        patch = torch.randn(5, 9, 9)
        hidden = model.initial_hidden(batch_size=1).squeeze(0)

        logits_advance, _ = model(patch, torch.tensor(Order.ADVANCE), hidden)
        logits_retreat, _ = model(patch, torch.tensor(Order.RETREAT), hidden)

        # With random weights, different order embeddings should produce
        # different outputs (extremely unlikely to be identical)
        assert not torch.allclose(logits_advance, logits_retreat)

    def test_initial_hidden_shape(self, model: SimpleExecutorModel) -> None:
        """Initial hidden state should have correct shape."""
        hidden = model.initial_hidden(batch_size=3)
        assert hidden.shape == (3, model.config.gru_hidden_size)
        assert torch.all(hidden == 0)

    def test_total_hidden_size(self, model: SimpleExecutorModel) -> None:
        """total_hidden_size should match GRU hidden size."""
        assert model.total_hidden_size == model.config.gru_hidden_size

    def test_hidden_state_evolves(self, model: SimpleExecutorModel) -> None:
        """Hidden state should change after a forward pass."""
        patch = torch.randn(5, 9, 9)
        order = torch.tensor(Order.HOLD)
        hidden = model.initial_hidden(batch_size=1).squeeze(0)

        _, new_hidden = model(patch, order, hidden)
        assert not torch.allclose(hidden, new_hidden)

    def test_all_orders_accepted(self, model: SimpleExecutorModel) -> None:
        """Model should accept all valid order indices without error."""
        patch = torch.randn(5, 9, 9)
        hidden = model.initial_hidden(batch_size=1).squeeze(0)

        for order in Order:
            logits, _ = model(patch, torch.tensor(order), hidden)
            assert logits.shape == (NUM_ACTIONS,)
