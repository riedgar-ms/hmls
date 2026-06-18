"""Tests for the simple squad planner model."""

from __future__ import annotations

import pytest
import torch

from hmls.nncore.squad.orders import NUM_ORDERS
from hmls.simplesquadplanner.model import SimplePlannerConfig, SimplePlannerModel

NUM_DIRECTIONS = 4


class TestSimplePlannerConfig:
    """Tests for planner configuration."""

    def test_defaults(self) -> None:
        """Default config should produce valid settings."""
        config = SimplePlannerConfig()
        assert config.patch_size == 9
        assert config.num_orders == NUM_ORDERS
        assert config.max_tanks == 5
        assert config.tank_feature_dim == 64
        assert config.mlp_hidden_dim == 64
        assert config.model_id == "hmls.simplesquadplanner"

    def test_serialisation_round_trip(self) -> None:
        """Config should survive JSON serialisation."""
        config = SimplePlannerConfig(tank_feature_dim=32, mlp_hidden_dim=32)
        json_str = config.model_dump_json()
        restored = SimplePlannerConfig.model_validate_json(json_str)
        assert restored.tank_feature_dim == 32
        assert restored.mlp_hidden_dim == 32

    def test_invalid_patch_size_rejected(self) -> None:
        """Patch size < 3 should be rejected."""
        with pytest.raises(ValueError, match="greater than or equal to 3"):
            SimplePlannerConfig(patch_size=1)

    def test_invalid_max_tanks_rejected(self) -> None:
        """max_tanks < 1 should be rejected."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            SimplePlannerConfig(max_tanks=0)


class TestSimplePlannerModel:
    """Tests for the planner neural network."""

    @pytest.fixture
    def model(self) -> SimplePlannerModel:
        """Create a small model for testing."""
        config = SimplePlannerConfig(
            patch_size=9,
            cnn_channels=(16, 32),
            tank_feature_dim=32,
            mlp_hidden_dim=32,
        )
        return SimplePlannerModel(config)

    def test_forward_single_tank(self, model: SimplePlannerModel) -> None:
        """Forward pass with 1 alive tank should produce correct shape."""
        patches = torch.randn(1, 5, 9, 9)
        positions = torch.tensor([[0.5, 0.5]])
        directions = torch.zeros(1, NUM_DIRECTIONS)
        directions[0, 0] = 1.0  # North

        logits = model(patches, positions, directions)
        assert logits.shape == (1, NUM_ORDERS)

    def test_forward_multiple_tanks(self, model: SimplePlannerModel) -> None:
        """Forward pass with multiple tanks should produce per-tank outputs."""
        for num_tanks in [2, 3, 4, 5]:
            patches = torch.randn(num_tanks, 5, 9, 9)
            positions = torch.rand(num_tanks, 2)
            directions = torch.zeros(num_tanks, NUM_DIRECTIONS)
            for i in range(num_tanks):
                directions[i, i % NUM_DIRECTIONS] = 1.0

            logits = model(patches, positions, directions)
            assert logits.shape == (num_tanks, NUM_ORDERS)

    def test_permutation_invariance_of_pooling(self, model: SimplePlannerModel) -> None:
        """Set-pooling should make global context invariant to tank order.

        Note: the per-tank *output* is NOT order-invariant (each tank gets
        its own logits based on its features), but the global context
        (mean pool) should be the same regardless of input ordering.
        """
        patches = torch.randn(3, 5, 9, 9)
        positions = torch.rand(3, 2)
        directions = torch.zeros(3, NUM_DIRECTIONS)
        directions[0, 0] = 1.0
        directions[1, 1] = 1.0
        directions[2, 2] = 1.0

        # Forward with original order
        logits_original = model(patches, positions, directions)

        # Permute inputs (swap tanks 0 and 2)
        perm = [2, 1, 0]
        patches_perm = patches[perm]
        positions_perm = positions[perm]
        directions_perm = directions[perm]

        logits_permuted = model(patches_perm, positions_perm, directions_perm)

        # Per-tank outputs should follow the permutation
        assert torch.allclose(logits_original[0], logits_permuted[2], atol=1e-5)
        assert torch.allclose(logits_original[2], logits_permuted[0], atol=1e-5)

    def test_different_positions_produce_different_outputs(self, model: SimplePlannerModel) -> None:
        """Different positions should influence the output."""
        patches = torch.randn(2, 5, 9, 9)
        directions = torch.zeros(2, NUM_DIRECTIONS)
        directions[0, 0] = 1.0
        directions[1, 0] = 1.0

        # Positions far apart
        positions_far = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        logits_far = model(patches, positions_far, directions)

        # Positions close together
        positions_close = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        logits_close = model(patches, positions_close, directions)

        # Outputs should differ
        assert not torch.allclose(logits_far, logits_close)

    def test_initial_hidden_is_empty(self, model: SimplePlannerModel) -> None:
        """Non-recurrent planner should return empty hidden state."""
        hidden = model.initial_hidden()
        assert hidden.numel() == 0
