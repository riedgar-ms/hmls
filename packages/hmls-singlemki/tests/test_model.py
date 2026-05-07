"""Tests for the model module."""

from __future__ import annotations

import torch

from hmls.nncore.constants import NUM_ACTIONS
from hmls.singlemki.encoding import NUM_CHANNELS
from hmls.singlemki.model import ModelConfig, TankPolicyNetwork


def test_model_forward_unbatched() -> None:
    """Model forward pass works for a single (unbatched) input."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    model.eval()

    patch_tensor = torch.randn(NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_hidden.shape == (config.gru_hidden_size,)


def test_model_forward_batched() -> None:
    """Model forward pass works for batched input."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    model.eval()

    batch_size = 4
    patch_tensor = torch.randn(batch_size, NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=batch_size)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (batch_size, NUM_ACTIONS)
    assert new_hidden.shape == (batch_size, config.gru_hidden_size)


def test_model_different_patch_sizes() -> None:
    """Model works with different patch sizes."""
    for ps in (5, 7, 9, 11):
        config = ModelConfig(patch_size=ps)
        model = TankPolicyNetwork(config)
        model.eval()

        x = torch.randn(NUM_CHANNELS, ps, ps)
        h = model.initial_hidden().squeeze(0)
        logits, _ = model(x, h)
        assert logits.shape == (NUM_ACTIONS,)


def test_initial_hidden_is_zero() -> None:
    """Initial hidden state is all zeros."""
    config = ModelConfig(gru_hidden_size=64)
    model = TankPolicyNetwork(config)
    h = model.initial_hidden(batch_size=2)
    assert h.shape == (2, 64)
    assert (h == 0).all()


def test_model_hidden_state_changes() -> None:
    """GRU hidden state should change after a forward pass."""
    config = ModelConfig(patch_size=9)
    model = TankPolicyNetwork(config)
    model.eval()

    x = torch.randn(NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)

    _, h_new = model(x, h)
    # Hidden state should not remain all zeros after processing input
    assert not torch.allclose(h_new, h)
