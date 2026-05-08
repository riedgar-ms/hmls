"""Tests for the Mk-II model module."""

from __future__ import annotations

import torch

from hmls.nncore.constants import NUM_ACTIONS
from hmls.singlemkii.encoding import NUM_CHANNELS
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork


def test_model_forward_unbatched() -> None:
    """Model forward pass works for a single (unbatched) input."""
    config = MkIIModelConfig(patch_size=9)
    model = MkIITankPolicyNetwork(config)
    model.eval()

    patch_tensor = torch.randn(NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_hidden.shape == (model.total_hidden_size,)


def test_model_forward_batched() -> None:
    """Model forward pass works for batched input."""
    config = MkIIModelConfig(patch_size=9)
    model = MkIITankPolicyNetwork(config)
    model.eval()

    batch_size = 4
    patch_tensor = torch.randn(batch_size, NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=batch_size)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (batch_size, NUM_ACTIONS)
    assert new_hidden.shape == (batch_size, model.total_hidden_size)


def test_model_different_patch_sizes() -> None:
    """Model works with different patch sizes."""
    for ps in (5, 7, 9, 11):
        config = MkIIModelConfig(patch_size=ps)
        model = MkIITankPolicyNetwork(config)
        model.eval()

        x = torch.randn(NUM_CHANNELS, ps, ps)
        h = model.initial_hidden().squeeze(0)
        logits, _ = model(x, h)
        assert logits.shape == (NUM_ACTIONS,)


def test_initial_hidden_is_zero() -> None:
    """Initial hidden state is all zeros with correct size."""
    config = MkIIModelConfig(gru1_hidden_size=64, gru2_hidden_size=32)
    model = MkIITankPolicyNetwork(config)
    h = model.initial_hidden(batch_size=2)
    assert h.shape == (2, 96)  # 64 + 32
    assert (h == 0).all()


def test_total_hidden_size() -> None:
    """Total hidden size is sum of both GRU hidden sizes."""
    config = MkIIModelConfig(gru1_hidden_size=100, gru2_hidden_size=50)
    model = MkIITankPolicyNetwork(config)
    assert model.total_hidden_size == 150


def test_model_hidden_state_changes() -> None:
    """Both GRU hidden states should change after a forward pass."""
    config = MkIIModelConfig(patch_size=9, gru1_hidden_size=64, gru2_hidden_size=32)
    model = MkIITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)

    _, h_new = model(x, h)
    assert not torch.allclose(h_new, h)

    # Check that both halves changed
    h1_new = h_new[:64]
    h2_new = h_new[64:]
    assert not torch.allclose(h1_new, torch.zeros_like(h1_new))
    assert not torch.allclose(h2_new, torch.zeros_like(h2_new))


def test_independent_gru_sizes() -> None:
    """GRU hidden sizes can be configured independently."""
    config = MkIIModelConfig(patch_size=9, gru1_hidden_size=256, gru2_hidden_size=32)
    model = MkIITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)
    logits, new_h = model(x, h)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_h.shape == (288,)  # 256 + 32


def test_config_defaults() -> None:
    """Config fields have correct defaults."""
    config = MkIIModelConfig()
    assert config.model_package == "hmls.singlemkii"
    assert config.gru1_hidden_size == 128
    assert config.gru2_hidden_size == 64
    assert config.conv_kernel_size == 3
    assert config.pool_kernel_size == 2
    assert config.pool_stride == 2


def test_config_roundtrip_json() -> None:
    """MkIIModelConfig round-trips through JSON."""
    config = MkIIModelConfig(
        patch_size=11,
        gru1_hidden_size=200,
        gru2_hidden_size=100,
        conv_kernel_size=5,
        pool_kernel_size=3,
        pool_stride=3,
    )
    json_str = config.model_dump_json()
    loaded = MkIIModelConfig.model_validate_json(json_str)
    assert loaded == config
