"""Tests for the model module."""

from __future__ import annotations

import torch

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.singlemki.model import MkIModelConfig, MkITankPolicyNetwork


def test_model_forward_unbatched() -> None:
    """Model forward pass works for a single (unbatched) input."""
    config = MkIModelConfig(patch_size=9)
    model = MkITankPolicyNetwork(config)
    model.eval()

    patch_tensor = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_hidden.shape == (config.gru_hidden_size,)


def test_model_forward_batched() -> None:
    """Model forward pass works for batched input."""
    config = MkIModelConfig(patch_size=9)
    model = MkITankPolicyNetwork(config)
    model.eval()

    batch_size = 4
    patch_tensor = torch.randn(batch_size, FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=batch_size)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (batch_size, NUM_ACTIONS)
    assert new_hidden.shape == (batch_size, config.gru_hidden_size)


def test_model_different_patch_sizes() -> None:
    """Model works with different patch sizes."""
    for ps in (5, 7, 9, 11):
        config = MkIModelConfig(patch_size=ps)
        model = MkITankPolicyNetwork(config)
        model.eval()

        x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, ps, ps)
        h = model.initial_hidden().squeeze(0)
        logits, _ = model(x, h)
        assert logits.shape == (NUM_ACTIONS,)


def test_initial_hidden_is_zero() -> None:
    """Initial hidden state is all zeros."""
    config = MkIModelConfig(gru_hidden_size=64)
    model = MkITankPolicyNetwork(config)
    h = model.initial_hidden(batch_size=2)
    assert h.shape == (2, 64)
    assert (h == 0).all()


def test_model_hidden_state_changes() -> None:
    """GRU hidden state should change after a forward pass."""
    config = MkIModelConfig(patch_size=9)
    model = MkITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)

    _, h_new = model(x, h)
    # Hidden state should not remain all zeros after processing input
    assert not torch.allclose(h_new, h)


def test_config_defaults() -> None:
    """New conv/pool config fields have defaults matching original hardcoded values."""
    config = MkIModelConfig()
    assert config.conv_kernel_size == 3
    assert config.pool_kernel_size == 2
    assert config.pool_stride == 2


def test_custom_conv_kernel_size() -> None:
    """Model works with a non-default conv kernel size."""
    config = MkIModelConfig(patch_size=9, conv_kernel_size=5)
    model = MkITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)
    logits, new_h = model(x, h)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_h.shape == (config.gru_hidden_size,)


def test_custom_pool_params() -> None:
    """Model works with non-default pool kernel size and stride."""
    config = MkIModelConfig(patch_size=9, pool_kernel_size=3, pool_stride=3)
    model = MkITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)
    logits, new_h = model(x, h)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_h.shape == (config.gru_hidden_size,)


def test_config_roundtrip_with_new_fields() -> None:
    """MkIModelConfig with non-default conv/pool fields round-trips through JSON."""
    config = MkIModelConfig(
        patch_size=11,
        conv_kernel_size=5,
        pool_kernel_size=3,
        pool_stride=3,
    )
    json_str = config.model_dump_json()
    loaded = MkIModelConfig.model_validate_json(json_str)

    assert loaded == config
    assert loaded.conv_kernel_size == 5
    assert loaded.pool_kernel_size == 3
    assert loaded.pool_stride == 3
