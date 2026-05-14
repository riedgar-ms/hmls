"""Tests for the Mk-III model module."""

from __future__ import annotations

import torch

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.singlemkiii.model import MkIIIModelConfig, MkIIITankPolicyNetwork


def test_model_forward_unbatched() -> None:
    """Model forward pass works for a single (unbatched) input."""
    config = MkIIIModelConfig(patch_size=9)
    model = MkIIITankPolicyNetwork(config)
    model.eval()

    patch_tensor = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=1).squeeze(0)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (NUM_ACTIONS,)
    assert new_hidden.shape == (model.total_hidden_size,)


def test_model_forward_batched() -> None:
    """Model forward pass works for batched input."""
    config = MkIIIModelConfig(patch_size=9)
    model = MkIIITankPolicyNetwork(config)
    model.eval()

    batch_size = 4
    patch_tensor = torch.randn(batch_size, FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    hidden = model.initial_hidden(batch_size=batch_size)

    logits, new_hidden = model(patch_tensor, hidden)

    assert logits.shape == (batch_size, NUM_ACTIONS)
    assert new_hidden.shape == (batch_size, model.total_hidden_size)


def test_model_different_patch_sizes() -> None:
    """Model works with different patch sizes."""
    for ps in (5, 7, 9, 11):
        config = MkIIIModelConfig(patch_size=ps)
        model = MkIIITankPolicyNetwork(config)
        model.eval()

        x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, ps, ps)
        h = model.initial_hidden().squeeze(0)
        logits, _ = model(x, h)
        assert logits.shape == (NUM_ACTIONS,)


def test_initial_hidden_is_zero() -> None:
    """Initial hidden state is all zeros with correct size."""
    config = MkIIIModelConfig(gru_hidden_size=64)
    model = MkIIITankPolicyNetwork(config)
    h = model.initial_hidden(batch_size=2)
    assert h.shape == (2, 64)
    assert (h == 0).all()


def test_total_hidden_size() -> None:
    """Total hidden size equals GRU hidden size."""
    config = MkIIIModelConfig(gru_hidden_size=100)
    model = MkIIITankPolicyNetwork(config)
    assert model.total_hidden_size == 100


def test_model_hidden_state_changes() -> None:
    """GRU hidden state should change after a forward pass."""
    config = MkIIIModelConfig(patch_size=9, gru_hidden_size=64)
    model = MkIIITankPolicyNetwork(config)
    model.eval()

    x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)

    _, h_new = model(x, h)
    assert not torch.allclose(h_new, h)


def test_input_size_matches_patch() -> None:
    """Internal input size should be channels * patch_size^2."""
    config = MkIIIModelConfig(patch_size=7)
    model = MkIIITankPolicyNetwork(config)
    expected = FiveChannelPatchEncoder.NUM_CHANNELS * 7 * 7
    assert model._input_size == expected


def test_config_defaults() -> None:
    """Config fields have correct defaults."""
    config = MkIIIModelConfig()
    assert config.model_id == "hmls.singlemkiii"
    assert config.gru_hidden_size == 128
    assert config.patch_size == 9


def test_config_roundtrip_json() -> None:
    """MkIIIModelConfig round-trips through JSON."""
    config = MkIIIModelConfig(
        patch_size=11,
        gru_hidden_size=200,
    )
    json_str = config.model_dump_json()
    loaded = MkIIIModelConfig.model_validate_json(json_str)
    assert loaded == config
