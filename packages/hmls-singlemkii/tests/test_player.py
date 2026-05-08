"""Tests for the Mk-II NNPlayer."""

from __future__ import annotations

import torch

from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork
from hmls.singlemkii.persistence import create_player


def test_create_player_factory() -> None:
    """create_player produces an NNPlayer wrapping the model."""
    model = MkIITankPolicyNetwork(MkIIModelConfig(patch_size=9))
    player = create_player(team="A", model=model, mode="play")
    assert player.patch_size == 9
    assert player.mode == "play"


def test_create_player_learn_mode() -> None:
    """create_player can produce a learning player."""
    model = MkIITankPolicyNetwork(MkIIModelConfig(patch_size=9))
    player = create_player(team="B", model=model, mode="learn")
    assert player.mode == "learn"


def test_player_hidden_state_reset() -> None:
    """reset_episode resets hidden state to zeros."""
    model = MkIITankPolicyNetwork(MkIIModelConfig(gru1_hidden_size=64, gru2_hidden_size=32))
    player = create_player(team="A", model=model, mode="play")
    player.reset_episode()
    # After reset, internal hidden should be zeros
    hidden = player._hidden  # type: ignore[attr-defined]
    assert hidden.shape == (96,)
    assert torch.all(hidden == 0)
