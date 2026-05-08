"""Neural-network player implementation for single-tank CNN→GRU→policy head.

The :class:`NNPlayer` is a concrete
:class:`~hmls.nncore.player.NNPlayerBase` subclass that selects actions
by running a forward pass through the
:class:`~hmls.singlemki.model.TankPolicyNetwork`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import Categorical

from hmls.core.visibility import TankPatch
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.player import NNPlayerBase
from hmls.singlemki.model import TankPolicyNetwork


class NNPlayer(NNPlayerBase):
    """A neural-network-based player using the CNN→GRU→policy-head model.

    Encodes egocentric visibility patches via
    :func:`~hmls.nncore.encoding.FiveChannelPatchEncoder.encode_patch` and runs them
    through a :class:`TankPolicyNetwork`.Maintains GRU hidden state
    across turns within an episode.

    Args:
        team: The team this player controls.
        model: The :class:`TankPolicyNetwork` to use for inference.
        mode: ``"play"`` for deterministic inference, ``"learn"`` for
            stochastic sampling with trajectory recording.
    """

    def __init__(
        self,
        team: str,
        model: TankPolicyNetwork,
        mode: Literal["play", "learn"] = "play",
    ) -> None:
        super().__init__(team, mode=mode)
        self._model = model
        self._hidden: torch.Tensor = model.initial_hidden(batch_size=1).squeeze(0)

    @property
    def model(self) -> TankPolicyNetwork:
        """The underlying neural network model."""
        return self._model

    @property
    def patch_size(self) -> int:
        """Expected patch side length (from model config)."""
        return self._model.config.patch_size

    # ── NNPlayerBase abstract method implementations ──────────────────

    def _forward_play(self, patch: TankPatch) -> int:
        """Deterministic action selection via argmax over logits."""
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        with torch.no_grad():
            logits, new_hidden = self._model(patch_tensor, self._hidden)
        self._hidden = new_hidden
        return int(logits.argmax().item())

    def _forward_learn(self, patch: TankPatch) -> tuple[int, float, torch.Tensor]:
        """Stochastic action selection with gradient-tracked log-prob."""
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        logits, new_hidden = self._model(patch_tensor, self._hidden)
        self._hidden = new_hidden.detach()
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action_tensor = dist.sample()  # type: ignore[no-untyped-call]
        action_idx = int(action_tensor.item())
        log_prob_tensor: torch.Tensor = dist.log_prob(action_tensor)  # type: ignore[no-untyped-call]
        return action_idx, float(log_prob_tensor.item()), log_prob_tensor

    def _reset_model_state(self) -> None:
        """Reset GRU hidden state to zeros."""
        self._hidden = self._model.initial_hidden(batch_size=1).squeeze(0)
