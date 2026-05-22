"""Concrete neural-network-based player implementation.

:class:`NNPlayer` is the standard concrete implementation that works
with any :class:`~hmls.nncore.model.TankModelBase` subclass.  It
encodes egocentric visibility patches and runs them through the model,
maintaining recurrent hidden state across turns within an episode.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import Categorical

from hmls.core.visibility import TankPatch
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.model import TankModelBase
from hmls.nncore.player_base import NNPlayerBase


class NNPlayer(NNPlayerBase):
    """A concrete neural-network-based player for any TankModelBase model.

    Encodes egocentric visibility patches via
    :func:`~hmls.nncore.encoding.FiveChannelPatchEncoder.encode_patch` and
    runs them through any :class:`~hmls.nncore.model.TankModelBase`
    subclass.  Maintains recurrent hidden state across turns within an
    episode.

    Args:
        team: The team this player controls.
        model: A :class:`TankModelBase` subclass to use for inference.
        mode: ``"play"`` for deterministic inference, ``"learn"`` for
            stochastic sampling with trajectory recording.
    """

    def __init__(
        self,
        team: str,
        model: TankModelBase,
        mode: Literal["play", "learn"] = "play",
    ) -> None:
        super().__init__(team, mode=mode)
        self._model = model
        self._hidden: torch.Tensor = model.initial_hidden(batch_size=1).squeeze(0)

    @property
    def model(self) -> TankModelBase:
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

    def _forward_learn(self, patch: TankPatch) -> tuple[int, float, torch.Tensor, torch.Tensor]:
        """Stochastic action selection with gradient-tracked log-prob and entropy."""
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        logits, new_hidden = self._model(patch_tensor, self._hidden)
        self._hidden = new_hidden.detach()
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action_tensor = dist.sample()  # type: ignore[no-untyped-call]
        action_idx = int(action_tensor.item())
        log_prob_tensor: torch.Tensor = dist.log_prob(action_tensor)  # type: ignore[no-untyped-call]
        entropy_tensor: torch.Tensor = dist.entropy()  # type: ignore[no-untyped-call]
        return action_idx, float(log_prob_tensor.item()), log_prob_tensor, entropy_tensor

    def _reset_model_state(self) -> None:
        """Reset recurrent hidden state to zeros."""
        self._hidden = self._model.initial_hidden(batch_size=1).squeeze(0)
