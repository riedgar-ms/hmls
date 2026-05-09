"""Stub neural-network player for testing.

Provides :class:`StubNNPlayer`, a concrete
:class:`~hmls.nncore.player.NNPlayerBase` subclass that records all
patches and actions for test assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.distributions import Categorical

from hmls.core.visibility import TankPatch
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.player import NNPlayerBase
from hmls.reinforcetrainer._testing.stub_model import StubTankModel


@dataclass
class ActionRecord:
    """A record of a single action selection.

    Attributes:
        patch: The egocentric patch the player received.
        action_index: The action index that was selected.
        mode: Whether this was a 'play' or 'learn' selection.
    """

    patch: TankPatch
    action_index: int
    mode: Literal["play", "learn"]


class StubNNPlayer(NNPlayerBase):
    """Recording stub player for testing.

    Wraps a :class:`StubTankModel` and records every patch received
    and action selected.  This enables test assertions about what the
    trainer passed to the player during a game.

    Args:
        team: The team this player controls.
        model: The stub model to use for forward passes.
        mode: Operating mode.
    """

    def __init__(
        self,
        team: str,
        model: StubTankModel,
        mode: Literal["play", "learn"] = "play",
    ) -> None:
        super().__init__(team, mode=mode)
        self._model = model
        self._hidden: torch.Tensor = model.initial_hidden(batch_size=1).squeeze(0)
        self._action_records: list[ActionRecord] = []

    @property
    def model(self) -> StubTankModel:
        """The underlying stub model."""
        return self._model

    @property
    def patch_size(self) -> int:
        """Expected patch side length (from model config)."""
        return self._model.config.patch_size

    @property
    def action_records(self) -> list[ActionRecord]:
        """All action selections made by this player.

        Each entry records the patch, chosen action index, and mode.
        """
        return self._action_records

    # ── NNPlayerBase abstract method implementations ──────────────────

    def _forward_play(self, patch: TankPatch) -> int:
        """Deterministic action selection via argmax over logits."""
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        with torch.no_grad():
            logits, new_hidden = self._model(patch_tensor, self._hidden)
        self._hidden = new_hidden
        action_idx = int(logits.argmax().item())
        self._action_records.append(ActionRecord(patch=patch, action_index=action_idx, mode="play"))
        return action_idx

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
        self._action_records.append(
            ActionRecord(patch=patch, action_index=action_idx, mode="learn")
        )
        return action_idx, float(log_prob_tensor.item()), log_prob_tensor, entropy_tensor

    def _reset_model_state(self) -> None:
        """Reset hidden state and action records."""
        self._hidden = self._model.initial_hidden(batch_size=1).squeeze(0)
        self._action_records = []
