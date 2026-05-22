"""Abstract base class for neural-network-based players.

:class:`NNPlayerBase` provides the model-agnostic infrastructure that
all NN tank players share: mode management, episode trajectory tracking,
and the ``choose_action`` skeleton.  Concrete subclasses implement the
model-specific forward pass by overriding a small set of abstract
methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from hmls.core.player import Player
from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView, TankPatch
from hmls.nncore.constants import ACTION_INDEX_TO_ACTION
from hmls.nncore.trajectory import Episode


class NNPlayerBase(Player):
    """Base class for neural-network-based players.

    Handles mode switching (play / learn), episode lifecycle,
    log-probability tensor management, and the ``choose_action``
    dispatch loop.  Subclasses only need to implement the model-specific
    forward pass and state-reset logic.

    Args:
        team: The team this player controls.
        mode: ``"play"`` for deterministic inference, ``"learn"`` for
            stochastic sampling with trajectory recording.
    """

    def __init__(
        self,
        team: str,
        mode: Literal["play", "learn"] = "play",
    ) -> None:
        super().__init__(team)
        self._mode = mode
        self._episode = Episode()
        self._log_prob_tensors: list[torch.Tensor] = []
        self._entropy_tensors: list[torch.Tensor] = []
        self._last_patch: TankPatch | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def mode(self) -> Literal["play", "learn"]:
        """Current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, value: Literal["play", "learn"]) -> None:
        """Switch between play and learn modes."""
        self._mode = value

    @property
    def episode(self) -> Episode:
        """The current episode trajectory (only meaningful in learn mode)."""
        return self._episode

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Expected patch side length.

        Must match the model's trained patch size.
        """
        ...

    @property
    def log_prob_tensors(self) -> list[torch.Tensor]:
        """Log-probability tensors from the computation graph (learn mode).

        These retain gradient information for backpropagation, unlike
        the float log_probs stored in the Episode.
        """
        return self._log_prob_tensors

    @property
    def entropy_tensors(self) -> list[torch.Tensor]:
        """Per-step entropy tensors from the action distributions (learn mode).

        Used by the entropy bonus in the REINFORCE loss to encourage
        the policy to maintain exploration across all actions, preventing
        collapse onto a narrow subset (e.g. always turning).
        """
        return self._entropy_tensors

    @property
    def last_patch(self) -> TankPatch | None:
        """The last egocentric patch seen by this player.

        Set during :meth:`choose_action`; ``None`` before the first step.
        """
        return self._last_patch

    # ── Episode lifecycle ─────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset state for a new episode.

        Clears trajectory data and delegates to
        :meth:`_reset_model_state` for model-specific cleanup (e.g.
        recurrent hidden states).  Call this at the start of each new
        game.
        """
        self._episode = Episode()
        self._log_prob_tensors = []
        self._entropy_tensors = []
        self._last_patch = None
        self._reset_model_state()

    # ── Action selection ──────────────────────────────────────────────

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action for the specified tank.

        Finds the tank's egocentric patch, validates it, and delegates
        to the model-specific forward pass (:meth:`_forward_play` or
        :meth:`_forward_learn`).

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The chosen :class:`Action`.

        Raises:
            ValueError: If the patch size doesn't match
                :attr:`patch_size`.
        """
        patch = None
        for p in view.patches:
            if p.tank_id == tank_id:
                patch = p
                break

        if patch is None:
            return Action.PASS

        self._last_patch = patch

        grid_size = len(patch.grid)
        if grid_size != self.patch_size:
            msg = (
                f"Patch size mismatch: expected {self.patch_size}, got {grid_size}. "
                f"The model was trained for patch_size={self.patch_size}."
            )
            raise ValueError(msg)

        if self._mode == "play":
            action_idx = self._forward_play(patch)
        else:
            action_idx, log_prob, log_prob_tensor, entropy_tensor = self._forward_learn(patch)
            self._log_prob_tensors.append(log_prob_tensor)
            self._entropy_tensors.append(entropy_tensor)
            self._episode.add_step(action_index=action_idx, log_prob=log_prob)

        return ACTION_INDEX_TO_ACTION[action_idx]

    # ── Abstract methods for subclasses ───────────────────────────────

    @abstractmethod
    def _forward_play(self, patch: TankPatch) -> int:
        """Select an action deterministically (no gradient tracking).

        Args:
            patch: The tank's egocentric visibility patch.

        Returns:
            The index of the chosen action.
        """
        ...

    @abstractmethod
    def _forward_learn(self, patch: TankPatch) -> tuple[int, float, torch.Tensor, torch.Tensor]:
        """Select an action stochastically for training.

        Must sample from the policy distribution and return the
        log-probability tensor with gradient information intact,
        along with the entropy of the distribution for regularisation.

        Args:
            patch: The tank's egocentric visibility patch.

        Returns:
            A tuple of ``(action_index, log_prob_float, log_prob_tensor,
            entropy_tensor)`` where *log_prob_tensor* retains ``grad_fn``
            for backpropagation and *entropy_tensor* is the entropy of
            the action distribution (used for the entropy bonus in the
            REINFORCE loss).
        """
        ...

    @abstractmethod
    def _reset_model_state(self) -> None:
        """Reset model-specific state for a new episode.

        Called by :meth:`reset_episode`.  Override to clear recurrent
        hidden states or other per-episode model state.
        """
        ...
