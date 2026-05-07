"""Abstract base class for neural-network-based players.

:class:`NNPlayerBase` provides the model-agnostic infrastructure that
all NN tank players share: mode management, episode trajectory tracking,
exploration bookkeeping, and the ``choose_action`` skeleton.  Concrete
subclasses implement the model-specific forward pass by overriding a
small set of abstract methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from hmls.core.player import Player
from hmls.core.tank import TankId
from hmls.core.types import Action, Position
from hmls.core.visibility import PlayerView, TankPatch, VisibleCell
from hmls.nncore.constants import ACTION_INDEX_TO_ACTION
from hmls.nncore.trajectory import Episode


class NNPlayerBase(Player):
    """Base class for neural-network-based players.

    Handles mode switching (play / learn), episode lifecycle, exploration
    tracking, log-probability tensor management, and the ``choose_action``
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
        self._explored_positions: set[Position] = set()
        self._episode = Episode()
        self._last_new_positions: int = 0
        self._log_prob_tensors: list[torch.Tensor] = []
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
    def explored_positions(self) -> set[Position]:
        """Set of all positions observed during the current episode."""
        return self._explored_positions

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
    def last_patch(self) -> TankPatch | None:
        """The last egocentric patch seen by this player.

        Set during :meth:`choose_action`; ``None`` before the first step.
        """
        return self._last_patch

    # ── Episode lifecycle ─────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset state for a new episode.

        Clears exploration tracking, trajectory data, and delegates to
        :meth:`_reset_model_state` for model-specific cleanup (e.g.
        recurrent hidden states).  Call this at the start of each new
        game.
        """
        self._explored_positions = set()
        self._episode = Episode()
        self._last_new_positions = 0
        self._log_prob_tensors = []
        self._last_patch = None
        self._reset_model_state()

    # ── Action selection ──────────────────────────────────────────────

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action for the specified tank.

        Finds the tank's egocentric patch, validates it, updates
        exploration tracking, and delegates to the model-specific
        forward pass (:meth:`_forward_play` or :meth:`_forward_learn`).

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
            raise ValueError(
                f"Patch size mismatch: expected {self.patch_size}, got {grid_size}. "
                f"The model was trained for patch_size={self.patch_size}."
            )

        new_positions = self._update_exploration(patch)
        self._last_new_positions = new_positions

        if self._mode == "play":
            action_idx = self._forward_play(patch)
        else:
            action_idx, log_prob, log_prob_tensor = self._forward_learn(patch)
            self._log_prob_tensors.append(log_prob_tensor)
            self._episode.add_step(action_index=action_idx, log_prob=log_prob)

        return ACTION_INDEX_TO_ACTION[action_idx]

    # ── Exploration tracking ──────────────────────────────────────────

    def last_step_new_positions(self) -> int:
        """Return the number of new positions discovered on the last step.

        This is a convenience for the reward function.  Returns 0 if no
        step has been taken yet.
        """
        return self._last_new_positions

    def _update_exploration(self, patch: TankPatch) -> int:
        """Update the set of explored positions from a patch.

        Adds all visible cell world positions to the explored set.
        Returns the number of newly discovered positions.

        Args:
            patch: The tank's visibility patch.

        Returns:
            Number of positions that were not previously in
            :attr:`explored_positions`.
        """
        half = len(patch.grid) // 2
        forward = patch.direction.forward_delta()
        right = patch.direction.turn_right().forward_delta()
        fx, fy = forward
        rx, ry = right

        new_count = 0
        for ego_row, row in enumerate(patch.grid):
            for ego_col, cell in enumerate(row):
                if isinstance(cell, VisibleCell):
                    fwd_steps = half - ego_row
                    rgt_steps = ego_col - half
                    world_x = patch.position.x + fwd_steps * fx + rgt_steps * rx
                    world_y = patch.position.y + fwd_steps * fy + rgt_steps * ry
                    pos = Position(world_x, world_y)
                    if pos not in self._explored_positions:
                        self._explored_positions.add(pos)
                        new_count += 1

        return new_count

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
    def _forward_learn(self, patch: TankPatch) -> tuple[int, float, torch.Tensor]:
        """Select an action stochastically for training.

        Must sample from the policy distribution and return the
        log-probability tensor with gradient information intact.

        Args:
            patch: The tank's egocentric visibility patch.

        Returns:
            A tuple of ``(action_index, log_prob_float, log_prob_tensor)``
            where *log_prob_tensor* retains ``grad_fn`` for
            backpropagation.
        """
        ...

    @abstractmethod
    def _reset_model_state(self) -> None:
        """Reset model-specific state for a new episode.

        Called by :meth:`reset_episode`.  Override to clear recurrent
        hidden states or other per-episode model state.
        """
        ...
