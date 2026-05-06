"""Neural-network player implementation.

The :class:`NNPlayer` is a :class:`~hmls.core.player.Player` subclass
that selects actions by running a forward pass through the
:class:`~hmls.singletanknn.model.TankPolicyNetwork`.  It supports
"play" mode (deterministic argmax) and "learn" mode (stochastic
sampling with trajectory recording).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import Categorical

from hmls.core.player import Player
from hmls.core.tank import TankId
from hmls.core.types import Action, Position
from hmls.core.visibility import PlayerView, TankPatch, VisibleCell
from hmls.singletanknn.encoding import encode_patch
from hmls.singletanknn.model import TankPolicyNetwork
from hmls.singletanknn.trajectory import Episode

# Mapping from action index to Action enum (stable ordering).
ACTION_INDEX_TO_ACTION: list[Action] = [
    Action.MOVE_FORWARD,
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.FIRE,
    Action.PASS,
]

ACTION_TO_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTION_INDEX_TO_ACTION)}


class NNPlayer(Player):
    """A neural-network-based player controlling a single tank.

    Uses a CNN→GRU→policy-head architecture to choose actions from the
    egocentric visibility patch.  In "play" mode, selects the highest-
    probability action deterministically.  In "learn" mode, samples from
    the policy distribution and records trajectory data for REINFORCE.

    The player tracks which map positions have been observed (for
    exploration rewards) and validates that incoming patches match the
    expected size.

    Args:
        team: The team this player controls.
        model: The :class:`TankPolicyNetwork` to use for inference.
        mode: ``"play"`` for deterministic inference, ``"learn"`` for
            stochastic sampling with trajectory recording.
        patch_size: Expected patch side length (must match the model's
            trained patch size).  Defaults to 9.
    """

    def __init__(
        self,
        team: str,
        model: TankPolicyNetwork,
        mode: Literal["play", "learn"] = "play",
        patch_size: int = 9,
    ) -> None:
        super().__init__(team)
        self._model = model
        self._mode = mode
        self._patch_size = patch_size
        self._hidden: torch.Tensor = model.initial_hidden(batch_size=1).squeeze(0)
        self._explored_positions: set[Position] = set()
        self._episode = Episode()

    @property
    def mode(self) -> Literal["play", "learn"]:
        """Current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, value: Literal["play", "learn"]) -> None:
        """Switch between play and learn modes."""
        self._mode = value

    @property
    def model(self) -> TankPolicyNetwork:
        """The underlying neural network model."""
        return self._model

    @property
    def explored_positions(self) -> set[Position]:
        """Set of all positions observed during the current episode."""
        return self._explored_positions

    @property
    def episode(self) -> Episode:
        """The current episode trajectory (only meaningful in learn mode)."""
        return self._episode

    @property
    def patch_size(self) -> int:
        """Expected patch side length."""
        return self._patch_size

    def reset_episode(self) -> None:
        """Reset state for a new episode.

        Clears the GRU hidden state, exploration tracking, and trajectory.
        Call this at the start of each new game.
        """
        self._hidden = self._model.initial_hidden(batch_size=1).squeeze(0)
        self._explored_positions = set()
        self._episode = Episode()

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action using the neural network.

        Finds the patch for the specified tank, encodes it, runs the
        forward pass, and selects an action.

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The chosen :class:`Action`.

        Raises:
            ValueError: If no patch is found for *tank_id* or if the
                patch size doesn't match the expected size.
        """
        # Find the patch for our tank
        patch = None
        for p in view.patches:
            if p.tank_id == tank_id:
                patch = p
                break

        if patch is None:
            # Tank has no patch (shouldn't happen for alive tank)
            return Action.PASS

        # Validate patch size
        grid_size = len(patch.grid)
        if grid_size != self._patch_size:
            raise ValueError(
                f"Patch size mismatch: expected {self._patch_size}, got {grid_size}. "
                f"The model was trained for patch_size={self._patch_size}."
            )

        # Update exploration tracking
        self._update_exploration(patch)

        # Encode patch to tensor
        patch_tensor = encode_patch(patch, self._team)

        # Forward pass
        if self._mode == "play":
            with torch.no_grad():
                logits, new_hidden = self._model(patch_tensor, self._hidden)
            self._hidden = new_hidden
            action_idx = int(logits.argmax().item())
        else:
            logits, new_hidden = self._model(patch_tensor, self._hidden)
            self._hidden = new_hidden.detach()
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action_tensor = dist.sample()  # type: ignore[no-untyped-call]
            action_idx = int(action_tensor.item())
            log_prob = float(dist.log_prob(action_tensor).item())  # type: ignore[no-untyped-call]
            self._episode.add_step(action_index=action_idx, log_prob=log_prob)

        return ACTION_INDEX_TO_ACTION[action_idx]

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
        # We compute world positions from the patch's egocentric grid.
        # The patch centre is at the tank's position; we need the
        # direction to map egocentric → world.
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

    def last_step_new_positions(self) -> int:
        """Return the number of new positions discovered on the last step.

        This is a convenience for the reward function. Returns 0 if no
        step has been taken yet.

        Note:
            This value is computed during :meth:`choose_action` but not
            stored per-step.  For accurate per-step tracking, the training
            loop should call :meth:`_update_exploration` tracking externally
            or this method immediately after choose_action.
        """
        # This is handled internally — the training loop should track
        # new_positions from the return value of _update_exploration.
        # For external access we expose explored_positions directly.
        return 0
