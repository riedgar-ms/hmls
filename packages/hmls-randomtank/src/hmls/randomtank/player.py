"""Rule-based player for the random tank.

:class:`RandomTankPlayer` inspects the egocentric visibility patch and
selects actions using simple probabilistic rules, without any neural
network inference.
"""

from __future__ import annotations

import random
from typing import Literal

import torch

from hmls.core.map import CellType
from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import (
    BoundaryCell,
    FogCell,
    PlayerView,
    TankPatch,
    VisibleCell,
)
from hmls.nncore.constants import ACTION_TO_INDEX
from hmls.nncore.player import NNPlayerBase
from hmls.randomtank.model import RandomTankModel, RandomTankModelConfig


class RandomTankPlayer(NNPlayerBase):
    """Rule-based tank player with configurable probabilities.

    Overrides :meth:`choose_action` to apply simple rules based on the
    egocentric visibility patch, bypassing the neural-network forward
    pass entirely.

    Decision logic (evaluated in order):

    1. **Alive enemy in front** → :attr:`~hmls.core.types.Action.FIRE`.
    2. **Cell in front is blocked** (boundary, impassable terrain, or
       occupied by any tank — alive or destroyed, any team) → turn
       left with probability ``config.prob_turn_left_on_blocked``,
       otherwise turn right.
    3. **Cell in front is passable** (passable terrain, no tank) →
       move forward with probability ``config.prob_forward_on_passable``,
       turn left with probability ``config.prob_turn_left_on_passable``,
       turn right with the remaining probability.

    Args:
        team: The team this player controls.
        model: The :class:`RandomTankModel` (used only for config access).
        mode: Operating mode (``"play"`` or ``"learn"``).  In either
            mode the same rule-based logic is used; ``"learn"`` mode
            additionally records trajectory data (though it will never
            be used for gradient updates).
        rng: Optional :class:`random.Random` instance for reproducible
            action selection.  If ``None``, a default instance is used.
    """

    def __init__(
        self,
        team: str,
        model: RandomTankModel,
        mode: Literal["play", "learn"] = "play",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(team, mode=mode)
        self._model = model
        self._rng = rng or random.Random()
        self._config: RandomTankModelConfig = model.config

    @property
    def model(self) -> RandomTankModel:
        """The underlying random tank model."""
        return self._model

    @property
    def patch_size(self) -> int:
        """Expected patch side length (from model config)."""
        return self._config.patch_size

    # ── Action selection (overrides NNPlayerBase) ─────────────────────

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action using rule-based logic.

        Inspects the cell directly in front of the tank in the
        egocentric patch and applies the probabilistic rules described
        in the class docstring.

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The chosen :class:`Action`.
        """
        # Find our tank's patch
        patch: TankPatch | None = None
        for p in view.patches:
            if p.tank_id == tank_id:
                patch = p
                break

        if patch is None:
            return Action.PASS

        self._last_patch = patch

        half = len(patch.grid) // 2
        front_cell = patch.grid[half - 1][half]

        # Determine the state of the cell in front
        action = self._decide_action(front_cell)

        # Record trajectory step in learn mode (with dummy log_prob)
        if self._mode == "learn":
            action_idx = ACTION_TO_INDEX[action]
            self._episode.add_step(action_index=action_idx, log_prob=0.0)
            # Provide dummy tensors for the training loop (never used
            # since train=False, but keeps the types consistent)
            self._log_prob_tensors.append(torch.tensor(0.0))
            self._entropy_tensors.append(torch.tensor(0.0))

        return action

    def _decide_action(self, front_cell: VisibleCell | FogCell | BoundaryCell) -> Action:
        """Apply the decision rules to the cell directly in front.

        Args:
            front_cell: The patch cell one step ahead of the tank.

        Returns:
            The chosen action.
        """
        # Boundary → always blocked
        if isinstance(front_cell, BoundaryCell):
            return self._turn_blocked()

        # Fog → treat as passable (adjacent cells should always be
        # visible, but handle defensively)
        if isinstance(front_cell, FogCell):
            return self._choose_passable()

        # VisibleCell — check contents
        assert isinstance(front_cell, VisibleCell)

        # Alive enemy → fire
        if front_cell.tank is not None and front_cell.tank.alive:
            if front_cell.tank.team != self._team:
                return Action.FIRE

        # Any tank (alive friendly or destroyed wreckage) → blocked
        if front_cell.tank is not None:
            return self._turn_blocked()

        # Impassable terrain → blocked
        if front_cell.cell_type == CellType.IMPASSABLE:
            return self._turn_blocked()

        # Passable and clear
        return self._choose_passable()

    def _turn_blocked(self) -> Action:
        """Choose a turn direction when the cell ahead is blocked.

        Returns:
            :attr:`Action.TURN_LEFT` with probability
            ``prob_turn_left_on_blocked``, otherwise
            :attr:`Action.TURN_RIGHT`.
        """
        if self._rng.random() < self._config.prob_turn_left_on_blocked:
            return Action.TURN_LEFT
        return Action.TURN_RIGHT

    def _choose_passable(self) -> Action:
        """Choose an action when the cell ahead is passable.

        Returns:
            :attr:`Action.MOVE_FORWARD` with probability
            ``prob_forward_on_passable``, :attr:`Action.TURN_LEFT` with
            probability ``prob_turn_left_on_passable``, otherwise
            :attr:`Action.TURN_RIGHT`.
        """
        roll = self._rng.random()
        if roll < self._config.prob_forward_on_passable:
            return Action.MOVE_FORWARD
        if roll < self._config.prob_forward_on_passable + self._config.prob_turn_left_on_passable:
            return Action.TURN_LEFT
        return Action.TURN_RIGHT

    # ── NNPlayerBase abstract method stubs ────────────────────────────
    #
    # These are never called because choose_action is overridden, but
    # they must exist to satisfy the abstract base class contract.

    def _forward_play(self, patch: TankPatch) -> int:
        """Not used — choose_action is overridden."""
        msg = "RandomTankPlayer does not use _forward_play"
        raise NotImplementedError(msg)

    def _forward_learn(self, patch: TankPatch) -> tuple[int, float, torch.Tensor, torch.Tensor]:
        """Not used — choose_action is overridden."""
        msg = "RandomTankPlayer does not use _forward_learn"
        raise NotImplementedError(msg)

    def _reset_model_state(self) -> None:
        """No model state to reset."""
