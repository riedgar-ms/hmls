"""Composite squad player: planner + executor orchestration.

:class:`SimpleSquadPlayer` is a :class:`~hmls.core.player.Player`
implementation that uses a hierarchical architecture:

1. A **planner** observes all alive friendly tanks and assigns orders
2. Per-tank **executors** translate orders + local patches into actions

The planner runs once per planning round (when the first tank on the
team acts in a turn).  Subsequent tanks in the same round reuse the
cached orders.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.distributions import Categorical

from hmls.core.player import Player
from hmls.core.tank import TankId
from hmls.core.types import Action, Direction
from hmls.core.visibility import PlayerView, TankPatch
from hmls.nncore.constants import ACTION_INDEX_TO_ACTION
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.squad.orders import Order
from hmls.nncore.trajectory import Episode
from hmls.simplesquadexecutor.model import SimpleExecutorModel
from hmls.simplesquadplanner.model import SimplePlannerModel

# Direction → index for one-hot encoding
_DIRECTION_INDEX: dict[Direction, int] = {
    Direction.NORTH: 0,
    Direction.EAST: 1,
    Direction.SOUTH: 2,
    Direction.WEST: 3,
}
NUM_DIRECTIONS = 4


class SimpleSquadPlayer(Player):
    """Composite player using hierarchical planner + executor architecture.

    Orchestrates a planner model (assigns orders to alive tanks) and
    a shared executor model (translates orders into low-level actions).
    Maintains per-tank hidden states and trajectories for training.

    Args:
        team: The team this player controls.
        planner: The planner model instance.
        executor: The executor model instance.
        mode: ``"play"`` for deterministic inference, ``"learn"`` for
            stochastic sampling with trajectory recording.
        map_width: Width of the game map (for position normalisation).
        map_height: Height of the game map (for position normalisation).
    """

    def __init__(
        self,
        team: str,
        planner: SimplePlannerModel,
        executor: SimpleExecutorModel,
        mode: Literal["play", "learn"] = "play",
        map_width: int = 1,
        map_height: int = 1,
    ) -> None:
        super().__init__(team)
        self._planner = planner
        self._executor = executor
        self._mode = mode
        self._map_width = max(map_width, 1)
        self._map_height = max(map_height, 1)

        # Per-tank state
        self._hidden_states: dict[TankId, torch.Tensor] = {}
        self._current_orders: dict[TankId, Order] = {}
        self._planning_done_this_round = False

        # Training trajectories (per-tank for executor)
        self._episodes: dict[TankId, Episode] = {}
        self._log_prob_tensors: dict[TankId, list[torch.Tensor]] = {}
        self._entropy_tensors: dict[TankId, list[torch.Tensor]] = {}

        # Planner trajectory (team-level)
        self._planner_episode = Episode()
        self._planner_log_prob_tensors: list[torch.Tensor] = []
        self._planner_entropy_tensors: list[torch.Tensor] = []

        # Track last patches for reward computation
        self._last_patches: dict[TankId, TankPatch] = {}

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
    def planner(self) -> SimplePlannerModel:
        """The planner model."""
        return self._planner

    @property
    def executor(self) -> SimpleExecutorModel:
        """The executor model."""
        return self._executor

    @property
    def episodes(self) -> dict[TankId, Episode]:
        """Per-tank executor episodes (for training)."""
        return self._episodes

    @property
    def log_prob_tensors(self) -> dict[TankId, list[torch.Tensor]]:
        """Per-tank executor log-prob tensors (retain grad)."""
        return self._log_prob_tensors

    @property
    def entropy_tensors(self) -> dict[TankId, list[torch.Tensor]]:
        """Per-tank executor entropy tensors."""
        return self._entropy_tensors

    @property
    def planner_episode(self) -> Episode:
        """Team-level planner episode (for training)."""
        return self._planner_episode

    @property
    def planner_log_prob_tensors(self) -> list[torch.Tensor]:
        """Planner log-prob tensors (retain grad)."""
        return self._planner_log_prob_tensors

    @property
    def planner_entropy_tensors(self) -> list[torch.Tensor]:
        """Planner entropy tensors."""
        return self._planner_entropy_tensors

    @property
    def current_orders(self) -> dict[TankId, Order]:
        """Currently assigned orders per tank."""
        return self._current_orders

    def last_patch(self, tank_id: TankId) -> TankPatch | None:
        """Get the last patch seen by a specific tank."""
        return self._last_patches.get(tank_id)

    @property
    def patch_size(self) -> int:
        """Expected patch side length."""
        return self._executor.config.patch_size

    # ── Episode lifecycle ─────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset all state for a new episode.

        Clears per-tank hidden states, trajectories, orders, and
        planner state.
        """
        self._hidden_states.clear()
        self._current_orders.clear()
        self._planning_done_this_round = False
        self._episodes.clear()
        self._log_prob_tensors.clear()
        self._entropy_tensors.clear()
        self._planner_episode = Episode()
        self._planner_log_prob_tensors = []
        self._planner_entropy_tensors = []
        self._last_patches.clear()

    def begin_round(self) -> None:
        """Signal the start of a new planning round.

        Must be called before the first ``choose_action`` call in each
        turn so the planner runs fresh for the new round.
        """
        self._planning_done_this_round = False

    # ── Action selection ──────────────────────────────────────────────

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action for the specified tank.

        On the first call per round, runs the planner to assign orders.
        Then runs the executor with the assigned order.

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The chosen :class:`Action`.
        """
        # Find this tank's patch
        patch: TankPatch | None = None
        for p in view.patches:
            if p.tank_id == tank_id:
                patch = p
                break

        if patch is None:
            return Action.PASS

        self._last_patches[tank_id] = patch

        # Run planner if not yet done this round
        if not self._planning_done_this_round:
            self._run_planner(view)
            self._planning_done_this_round = True

        # Get assigned order (default to ADVANCE if not assigned)
        order = self._current_orders.get(tank_id, Order.ADVANCE)

        # Run executor
        if self._mode == "play":
            action_idx = self._executor_forward_play(tank_id, patch, order)
        else:
            action_idx = self._executor_forward_learn(tank_id, patch, order)

        return ACTION_INDEX_TO_ACTION[action_idx]

    # ── Planner ───────────────────────────────────────────────────────

    def _run_planner(self, view: PlayerView) -> None:
        """Run the planner to assign orders to all alive tanks.

        Args:
            view: The current player view.
        """
        alive_patches = list(view.patches)
        if not alive_patches:
            return

        # Encode patches
        patch_tensors = torch.stack(
            [FiveChannelPatchEncoder.encode_patch(p, self._team) for p in alive_patches]
        )

        # Encode positions (normalised)
        positions = torch.tensor(
            [
                [p.position[0] / self._map_width, p.position[1] / self._map_height]
                for p in alive_patches
            ],
            dtype=torch.float32,
        )

        # Encode directions (one-hot)
        directions = torch.zeros(len(alive_patches), NUM_DIRECTIONS, dtype=torch.float32)
        for i, p in enumerate(alive_patches):
            dir_idx = _DIRECTION_INDEX.get(p.direction, 0)
            directions[i, dir_idx] = 1.0

        if self._mode == "play":
            self._planner_forward_play(alive_patches, patch_tensors, positions, directions)
        else:
            self._planner_forward_learn(alive_patches, patch_tensors, positions, directions)

    def _planner_forward_play(
        self,
        alive_patches: list[TankPatch],
        patch_tensors: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> None:
        """Deterministic planner: argmax per-tank order."""
        with torch.no_grad():
            order_logits = self._planner(patch_tensors, positions, directions)

        for i, p in enumerate(alive_patches):
            order_idx = int(order_logits[i].argmax().item())
            self._current_orders[p.tank_id] = Order(order_idx)

    def _planner_forward_learn(
        self,
        alive_patches: list[TankPatch],
        patch_tensors: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> None:
        """Stochastic planner: sample orders, record log-probs."""
        order_logits = self._planner(patch_tensors, positions, directions)

        total_log_prob = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)

        for i, p in enumerate(alive_patches):
            probs = torch.softmax(order_logits[i], dim=-1)
            dist = Categorical(probs)
            order_tensor = dist.sample()  # type: ignore[no-untyped-call]
            order_idx = int(order_tensor.item())
            self._current_orders[p.tank_id] = Order(order_idx)

            log_prob: torch.Tensor = dist.log_prob(order_tensor)  # type: ignore[no-untyped-call]
            entropy: torch.Tensor = dist.entropy()  # type: ignore[no-untyped-call]
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy

        # Record team-level planner step
        avg_log_prob = total_log_prob / len(alive_patches)
        avg_entropy = total_entropy / len(alive_patches)
        self._planner_log_prob_tensors.append(avg_log_prob)
        self._planner_entropy_tensors.append(avg_entropy)
        self._planner_episode.add_step(
            action_index=0,  # placeholder (planner actions are orders, not indexed here)
            log_prob=float(avg_log_prob.item()),
        )

    # ── Executor ──────────────────────────────────────────────────────

    def _ensure_tank_state(self, tank_id: TankId) -> None:
        """Initialise per-tank state if this is the first call for this tank."""
        if tank_id not in self._hidden_states:
            self._hidden_states[tank_id] = self._executor.initial_hidden(batch_size=1).squeeze(0)
            self._episodes[tank_id] = Episode()
            self._log_prob_tensors[tank_id] = []
            self._entropy_tensors[tank_id] = []

    def _executor_forward_play(self, tank_id: TankId, patch: TankPatch, order: Order) -> int:
        """Deterministic executor: argmax over action logits."""
        self._ensure_tank_state(tank_id)
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        order_tensor = torch.tensor(int(order), dtype=torch.long)
        hidden = self._hidden_states[tank_id]

        with torch.no_grad():
            logits, new_hidden = self._executor(patch_tensor, order_tensor, hidden)

        self._hidden_states[tank_id] = new_hidden
        return int(logits.argmax().item())

    def _executor_forward_learn(self, tank_id: TankId, patch: TankPatch, order: Order) -> int:
        """Stochastic executor: sample action, record log-prob."""
        self._ensure_tank_state(tank_id)
        patch_tensor = FiveChannelPatchEncoder.encode_patch(patch, self._team)
        order_tensor = torch.tensor(int(order), dtype=torch.long)
        hidden = self._hidden_states[tank_id]

        logits, new_hidden = self._executor(patch_tensor, order_tensor, hidden)
        self._hidden_states[tank_id] = new_hidden.detach()

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action_tensor = dist.sample()  # type: ignore[no-untyped-call]
        action_idx = int(action_tensor.item())
        log_prob_tensor: torch.Tensor = dist.log_prob(action_tensor)  # type: ignore[no-untyped-call]
        entropy_tensor: torch.Tensor = dist.entropy()  # type: ignore[no-untyped-call]

        # Record in this tank's trajectory
        self._log_prob_tensors[tank_id].append(log_prob_tensor)
        self._entropy_tensors[tank_id].append(entropy_tensor)
        self._episodes[tank_id].add_step(
            action_index=action_idx,
            log_prob=float(log_prob_tensor.item()),
        )

        return action_idx
