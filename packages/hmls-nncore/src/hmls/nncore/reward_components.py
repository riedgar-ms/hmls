"""Composable reward components for NN-controlled tank players.

Defines the :class:`RewardComponent` abstract base class and concrete
implementations for each reward category: game state, firing, exploration,
actions, and situational awareness.

Each component is independently testable and owns any per-episode state
it requires (e.g. streak counters, exploration sets).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action, Position
from hmls.core.visibility import TankPatch, VisibleCell, ego_to_world_position
from hmls.nncore.reward import RewardConfig

# ── Context ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RewardContext:
    """Pre-computed per-step state shared by all reward components.

    Built by :class:`RewardFunction` before invoking components so that
    expensive shared computations (e.g. determining whether the action
    is a "meaningful reset") happen exactly once.

    Attributes:
        entry: The history entry for the current step.
        patch: The egocentric visibility patch for the acting tank.
        team: The team of the acting player.
        config: The full reward configuration.
        is_meaningful_reset: Whether this step should reset escalating
            streaks (a fire hit or a valid forward move).
    """

    entry: HistoryEntry
    patch: TankPatch
    team: str
    config: RewardConfig
    is_meaningful_reset: bool


# ── ABC ───────────────────────────────────────────────────────────────


class RewardComponent(ABC):
    """Abstract base class for a single reward component.

    Subclasses implement :meth:`compute` to return their contribution
    to the total step reward.  Stateful components override :meth:`reset`
    to clear per-episode state.
    """

    @abstractmethod
    def compute(self, context: RewardContext) -> float:
        """Compute this component's reward contribution for one step.

        Args:
            context: Pre-computed per-step state.

        Returns:
            The reward value (may be positive, negative, or zero).
        """

    def reset(self) -> None:
        """Reset per-episode state.  Default: no-op."""


# ── Concrete components ───────────────────────────────────────────────


class GameStateStepReward(RewardComponent):
    """Per-step time cost and invalid-move penalty.

    Stateless — reads only from :attr:`RewardContext.config`.
    """

    def compute(self, context: RewardContext) -> float:
        """Return per-step cost plus penalty for invalid actions."""
        reward = context.config.game_state.step
        if not context.entry.valid:
            reward += context.config.game_state.invalid_move
        return reward


class FiringReward(RewardComponent):
    """Firing-outcome rewards: hit, miss, neglect, and miss streaks.

    Owns the consecutive-miss streak counter per tank.
    """

    def __init__(self) -> None:
        self._miss_streaks: dict[str, int] = {}

    def reset(self) -> None:
        """Clear miss streak counters."""
        self._miss_streaks = {}

    def compute(self, context: RewardContext) -> float:
        """Return firing reward based on hit/miss/neglect and streak."""
        cfg = context.config.firing
        entry = context.entry
        tank_id = entry.tank_id
        reward = 0.0

        if entry.hit is True:
            reward += cfg.hit
        elif entry.hit is False:
            reward += cfg.miss
        elif _enemy_directly_ahead(context.patch, context.team):
            reward += cfg.neglect

        # Consecutive miss streak
        if entry.hit is False:
            streak = self._miss_streaks.get(tank_id, 0) + 1
            self._miss_streaks[tank_id] = streak
            reward += cfg.consecutive_miss * streak
        elif context.is_meaningful_reset:
            self._miss_streaks[tank_id] = 0

        return reward


class ExplorationReward(RewardComponent):
    """Exploration bonuses for newly seen and newly occupied cells.

    Owns the sets of seen/occupied positions and must receive the
    patch via :meth:`observe_patch` before :meth:`compute` is called.
    """

    def __init__(self) -> None:
        self._seen_positions: set[Position] = set()
        self._occupied_positions: set[Position] = set()
        self._last_new_seen: int = 0
        self._last_new_occupied: int = 0

    def reset(self) -> None:
        """Clear exploration state for a new episode."""
        self._seen_positions = set()
        self._occupied_positions = set()
        self._last_new_seen = 0
        self._last_new_occupied = 0

    def observe_patch(self, patch: TankPatch) -> None:
        """Update exploration state from the observed visibility patch.

        Must be called before :meth:`compute` for the same step.

        Args:
            patch: The egocentric visibility patch.
        """
        new_seen = 0
        for ego_row, row in enumerate(patch.grid):
            for ego_col, cell in enumerate(row):
                if isinstance(cell, VisibleCell):
                    pos = ego_to_world_position(patch, ego_row, ego_col)
                    if pos not in self._seen_positions:
                        self._seen_positions.add(pos)
                        new_seen += 1

        self._last_new_seen = new_seen

        # Track occupied cells (movement-based exploration)
        tank_pos = patch.position
        if tank_pos not in self._occupied_positions:
            self._occupied_positions.add(tank_pos)
            self._last_new_occupied = 1
        else:
            self._last_new_occupied = 0

    def compute(self, context: RewardContext) -> float:
        """Return exploration bonuses based on newly seen/occupied cells."""
        cfg = context.config.exploration
        return cfg.see_cell * self._last_new_seen + cfg.occupy_cell * self._last_new_occupied


class ActionReward(RewardComponent):
    """Per-action rewards and escalating turn/pass streak penalties.

    Owns the consecutive-turn and consecutive-pass streak counters.
    """

    _TURN_ACTIONS: frozenset[Action] = frozenset({Action.TURN_LEFT, Action.TURN_RIGHT})

    def __init__(self) -> None:
        self._turn_streaks: dict[str, int] = {}
        self._pass_streaks: dict[str, int] = {}

    def reset(self) -> None:
        """Clear turn and pass streak counters."""
        self._turn_streaks = {}
        self._pass_streaks = {}

    def compute(self, context: RewardContext) -> float:
        """Return per-action reward plus escalating streak penalties."""
        cfg = context.config.actions
        entry = context.entry
        tank_id = entry.tank_id
        reward = 0.0

        # Deliberate pass
        if entry.requested_action == Action.PASS and entry.valid:
            reward += cfg.pass_action

        # Per-action rewards
        if entry.valid:
            action_rewards = {
                Action.TURN_LEFT: cfg.turn_left,
                Action.TURN_RIGHT: cfg.turn_right,
                Action.MOVE_FORWARD: cfg.move_forward,
                Action.FIRE: cfg.fire,
            }
            reward += action_rewards.get(entry.requested_action, 0.0)

        # Escalating consecutive-turn streak
        if entry.requested_action in self._TURN_ACTIONS and entry.valid:
            streak = self._turn_streaks.get(tank_id, 0) + 1
            self._turn_streaks[tank_id] = streak
            reward += cfg.consecutive_turn * streak
        elif context.is_meaningful_reset:
            self._turn_streaks[tank_id] = 0

        # Escalating consecutive-pass streak
        if entry.requested_action == Action.PASS and entry.valid:
            streak = self._pass_streaks.get(tank_id, 0) + 1
            self._pass_streaks[tank_id] = streak
            reward += cfg.consecutive_pass * streak
        elif context.is_meaningful_reset:
            self._pass_streaks[tank_id] = 0

        return reward


class SituationalReward(RewardComponent):
    """Reward for alive enemy tanks visible in the forward cone.

    Stateless — delegates to :func:`_score_enemies_in_cone`.
    """

    def compute(self, context: RewardContext) -> float:
        """Return distance-weighted score of enemies in forward cone."""
        cfg = context.config.situational
        cone_score = _score_enemies_in_cone(
            context.patch,
            context.team,
            cfg.enemy_in_cone_distance_discount,
        )
        return cfg.enemy_in_cone * cone_score


# ── Helper functions ──────────────────────────────────────────────────


def _enemy_directly_ahead(patch: TankPatch, team: str) -> bool:
    """Check whether an alive enemy tank occupies the cell directly ahead.

    The cell directly ahead in egocentric coordinates is at
    ``grid[half - 1][half]`` (one row above the centre, same column).

    Args:
        patch: The egocentric visibility patch.
        team: The observing player's team.

    Returns:
        ``True`` if an alive enemy occupies the cell directly ahead.
    """
    half = len(patch.grid) // 2
    ahead_row = half - 1
    ahead_col = half
    cell = patch.grid[ahead_row][ahead_col]
    if isinstance(cell, VisibleCell) and cell.tank is not None:
        return cell.tank.alive and cell.tank.team != team
    return False


def _score_enemies_in_cone(
    patch: TankPatch,
    team: str,
    distance_discount: float,
) -> float:
    """Score alive enemy tanks visible in the forward cone.

    The forward cone comprises all visible cells in rows above the
    patch centre (``ego_row < half``).  Each enemy's contribution is
    ``distance_discount ** manhattan_distance``, where the Manhattan
    distance is computed in egocentric grid coordinates from the
    player's centre cell.

    Args:
        patch: The egocentric visibility patch.
        team: The observing player's team.
        distance_discount: Per-unit-distance discount factor.

    Returns:
        Distance-discounted score of alive enemy tanks in the cone.
    """
    half = len(patch.grid) // 2
    score = 0.0
    for ego_row in range(half):
        for ego_col, cell in enumerate(patch.grid[ego_row]):
            if isinstance(cell, VisibleCell) and cell.tank is not None:
                if cell.tank.alive and cell.tank.team != team:
                    manhattan = (half - ego_row) + abs(ego_col - half)
                    score += distance_discount**manhattan
    return score
