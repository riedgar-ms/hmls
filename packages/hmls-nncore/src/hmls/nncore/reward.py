"""Reward functions for NN-controlled tank players.

Defines :class:`RewardConfig`, a nested Pydantic configuration model
with sections for actions, firing, game state, exploration, and
situational rewards, and :class:`RewardFunction`, the single concrete
reward implementation that uses it.

The :class:`RewardFunction` tracks per-episode state (explored cells,
consecutive turn streaks) via :meth:`~RewardFunction.reset` and
:meth:`~RewardFunction.observe_patch` lifecycle hooks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action, Position
from hmls.core.visibility import TankPatch, VisibleCell

# ── Nested reward config sections ────────────────────────────────────


class ActionsRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Per-action reward configuration.

    Attributes:
        move_forward: Reward for choosing to move forward.
        turn_left: Reward for choosing to turn left.
        turn_right: Reward for choosing to turn right.
        fire: Reward for choosing to fire (independent of hit/miss,
            which are in :class:`FiringRewardConfig`).
        pass_action: Reward for deliberately choosing to pass a turn
            (not applied when an invalid action is converted to pass).
        consecutive_turn: Escalating reward multiplier for consecutive
            turn actions (typically negative).  When a tank takes N
            consecutive turns (``TURN_LEFT`` or ``TURN_RIGHT``), the
            Nth turn incurs an additional reward of
            ``consecutive_turn × N``.

            The streak resets to 0 only on *meaningful* non-turn
            actions: a fire that hits or a valid move forward.  Other
            actions leave the streak unchanged but do not incur the
            escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    move_forward: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    fire: float = 0.0
    pass_action: float = -0.02
    consecutive_turn: float = 0.0


class FiringRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Firing-outcome reward configuration.

    Attributes:
        hit: Reward for hitting an enemy tank.
        miss: Reward (negative) for firing and missing.
        neglect: Reward (negative) for not firing when an alive enemy
            tank is directly ahead and could have been hit.
    """

    hit: float = 0.5
    miss: float = -0.05
    neglect: float = -0.1


class GameStateRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Game-state reward configuration.

    Attributes:
        win: Reward for winning the game.
        loss: Reward (negative) for losing the game.
        invalid_move: Reward (negative) for attempting an invalid action.
        step: Per-step reward (negative to encourage faster play).
        death: Reward (negative) when the player's tank dies.
    """

    win: float = 1.0
    loss: float = -1.0
    invalid_move: float = -0.1
    step: float = -0.01
    death: float = -1.0


class ExplorationRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Exploration reward configuration.

    Attributes:
        see_cell: Reward per newly *seen* cell in the visibility patch
            (cells visible but not necessarily stepped on).
        occupy_cell: Reward per newly *occupied* cell (cells the tank
            physically moves onto for the first time).
    """

    see_cell: float = 0.02
    occupy_cell: float = 0.0


class SituationalRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Situational reward configuration.

    Attributes:
        enemy_in_cone: Per-enemy reward for each alive enemy tank
            visible in the forward cone of the egocentric patch.
    """

    enemy_in_cone: float = 0.01


class RewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Top-level reward configuration with nested sections.

    All sections have sensible defaults so ``RewardConfig()`` produces
    a usable configuration out of the box.

    Attributes:
        actions: Per-action rewards (move, turn, fire, pass,
            consecutive turn penalty).
        firing: Firing-outcome rewards (hit, miss, neglect).
        game_state: Game-state rewards (win, loss, invalid move,
            per-step cost, death).
        exploration: Exploration rewards (see cell, occupy cell).
        situational: Situational rewards (enemy in cone).
    """

    actions: ActionsRewardConfig = Field(default_factory=ActionsRewardConfig)
    firing: FiringRewardConfig = Field(default_factory=FiringRewardConfig)
    game_state: GameStateRewardConfig = Field(default_factory=GameStateRewardConfig)
    exploration: ExplorationRewardConfig = Field(default_factory=ExplorationRewardConfig)
    situational: SituationalRewardConfig = Field(default_factory=SituationalRewardConfig)


# ── Reward function ──────────────────────────────────────────────────


class RewardFunction:
    """Shaped reward function with exploration bonus and streak tracking.

    Computes per-step and episode-end rewards based on a
    :class:`RewardConfig`.  Internally tracks which grid cells have
    been seen (for ``see_cell`` rewards) and which have been physically
    occupied (for ``occupy_cell`` rewards), as well as consecutive
    turn streaks for escalating penalties.

    Args:
        config: A :class:`RewardConfig` instance.  Uses defaults if
            not provided.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config: RewardConfig = config or RewardConfig()
        self._seen_positions: set[Position] = set()
        self._occupied_positions: set[Position] = set()
        self._last_new_seen: int = 0
        self._last_new_occupied: int = 0
        self._turn_streaks: dict[str, int] = {}

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._seen_positions = set()
        self._occupied_positions = set()
        self._last_new_seen = 0
        self._last_new_occupied = 0
        self._turn_streaks = {}

    def observe_patch(self, patch: TankPatch) -> None:
        """Update exploration state from the observed visibility patch.

        Records newly seen cells (all visible cells in the patch) and
        newly occupied cells (the tank's current position).

        Args:
            patch: The egocentric visibility patch.
        """
        # Track seen cells (visibility-based exploration)
        half = len(patch.grid) // 2
        forward = patch.direction.forward_delta()
        right = patch.direction.turn_right().forward_delta()
        fx, fy = forward
        rx, ry = right

        new_seen = 0
        for ego_row, row in enumerate(patch.grid):
            for ego_col, cell in enumerate(row):
                if isinstance(cell, VisibleCell):
                    fwd_steps = half - ego_row
                    rgt_steps = ego_col - half
                    world_x = patch.position.x + fwd_steps * fx + rgt_steps * rx
                    world_y = patch.position.y + fwd_steps * fy + rgt_steps * ry
                    pos = Position(world_x, world_y)
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

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute the shaped step reward.

        Components (all added to the total):

        - ``game_state.step``: per-step time cost
        - ``firing.hit / miss / neglect``: firing outcome
        - ``game_state.invalid_move``: for invalid actions
        - ``actions.pass_action``: for deliberate pass
        - ``actions.turn_left / turn_right / move_forward / fire``:
          per-action rewards
        - ``actions.consecutive_turn``: escalating streak penalty
        - ``exploration.see_cell``: per newly seen cell
        - ``exploration.occupy_cell``: per newly occupied cell
        - ``situational.enemy_in_cone``: per visible enemy in cone
        """
        cfg = self.config
        reward = cfg.game_state.step

        # Fire outcome
        if entry.hit is True:
            reward += cfg.firing.hit
        elif entry.hit is False:
            reward += cfg.firing.miss
        elif _enemy_directly_ahead(patch, team):
            reward += cfg.firing.neglect

        # Exploration bonuses
        reward += cfg.exploration.see_cell * self._last_new_seen
        reward += cfg.exploration.occupy_cell * self._last_new_occupied

        # Invalid action
        if not entry.valid:
            reward += cfg.game_state.invalid_move

        # Deliberate pass
        if entry.requested_action == Action.PASS and entry.valid:
            reward += cfg.actions.pass_action

        # Per-action rewards
        if entry.requested_action == Action.TURN_LEFT and entry.valid:
            reward += cfg.actions.turn_left
        elif entry.requested_action == Action.TURN_RIGHT and entry.valid:
            reward += cfg.actions.turn_right
        elif entry.requested_action == Action.MOVE_FORWARD and entry.valid:
            reward += cfg.actions.move_forward
        elif entry.requested_action == Action.FIRE and entry.valid:
            reward += cfg.actions.fire

        # Escalating consecutive-turn reward
        _TURN_ACTIONS = frozenset({Action.TURN_LEFT, Action.TURN_RIGHT})
        tank_id = entry.tank_id
        if entry.requested_action in _TURN_ACTIONS and entry.valid:
            streak = self._turn_streaks.get(tank_id, 0) + 1
            self._turn_streaks[tank_id] = streak
            reward += cfg.actions.consecutive_turn * streak
        elif entry.hit is True or (entry.requested_action == Action.MOVE_FORWARD and entry.valid):
            self._turn_streaks[tank_id] = 0

        # Enemy in forward cone
        cone_enemies = _count_enemies_in_cone(patch, team)
        reward += cfg.situational.enemy_in_cone * cone_enemies

        return reward

    def compute_episode_end_reward(
        self,
        won: bool | None,
    ) -> float:
        """Compute terminal reward based on game outcome."""
        if won is True:
            return self.config.game_state.win
        elif won is False:
            return self.config.game_state.loss
        else:
            return 0.0


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


def _count_enemies_in_cone(patch: TankPatch, team: str) -> int:
    """Count alive enemy tanks visible in the forward cone.

    The forward cone comprises all visible cells in rows above the
    patch centre (``ego_row < half``).  Fog cells are naturally
    excluded because they are :class:`FogCell`, not :class:`VisibleCell`.

    Args:
        patch: The egocentric visibility patch.
        team: The observing player's team.

    Returns:
        Number of alive enemy tanks visible in the forward cone.
    """
    half = len(patch.grid) // 2
    count = 0
    for ego_row in range(half):
        for cell in patch.grid[ego_row]:
            if isinstance(cell, VisibleCell) and cell.tank is not None:
                if cell.tank.alive and cell.tank.team != team:
                    count += 1
    return count
