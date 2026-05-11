"""Reward functions for NN-controlled tank players.

Defines a :class:`RewardFunction` abstract base class and a
:class:`BasicReward` implementation that provides shaped rewards
including exploration bonuses.

Reward classes that need to track state across an episode (e.g.
explored grid cells) should do so internally, using the
:meth:`~RewardFunction.reset` and :meth:`~RewardFunction.observe_patch`
lifecycle hooks.

New reward types are added by:

1. Defining a new config model (subclass of :class:`~pydantic.BaseModel`)
   with a ``reward_type`` :class:`~typing.Literal` field set to a
   unique tag string.
2. Implementing a corresponding :class:`RewardFunction` subclass.
3. Adding the config to the :data:`RewardConfig` discriminated union.
4. Registering the mapping in :func:`create_reward`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Discriminator, Tag

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action, Position
from hmls.core.visibility import TankPatch, VisibleCell


class BasicRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Serialisable configuration for :class:`BasicReward`.

    All fields have sensible defaults so ``BasicRewardConfig()``
    produces a usable configuration out of the box.

    All values are rewards: positive values reinforce behaviour,
    negative values discourage it.

    Attributes:
        reward_type: Discriminator tag — always ``"basic"`` for this
            config variant.
        fire_hit_reward: Reward for hitting an enemy tank.
        death_reward: Reward (negative) when the player's tank dies.
        win_reward: Reward for winning the game.
        loss_reward: Reward (negative) for losing the game.
        step_reward: Per-step reward (negative to encourage faster play).
        exploration_reward: Reward per newly discovered cell.
        invalid_move_reward: Reward (negative) for attempting an invalid
            action (applied in addition to the step reward).
        fire_miss_reward: Reward (negative) for firing and missing
            (applied in addition to the step reward).
        fire_neglect_reward: Reward (negative) for not firing when an
            alive enemy tank is directly ahead and could have been hit.
        pass_reward: Reward (negative) for deliberately choosing to pass
            a turn (not applied when an invalid action is converted to
            pass).
        enemy_in_cone_reward: Per-enemy reward for each alive enemy tank
            visible in the forward cone of the egocentric patch.
        turn_left_reward: Reward for choosing to turn left.
        turn_right_reward: Reward for choosing to turn right.
        move_forward_reward: Reward for choosing to move forward.
        consecutive_turn_reward: Escalating reward multiplier for
            consecutive turn actions (typically negative).  When a tank
            takes N consecutive turns (``TURN_LEFT`` or ``TURN_RIGHT``),
            the Nth turn incurs an additional reward of
            ``consecutive_turn_reward × N``.  For example, with
            ``consecutive_turn_reward = -0.02`` and
            ``turn_left_reward = -0.02``:

            - Turn 1 (streak=1): ``-0.02 + (-0.02 × 1) = -0.04``
            - Turn 2 (streak=2): ``-0.02 + (-0.02 × 2) = -0.06``
            - Turn 3 (streak=3): ``-0.02 + (-0.02 × 3) = -0.08``

            The streak resets to 0 only on *meaningful* non-turn
            actions: a fire that hits (``entry.hit is True``) or a
            valid move forward.  Other actions — pass, fire-miss,
            invalid move — leave the streak unchanged but do not
            incur the escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    reward_type: Literal["basic"] = "basic"
    fire_hit_reward: float = 0.5
    death_reward: float = -1.0
    win_reward: float = 1.0
    loss_reward: float = -1.0
    step_reward: float = -0.01
    exploration_reward: float = 0.02
    invalid_move_reward: float = -0.1
    fire_miss_reward: float = -0.05
    fire_neglect_reward: float = -0.1
    pass_reward: float = -0.02
    enemy_in_cone_reward: float = 0.01
    turn_left_reward: float = 0.0
    turn_right_reward: float = 0.0
    move_forward_reward: float = 0.0
    consecutive_turn_reward: float = 0.0


class RewardFunction(ABC):
    """Base class for computing per-step and episode-end rewards.

    Subclasses receive game information and return scalar rewards
    that the training loop accumulates into the trajectory.

    Reward functions that need to track state across an episode
    (such as explored positions) should override :meth:`reset` and
    :meth:`observe_patch`.
    """

    def reset(self) -> None:
        """Reset internal state for a new episode.

        Called at the start of each episode before any steps are taken.
        The default implementation is a no-op; override in subclasses
        that maintain per-episode state.
        """

    def observe_patch(self, patch: TankPatch) -> None:
        """Observe a visibility patch for internal bookkeeping.

        Called each step *before* :meth:`compute_step_reward`, giving
        the reward function an opportunity to update internal state
        (e.g. exploration tracking) based on what was seen.

        The default implementation is a no-op.

        Args:
            patch: The egocentric visibility patch the player saw
                before choosing the action.
        """

    @abstractmethod
    def compute_step_reward(
        self,
        entry: HistoryEntry,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute the reward for a single action step.

        Args:
            entry: The history entry from the engine after the action.
            patch: The egocentric visibility patch the player saw before
                choosing the action.
            team: The team the player belongs to (used to distinguish
                friendly vs enemy tanks in the patch).

        Returns:
            A scalar reward value.
        """
        ...

    @abstractmethod
    def compute_episode_end_reward(
        self,
        won: bool | None,
    ) -> float:
        """Compute the reward at the end of an episode.

        Args:
            won: ``True`` if the player's team won, ``False`` if they
                lost, ``None`` for a draw.

        Returns:
            A scalar reward value.
        """
        ...


class BasicReward(RewardFunction):
    """Shaped reward function with exploration bonus.

    Provides immediate feedback for hits, deaths, exploration, and
    a terminal reward for winning/losing.  Internally tracks which
    grid cells have been seen to compute exploration bonuses.

    Configuration is held in a :class:`BasicRewardConfig` Pydantic model
    for easy serialisation and storage of training parameters.

    Args:
        config: A :class:`BasicRewardConfig` instance. Uses defaults
            if not provided.
    """

    def __init__(self, config: BasicRewardConfig | None = None) -> None:
        self.config: BasicRewardConfig = config or BasicRewardConfig()
        self._explored_positions: set[Position] = set()
        self._last_new_positions: int = 0
        # Per-tank consecutive-turn streak for escalating reward.
        # See BasicRewardConfig.consecutive_turn_reward for details.
        self._turn_streaks: dict[str, int] = {}

    @property
    def fire_hit_reward(self) -> float:
        """Reward for hitting an enemy tank."""
        return self.config.fire_hit_reward

    @property
    def death_reward(self) -> float:
        """Reward (negative) when the player's tank dies."""
        return self.config.death_reward

    @property
    def win_reward(self) -> float:
        """Reward for winning the game."""
        return self.config.win_reward

    @property
    def loss_reward(self) -> float:
        """Reward (negative) for losing the game."""
        return self.config.loss_reward

    @property
    def step_reward(self) -> float:
        """Per-step reward (negative to encourage faster play)."""
        return self.config.step_reward

    @property
    def exploration_reward(self) -> float:
        """Reward per newly discovered cell."""
        return self.config.exploration_reward

    @property
    def invalid_move_reward(self) -> float:
        """Reward (negative) for attempting an invalid action."""
        return self.config.invalid_move_reward

    @property
    def fire_miss_reward(self) -> float:
        """Reward (negative) for firing and missing."""
        return self.config.fire_miss_reward

    @property
    def fire_neglect_reward(self) -> float:
        """Reward (negative) for not firing when an enemy is directly ahead."""
        return self.config.fire_neglect_reward

    @property
    def pass_reward(self) -> float:
        """Reward (negative) for deliberately choosing to pass."""
        return self.config.pass_reward

    @property
    def enemy_in_cone_reward(self) -> float:
        """Per-enemy reward for visible enemies in the forward cone."""
        return self.config.enemy_in_cone_reward

    @property
    def turn_left_reward(self) -> float:
        """Reward for choosing to turn left."""
        return self.config.turn_left_reward

    @property
    def turn_right_reward(self) -> float:
        """Reward for choosing to turn right."""
        return self.config.turn_right_reward

    @property
    def move_forward_reward(self) -> float:
        """Reward for choosing to move forward."""
        return self.config.move_forward_reward

    @property
    def consecutive_turn_reward(self) -> float:
        """Escalating per-turn reward multiplier for consecutive turns."""
        return self.config.consecutive_turn_reward

    @property
    def explored_positions(self) -> set[Position]:
        """Set of all positions observed during the current episode."""
        return self._explored_positions

    @property
    def total_explored(self) -> int:
        """Total number of unique positions explored this episode."""
        return len(self._explored_positions)

    def reset(self) -> None:
        """Reset exploration and streak tracking for a new episode."""
        self._explored_positions = set()
        self._last_new_positions = 0
        self._turn_streaks = {}

    def observe_patch(self, patch: TankPatch) -> None:
        """Update exploration state from the observed visibility patch.

        Computes world positions for all visible cells in the
        egocentric patch and records newly discovered ones.

        Args:
            patch: The egocentric visibility patch.
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

        self._last_new_positions = new_count

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute shaped step reward.

        Components (all added to the total):
        - fire_hit_reward: for hitting an enemy
        - fire_miss_reward: for firing and missing (negative)
        - invalid_move_reward: for an invalid action (negative)
        - exploration_reward: per newly discovered cell
        - step_reward: per-step time cost (negative)
        - fire_neglect_reward: when enemy directly ahead but action was
          not fire (negative)
        - pass_reward: for deliberate pass actions (negative)
        - turn_left_reward: for turning left
        - turn_right_reward: for turning right
        - move_forward_reward: for moving forward
        - enemy_in_cone_reward: per visible enemy in the forward cone
        - consecutive_turn_reward: escalating reward for consecutive
          turns (reward × streak count; see config docstring)
        """
        reward = self.step_reward

        # Fire outcome
        if entry.hit is True:
            reward += self.fire_hit_reward
        elif entry.hit is False:
            reward += self.fire_miss_reward
        elif _enemy_directly_ahead(patch, team):
            # hit is None means a non-fire action was taken;
            # penalize missing the opportunity when enemy directly ahead
            reward += self.fire_neglect_reward

        # Exploration bonus
        reward += self.exploration_reward * self._last_new_positions

        # Invalid action reward
        if not entry.valid:
            reward += self.invalid_move_reward

        # Deliberate pass reward
        if entry.requested_action == Action.PASS and entry.valid:
            reward += self.pass_reward

        # Movement action rewards
        if entry.requested_action == Action.TURN_LEFT and entry.valid:
            reward += self.turn_left_reward
        elif entry.requested_action == Action.TURN_RIGHT and entry.valid:
            reward += self.turn_right_reward
        elif entry.requested_action == Action.MOVE_FORWARD and entry.valid:
            reward += self.move_forward_reward

        # ── Escalating consecutive-turn reward ───────────────────
        #
        # Tracks a per-tank streak of consecutive turn actions.  Each
        # successive turn incurs reward × streak_count, creating a
        # progressively stronger signal to stop spinning.
        #
        # The streak only resets on *meaningful* non-turn actions:
        # fire-and-hit or a valid move forward.  Other actions (pass,
        # fire-miss, invalid move) leave the streak unchanged — they
        # don't prove the tank has stopped spinning, but they also
        # don't receive the escalating penalty.
        #
        _TURN_ACTIONS = frozenset({Action.TURN_LEFT, Action.TURN_RIGHT})
        tank_id = entry.tank_id
        if entry.requested_action in _TURN_ACTIONS and entry.valid:
            streak = self._turn_streaks.get(tank_id, 0) + 1
            self._turn_streaks[tank_id] = streak
            reward += self.consecutive_turn_reward * streak
        elif entry.hit is True or (entry.requested_action == Action.MOVE_FORWARD and entry.valid):
            self._turn_streaks[tank_id] = 0

        # Enemy in forward cone reward
        cone_enemies = _count_enemies_in_cone(patch, team)
        reward += self.enemy_in_cone_reward * cone_enemies

        return reward

    def compute_episode_end_reward(
        self,
        won: bool | None,
    ) -> float:
        """Compute terminal reward based on game outcome."""
        if won is True:
            return self.win_reward
        elif won is False:
            return self.loss_reward
        else:
            return 0.0


# ── Focused reward ────────────────────────────────────────────────────


class FocusedRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Configuration for :class:`FocusedReward`.

    A streamlined reward that keeps core combat and validity signals,
    uses movement-based exploration (only cells the tank physically
    moves onto count), and adds a liveliness reward based on position
    diversity in a sliding window of recent steps.

    Attributes:
        reward_type: Discriminator tag — always ``"focused"``.
        fire_hit_reward: Reward for hitting an enemy tank.
        death_reward: Reward (negative) when the player's tank dies.
        win_reward: Reward for winning the game.
        loss_reward: Reward (negative) for losing the game.
        step_reward: Per-step reward (negative to encourage faster play).
        invalid_move_reward: Reward (negative) for attempting an invalid
            action (applied in addition to the step reward).
        fire_miss_reward: Reward (negative) for firing and missing.
        fire_neglect_reward: Reward (negative) for not firing when an
            alive enemy tank is directly ahead and could have been hit.
        exploration_reward: Reward per newly visited cell.  Only cells
            the tank actually moves onto count (not merely seen cells).
        liveliness_window: Number of recent steps over which to measure
            position diversity.
        liveliness_target: Target number of unique positions within the
            liveliness window.
        liveliness_reward: Scaling factor for the liveliness signal.
            The per-step liveliness reward is
            ``liveliness_reward × (unique_positions - liveliness_target)``,
            which is positive when the tank exceeds the target and
            negative when it falls short.
    """

    reward_type: Literal["focused"] = "focused"
    fire_hit_reward: float = 0.5
    death_reward: float = -1.0
    win_reward: float = 1.0
    loss_reward: float = -1.0
    step_reward: float = -0.01
    invalid_move_reward: float = -0.05
    fire_miss_reward: float = -0.1
    fire_neglect_reward: float = -0.7
    exploration_reward: float = 0.05
    liveliness_window: int = 10
    liveliness_target: int = 5
    liveliness_reward: float = 0.1


class FocusedReward(RewardFunction):
    """Streamlined reward with movement-based exploration and liveliness.

    Compared to :class:`BasicReward`, this removes per-action rewards
    (pass, turn, move-forward, enemy-in-cone, consecutive-turn-penalty)
    and changes exploration to only count cells physically visited by
    the tank.  It adds a *liveliness* signal that encourages positional
    diversity over a sliding window of recent steps.

    Args:
        config: A :class:`FocusedRewardConfig` instance.
    """

    def __init__(self, config: FocusedRewardConfig | None = None) -> None:
        self.config: FocusedRewardConfig = config or FocusedRewardConfig()
        self._explored_positions: set[Position] = set()
        self._last_new_positions: int = 0
        self._position_window: deque[Position] = deque(maxlen=self.config.liveliness_window)

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._explored_positions = set()
        self._last_new_positions = 0
        self._position_window = deque(maxlen=self.config.liveliness_window)

    def observe_patch(self, patch: TankPatch) -> None:
        """Record the tank's position for exploration and liveliness.

        Exploration credit is given only when the tank's current
        position (from the patch) is new.  The position is also
        appended to the liveliness sliding window every step.

        Args:
            patch: The egocentric visibility patch.
        """
        pos = patch.position
        self._position_window.append(pos)

        if pos not in self._explored_positions:
            self._explored_positions.add(pos)
            self._last_new_positions = 1
        else:
            self._last_new_positions = 0

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute the focused step reward.

        Components (all added to the total):

        - ``step_reward``: per-step time cost
        - ``fire_hit_reward``: for hitting an enemy
        - ``fire_miss_reward``: for firing and missing
        - ``fire_neglect_reward``: when enemy directly ahead but action
          was not fire
        - ``invalid_move_reward``: for an invalid action
        - ``exploration_reward``: per newly visited cell (movement only)
        - liveliness reward once the window is full
        """
        reward = self.config.step_reward

        # Fire outcome
        if entry.hit is True:
            reward += self.config.fire_hit_reward
        elif entry.hit is False:
            reward += self.config.fire_miss_reward
        elif _enemy_directly_ahead(patch, team):
            reward += self.config.fire_neglect_reward

        # Invalid action
        if not entry.valid:
            reward += self.config.invalid_move_reward

        # Exploration bonus (movement-only)
        reward += self.config.exploration_reward * self._last_new_positions

        # Liveliness reward (only once window is full)
        if len(self._position_window) == self._position_window.maxlen:
            unique = len(set(self._position_window))
            reward += self.config.liveliness_reward * (unique - self.config.liveliness_target)

        return reward

    def compute_episode_end_reward(self, won: bool | None) -> float:
        """Compute terminal reward based on game outcome."""
        if won is True:
            return self.config.win_reward
        elif won is False:
            return self.config.loss_reward
        else:
            return 0.0


# ── Discriminated union & factory ─────────────────────────────────────

RewardConfig = Annotated[
    Union[
        Annotated[BasicRewardConfig, Tag("basic")],
        Annotated[FocusedRewardConfig, Tag("focused")],
    ],
    Discriminator("reward_type"),
]
"""Discriminated union of all reward config types.

The ``reward_type`` field in the JSON selects which concrete config
class to deserialise into.  To add a new reward type, define its
config model with a unique ``reward_type: Literal[...]`` field and
add it to this union.
"""


def create_reward(config: RewardConfig) -> RewardFunction:
    """Instantiate a :class:`RewardFunction` from a config union member.

    Maps each concrete config type to its corresponding reward class.

    Args:
        config: A member of the :data:`RewardConfig` discriminated union.

    Returns:
        A :class:`RewardFunction` instance configured by *config*.

    Raises:
        TypeError: If *config* is not a recognised reward config type.
    """
    if isinstance(config, BasicRewardConfig):
        return BasicReward(config)
    if isinstance(config, FocusedRewardConfig):
        return FocusedReward(config)
    raise TypeError(f"Unknown reward config type: {type(config).__name__}")


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
