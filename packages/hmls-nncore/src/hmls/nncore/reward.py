"""Reward functions for NN-controlled tank players.

Defines :class:`RewardConfig`, a nested Pydantic configuration model
with sections for actions, firing, game state, exploration, and
situational rewards, and :class:`RewardFunction`, which orchestrates
composable :class:`~hmls.nncore.reward_components.RewardComponent`
instances.

The :class:`RewardFunction` delegates per-episode state management to
its components via :meth:`~RewardFunction.reset` and
:meth:`~RewardFunction.observe_patch` lifecycle hooks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action
from hmls.core.visibility import TankPatch

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
        consecutive_pass: Escalating reward multiplier for consecutive
            valid pass actions (typically negative).  When a tank takes
            N consecutive deliberate passes, the Nth pass incurs an
            additional reward of ``consecutive_pass × N``.

            The streak resets to 0 on a fire that hits or a valid move
            forward.  Other actions leave the streak unchanged but do
            not incur the escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    move_forward: float = 0.0
    turn_left: float = 0.0
    turn_right: float = 0.0
    fire: float = 0.0
    pass_action: float = -0.02
    consecutive_turn: float = 0.0
    consecutive_pass: float = 0.0


class FiringRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Firing-outcome reward configuration.

    Attributes:
        hit: Reward for hitting an enemy tank.
        miss: Reward (negative) for firing and missing.
        neglect: Reward (negative) for not firing when an alive enemy
            tank is directly ahead and could have been hit.
        consecutive_miss: Escalating reward multiplier for consecutive
            fire misses (typically negative).  When a tank fires and
            misses N consecutive times, the Nth miss incurs an
            additional reward of ``consecutive_miss × N``.

            The streak resets to 0 on a fire that hits or a valid move
            forward.  Other actions leave the streak unchanged but do
            not incur the escalating penalty.

            Set to ``0.0`` (the default) to disable.
    """

    hit: float = 0.5
    miss: float = -0.05
    neglect: float = -0.1
    consecutive_miss: float = 0.0


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
        enemy_in_cone_distance_discount: Discount factor applied
            per unit of Manhattan distance from the player to the
            enemy in egocentric coordinates.  Each enemy's
            contribution is ``enemy_in_cone *
            enemy_in_cone_distance_discount ** manhattan_distance``.
            A value of ``1.0`` (the default) disables discounting.
            Values below ``1.0`` make distant enemies worth less.
    """

    enemy_in_cone: float = 0.01
    enemy_in_cone_distance_discount: float = 1.0


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
    """Shaped reward function composed of independent reward components.

    Orchestrates :class:`~hmls.nncore.reward_components.RewardComponent`
    instances, building a shared
    :class:`~hmls.nncore.reward_components.RewardContext` for each step
    and summing their contributions.

    Args:
        config: A :class:`RewardConfig` instance.  Uses defaults if
            not provided.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        from hmls.nncore.reward_components import (
            ActionReward,
            ExplorationReward,
            FiringReward,
            GameStateStepReward,
            RewardComponent,
            SituationalReward,
        )

        self.config: RewardConfig = config or RewardConfig()
        self._exploration = ExplorationReward()
        self._components: list[RewardComponent] = [
            GameStateStepReward(),
            FiringReward(),
            self._exploration,
            ActionReward(),
            SituationalReward(),
        ]

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        for component in self._components:
            component.reset()

    def observe_patch(self, patch: TankPatch) -> None:
        """Update exploration state from the observed visibility patch.

        Must be called before :meth:`compute_step_reward` for the same
        step so that exploration bonuses reflect newly seen cells.

        Args:
            patch: The egocentric visibility patch.
        """
        self._exploration.observe_patch(patch)

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute the shaped step reward by summing all components.

        Args:
            entry: The history entry for the current step.
            patch: The egocentric visibility patch.
            team: The team of the acting player.

        Returns:
            Total step reward (sum of all component contributions).
        """
        from hmls.nncore.reward_components import RewardContext

        is_meaningful_reset = entry.hit is True or (
            entry.requested_action == Action.MOVE_FORWARD and entry.valid
        )

        context = RewardContext(
            entry=entry,
            patch=patch,
            team=team,
            config=self.config,
            is_meaningful_reset=is_meaningful_reset,
        )

        return sum(component.compute(context) for component in self._components)

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
