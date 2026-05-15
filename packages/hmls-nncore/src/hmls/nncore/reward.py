"""Reward function for NN-controlled tank players.

Defines :class:`RewardFunction`, which orchestrates composable
:class:`~hmls.nncore.reward_components.RewardComponent` instances
using a :class:`~hmls.nncore.reward_config.RewardConfig`.

The :class:`RewardFunction` delegates per-episode state management to
its components via :meth:`~RewardFunction.reset` and
:meth:`~RewardFunction.observe_patch` lifecycle hooks.
"""

from __future__ import annotations

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action
from hmls.core.visibility import TankPatch
from hmls.nncore.reward_components import (
    ActionReward,
    ExplorationReward,
    FiringReward,
    GameStateStepReward,
    RewardComponent,
    RewardContext,
    SituationalReward,
)
from hmls.nncore.reward_config import RewardConfig

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
