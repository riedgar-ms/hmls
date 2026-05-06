"""Reward functions for the NN player.

Defines a :class:`RewardFunction` abstract base class and a
:class:`DefaultReward` implementation that provides shaped rewards
including exploration bonuses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hmls.core.engine import HistoryEntry
from hmls.core.types import Position


class RewardFunction(ABC):
    """Base class for computing per-step and episode-end rewards.

    Subclasses receive game information and return scalar rewards
    that the training loop accumulates into the trajectory.
    """

    @abstractmethod
    def compute_step_reward(
        self,
        entry: HistoryEntry,
        explored_positions: set[Position],
        new_positions_this_step: int,
    ) -> float:
        """Compute the reward for a single action step.

        Args:
            entry: The history entry from the engine after the action.
            explored_positions: All positions the player has seen so far
                (including this step).
            new_positions_this_step: Number of newly discovered positions
                this step (cells not previously in explored_positions).

        Returns:
            A scalar reward value.
        """
        ...

    @abstractmethod
    def compute_episode_end_reward(
        self,
        won: bool | None,
        total_explored: int,
    ) -> float:
        """Compute the reward at the end of an episode.

        Args:
            won: ``True`` if the player's team won, ``False`` if they
                lost, ``None`` for a draw.
            total_explored: Total number of unique positions explored
                during the episode.

        Returns:
            A scalar reward value.
        """
        ...


class DefaultReward(RewardFunction):
    """Shaped reward function with exploration bonus.

    Provides immediate feedback for hits, deaths, exploration, and
    a terminal bonus/penalty for winning/losing.

    Args:
        hit_reward: Reward for hitting an enemy tank.
        death_penalty: Penalty when the player's tank dies.
        win_reward: Reward for winning the game.
        loss_penalty: Penalty for losing the game.
        step_penalty: Small per-step penalty to encourage faster play.
        exploration_bonus: Reward per newly discovered cell.
    """

    def __init__(
        self,
        hit_reward: float = 0.5,
        death_penalty: float = -1.0,
        win_reward: float = 1.0,
        loss_penalty: float = -1.0,
        step_penalty: float = -0.01,
        exploration_bonus: float = 0.02,
    ) -> None:
        self.hit_reward = hit_reward
        self.death_penalty = death_penalty
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.step_penalty = step_penalty
        self.exploration_bonus = exploration_bonus

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        explored_positions: set[Position],
        new_positions_this_step: int,
    ) -> float:
        """Compute shaped step reward.

        Rewards:
        - Hit an enemy: +hit_reward
        - Own tank died (invalid action resulted in penalty or got shot):
          not directly observable from a single step here, so we check
          if the action was invalid.
        - Exploration: +exploration_bonus per new cell
        - Time penalty: +step_penalty (negative value)
        """
        reward = self.step_penalty

        # Hit reward
        if entry.hit is True:
            reward += self.hit_reward

        # Exploration bonus
        reward += self.exploration_bonus * new_positions_this_step

        # Invalid action penalty (half of step penalty extra)
        if not entry.valid:
            reward += self.step_penalty

        return reward

    def compute_episode_end_reward(
        self,
        won: bool | None,
        total_explored: int,
    ) -> float:
        """Compute terminal reward based on game outcome."""
        if won is True:
            return self.win_reward
        elif won is False:
            return self.loss_penalty
        else:
            return 0.0
