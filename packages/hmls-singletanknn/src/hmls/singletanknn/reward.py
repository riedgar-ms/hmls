"""Reward functions for the NN player.

Defines a :class:`RewardFunction` abstract base class and a
:class:`DefaultReward` implementation that provides shaped rewards
including exploration bonuses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from hmls.core.engine import HistoryEntry
from hmls.core.types import Position


class DefaultRewardConfig(BaseModel, frozen=True):
    """Serialisable configuration for :class:`DefaultReward`.

    All fields have sensible defaults so ``DefaultRewardConfig()``
    produces a usable configuration out of the box.

    Attributes:
        hit_reward: Reward for hitting an enemy tank.
        death_penalty: Penalty when the player's tank dies.
        win_reward: Reward for winning the game.
        loss_penalty: Penalty for losing the game.
        step_penalty: Small per-step penalty to encourage faster play.
        exploration_bonus: Reward per newly discovered cell.
        invalid_move_penalty: Penalty for attempting an invalid action
            (applied in addition to the step penalty).
        fire_miss_penalty: Penalty for firing and missing (applied in
            addition to the step penalty).
    """

    hit_reward: float = 0.5
    death_penalty: float = -1.0
    win_reward: float = 1.0
    loss_penalty: float = -1.0
    step_penalty: float = -0.01
    exploration_bonus: float = 0.02
    invalid_move_penalty: float = -0.1
    fire_miss_penalty: float = -0.05


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

    Configuration is held in a :class:`DefaultRewardConfig` Pydantic model
    for easy serialisation and storage of training parameters.

    Args:
        config: A :class:`DefaultRewardConfig` instance. Uses defaults
            if not provided.
    """

    def __init__(self, config: DefaultRewardConfig | None = None) -> None:
        self.config: DefaultRewardConfig = config or DefaultRewardConfig()

    @property
    def hit_reward(self) -> float:
        """Reward for hitting an enemy tank."""
        return self.config.hit_reward

    @property
    def death_penalty(self) -> float:
        """Penalty when the player's tank dies."""
        return self.config.death_penalty

    @property
    def win_reward(self) -> float:
        """Reward for winning the game."""
        return self.config.win_reward

    @property
    def loss_penalty(self) -> float:
        """Penalty for losing the game."""
        return self.config.loss_penalty

    @property
    def step_penalty(self) -> float:
        """Small per-step penalty to encourage faster play."""
        return self.config.step_penalty

    @property
    def exploration_bonus(self) -> float:
        """Reward per newly discovered cell."""
        return self.config.exploration_bonus

    @property
    def invalid_move_penalty(self) -> float:
        """Penalty for attempting an invalid action."""
        return self.config.invalid_move_penalty

    @property
    def fire_miss_penalty(self) -> float:
        """Penalty for firing and missing."""
        return self.config.fire_miss_penalty

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        explored_positions: set[Position],
        new_positions_this_step: int,
    ) -> float:
        """Compute shaped step reward.

        Rewards:
        - Hit an enemy: +hit_reward
        - Fire and miss: +fire_miss_penalty (negative value)
        - Invalid action: +invalid_move_penalty (negative value)
        - Exploration: +exploration_bonus per new cell
        - Time penalty: +step_penalty (negative value)
        """
        reward = self.step_penalty

        # Hit reward
        if entry.hit is True:
            reward += self.hit_reward
        # Fire miss penalty
        elif entry.hit is False:
            reward += self.fire_miss_penalty

        # Exploration bonus
        reward += self.exploration_bonus * new_positions_this_step

        # Invalid action penalty
        if not entry.valid:
            reward += self.invalid_move_penalty

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
