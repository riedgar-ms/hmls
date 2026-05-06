"""Reward functions for the NN player.

Defines a :class:`RewardFunction` abstract base class and a
:class:`DefaultReward` implementation that provides shaped rewards
including exploration bonuses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from hmls.core.engine import HistoryEntry
from hmls.core.types import Action, Position
from hmls.core.visibility import TankPatch, VisibleCell


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
        missed_fire_penalty: Penalty for not firing when an alive enemy
            tank is directly ahead and could have been hit.
        pass_penalty: Penalty for deliberately choosing to pass a turn
            (not applied when an invalid action is converted to pass).
        enemy_in_cone_reward: Per-enemy reward for each alive enemy tank
            visible in the forward cone of the egocentric patch.
    """

    hit_reward: float = 0.5
    death_penalty: float = -1.0
    win_reward: float = 1.0
    loss_penalty: float = -1.0
    step_penalty: float = -0.01
    exploration_bonus: float = 0.02
    invalid_move_penalty: float = -0.1
    fire_miss_penalty: float = -0.05
    missed_fire_penalty: float = -0.1
    pass_penalty: float = -0.02
    enemy_in_cone_reward: float = 0.01


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
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute the reward for a single action step.

        Args:
            entry: The history entry from the engine after the action.
            explored_positions: All positions the player has seen so far
                (including this step).
            new_positions_this_step: Number of newly discovered positions
                this step (cells not previously in explored_positions).
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

    @property
    def missed_fire_penalty(self) -> float:
        """Penalty for not firing when an enemy is directly ahead."""
        return self.config.missed_fire_penalty

    @property
    def pass_penalty(self) -> float:
        """Penalty for deliberately choosing to pass."""
        return self.config.pass_penalty

    @property
    def enemy_in_cone_reward(self) -> float:
        """Per-enemy reward for visible enemies in the forward cone."""
        return self.config.enemy_in_cone_reward

    def compute_step_reward(
        self,
        entry: HistoryEntry,
        explored_positions: set[Position],
        new_positions_this_step: int,
        patch: TankPatch,
        team: str,
    ) -> float:
        """Compute shaped step reward.

        Rewards:
        - Hit an enemy: +hit_reward
        - Fire and miss: +fire_miss_penalty (negative value)
        - Invalid action: +invalid_move_penalty (negative value)
        - Exploration: +exploration_bonus per new cell
        - Time penalty: +step_penalty (negative value)
        - Missed fire: +missed_fire_penalty when enemy directly ahead
          but action was not fire
        - Pass: +pass_penalty for deliberate pass actions
        - Enemies in cone: +enemy_in_cone_reward per visible enemy in
          the forward cone
        """
        reward = self.step_penalty

        # Fire outcome
        if entry.hit is True:
            reward += self.hit_reward
        elif entry.hit is False:
            reward += self.fire_miss_penalty
        elif _enemy_directly_ahead(patch, team):
            # hit is None means a non-fire action was taken;
            # penalize missing the opportunity when enemy directly ahead
            reward += self.missed_fire_penalty

        # Exploration bonus
        reward += self.exploration_bonus * new_positions_this_step

        # Invalid action penalty
        if not entry.valid:
            reward += self.invalid_move_penalty

        # Deliberate pass penalty
        if entry.requested_action == Action.PASS and entry.valid:
            reward += self.pass_penalty

        # Enemy in forward cone reward
        cone_enemies = _count_enemies_in_cone(patch, team)
        reward += self.enemy_in_cone_reward * cone_enemies

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
