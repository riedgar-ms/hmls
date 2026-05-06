"""Trajectory storage for REINFORCE training.

Stores per-step data (action index, log-probability) during an episode,
and provides utilities for computing discounted returns once rewards are
assigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    """A single step in an episode trajectory.

    Attributes:
        action_index: Index of the chosen action (0–4).
        log_prob: Log-probability of the chosen action under the policy.
        reward: Reward received after this step (filled in after the
            engine processes the action).
    """

    action_index: int
    log_prob: float
    reward: float = 0.0


@dataclass
class Episode:
    """A complete episode trajectory with computed returns.

    Attributes:
        steps: Ordered list of trajectory steps from the episode.
    """

    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_step(self, action_index: int, log_prob: float) -> None:
        """Append a new step (reward to be filled later).

        Args:
            action_index: Index of the chosen action.
            log_prob: Log-probability under the current policy.
        """
        self.steps.append(TrajectoryStep(action_index=action_index, log_prob=log_prob))

    def set_reward(self, step_index: int, reward: float) -> None:
        """Set the reward for a specific step.

        Args:
            step_index: Index into :attr:`steps`.
            reward: The reward value to assign.

        Raises:
            IndexError: If *step_index* is out of range.
        """
        self.steps[step_index].reward = reward

    def rewards(self) -> list[float]:
        """Return the list of rewards in step order."""
        return [s.reward for s in self.steps]

    def log_probs(self) -> list[float]:
        """Return the list of log-probabilities in step order."""
        return [s.log_prob for s in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


def compute_returns(rewards: list[float], gamma: float = 0.99) -> list[float]:
    """Compute discounted cumulative returns for a reward sequence.

    The return at step *t* is defined as:
        G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

    Args:
        rewards: Per-step rewards in chronological order.
        gamma: Discount factor (0 < gamma ≤ 1).

    Returns:
        List of discounted returns, same length as *rewards*.
    """
    returns: list[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return returns
