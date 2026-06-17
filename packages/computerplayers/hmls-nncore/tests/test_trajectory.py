"""Tests for the trajectory module."""

from __future__ import annotations

from hmls.nncore.trajectory import Episode, TrajectoryStep, compute_returns


def test_trajectory_step_default_reward() -> None:
    """TrajectoryStep defaults reward to 0."""
    step = TrajectoryStep(action_index=2, log_prob=-0.5)
    assert step.reward == 0.0


def test_episode_add_and_set_reward() -> None:
    """Episode correctly adds steps and sets rewards."""
    ep = Episode()
    ep.add_step(action_index=0, log_prob=-1.0)
    ep.add_step(action_index=3, log_prob=-0.2)
    assert len(ep) == 2

    ep.set_reward(0, 0.5)
    ep.set_reward(1, 1.0)
    assert ep.rewards() == [0.5, 1.0]
    assert ep.log_probs() == [-1.0, -0.2]


def test_compute_returns_single_step() -> None:
    """Single step: return equals the reward."""
    returns = compute_returns([1.0], gamma=0.99)
    assert len(returns) == 1
    assert abs(returns[0] - 1.0) < 1e-7


def test_compute_returns_multi_step() -> None:
    """Multi-step discounted returns are correct."""
    rewards = [1.0, 2.0, 3.0]
    gamma = 0.9
    # G2 = 3.0
    # G1 = 2.0 + 0.9 * 3.0 = 4.7
    # G0 = 1.0 + 0.9 * 4.7 = 5.23
    returns = compute_returns(rewards, gamma=gamma)
    assert abs(returns[2] - 3.0) < 1e-7
    assert abs(returns[1] - 4.7) < 1e-7
    assert abs(returns[0] - 5.23) < 1e-7


def test_compute_returns_gamma_zero() -> None:
    """With gamma=0, each return equals the immediate reward."""
    rewards = [1.0, 2.0, 3.0]
    returns = compute_returns(rewards, gamma=0.0)
    assert returns == [1.0, 2.0, 3.0]


def test_compute_returns_gamma_one() -> None:
    """With gamma=1, return is sum of all future rewards."""
    rewards = [1.0, 1.0, 1.0]
    returns = compute_returns(rewards, gamma=1.0)
    assert abs(returns[0] - 3.0) < 1e-7
    assert abs(returns[1] - 2.0) < 1e-7
    assert abs(returns[2] - 1.0) < 1e-7
