"""REINFORCE policy gradient update logic.

Provides :func:`reinforce_update` which computes the policy gradient
loss from an episode's log-probabilities and discounted returns, and
performs a single optimizer step.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from hmls.nncore.trajectory import Episode, compute_returns


def reinforce_update(
    episode: Episode,
    optimizer: Optimizer,
    gamma: float = 0.99,
    log_prob_tensors: list[torch.Tensor] | None = None,
) -> float:
    """Perform a single REINFORCE update from one episode.

    Computes discounted returns, normalises them (baseline subtraction
    via mean/std), and backpropagates the policy gradient loss.

    Args:
        episode: The completed episode with rewards assigned.
        optimizer: The optimizer managing the model's parameters.
        gamma: Discount factor for return computation.
        log_prob_tensors: Tensor log-probabilities from the computation
            graph (retaining grad_fn).  If not provided, falls back to
            the float log_probs in the episode (no gradient flow).

    Returns:
        The scalar policy gradient loss value (for logging).
    """
    if len(episode) == 0:
        return 0.0

    rewards = episode.rewards()

    returns = compute_returns(rewards, gamma)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    # Normalise returns (baseline subtraction)
    if len(returns_tensor) > 1:
        std = returns_tensor.std()
        if std > 1e-8:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / std
        else:
            returns_tensor = returns_tensor - returns_tensor.mean()

    # Use tensor log_probs if available (enables gradient flow)
    if log_prob_tensors is not None and len(log_prob_tensors) == len(episode):
        log_probs_stacked = torch.stack(log_prob_tensors)
    else:
        log_probs_stacked = torch.tensor(episode.log_probs(), dtype=torch.float32)

    # Compute policy gradient loss: -sum(log_prob * return)
    loss = -(log_probs_stacked * returns_tensor.detach()).sum()

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    return float(loss.item())
