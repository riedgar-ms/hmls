"""REINFORCE policy gradient updates for the squad architecture.

Provides update functions for both the executor (multi-tank accumulated
gradient) and planner (team-level trajectory).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from hmls.nncore.trajectory import Episode, compute_returns
from hmls.reinforcetrainer.updater import ReturnBaseline


def executor_update(
    episodes: dict[str, Episode],
    log_prob_tensors: dict[str, list[torch.Tensor]],
    entropy_tensors: dict[str, list[torch.Tensor]],
    optimizer: Optimizer,
    gamma: float,
    baseline: ReturnBaseline,
    entropy_coeff: float = 0.01,
    reduction: str = "sum",
    max_grad_norm: float | None = None,
) -> float:
    """Perform a single REINFORCE update for the shared executor.

    Accumulates the loss across all tanks (each is an independent
    trajectory from the same policy) and performs one optimizer step.

    Args:
        episodes: Per-tank episodes (keyed by tank_id).
        log_prob_tensors: Per-tank log-prob tensors (retain grad).
        entropy_tensors: Per-tank entropy tensors.
        optimizer: Executor optimizer.
        gamma: Discount factor.
        baseline: Shared cross-episode return baseline.
        entropy_coeff: Entropy bonus weight.
        reduction: ``"sum"`` or ``"mean"`` across time steps.
        max_grad_norm: Optional gradient clipping threshold.

    Returns:
        Total loss value (float).
    """
    total_loss = torch.tensor(0.0)
    total_steps = 0

    for tank_id, episode in episodes.items():
        if len(episode) == 0:
            continue
        tank_log_probs = log_prob_tensors.get(tank_id, [])
        tank_entropies = entropy_tensors.get(tank_id, [])
        if not tank_log_probs:
            continue

        rewards = episode.rewards()
        returns_list = compute_returns(rewards, gamma)
        returns = torch.tensor(returns_list, dtype=torch.float32)

        # Update baseline and compute advantages
        baseline.update(returns)
        if baseline.mean is not None and baseline.std is not None:
            advantages = (returns - baseline.mean) / baseline.std
        else:
            advantages = returns

        log_probs_stacked = torch.stack(tank_log_probs)
        entropies_stacked = torch.stack(tank_entropies)

        # Policy gradient loss
        pg_loss = -(log_probs_stacked * advantages.detach()).sum()
        entropy_bonus = -entropy_coeff * entropies_stacked.sum()
        total_loss = total_loss + pg_loss + entropy_bonus
        total_steps += len(episode)

    if total_steps == 0:
        return 0.0

    if reduction == "mean" and total_steps > 0:
        total_loss = total_loss / total_steps

    optimizer.zero_grad()
    total_loss.backward()  # type: ignore[no-untyped-call]
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            max_grad_norm,
        )
    optimizer.step()

    return float(total_loss.item())


def planner_update(
    episode: Episode,
    log_prob_tensors: list[torch.Tensor],
    entropy_tensors: list[torch.Tensor],
    optimizer: Optimizer,
    gamma: float,
    baseline: ReturnBaseline,
    entropy_coeff: float = 0.01,
    reduction: str = "sum",
    max_grad_norm: float | None = None,
) -> float:
    """Perform a single REINFORCE update for the planner.

    The planner has a single team-level trajectory where rewards
    are the aggregated (mean) per-step executor rewards across
    all alive tanks.

    Args:
        episode: Planner's team-level episode.
        log_prob_tensors: Planner log-prob tensors (retain grad).
        entropy_tensors: Planner entropy tensors.
        optimizer: Planner optimizer.
        gamma: Discount factor.
        baseline: Planner's cross-episode return baseline.
        entropy_coeff: Entropy bonus weight.
        reduction: ``"sum"`` or ``"mean"`` across time steps.
        max_grad_norm: Optional gradient clipping threshold.

    Returns:
        Loss value (float).
    """
    if len(episode) == 0 or not log_prob_tensors:
        return 0.0

    rewards = episode.rewards()
    returns_list = compute_returns(rewards, gamma)
    returns = torch.tensor(returns_list, dtype=torch.float32)

    baseline.update(returns)
    if baseline.mean is not None and baseline.std is not None:
        advantages = (returns - baseline.mean) / baseline.std
    else:
        advantages = returns

    log_probs_stacked = torch.stack(log_prob_tensors)
    entropies_stacked = torch.stack(entropy_tensors)

    pg_loss = -(log_probs_stacked * advantages.detach()).sum()
    entropy_bonus = -entropy_coeff * entropies_stacked.sum()
    loss = pg_loss + entropy_bonus

    if reduction == "mean" and len(episode) > 0:
        loss = loss / len(episode)

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            max_grad_norm,
        )
    optimizer.step()

    return float(loss.item())
