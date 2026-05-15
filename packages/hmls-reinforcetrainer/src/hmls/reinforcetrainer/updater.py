"""REINFORCE policy gradient update logic.

Provides :func:`reinforce_update` which computes the policy gradient
loss from an episode's log-probabilities and discounted returns, and
performs a single optimizer step.

Also provides :class:`ReturnBaseline`, a cross-episode exponential
moving average baseline that replaces the naïve per-episode mean/std
normalisation used in earlier versions.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch.optim import Optimizer

from hmls.nncore.trajectory import Episode, compute_returns


class ReturnBaseline:
    """Cross-episode exponential moving average (EMA) baseline.

    Tracks a running mean of discounted returns across episodes and
    uses it to compute advantages.  This replaces the earlier approach
    of normalising returns **within** each episode (subtracting that
    episode's own mean and dividing by its std), which destroys the
    learning signal on degenerate episodes.

    **Why this matters — the spinning problem:**

    Without a cross-episode baseline, per-episode normalisation
    creates arbitrary positive/negative splits that don't reflect
    action quality.  Consider a 5-step episode where every action is
    TURN_LEFT with per-step reward −0.03 and a terminal loss of −1.0::

        Step rewards:  [-0.03, -0.03, -0.03, -0.03, -1.03]
        Returns:       [-1.130, -1.105, -1.080, -1.055, -1.030]
        Per-ep norm'd: [-1.35,  -0.68,   0.00,  +0.68,  +1.35]

    Steps 3–4 receive *positive* advantages, so REINFORCE actively
    reinforces the turn actions taken at those steps.  The net gradient
    approximately cancels — the NN learns nothing about spinning being
    bad.

    A cross-episode baseline avoids this: if the running average return
    is, say, −0.5, then all five returns are well below baseline and
    all spinning actions are uniformly discouraged.

    Args:
        alpha: EMA decay factor (0 < α < 1).  Higher values make the
            baseline adapt more slowly.  ``0.99`` is a reasonable
            default: the baseline averages over roughly the last 100
            episodes.
    """

    def __init__(self, alpha: float = 0.99) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")  # noqa: EM102
        self._alpha = alpha
        self._mean: float | None = None
        self._var: float | None = None

    @property
    def mean(self) -> float | None:
        """Current EMA of mean returns, or ``None`` before first update."""
        return self._mean

    @property
    def std(self) -> float | None:
        """Current EMA of std of returns, or ``None`` before first update."""
        if self._var is None:
            return None
        import math

        return math.sqrt(self._var if self._var > 1e-8 else 1e-8)

    def update(self, returns: torch.Tensor) -> None:
        """Update the running baseline with a new episode's returns.

        Args:
            returns: 1-D tensor of discounted returns from one episode.
        """
        ep_mean = float(returns.mean().item())
        ep_var = float(returns.var().item()) if len(returns) > 1 else 0.0

        if self._mean is None:
            # First episode: initialise directly instead of blending
            # with a (non-existent) previous estimate.
            self._mean = ep_mean
            self._var = ep_var
        else:
            self._mean = self._alpha * self._mean + (1.0 - self._alpha) * ep_mean
            assert self._var is not None  # set alongside _mean
            self._var = self._alpha * self._var + (1.0 - self._alpha) * ep_var

    def compute_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        """Subtract the running baseline and scale by running std.

        During warm-up (before the first :meth:`update`), falls back to
        per-episode mean subtraction (equivalent to the old behaviour)
        so that early episodes still get *some* gradient signal.

        Args:
            returns: 1-D tensor of discounted returns for one episode.

        Returns:
            Advantage tensor of the same shape.
        """
        if self._mean is None:
            # Warm-up: no cross-episode data yet — use per-episode mean
            # as a one-off fallback.  This is the old behaviour and is
            # only hit for the very first episode.
            advantages = returns - returns.mean()
        else:
            advantages = returns - self._mean

        # Scale by running std for stability.  If we don't have a
        # running variance yet (first episode), just use the raw
        # advantages.
        if self._var is not None and self._var > 1e-8:
            advantages = advantages / (self._var**0.5)

        return advantages


def reinforce_update(
    episode: Episode,
    optimizer: Optimizer,
    gamma: float = 0.99,
    log_prob_tensors: list[torch.Tensor] | None = None,
    entropy_tensors: list[torch.Tensor] | None = None,
    entropy_coeff: float = 0.0,
    baseline: ReturnBaseline | None = None,
    reduction: Literal["sum", "mean"] = "sum",
    max_grad_norm: float | None = None,
) -> float:
    """Perform a single REINFORCE update from one episode.

    Computes discounted returns, computes advantages using either a
    cross-episode :class:`ReturnBaseline` (preferred) or per-episode
    normalisation (legacy fallback), and backpropagates the policy
    gradient loss.

    When a :class:`ReturnBaseline` is provided, the baseline is
    updated with this episode's returns **after** computing advantages
    (so this episode's advantages are relative to previous episodes,
    not itself).

    An optional entropy bonus can be added to the loss to prevent
    policy collapse.  Without entropy regularisation, the policy can
    converge onto a narrow subset of actions (e.g. always turning)
    and never explore alternatives.  The entropy bonus encourages the
    policy to maintain a spread over the action space:

        loss = −Σ(log_prob × advantage) − β × Σ(entropy)

    where β is ``entropy_coeff``.  The negative sign means that
    *maximising* entropy is equivalent to *minimising* the loss.

    Args:
        episode: The completed episode with rewards assigned.
        optimizer: The optimizer managing the model's parameters.
        gamma: Discount factor for return computation.
        log_prob_tensors: Tensor log-probabilities from the computation
            graph (retaining grad_fn).  If not provided, falls back to
            the float log_probs in the episode (no gradient flow).
        entropy_tensors: Per-step entropy tensors from the action
            distributions.  Required when ``entropy_coeff > 0``.
        entropy_coeff: Weight of the entropy bonus in the loss.  Set
            to 0 (the default) to disable entropy regularisation.
        baseline: A :class:`ReturnBaseline` for cross-episode advantage
            computation.  If ``None``, falls back to per-episode
            mean/std normalisation (the old, less effective approach).
        reduction: How to aggregate the per-step loss across time
            steps.  ``"sum"`` (default) matches the original Williams
            (1992) REINFORCE derivation.  ``"mean"`` provides more
            stable per-step gradient magnitudes when episode lengths
            vary.
        max_grad_norm: If set, clip the total gradient norm to this
            value using ``torch.nn.utils.clip_grad_norm_`` after
            ``loss.backward()`` and before ``optimizer.step()``.
            ``None`` (default) disables clipping.

    Returns:
        The scalar policy gradient loss value (for logging).
    """
    if len(episode) == 0:
        return 0.0

    rewards = episode.rewards()

    returns = compute_returns(rewards, gamma)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)

    # ── Advantage computation ────────────────────────────────────
    #
    # Two strategies:
    #
    # 1. Cross-episode baseline (preferred): subtract a running mean
    #    of returns from previous episodes, so uniformly-bad episodes
    #    produce uniformly-negative advantages.  This is the standard
    #    "REINFORCE with baseline" from Sutton & Barto.
    #
    # 2. Per-episode normalisation (legacy fallback): subtract this
    #    episode's own mean and divide by its std.  This creates
    #    arbitrary ± splits on degenerate episodes (see the
    #    ReturnBaseline docstring for why this fails).
    #
    if baseline is not None:
        advantages = baseline.compute_advantages(returns_tensor)
        # Update the baseline *after* computing this episode's
        # advantages, so the episode is scored against prior data.
        baseline.update(returns_tensor)
    else:
        # Legacy per-episode normalisation
        advantages = returns_tensor
        if len(returns_tensor) > 1:
            std = returns_tensor.std()
            if std > 1e-8:
                advantages = (returns_tensor - returns_tensor.mean()) / std
            else:
                advantages = returns_tensor - returns_tensor.mean()

    # Use tensor log_probs if available (enables gradient flow)
    if log_prob_tensors is not None and len(log_prob_tensors) == len(episode):
        log_probs_stacked = torch.stack(log_prob_tensors)
    else:
        log_probs_stacked = torch.tensor(episode.log_probs(), dtype=torch.float32)

    # ── Policy gradient loss ─────────────────────────────────────
    #
    # Core REINFORCE: −Σ(log π(a|s) × advantage)
    #
    # The reduction controls whether the per-step terms are summed
    # or averaged.  "sum" matches the original Williams (1992)
    # derivation; "mean" normalises by episode length for more
    # stable gradients across varying episode lengths.
    #
    reduce = torch.Tensor.sum if reduction == "sum" else torch.Tensor.mean
    policy_loss = -reduce(log_probs_stacked * advantages.detach())

    # ── Entropy bonus ────────────────────────────────────────────
    #
    # Subtracting the entropy bonus from the loss encourages the
    # optimiser to *increase* entropy (i.e. spread probability mass
    # across actions), counteracting policy collapse.  Without this,
    # a policy that starts slightly favouring turns (2 of 5 actions)
    # has no force pushing it to explore other actions.
    #
    entropy_bonus = torch.tensor(0.0)
    if entropy_coeff > 0.0 and entropy_tensors is not None:
        if len(entropy_tensors) == len(episode):
            entropy_bonus = reduce(torch.stack(entropy_tensors))

    loss = policy_loss - entropy_coeff * entropy_bonus

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]

    # ── Gradient clipping ────────────────────────────────────────
    #
    # Clip the total gradient norm before the optimizer step to
    # prevent outlier episodes from destabilising training.  This
    # is especially important with sum-reduction, where long
    # episodes can produce very large gradients.
    #
    if max_grad_norm is not None:
        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

    optimizer.step()

    return float(loss.item())
