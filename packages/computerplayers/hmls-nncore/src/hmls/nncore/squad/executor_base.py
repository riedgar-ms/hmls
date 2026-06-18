"""Abstract base class for executor models in squad architectures.

Defines :class:`ExecutorModelConfig` and :class:`ExecutorModelBase`,
the foundational abstractions for executor neural networks that translate
a planner's high-level order plus a local egocentric patch into
low-level tank actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from hmls.nncore.squad.orders import NUM_ORDERS


class ExecutorModelConfig(BaseModel, frozen=True, extra="forbid"):
    """Base configuration shared by all executor models.

    Every concrete executor config must extend this class so that
    generic infrastructure can validate common settings.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Identifier for persistence registry resolution.
        num_orders: Number of discrete orders the executor accepts.
            Must match the planner's output vocabulary size.
    """

    patch_size: int = Field(default=9, ge=3)
    model_id: str
    num_orders: int = Field(default=NUM_ORDERS, ge=1)


class ExecutorModelBase(ABC, nn.Module):
    """Abstract base class for squad executor neural networks.

    Executor models receive an encoded egocentric patch, a discrete
    order from the planner, and a recurrent hidden state.  They produce
    action logits over the 5 low-level actions and an updated hidden
    state.

    Subclasses must:
    - Store a ``config`` attribute of an :class:`ExecutorModelConfig` subclass.
    - Implement :meth:`forward` with the specified signature.
    - Implement :meth:`initial_hidden` to provide episode-start states.
    """

    config: ExecutorModelConfig

    @abstractmethod
    def forward(
        self,
        patch_tensor: torch.Tensor,
        order: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patch + order → (action_logits, new_hidden).

        Args:
            patch_tensor: Encoded patch tensor of shape
                ``[batch, channels, patch_size, patch_size]`` or
                ``[channels, patch_size, patch_size]`` (unbatched).
            order: Order index tensor of shape ``[batch]`` or scalar
                (unbatched).  Values in ``[0, num_orders)``.
            hidden: Hidden state tensor whose shape is defined by the
                concrete model.

        Returns:
            A tuple of ``(logits, new_hidden)`` where logits has shape
            ``[batch, NUM_ACTIONS]`` (or ``[NUM_ACTIONS]`` if unbatched)
            and ``new_hidden`` has the same shape as the input hidden.
        """
        ...

    @abstractmethod
    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return the initial hidden state for the start of an episode.

        Args:
            batch_size: Number of parallel episodes (default 1).

        Returns:
            A tensor of shape ``[batch_size, hidden_dim]`` filled with
            the model's preferred initial values (typically zeros).
        """
        ...

    @property
    @abstractmethod
    def total_hidden_size(self) -> int:
        """Total dimensionality of the hidden state tensor."""
        ...
