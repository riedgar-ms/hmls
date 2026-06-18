"""Abstract base class for planner models in squad architectures.

Defines :class:`PlannerModelConfig` and :class:`PlannerModelBase`,
the foundational abstractions for planner neural networks that observe
all friendly tanks and produce per-tank order assignments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from hmls.nncore.squad.orders import NUM_ORDERS


class PlannerModelConfig(BaseModel, frozen=True, extra="forbid"):
    """Base configuration shared by all planner models.

    Every concrete planner config must extend this class so that
    generic infrastructure can validate common settings.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Identifier for persistence registry resolution.
        num_orders: Number of discrete orders the planner can issue.
        max_tanks: Maximum number of tanks per team the planner
            supports.  The model must handle any count from 1 to
            this value.
    """

    patch_size: int = Field(default=9, ge=3)
    model_id: str
    num_orders: int = Field(default=NUM_ORDERS, ge=1)
    max_tanks: int = Field(default=5, ge=1)


class PlannerModelBase(ABC, nn.Module):
    """Abstract base class for squad planner neural networks.

    Planner models observe all alive friendly tanks (patches +
    positional metadata) and produce per-tank order logits.  They
    must handle variable numbers of alive tanks without requiring
    padding or masking.

    Subclasses must:
    - Store a ``config`` attribute of a :class:`PlannerModelConfig` subclass.
    - Implement :meth:`forward` with the specified signature.
    """

    config: PlannerModelConfig

    @abstractmethod
    def forward(
        self,
        patch_tensors: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: all alive tank observations → per-tank order logits.

        Args:
            patch_tensors: Encoded patches for all alive tanks, shape
                ``[num_alive, channels, patch_size, patch_size]``.
            positions: Normalised (x, y) positions for each alive tank,
                shape ``[num_alive, 2]``.  Values in ``[0, 1]``.
            directions: Direction encoding for each alive tank, shape
                ``[num_alive, direction_dim]``.  Encoding scheme is
                defined by the concrete implementation (e.g. one-hot
                over 4 cardinal directions, or sin/cos).

        Returns:
            Per-tank order logits of shape
            ``[num_alive, num_orders]``.
        """
        ...

    @abstractmethod
    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return the initial hidden state for the planner.

        Planners may or may not use recurrent state.  If not recurrent,
        return a zero-size tensor.

        Args:
            batch_size: Number of parallel episodes (default 1).

        Returns:
            Hidden state tensor (may be empty for non-recurrent planners).
        """
        ...
