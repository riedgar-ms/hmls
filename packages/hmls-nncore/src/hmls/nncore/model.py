"""Abstract base classes for tank policy networks.

Defines :class:`TankModelConfig` and :class:`TankModelBase`, the
foundational abstractions that all tank model packages must implement.
This allows the training infrastructure to work with any tank model
without depending on concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class TankModelConfig(BaseModel, frozen=True, extra="forbid"):
    """Base configuration shared by all tank policy networks.

    Every concrete model config (e.g. ``MkIModelConfig``,
    ``MkIIModelConfig``) must extend this class so that generic loading
    infrastructure can discover the correct model package.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_package: Fully-qualified Python package that defines the
            model (e.g. ``"hmls.singlemki"``).  Used by the generic
            persistence layer to locate the correct load/save routines.
    """

    patch_size: int = Field(default=9, ge=3)
    model_package: str


class TankModelBase(ABC, nn.Module):
    """Abstract base class for all tank policy networks.

    Concrete subclasses implement the neural network architecture
    (e.g. CNN→GRU→policy-head) while this base defines the interface
    that the training infrastructure and player classes depend on.

    Subclasses must:
    - Store a ``config`` attribute of a :class:`TankModelConfig` subclass.
    - Implement :meth:`forward` with the specified signature.
    - Implement :meth:`initial_hidden` to provide episode-start states.
    """

    config: TankModelConfig

    @abstractmethod
    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encoded patch → (action_logits, new_hidden).

        Args:
            patch_tensor: Encoded patch tensor of shape
                ``[batch, channels, patch_size, patch_size]`` or
                ``[channels, patch_size, patch_size]`` (unbatched).
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
        """Total dimensionality of the hidden state tensor.

        For models with a single recurrent layer this equals the GRU
        hidden size.  For stacked architectures it is the sum of all
        recurrent hidden sizes.
        """
        ...
