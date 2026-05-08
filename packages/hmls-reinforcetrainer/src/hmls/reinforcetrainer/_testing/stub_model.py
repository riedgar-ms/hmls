"""Stub tank policy network for testing.

Provides :class:`StubModelConfig` and :class:`StubTankModel`, a minimal
implementation of :class:`~hmls.nncore.model.TankModelBase` that uses a
single linear layer.  This is fast, supports gradient flow, and avoids
depending on any concrete model package.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.model import TankModelBase, TankModelConfig


class StubModelConfig(TankModelConfig, frozen=True):
    """Configuration for the stub test model.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_package: Python package path for dynamic dispatch.
        hidden_size: Dimensionality of the hidden state.
    """

    model_package: str = "hmls.reinforcetrainer._testing"
    hidden_size: int = 16


class StubTankModel(TankModelBase):
    """Minimal neural network model for testing.

    A single linear layer maps flattened patch + hidden to action logits
    and a new hidden state.  This is sufficient for testing gradient flow,
    optimizer steps, and the trainer's integration with the model interface.

    Args:
        config: Stub model configuration.
    """

    def __init__(self, config: StubModelConfig | None = None) -> None:
        super().__init__()
        self.config: StubModelConfig = config or StubModelConfig()

        input_size = (
            FiveChannelPatchEncoder.NUM_CHANNELS * self.config.patch_size * self.config.patch_size
        )
        self.feature_layer = nn.Linear(input_size + self.config.hidden_size, 32)
        self.logit_head = nn.Linear(32, NUM_ACTIONS)
        self.hidden_head = nn.Linear(32, self.config.hidden_size)

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patch + hidden → (logits, new_hidden).

        Args:
            patch_tensor: Encoded patch tensor of shape
                ``[batch, channels, patch_size, patch_size]`` or
                ``[channels, patch_size, patch_size]`` (unbatched).
            hidden: Hidden state tensor of shape ``[batch, hidden_size]``
                or ``[hidden_size]``.

        Returns:
            A tuple of ``(logits, new_hidden)``.
        """
        unbatched = patch_tensor.dim() == 3
        if unbatched:
            patch_tensor = patch_tensor.unsqueeze(0)
            hidden = hidden.unsqueeze(0)

        flat = patch_tensor.view(patch_tensor.size(0), -1)
        combined = torch.cat([flat, hidden], dim=-1)
        features = torch.relu(self.feature_layer(combined))
        logits = self.logit_head(features)
        new_hidden = self.hidden_head(features)

        if unbatched:
            return logits.squeeze(0), new_hidden.squeeze(0)
        return logits, new_hidden

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zero-initialised hidden state.

        Args:
            batch_size: Number of parallel episodes.

        Returns:
            Tensor of shape ``[batch_size, hidden_size]``.
        """
        return torch.zeros(batch_size, self.config.hidden_size)

    @property
    def total_hidden_size(self) -> int:
        """Total dimensionality of the hidden state."""
        return self.config.hidden_size
