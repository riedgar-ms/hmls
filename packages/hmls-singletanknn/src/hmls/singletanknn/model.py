"""Neural network model: CNN → GRU → policy head.

The :class:`TankPolicyNetwork` processes an encoded patch tensor through
convolutional layers (spatial features), a GRU (temporal memory across
turns), and a linear policy head that outputs logits over the 5 actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from hmls.singletanknn.encoding import NUM_CHANNELS

#: The number of discrete actions available to a tank (fixed by game rules).
NUM_ACTIONS: int = 5


class ModelConfig(BaseModel, frozen=True):
    """Hyperparameters for :class:`TankPolicyNetwork`.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        cnn_channels: Number of output channels for each conv layer.
        gru_hidden_size: Dimensionality of the GRU hidden state.
    """

    patch_size: int = Field(default=9, ge=3)
    cnn_channels: tuple[int, ...] = (32, 64)
    gru_hidden_size: int = 128


class TankPolicyNetwork(nn.Module):
    """CNN → GRU → policy head for single-tank action selection.

    The CNN extracts spatial features from the encoded patch.  These are
    flattened and passed through a GRU cell (maintaining hidden state
    across turns within an episode).  The GRU output feeds a linear
    layer producing logits over the action space.

    Args:
        config: Model hyperparameters.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Build CNN layers
        layers: list[nn.Module] = []
        in_channels = NUM_CHANNELS
        for out_channels in self.config.cnn_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # Compute flattened CNN output size by doing a dummy forward pass
        dummy = torch.zeros(1, NUM_CHANNELS, self.config.patch_size, self.config.patch_size)
        with torch.no_grad():
            cnn_out = self.cnn(dummy)
        self._cnn_output_size = cnn_out.numel()

        # GRU cell: takes flattened CNN features as input
        self.gru = nn.GRUCell(self._cnn_output_size, self.config.gru_hidden_size)

        # Policy head: GRU hidden → action logits
        self.policy_head = nn.Linear(self.config.gru_hidden_size, NUM_ACTIONS)

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patch → (action_logits, new_hidden).

        Args:
            patch_tensor: Encoded patch tensor of shape
                ``[batch, NUM_CHANNELS, patch_size, patch_size]`` or
                ``[NUM_CHANNELS, patch_size, patch_size]`` (unbatched).
            hidden: GRU hidden state, shape ``[batch, gru_hidden_size]``
                or ``[gru_hidden_size]``.

        Returns:
            A tuple of ``(logits, new_hidden)`` where logits has shape
            ``[batch, NUM_ACTIONS]`` and new_hidden has the same shape
            as the input hidden state.
        """
        # Ensure batch dimension
        unbatched = patch_tensor.dim() == 3
        if unbatched:
            patch_tensor = patch_tensor.unsqueeze(0)
            hidden = hidden.unsqueeze(0)

        # CNN feature extraction
        cnn_features = self.cnn(patch_tensor)
        cnn_flat = cnn_features.view(cnn_features.size(0), -1)

        # GRU step
        new_hidden = self.gru(cnn_flat, hidden)

        # Policy head
        logits = self.policy_head(new_hidden)

        if unbatched:
            return logits.squeeze(0), new_hidden.squeeze(0)
        return logits, new_hidden

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zero-initialised GRU hidden state.

        Args:
            batch_size: Number of parallel episodes (default 1).

        Returns:
            Tensor of shape ``[batch_size, gru_hidden_size]`` filled
            with zeros.
        """
        return torch.zeros(batch_size, self.config.gru_hidden_size)
