"""Neural network model: Flatten → GRU → policy head.

The :class:`MkIIITankPolicyNetwork` processes an encoded patch tensor by
flattening it directly (no CNN) and feeding the resulting vector into a
GRU cell (temporal memory across turns), followed by a linear policy head
that outputs logits over the 5 actions.

Compared to the Mk-I and Mk-II architectures, the Mk-III removes all
convolutional layers.  The visual channels are fed directly into the
recurrent layer, making the model simpler and faster but without learned
spatial feature extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.model import TankModelBase, TankModelConfig


class MkIIIModelConfig(TankModelConfig, frozen=True, extra="forbid"):
    """Hyperparameters for :class:`MkIIITankPolicyNetwork`.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Model identifier for the persistence registry.
        gru_hidden_size: Dimensionality of the GRU hidden state.
    """

    model_id: str = "hmls.singlemkiii"
    gru_hidden_size: int = 128


class MkIIITankPolicyNetwork(TankModelBase):
    """Flatten → GRU → policy head for single-tank action selection.

    The encoded patch tensor is flattened into a 1-D vector and passed
    directly to a GRU cell (maintaining hidden state across turns within
    an episode).  The GRU output feeds a linear layer producing logits
    over the action space.

    This architecture trades spatial feature extraction for simplicity
    and speed — useful as a baseline or for small patch sizes where
    convolution may not add value.

    Args:
        config: Model hyperparameters.
    """

    def __init__(self, config: MkIIIModelConfig | None = None) -> None:
        super().__init__()
        self.config: MkIIIModelConfig = config or MkIIIModelConfig()

        # Flattened input size: channels × patch_size × patch_size
        self._input_size = (
            FiveChannelPatchEncoder.NUM_CHANNELS * self.config.patch_size * self.config.patch_size
        )

        # GRU cell: takes flattened patch as input
        self.gru = nn.GRUCell(self._input_size, self.config.gru_hidden_size)

        # Policy head: GRU hidden → action logits
        self.policy_head = nn.Linear(self.config.gru_hidden_size, NUM_ACTIONS)

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: patch → (action_logits, new_hidden).

        Args:
            patch_tensor: Encoded patch tensor of shape
                ``[batch, FiveChannelPatchEncoder.NUM_CHANNELS, patch_size, patch_size]`` or
                ``[FiveChannelPatchEncoder.NUM_CHANNELS, patch_size, patch_size]`` (unbatched).
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

        # Flatten patch directly (no CNN)
        flat = patch_tensor.view(patch_tensor.size(0), -1)

        # GRU step
        new_hidden = self.gru(flat, hidden)

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

    @property
    def total_hidden_size(self) -> int:
        """Total hidden state dimensionality (single GRU)."""
        return self.config.gru_hidden_size
