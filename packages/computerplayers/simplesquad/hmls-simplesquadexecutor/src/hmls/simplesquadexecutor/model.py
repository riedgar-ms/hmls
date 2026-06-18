"""Neural network model: CNN + order embedding → GRU → policy head.

The :class:`SimpleExecutorModel` processes an encoded patch tensor through
convolutional layers (spatial features), concatenates a learned order
embedding, feeds the result through a GRU (temporal memory across turns),
and produces action logits over the 5 low-level actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import Field

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.squad.executor_base import ExecutorModelBase, ExecutorModelConfig


class SimpleExecutorConfig(ExecutorModelConfig, frozen=True, extra="forbid"):
    """Hyperparameters for :class:`SimpleExecutorModel`.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Model identifier for persistence.
        num_orders: Number of discrete orders accepted.
        cnn_channels: Number of output channels for each conv layer.
        gru_hidden_size: Dimensionality of the GRU hidden state.
        order_embedding_dim: Dimensionality of the order embedding vector.
        conv_kernel_size: Kernel size for Conv2d layers (must be odd).
        pool_kernel_size: Kernel size for MaxPool2d layers.
        pool_stride: Stride for MaxPool2d layers.
    """

    model_id: str = "hmls.simplesquadexecutor"
    cnn_channels: tuple[int, ...] = (32, 64)
    gru_hidden_size: int = 128
    order_embedding_dim: int = 16
    conv_kernel_size: int = Field(default=3, ge=1)
    pool_kernel_size: int = Field(default=2, ge=1)
    pool_stride: int = Field(default=2, ge=1)


class SimpleExecutorModel(ExecutorModelBase):
    """CNN + order embedding → GRU → policy head for order-conditioned execution.

    The CNN extracts spatial features from the encoded patch.  These are
    flattened and concatenated with a learned order embedding.  The
    combined vector passes through a GRU cell (maintaining hidden state
    across turns), and the output feeds a linear layer producing logits
    over the action space.

    Args:
        config: Model hyperparameters.
    """

    def __init__(self, config: SimpleExecutorConfig | None = None) -> None:
        super().__init__()
        self.config: SimpleExecutorConfig = config or SimpleExecutorConfig()

        # Order embedding
        self.order_embedding = nn.Embedding(self.config.num_orders, self.config.order_embedding_dim)

        # Build CNN layers
        layers: list[nn.Module] = []
        in_channels = FiveChannelPatchEncoder.NUM_CHANNELS
        for out_channels in self.config.cnn_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.config.conv_kernel_size,
                    padding=self.config.conv_kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(
                nn.MaxPool2d(
                    kernel_size=self.config.pool_kernel_size,
                    stride=self.config.pool_stride,
                )
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # Compute flattened CNN output size
        dummy = torch.zeros(
            1, FiveChannelPatchEncoder.NUM_CHANNELS, self.config.patch_size, self.config.patch_size
        )
        with torch.no_grad():
            cnn_out = self.cnn(dummy)
        self._cnn_output_size = cnn_out.numel()

        # GRU input: CNN features + order embedding
        gru_input_size = self._cnn_output_size + self.config.order_embedding_dim
        self.gru = nn.GRUCell(gru_input_size, self.config.gru_hidden_size)

        # Policy head: GRU hidden → action logits
        self.policy_head = nn.Linear(self.config.gru_hidden_size, NUM_ACTIONS)

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
            order: Order index tensor of shape ``[batch]`` or scalar.
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
            order = order.unsqueeze(0) if order.dim() == 0 else order.unsqueeze(0)

        # CNN feature extraction
        cnn_features = self.cnn(patch_tensor)
        cnn_flat = cnn_features.view(cnn_features.size(0), -1)

        # Order embedding
        order_emb = self.order_embedding(order)

        # Concatenate CNN features + order embedding
        combined = torch.cat([cnn_flat, order_emb], dim=-1)

        # GRU step
        new_hidden = self.gru(combined, hidden)

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
            Tensor of shape ``[batch_size, gru_hidden_size]``.
        """
        return torch.zeros(batch_size, self.config.gru_hidden_size)

    @property
    def total_hidden_size(self) -> int:
        """Total hidden state dimensionality."""
        return self.config.gru_hidden_size
