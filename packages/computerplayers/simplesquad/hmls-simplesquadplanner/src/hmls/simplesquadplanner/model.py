"""Neural network model: set-pooling planner for multi-tank squads.

The :class:`SimplePlannerModel` observes all alive friendly tanks
(patches + positional metadata), encodes each independently, pools
across the set, and produces per-tank order logits.

Architecture:
1. Per-tank encoding: CNN(patch) ∥ position_features ∥ direction_features
2. Set-pooling: mean across all alive tank encodings → global context
3. Per-tank decision: [tank_i encoding ∥ global context] → MLP → order logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import Field

from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.nncore.squad.planner_base import PlannerModelBase, PlannerModelConfig

# 4 cardinal directions encoded as one-hot
NUM_DIRECTIONS = 4


class SimplePlannerConfig(PlannerModelConfig, frozen=True, extra="forbid"):
    """Hyperparameters for :class:`SimplePlannerModel`.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Model identifier for persistence.
        num_orders: Number of discrete orders the planner can issue.
        max_tanks: Maximum tanks per team supported.
        cnn_channels: Output channels for each conv layer in the
            per-tank patch encoder.
        cnn_kernel_size: Kernel size for Conv2d layers.
        pool_kernel_size: Kernel size for MaxPool2d layers.
        pool_stride: Stride for MaxPool2d layers.
        tank_feature_dim: Output dim of the per-tank feature encoder
            (before set-pooling).
        mlp_hidden_dim: Hidden layer size in the per-tank decision MLP.
    """

    model_id: str = "hmls.simplesquadplanner"
    cnn_channels: tuple[int, ...] = (32, 64)
    cnn_kernel_size: int = Field(default=3, ge=1)
    pool_kernel_size: int = Field(default=2, ge=1)
    pool_stride: int = Field(default=2, ge=1)
    tank_feature_dim: int = Field(default=64, ge=8)
    mlp_hidden_dim: int = Field(default=64, ge=8)


class SimplePlannerModel(PlannerModelBase):
    """Set-pooling planner: encode tanks → mean-pool → per-tank order logits.

    Each alive tank's patch is encoded via a shared CNN.  Positional
    metadata (normalised x/y position, one-hot direction) is concatenated.
    A linear projection maps to ``tank_feature_dim``.  The mean across
    all alive tanks forms a global context.  Each tank then gets an MLP
    that takes [its features ∥ global context] and outputs order logits.

    This naturally handles variable tank counts without masking or padding.

    Args:
        config: Model hyperparameters.
    """

    def __init__(self, config: SimplePlannerConfig | None = None) -> None:
        super().__init__()
        self.config: SimplePlannerConfig = config or SimplePlannerConfig()

        # Per-tank CNN encoder (shared across all tanks)
        layers: list[nn.Module] = []
        in_channels = FiveChannelPatchEncoder.NUM_CHANNELS
        for out_channels in self.config.cnn_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.config.cnn_kernel_size,
                    padding=self.config.cnn_kernel_size // 2,
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
        self._cnn_output_size = cnn_out.view(1, -1).size(1)

        # Metadata input: 2 (normalised x, y) + NUM_DIRECTIONS (one-hot direction)
        metadata_dim = 2 + NUM_DIRECTIONS

        # Project [CNN features ∥ metadata] → tank_feature_dim
        self.tank_encoder = nn.Sequential(
            nn.Linear(self._cnn_output_size + metadata_dim, self.config.tank_feature_dim),
            nn.ReLU(),
        )

        # Per-tank decision MLP: [tank_features ∥ global_context] → order logits
        decision_input_dim = self.config.tank_feature_dim * 2  # tank + pooled context
        self.decision_mlp = nn.Sequential(
            nn.Linear(decision_input_dim, self.config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.mlp_hidden_dim, self.config.num_orders),
        )

    def forward(
        self,
        patch_tensors: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: all alive tanks → per-tank order logits.

        Args:
            patch_tensors: Encoded patches, shape
                ``[num_alive, channels, patch_size, patch_size]``.
            positions: Normalised (x, y) positions, shape
                ``[num_alive, 2]``.  Values in ``[0, 1]``.
            directions: One-hot direction encoding, shape
                ``[num_alive, 4]``.

        Returns:
            Per-tank order logits, shape ``[num_alive, num_orders]``.
        """
        num_alive = patch_tensors.size(0)

        # CNN encode each tank's patch
        cnn_features = self.cnn(patch_tensors)
        cnn_flat = cnn_features.view(num_alive, -1)

        # Concatenate with positional metadata
        metadata = torch.cat([positions, directions], dim=-1)
        combined = torch.cat([cnn_flat, metadata], dim=-1)

        # Project to tank feature space
        tank_features = self.tank_encoder(combined)

        # Set-pooling: mean across all alive tanks
        global_context = tank_features.mean(dim=0, keepdim=True)
        global_context_expanded = global_context.expand(num_alive, -1)

        # Per-tank decision: [tank_features ∥ global_context] → order logits
        decision_input = torch.cat([tank_features, global_context_expanded], dim=-1)
        order_logits: torch.Tensor = self.decision_mlp(decision_input)

        return order_logits

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return empty hidden state (planner is non-recurrent).

        The simple planner is a feedforward model with no temporal
        memory — it makes decisions based on the current state only.

        Args:
            batch_size: Ignored (included for interface compatibility).

        Returns:
            An empty tensor (size 0).
        """
        return torch.empty(0)
