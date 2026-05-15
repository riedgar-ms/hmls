"""Model configuration and stub model for the random tank.

:class:`RandomTankModelConfig` defines the three probability parameters
that control the tank's behaviour.  :class:`RandomTankModel` is a
minimal :class:`~hmls.nncore.model.TankModelBase` subclass with no
meaningful learned parameters — it exists only to satisfy the
persistence/dispatch infrastructure that is typed around
:class:`~hmls.nncore.model.TankModelBase`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import Field, model_validator

from hmls.nncore.constants import NUM_ACTIONS
from hmls.nncore.model import TankModelBase, TankModelConfig


class RandomTankModelConfig(TankModelConfig, frozen=True, extra="forbid"):
    """Configuration for the rule-based random tank.

    Attributes:
        patch_size: Side length of the input patch (must be odd, ≥ 3).
        model_id: Model identifier for the persistence registry.
        prob_forward_on_passable: Probability of moving forward when
            the cell ahead is passable and empty.
        prob_turn_left_on_passable: Probability of turning left when
            the cell ahead is passable and empty.  The probability of
            turning right is ``1 - prob_forward_on_passable -
            prob_turn_left_on_passable``.
        prob_turn_left_on_blocked: Probability of turning left (vs
            right) when the cell ahead is impassable or occupied.
    """

    model_id: str = "hmls.randomtank"
    prob_forward_on_passable: float = Field(default=0.7, ge=0.0, le=1.0)
    prob_turn_left_on_passable: float = Field(default=0.15, ge=0.0, le=1.0)
    prob_turn_left_on_blocked: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_passable_probs(self) -> RandomTankModelConfig:
        """Ensure passable probabilities sum to at most 1.0."""
        total = self.prob_forward_on_passable + self.prob_turn_left_on_passable
        if total > 1.0:
            msg = (
                f"prob_forward_on_passable ({self.prob_forward_on_passable}) + "
                f"prob_turn_left_on_passable ({self.prob_turn_left_on_passable}) "
                f"= {total}, which exceeds 1.0"
            )
            raise ValueError(msg)
        return self


class RandomTankModel(TankModelBase):
    """Minimal stub model for the rule-based random tank.

    This model has no meaningful learned parameters.  It exists only
    to satisfy the :class:`~hmls.nncore.model.TankModelBase` interface
    required by the persistence and dispatch infrastructure.

    The ``forward`` method returns uniform logits and a zero hidden
    state.  In practice it is never called — the
    :class:`~hmls.randomtank.player.RandomTankPlayer` overrides
    ``choose_action`` to bypass the neural-network forward pass
    entirely.

    Args:
        config: Random tank model configuration.
    """

    def __init__(self, config: RandomTankModelConfig | None = None) -> None:
        super().__init__()
        self.config: RandomTankModelConfig = config or RandomTankModelConfig()
        # A single unused parameter so PyTorch does not complain about
        # an empty Module (e.g. when calling .parameters()).
        self._placeholder = nn.Parameter(torch.zeros(1))

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return uniform logits and unchanged hidden state.

        This method is not used during normal operation — the
        :class:`~hmls.randomtank.player.RandomTankPlayer` bypasses the
        forward pass entirely.

        Args:
            patch_tensor: Encoded patch tensor (ignored).
            hidden: Hidden state tensor (returned unchanged).

        Returns:
            A tuple of ``(uniform_logits, hidden)``.
        """
        logits = torch.zeros(NUM_ACTIONS)
        return logits, hidden

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return a zero-initialised hidden state.

        Args:
            batch_size: Number of parallel episodes.

        Returns:
            Tensor of shape ``[batch_size, 1]``.
        """
        return torch.zeros(batch_size, 1)

    @property
    def total_hidden_size(self) -> int:
        """Total dimensionality of the hidden state (always 1)."""
        return 1
