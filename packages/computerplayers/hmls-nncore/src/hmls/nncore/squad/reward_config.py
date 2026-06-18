"""Order-conditioned reward configuration types.

Defines Pydantic models for per-order reward modifiers that allow
the reward signal to be shaped based on the current order assigned
to a tank.  These types are defined here for forward compatibility
but are not required by the initial training implementation (which
uses simple aggregated executor rewards for the planner).

When enabled, order-conditioned rewards modify the standard step
reward based on the tank's current order.  For example:
- SCOUT order: amplify rewards for seeing new cells
- HOLD order: penalise movement
- ENGAGE order: amplify rewards for hits and closing distance
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hmls.nncore.squad.orders import Order


class OrderRewardModifier(BaseModel, frozen=True, extra="forbid"):
    """Reward modifier for a single order type.

    Each modifier scales specific reward components when the tank
    is executing the associated order.

    Attributes:
        exploration_scale: Multiplier for exploration rewards.
        firing_hit_scale: Multiplier for firing hit rewards.
        firing_miss_scale: Multiplier for firing miss penalties.
        movement_scale: Multiplier for movement rewards.
        hold_penalty: Per-step penalty for moving when ordered to HOLD.
    """

    exploration_scale: float = Field(
        default=1.0,
        title="Exploration Scale",
        description="Multiplier for exploration rewards under this order.",
    )
    firing_hit_scale: float = Field(
        default=1.0,
        title="Firing Hit Scale",
        description="Multiplier for firing hit rewards under this order.",
    )
    firing_miss_scale: float = Field(
        default=1.0,
        title="Firing Miss Scale",
        description="Multiplier for firing miss penalties under this order.",
    )
    movement_scale: float = Field(
        default=1.0,
        title="Movement Scale",
        description="Multiplier for movement rewards under this order.",
    )
    hold_penalty: float = Field(
        default=0.0,
        title="Hold Penalty",
        description="Per-step penalty for moving when ordered to HOLD.",
    )


class OrderConditionedRewardConfig(BaseModel, frozen=True, extra="forbid"):
    """Configuration for order-conditioned reward shaping.

    Maps each order to a set of reward modifiers.  Orders not
    explicitly listed use default (unmodified) scaling.

    Attributes:
        enabled: Whether order-conditioned rewards are active.
        modifiers: Per-order reward modifier overrides.
    """

    enabled: bool = Field(
        default=False,
        title="Enabled",
        description=(
            "Whether order-conditioned reward shaping is active. "
            "When disabled, all orders use the base reward function unmodified."
        ),
    )
    modifiers: dict[Order, OrderRewardModifier] = Field(
        default_factory=dict,
        title="Order Modifiers",
        description=(
            "Per-order reward modifier overrides. Orders not listed "
            "use default scaling (all multipliers = 1.0)."
        ),
    )

    def get_modifier(self, order: Order) -> OrderRewardModifier:
        """Get the reward modifier for a given order.

        Returns the configured modifier if one exists, otherwise
        returns a default (unmodified) instance.

        Args:
            order: The order to look up.

        Returns:
            The :class:`OrderRewardModifier` for this order.
        """
        return self.modifiers.get(order, OrderRewardModifier())
