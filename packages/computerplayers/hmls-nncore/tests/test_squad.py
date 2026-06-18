"""Tests for the hmls.nncore.squad submodule."""

from __future__ import annotations

import pytest

from hmls.nncore.squad import (
    NUM_ORDERS,
    ExecutorModelBase,
    ExecutorModelConfig,
    Order,
    OrderConditionedRewardConfig,
    OrderRewardModifier,
    PlannerModelBase,
    PlannerModelConfig,
    SquadPersistenceBase,
    discover_squads,
)


class TestOrderEnum:
    """Tests for the Order enum."""

    def test_order_count(self) -> None:
        """There should be exactly 8 orders."""
        assert len(Order) == 8
        assert NUM_ORDERS == 8

    def test_order_values_are_sequential(self) -> None:
        """Order values should be 0-indexed and sequential."""
        values = sorted(o.value for o in Order)
        assert values == list(range(8))

    def test_all_expected_orders_present(self) -> None:
        """All documented orders should exist."""
        expected = {
            "ADVANCE",
            "RETREAT",
            "HOLD",
            "ENGAGE",
            "EVADE",
            "SCOUT",
            "FLANK_LEFT",
            "FLANK_RIGHT",
        }
        actual = {o.name for o in Order}
        assert actual == expected

    def test_order_is_int_enum(self) -> None:
        """Orders should be usable as integer indices."""
        assert int(Order.ADVANCE) == 0
        assert int(Order.FLANK_RIGHT) == 7
        assert int(Order.ENGAGE) == 3


class TestOrderConditionedRewardConfig:
    """Tests for order-conditioned reward config types."""

    def test_default_config_disabled(self) -> None:
        """Default config should have rewards disabled."""
        config = OrderConditionedRewardConfig()
        assert config.enabled is False
        assert config.modifiers == {}

    def test_get_modifier_returns_default_for_unconfigured(self) -> None:
        """Unconfigured orders should return default modifier."""
        config = OrderConditionedRewardConfig()
        modifier = config.get_modifier(Order.ADVANCE)
        assert modifier.exploration_scale == 1.0
        assert modifier.firing_hit_scale == 1.0
        assert modifier.hold_penalty == 0.0

    def test_get_modifier_returns_configured_value(self) -> None:
        """Configured orders should return their specific modifier."""
        scout_mod = OrderRewardModifier(exploration_scale=2.0)
        config = OrderConditionedRewardConfig(
            enabled=True,
            modifiers={Order.SCOUT: scout_mod},
        )
        assert config.get_modifier(Order.SCOUT).exploration_scale == 2.0
        # Other orders still get default
        assert config.get_modifier(Order.ENGAGE).exploration_scale == 1.0

    def test_config_serialisation_round_trip(self) -> None:
        """Config should survive JSON serialisation round-trip."""
        original = OrderConditionedRewardConfig(
            enabled=True,
            modifiers={
                Order.HOLD: OrderRewardModifier(hold_penalty=-0.05, movement_scale=0.5),
                Order.ENGAGE: OrderRewardModifier(firing_hit_scale=2.0),
            },
        )
        json_str = original.model_dump_json()
        restored = OrderConditionedRewardConfig.model_validate_json(json_str)
        assert restored.enabled is True
        assert restored.get_modifier(Order.HOLD).hold_penalty == -0.05
        assert restored.get_modifier(Order.ENGAGE).firing_hit_scale == 2.0


class TestSquadRegistry:
    """Tests for squad entry-point discovery."""

    def test_discover_squads_returns_dict(self) -> None:
        """discover_squads should return a dict (possibly empty if no squads installed)."""
        result = discover_squads()
        assert isinstance(result, dict)


class TestBaseClassesAreAbstract:
    """Verify that base classes cannot be instantiated directly."""

    def test_executor_model_base_is_abstract(self) -> None:
        """ExecutorModelBase should not be instantiable."""
        with pytest.raises(TypeError):
            ExecutorModelBase()  # type: ignore[abstract]

    def test_planner_model_base_is_abstract(self) -> None:
        """PlannerModelBase should not be instantiable."""
        with pytest.raises(TypeError):
            PlannerModelBase()  # type: ignore[abstract]

    def test_squad_persistence_base_is_abstract(self) -> None:
        """SquadPersistenceBase should not be instantiable."""
        with pytest.raises(TypeError):
            SquadPersistenceBase()  # type: ignore[abstract]


class TestConfigModels:
    """Tests for config Pydantic models."""

    def test_executor_config_defaults(self) -> None:
        """ExecutorModelConfig should have sensible defaults."""
        config = ExecutorModelConfig(model_id="test")
        assert config.patch_size == 9
        assert config.num_orders == NUM_ORDERS

    def test_planner_config_defaults(self) -> None:
        """PlannerModelConfig should have sensible defaults."""
        config = PlannerModelConfig(model_id="test")
        assert config.patch_size == 9
        assert config.num_orders == NUM_ORDERS
        assert config.max_tanks == 5

    def test_executor_config_validation(self) -> None:
        """ExecutorModelConfig should reject invalid patch_size."""
        with pytest.raises(ValueError, match="greater than or equal to 3"):
            ExecutorModelConfig(model_id="test", patch_size=1)

    def test_planner_config_validation(self) -> None:
        """PlannerModelConfig should reject invalid max_tanks."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            PlannerModelConfig(model_id="test", max_tanks=0)
