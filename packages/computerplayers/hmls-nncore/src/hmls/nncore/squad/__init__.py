"""Squad architecture support for multi-tank teams.

This submodule provides the shared types and abstract base classes
that all squad implementations depend on:

- :class:`Order` — discrete order vocabulary (8 tactical orders)
- :class:`PlannerModelBase` — abstract planner neural network
- :class:`ExecutorModelBase` — abstract executor neural network
- :class:`SquadPersistenceBase` — persistence contract for squad model pairs
- :func:`resolve_squad_id` — entry-point-based squad discovery
- :class:`OrderConditionedRewardConfig` — future reward shaping types

Concrete squad implementations (planner architectures, executor
architectures, composite players) live in separate packages and
register via the ``hmls.squads`` entry-point group.
"""

from hmls.nncore.squad.executor_base import ExecutorModelBase, ExecutorModelConfig
from hmls.nncore.squad.orders import NUM_ORDERS, Order
from hmls.nncore.squad.persistence_base import SquadPersistenceBase
from hmls.nncore.squad.planner_base import PlannerModelBase, PlannerModelConfig
from hmls.nncore.squad.registry import (
    SquadRegistryError,
    discover_squads,
    list_available_squads,
    resolve_squad_id,
)
from hmls.nncore.squad.reward_config import (
    OrderConditionedRewardConfig,
    OrderRewardModifier,
)

__all__ = [
    "NUM_ORDERS",
    "ExecutorModelBase",
    "ExecutorModelConfig",
    "Order",
    "OrderConditionedRewardConfig",
    "OrderRewardModifier",
    "PlannerModelBase",
    "PlannerModelConfig",
    "SquadPersistenceBase",
    "SquadRegistryError",
    "discover_squads",
    "list_available_squads",
    "resolve_squad_id",
]
