"""Generators subpackage — pluggable map generation strategies.

Strategy modules placed in this directory are auto-discovered at import
time using :func:`pkgutil.iter_modules`.  Each module that decorates a
:class:`~.base.MapStrategy` subclass with :func:`~.base.register_strategy`
will have that class added to :data:`STRATEGY_REGISTRY` automatically —
no manual edits to this file are required when adding new strategies.

Re-exports all public names so callers can use::

    from hmls.mapgenerator.generators import generate_map, BlobAndLineStrategy
"""

import importlib
import pkgutil
from typing import Annotated

from pydantic import Field

from hmls.core import GameMap
from hmls.mapgenerator.generators.base import (
    STRATEGY_REGISTRY,
    MapStrategy,
    StrategyConfigBase,
    StrategyParam,
    generate_map,
    register_strategy,
)
from hmls.mapgenerator.generators.blob_and_line import (
    BlobAndLineConfig,
    BlobAndLineStrategy,
)
from hmls.mapgenerator.generators.perlin import (
    PerlinNoiseConfig,
    PerlinNoiseStrategy,
)

# Auto-import all sibling modules in this package so that their
# @register_strategy decorators execute and populate STRATEGY_REGISTRY.
# See :func:`~.base.register_strategy` for the decorator that performs
# the actual registration.
for _info in pkgutil.iter_modules(__path__, __name__ + "."):
    importlib.import_module(_info.name)
del _info

# ── Discriminated union of all strategy configs ───────────────────────

StrategyConfig = Annotated[
    BlobAndLineConfig | PerlinNoiseConfig,
    Field(discriminator="type"),
]
"""Discriminated union of all strategy configuration models.

Pydantic uses the ``type`` field as a discriminator to dispatch JSON
like ``{"type": "perlin_noise", "scale": 0.1}`` to the correct
concrete config class.  Each member is a frozen
:class:`StrategyConfigBase` subclass with a ``create_strategy()``
factory method.
"""


# ── Config-based map generation entrypoint ────────────────────────────


def generate_map_from_config(
    width: int,
    height: int,
    *,
    impassable_fraction: float = 0.3,
    strategy_config: StrategyConfigBase | None = None,
    connected_obstacles: bool = False,
    seed: int | None = None,
) -> GameMap:
    """Generate a map using a strategy configuration object.

    Thin wrapper around :func:`generate_map` that accepts a
    :class:`StrategyConfigBase` instance (typically obtained by
    parsing JSON through the :data:`StrategyConfig` discriminated
    union) instead of a pre-instantiated :class:`MapStrategy`.

    Args:
        width: Number of columns.
        height: Number of rows.
        impassable_fraction: Target fraction of impassable cells (0.0–1.0).
        strategy_config: A concrete :class:`StrategyConfigBase` subclass
            instance.  If ``None``, defaults to :class:`BlobAndLineConfig`
            with default parameters.
        connected_obstacles: If ``True``, attempt to bridge disjoint
            impassable regions before enforcing passable connectivity.
        seed: Random seed for reproducibility.  ``None`` for non-deterministic.

    Returns:
        A :class:`~hmls.core.GameMap` with passable terrain guaranteed to
        be fully 4-connected.
    """
    if strategy_config is None:
        strategy_config = BlobAndLineConfig()
    strategy = strategy_config.create_strategy()
    return generate_map(
        width,
        height,
        impassable_fraction=impassable_fraction,
        connected_obstacles=connected_obstacles,
        seed=seed,
        strategy=strategy,
    )


__all__ = [
    "STRATEGY_REGISTRY",
    "BlobAndLineConfig",
    "BlobAndLineStrategy",
    "MapStrategy",
    "PerlinNoiseConfig",
    "PerlinNoiseStrategy",
    "StrategyConfig",
    "StrategyConfigBase",
    "StrategyParam",
    "generate_map",
    "generate_map_from_config",
    "register_strategy",
]
