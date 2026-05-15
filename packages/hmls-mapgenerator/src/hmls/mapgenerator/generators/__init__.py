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

from hmls.mapgenerator.generators.base import (
    STRATEGY_REGISTRY,
    MapStrategy,
    StrategyParam,
    generate_map,
    register_strategy,
)

# Auto-import all sibling modules in this package so that their
# @register_strategy decorators execute and populate STRATEGY_REGISTRY.
# See :func:`~.base.register_strategy` for the decorator that performs
# the actual registration.
for _info in pkgutil.iter_modules(__path__, __name__ + "."):
    importlib.import_module(_info.name)
del _info

# Explicit imports for type-checking convenience and IDE support.
from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy  # noqa: E402
from hmls.mapgenerator.generators.perlin import PerlinNoiseStrategy  # noqa: E402

__all__ = [
    "STRATEGY_REGISTRY",
    "BlobAndLineStrategy",
    "MapStrategy",
    "PerlinNoiseStrategy",
    "StrategyParam",
    "generate_map",
    "register_strategy",
]
