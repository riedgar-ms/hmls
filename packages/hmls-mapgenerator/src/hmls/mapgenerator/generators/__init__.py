"""Generators subpackage — pluggable map generation strategies.

Re-exports all public names so callers can use::

    from hmls.mapgenerator.generators import generate_map, BlobAndLineStrategy
"""

from hmls.mapgenerator.generators.base import (
    STRATEGY_REGISTRY,
    MapStrategy,
    StrategyParam,
    generate_map,
)
from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy
from hmls.mapgenerator.generators.perlin import PerlinNoiseStrategy

__all__ = [
    "generate_map",
    "MapStrategy",
    "StrategyParam",
    "STRATEGY_REGISTRY",
    "BlobAndLineStrategy",
    "PerlinNoiseStrategy",
]
