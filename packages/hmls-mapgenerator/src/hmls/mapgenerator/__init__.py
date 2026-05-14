"""hmls.mapgenerator — A randomised game map generator with Textual TUI."""

from hmls.mapgenerator.generators import (
    STRATEGY_REGISTRY,
    BlobAndLineStrategy,
    MapStrategy,
    PerlinNoiseStrategy,
    StrategyParam,
    generate_map,
    register_strategy,
)

__all__ = [
    "generate_map",
    "MapStrategy",
    "StrategyParam",
    "STRATEGY_REGISTRY",
    "register_strategy",
    "BlobAndLineStrategy",
    "PerlinNoiseStrategy",
]
