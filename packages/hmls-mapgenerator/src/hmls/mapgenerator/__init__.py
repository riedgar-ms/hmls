"""hmls.mapgenerator — A randomised game map generator with Textual TUI."""

from hmls.mapgenerator.generators import (
    STRATEGY_REGISTRY,
    BlobAndLineConfig,
    BlobAndLineStrategy,
    MapStrategy,
    PerlinNoiseConfig,
    PerlinNoiseStrategy,
    StrategyConfig,
    StrategyConfigBase,
    StrategyParam,
    generate_map,
    generate_map_from_config,
    register_strategy,
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
