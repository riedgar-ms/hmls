"""Map generation entry point.

This module provides the :func:`generate_map` convenience function,
separated from the base classes in :mod:`.base` to avoid a circular
import (``base`` defines the abstract :class:`~.base.MapStrategy` that
concrete strategies subclass, while ``generate_map`` needs to reference
a concrete default strategy).
"""

from __future__ import annotations

import random
import warnings

from hmls.core import GameMap
from hmls.mapgenerator.connectivity import (
    connect_impassable_regions,
    ensure_passable_connectivity,
)
from hmls.mapgenerator.generators.base import MapStrategy
from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy


def generate_map(
    width: int,
    height: int,
    *,
    impassable_fraction: float = 0.3,
    shape: float | None = None,
    connected_obstacles: bool = False,
    seed: int | None = None,
    strategy: MapStrategy | None = None,
) -> GameMap:
    """Generate a randomised grid map.

    Args:
        width: Number of columns.
        height: Number of rows.
        impassable_fraction: Target fraction of impassable cells (0.0–1.0).
        shape: **Deprecated.**  Obstacle geometry blend for the default
            :class:`BlobAndLineStrategy`.  Pass
            ``strategy=BlobAndLineStrategy(shape=X)`` instead.  If provided
            without a strategy, auto-constructs the default strategy with
            this shape value.  Cannot be used together with an explicit
            strategy.
        connected_obstacles: If ``True``, attempt to bridge disjoint
            impassable regions before enforcing passable connectivity.
            Note: the passable connectivity step (which always runs last)
            may re-split impassable terrain by carving corridors through it,
            so this is a best-effort setting rather than a hard guarantee.
        seed: Random seed for reproducibility.  ``None`` for non-deterministic.
        strategy: Obstacle placement strategy.  Defaults to
            :class:`BlobAndLineStrategy` if not provided.

    Returns:
        A :class:`~hmls.core.GameMap` with passable terrain guaranteed to
        be fully 4-connected.

    Raises:
        ValueError: If *impassable_fraction* is outside ``[0.0, 1.0]``.
        TypeError: If both *shape* and *strategy* are provided.
    """
    if not 0.0 <= impassable_fraction <= 1.0:
        msg = f"impassable_fraction must be 0.0–1.0, got {impassable_fraction}"
        raise ValueError(msg)

    # Handle deprecated ``shape`` parameter
    if shape is not None and strategy is not None:
        msg = (
            "Cannot pass both 'shape' and 'strategy'. "
            "Use strategy=BlobAndLineStrategy(shape=X) instead."
        )
        raise TypeError(msg)
    if shape is not None:
        warnings.warn(
            "The 'shape' parameter is deprecated. "
            "Use strategy=BlobAndLineStrategy(shape=X) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        strategy = BlobAndLineStrategy(shape=shape)

    rng = random.Random(seed)
    game_map = GameMap(width=width, height=height)

    # Phase 1: Place obstacles using the chosen strategy
    if strategy is None:
        strategy = BlobAndLineStrategy()
    strategy.place_obstacles(game_map, impassable_fraction, rng)

    # Phase 2 (optional): Connect disjoint impassable regions
    if connected_obstacles:
        connect_impassable_regions(game_map)

    # Phase 3: Guarantee passable connectivity by carving corridors
    ensure_passable_connectivity(game_map)

    return game_map
