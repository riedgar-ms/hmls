"""Map generation base classes and entry point.

This module provides the :func:`generate_map` entry point and defines the
:class:`MapStrategy` abstract base class for pluggable obstacle-placement
algorithms.

Architecture
------------
The generation pipeline has three phases:

1. **Obstacle placement** — delegated to a :class:`MapStrategy` subclass.
   The strategy receives an all-passable :class:`~hmls.core.GameMap` and is
   responsible for marking cells as impassable until the target fraction is
   approximately reached.

2. **Impassable connectivity** (optional) — if requested, disjoint impassable
   regions are bridged so they form a single connected mass.  This is handled
   by :func:`~hmls.mapgenerator.connectivity.connect_impassable_regions`.

3. **Passable connectivity enforcement** — always runs last.  Guarantees that
   all passable cells form a single 4-connected component by carving corridors
   between disconnected regions.  This may slightly reduce the actual
   impassable fraction below the target.

Writing a new strategy
----------------------
Subclass :class:`MapStrategy` and implement ``place_obstacles()``.
Then pass an instance to ``generate_map(strategy=...)``.

To make your strategy configurable via the TUI:

1. Accept parameters in ``__init__()`` and validate them.
2. Override the ``params`` class variable with a tuple of
   :class:`StrategyParam` descriptors.  The TUI reads these to build input
   widgets dynamically.
3. Register the strategy in :data:`STRATEGY_REGISTRY`.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

from hmls.core import GameMap
from hmls.mapgenerator.connectivity import (
    connect_impassable_regions,
    ensure_passable_connectivity,
)

# ── Strategy parameter descriptor ─────────────────────────────────────


@dataclass(frozen=True)
class StrategyParam:
    """Describes a configurable parameter for a map generation strategy.

    This is metadata used by the TUI (and potentially other UIs) to
    dynamically build input widgets for strategy-specific settings.
    Every :class:`MapStrategy` subclass exposes a ``params`` class variable
    (defaulting to an empty tuple) that the TUI reads to build input widgets.

    Attributes:
        name: Constructor keyword argument name.
        label: Human-readable label for display.
        param_type: Expected Python type (``float`` or ``int``).
        default: Default value if the user provides nothing.
        min_val: Minimum allowed value (inclusive), or ``None`` for unbounded.
        max_val: Maximum allowed value (inclusive), or ``None`` for unbounded.
    """

    name: str
    label: str
    param_type: type
    default: float | int
    min_val: float | int | None = None
    max_val: float | int | None = None


# ── Strategy base class ───────────────────────────────────────────────


class MapStrategy(ABC):
    """Abstract base class for map generation strategies.

    A strategy is responsible for placing impassable terrain on an
    initially all-passable :class:`~hmls.core.GameMap`.  The
    :func:`generate_map` function handles connectivity enforcement
    separately.

    Subclasses **must** implement :meth:`place_obstacles`.  They may also
    override :attr:`params` with a tuple of :class:`StrategyParam`
    descriptors so the TUI can build input widgets for strategy-specific
    settings.  Strategy-specific configuration (e.g. shape, scale, octaves)
    belongs on the concrete subclass.

    Attributes:
        params: Tuple of :class:`StrategyParam` descriptors for this
            strategy's configurable parameters.  Defaults to an empty tuple
            (no configurable parameters).
    """

    params: tuple[StrategyParam, ...] = ()

    @abstractmethod
    def place_obstacles(
        self,
        game_map: GameMap,
        fraction: float,
        rng: random.Random,
    ) -> None:
        """Place impassable cells on *game_map*.

        Args:
            game_map: An all-passable map to modify in place.
            fraction: Target fraction of cells that should be impassable
                (0.0 to 1.0).  The strategy should get close to this value
                but need not be exact.
            rng: Seeded :class:`random.Random` instance for reproducibility.
        """


# ── Strategy registry ─────────────────────────────────────────────────

# Maps display names to strategy classes.  The TUI uses this to populate
# the strategy selector.  Each registered class exposes ``params``
# (inherited from MapStrategy) and a constructor that accepts those
# params as keyword arguments.
STRATEGY_REGISTRY: dict[str, type[MapStrategy]] = {}


# ── Main entry point ──────────────────────────────────────────────────


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
    # Lazy import to avoid circular dependency
    from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy

    if not 0.0 <= impassable_fraction <= 1.0:
        raise ValueError(f"impassable_fraction must be 0.0–1.0, got {impassable_fraction}")

    # Handle deprecated ``shape`` parameter
    if shape is not None and strategy is not None:
        raise TypeError(
            "Cannot pass both 'shape' and 'strategy'. "
            "Use strategy=BlobAndLineStrategy(shape=X) instead."
        )
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
