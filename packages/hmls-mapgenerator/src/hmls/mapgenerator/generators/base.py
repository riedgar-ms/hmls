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
Subclass :class:`MapStrategy`, set a ``display_name`` class variable,
implement ``place_obstacles()``, and decorate with :func:`register_strategy`.

To make your strategy configurable via the TUI:

1. Accept parameters in ``__init__()`` and validate them.
2. Override :meth:`~MapStrategy.get_params` to return a list of
   :class:`StrategyParam` descriptors.  The TUI reads these to build input
   widgets dynamically.
3. Place the module in the ``generators/`` directory — it will be
   auto-discovered via :func:`pkgutil.iter_modules` (see
   :mod:`hmls.mapgenerator.generators.__init__`).
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

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
    Every :class:`MapStrategy` subclass exposes a :meth:`~MapStrategy.get_params`
    classmethod (defaulting to an empty list) that the TUI reads to build
    input widgets.

    Attributes:
        name: Constructor keyword argument name.
        label: Human-readable label for display.
        param_type: Expected Python type (``float`` or ``int``).
        default: Default value if the user provides nothing.
        min_val: Minimum allowed value (inclusive), or ``None`` for unbounded.
        max_val: Maximum allowed value (inclusive), or ``None`` for unbounded.
        hint: Optional explanatory text for the parameter (e.g. shown as
            a subtitle or tooltip in the TUI).
    """

    name: str
    label: str
    param_type: type
    default: float | int
    min_val: float | int | None = None
    max_val: float | int | None = None
    hint: str | None = None


# ── Strategy base class ───────────────────────────────────────────────


class MapStrategy(ABC):
    """Abstract base class for map generation strategies.

    A strategy is responsible for placing impassable terrain on an
    initially all-passable :class:`~hmls.core.GameMap`.  The
    :func:`generate_map` function handles connectivity enforcement
    separately.

    Subclasses **must** implement :meth:`place_obstacles` and define a
    :attr:`display_name` class variable (used by the TUI strategy selector).

    Strategy-specific parameters are defined on the corresponding
    :class:`StrategyConfigBase` subclass (referenced via :attr:`config_class`).
    The :meth:`get_params` method is auto-derived from the config model's
    Pydantic field metadata — subclasses should **not** override it.

    Attributes:
        display_name: Human-readable name shown in the TUI strategy selector.
            Concrete subclasses must define this as a class variable.
        config_class: The corresponding :class:`StrategyConfigBase` subclass
            that defines serialisable parameters for this strategy.
            ``None`` for strategies without configurable parameters.
    """

    display_name: ClassVar[str]
    config_class: ClassVar[type[StrategyConfigBase] | None] = None

    @classmethod
    def get_params(cls) -> list[StrategyParam]:
        """Return the configurable parameters for this strategy.

        Delegates to the :meth:`StrategyConfigBase.get_params` method
        on :attr:`config_class` if one is set.  Returns an empty list
        for strategies without a config class.

        Returns:
            A list of :class:`StrategyParam` descriptors.
        """
        if cls.config_class is not None:
            return cls.config_class.get_params()
        return []

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


# ── Strategy configuration base class ────────────────────────────────


def _extract_bounds(field_info: FieldInfo) -> tuple[float | int | None, float | int | None]:
    """Extract min/max bounds from Pydantic field metadata.

    Reads ``annotated_types.Ge``, ``Gt``, ``Le``, and ``Lt`` markers
    from *field_info.metadata* and returns ``(min_val, max_val)``.

    - ``Ge(ge=X)`` → ``min_val = X``
    - ``Gt(gt=X)`` → ``min_val = X`` (treated as inclusive for TUI clamping)
    - ``Le(le=X)`` → ``max_val = X``
    - ``Lt(lt=X)`` → ``max_val = X`` (treated as inclusive for TUI clamping)

    Args:
        field_info: A Pydantic :class:`~pydantic.fields.FieldInfo` instance.

    Returns:
        A ``(min_val, max_val)`` tuple, with ``None`` for unbounded sides.
    """
    min_val: float | int | None = None
    max_val: float | int | None = None
    for meta in field_info.metadata:
        if hasattr(meta, "ge"):
            min_val = meta.ge
        elif hasattr(meta, "gt"):
            min_val = meta.gt
        if hasattr(meta, "le"):
            max_val = meta.le
        elif hasattr(meta, "lt"):
            max_val = meta.lt
    return min_val, max_val


class StrategyConfigBase(BaseModel, ABC, frozen=True, extra="forbid"):
    """Abstract base for strategy configuration models.

    Each concrete subclass must:

    - Define a ``type`` field as a :class:`~typing.Literal` with a unique
      snake_case identifier (e.g. ``Literal["blob_and_line"]``).
    - Implement :meth:`create_strategy` to return a fully configured
      :class:`MapStrategy` instance.

    The ``type`` field serves as the Pydantic discriminator for the
    :data:`~hmls.mapgenerator.generators.StrategyConfig` union type,
    allowing JSON like ``{"type": "perlin_noise", "scale": 0.1}`` to be
    parsed into the correct concrete config class automatically.

    Subclasses inherit ``frozen=True`` and ``extra="forbid"`` for
    immutability and strict JSON parsing.
    """

    @abstractmethod
    def create_strategy(self) -> MapStrategy:
        """Instantiate and return a :class:`MapStrategy` with this config's parameters."""

    @classmethod
    def get_params(cls) -> list[StrategyParam]:
        """Derive :class:`StrategyParam` descriptors from Pydantic field metadata.

        Introspects the model's field definitions, skipping the ``type``
        discriminator, and builds a :class:`StrategyParam` for each
        remaining field.  Uses :attr:`~pydantic.fields.FieldInfo.title`
        as the TUI label and :attr:`~pydantic.fields.FieldInfo.description`
        as the hint text.

        Returns:
            A list of :class:`StrategyParam` descriptors.
        """
        params: list[StrategyParam] = []
        for name, field_info in cls.model_fields.items():
            if name == "type":
                continue
            annotation = field_info.annotation
            param_type: type = annotation if annotation in (int, float) else float
            label = field_info.title or name
            hint = field_info.description
            default = field_info.default
            min_val, max_val = _extract_bounds(field_info)
            params.append(
                StrategyParam(
                    name=name,
                    label=label,
                    param_type=param_type,
                    default=default,
                    min_val=min_val,
                    max_val=max_val,
                    hint=hint,
                )
            )
        return params


# ── Strategy registry ─────────────────────────────────────────────────

# Maps display names to strategy classes.  The TUI uses this to populate
# the strategy selector.  Each registered class exposes ``get_params()``
# (inherited from MapStrategy) and a constructor that accepts those
# params as keyword arguments.
STRATEGY_REGISTRY: dict[str, type[MapStrategy]] = {}


def register_strategy(cls: type[MapStrategy]) -> type[MapStrategy]:
    """Class decorator that registers a :class:`MapStrategy` subclass.

    Reads :attr:`~MapStrategy.display_name` from the decorated class and
    adds it to :data:`STRATEGY_REGISTRY` under that name.

    Usage::

        @register_strategy
        class MyStrategy(MapStrategy):
            display_name = "My Strategy"
            ...

    .. note::

        Registration happens at import time.  Strategy modules are
        auto-imported by :mod:`hmls.mapgenerator.generators.__init__`
        using :func:`pkgutil.iter_modules`, so simply placing a new
        module in the ``generators/`` directory is sufficient for it to
        be discovered — no manual edits to ``__init__.py`` required.

    Raises:
        TypeError: If *cls* does not define a ``display_name`` class variable.
    """
    if not hasattr(cls, "display_name") or not isinstance(cls.display_name, str):
        msg = (
            f"{cls.__name__} must define a 'display_name' class variable "
            f"(str) to be registered as a map strategy."
        )
        raise TypeError(msg)
    STRATEGY_REGISTRY[cls.display_name] = cls
    return cls


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
