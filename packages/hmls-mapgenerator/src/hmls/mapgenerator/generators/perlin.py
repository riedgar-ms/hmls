"""Perlin noise map generation strategy.

This module implements a :class:`PerlinNoiseStrategy` that uses 2D Perlin
noise to generate organic, natural-looking terrain.  The algorithm produces
smooth, continuous noise values across the grid and thresholds them to create
impassable regions.

Algorithm overview
------------------
1. **Permutation table** — A shuffled array of integers ``[0, 255]`` is used
   as a hash function to map grid coordinates to pseudo-random gradient
   indices.  The table is seeded from the caller-provided RNG for
   reproducibility.

2. **Gradient noise** — For each grid cell, the algorithm:
   a. Determines which unit square of the noise grid the cell falls in.
   b. Computes dot products between distance vectors and pseudo-random
      gradient vectors at each corner of the unit square.
   c. Interpolates the dot products using a smooth fade curve
      (``6t⁵ − 15t⁴ + 10t³``) to produce a single noise value.

3. **Octave layering (fractal noise)** — Multiple layers of noise at
   increasing frequencies and decreasing amplitudes are summed together.
   The ``octaves`` parameter controls how many layers are used.

4. **Thresholding** — All noise values are collected and sorted.  The
   lowest ``floor(fraction × total_cells)`` cells are marked impassable,
   so the result closely approximates (but may slightly undershoot) the
   requested fraction.  Connectivity enforcement may further adjust the
   final count.

References
----------
- Ken Perlin, "Improving Noise" (2002).
- Pure-Python clean-room implementation with no external dependencies.
"""

from __future__ import annotations

import math
import random
from typing import ClassVar, Literal

from pydantic import Field

from hmls.core import CellType, GameMap
from hmls.mapgenerator.generators.base import (
    MapStrategy,
    StrategyConfigBase,
    register_strategy,
)

# ── Perlin noise primitives───────────────────────────────────────────

# 12 gradient vectors for 2D Perlin noise (edges of a cube projected to 2D).
_GRAD2: list[tuple[int, int]] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
]


def _build_permutation_table(rng: random.Random) -> list[int]:
    """Build a shuffled permutation table for noise hashing.

    Args:
        rng: Seeded random number generator for reproducibility.

    Returns:
        A list of 512 integers — the shuffled ``[0, 255]`` repeated twice.
    """
    perm = list(range(256))
    rng.shuffle(perm)
    return perm + perm


def _fade(t: float) -> float:
    """Perlin's improved fade/ease curve: ``6t⁵ − 15t⁴ + 10t³``."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by factor *t*."""
    return a + t * (b - a)


def _grad2d(hash_val: int, x: float, y: float) -> float:
    """Dot product of a pseudo-random gradient with ``(x, y)``."""
    g = _GRAD2[hash_val % 12]
    return g[0] * x + g[1] * y


def _perlin2d(x: float, y: float, perm: list[int]) -> float:
    """Evaluate 2D Perlin noise at the point ``(x, y)``.

    Returns a value roughly in ``[-1, 1]``.

    Args:
        x: Horizontal coordinate in noise space.
        y: Vertical coordinate in noise space.
        perm: Permutation table from :func:`_build_permutation_table`.
    """
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255

    xf = x - math.floor(x)
    yf = y - math.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = _lerp(_grad2d(aa, xf, yf), _grad2d(ba, xf - 1, yf), u)
    x2 = _lerp(_grad2d(ab, xf, yf - 1), _grad2d(bb, xf - 1, yf - 1), u)

    return _lerp(x1, x2, v)


def _fractal_noise2d(
    x: float,
    y: float,
    perm: list[int],
    octaves: int,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> float:
    """Evaluate fractal (layered) 2D Perlin noise.

    Sums multiple octaves of Perlin noise at increasing frequencies
    and decreasing amplitudes to produce fractal detail.

    Args:
        x: Horizontal coordinate in noise space.
        y: Vertical coordinate in noise space.
        perm: Permutation table.
        octaves: Number of noise layers to sum.
        lacunarity: Frequency multiplier between octaves.
        persistence: Amplitude multiplier between octaves.
    """
    value = 0.0
    frequency = 1.0
    amplitude = 1.0

    for _ in range(octaves):
        value += amplitude * _perlin2d(x * frequency, y * frequency, perm)
        frequency *= lacunarity
        amplitude *= persistence

    return value


# ── Pydantic configuration model ─────────────────────────────────────


class PerlinNoiseConfig(StrategyConfigBase, frozen=True, extra="forbid"):
    """Configuration for the Perlin Noise map generation strategy.

    Serialisable Pydantic model that captures the parameters for
    :class:`PerlinNoiseStrategy`.  The ``type`` literal serves as
    the discriminator in the :data:`~hmls.mapgenerator.generators.StrategyConfig`
    union.

    Attributes:
        type: Discriminator literal, always ``"perlin_noise"``.
        scale: Noise scale / zoom level.  Lower values produce larger,
            smoother terrain features.
        octaves: Number of fractal noise layers summed together.
    """

    type: Literal["perlin_noise"] = "perlin_noise"
    scale: float = Field(
        default=0.05,
        ge=0.02,
        le=0.2,
        title="Noise scale",
        description="Lower values produce larger features",
    )
    octaves: int = Field(
        default=4,
        ge=1,
        le=8,
        title="Octaves",
        description="Number of fractal noise layers",
    )

    def create_strategy(self) -> PerlinNoiseStrategy:
        """Create a :class:`PerlinNoiseStrategy` with the configured parameters."""
        return PerlinNoiseStrategy(scale=self.scale, octaves=self.octaves)


# ── Strategy class ────────────────────────────────────────────────────


@register_strategy
class PerlinNoiseStrategy(MapStrategy):
    """Obstacle placement using 2D Perlin noise thresholding.

    Generates organic, natural-looking terrain by sampling fractal
    Perlin noise across the grid and marking cells below a computed
    threshold as impassable.

    Attributes:
        scale: Controls the "zoom level" of the noise.  Lower values produce
            larger, smoother terrain features.
        octaves: Number of noise layers summed together.
    """

    display_name = "Perlin Noise"
    config_class: ClassVar[type[StrategyConfigBase]] = PerlinNoiseConfig

    def __init__(self, scale: float = 0.05, octaves: int = 4) -> None:
        """Create a Perlin noise strategy.

        Args:
            scale: Noise scale / zoom level (must be positive).
            octaves: Number of fractal noise layers (must be >= 1).

        Raises:
            ValueError: If parameters are out of range.
        """
        if not 0.0 < scale:
            raise ValueError(f"scale must be positive, got {scale}")  # noqa: EM102
        if not 1 <= octaves:
            raise ValueError(f"octaves must be >= 1, got {octaves}")  # noqa: EM102
        self.scale = scale
        self.octaves = octaves

    def place_obstacles(
        self,
        game_map: GameMap,
        fraction: float,
        rng: random.Random,
    ) -> None:
        """Place impassable cells using Perlin noise thresholding.

        Args:
            game_map: An all-passable map to modify in place.
            fraction: Target fraction of impassable cells (0.0–1.0).
            rng: Seeded RNG for reproducibility.
        """
        if fraction <= 0.0:
            return

        perm = _build_permutation_table(rng)

        offset_x = rng.uniform(0, 1000)
        offset_y = rng.uniform(0, 1000)

        noise_values: list[tuple[float, int, int]] = []
        for x, y in game_map.all_positions():
            nx = (x + offset_x) * self.scale
            ny = (y + offset_y) * self.scale
            value = _fractal_noise2d(nx, ny, perm, self.octaves)
            noise_values.append((value, x, y))

        noise_values.sort(key=lambda t: t[0])
        target_count = int(fraction * game_map.total_cells)

        for i in range(min(target_count, len(noise_values))):
            _, cx, cy = noise_values[i]
            game_map[cx, cy] = CellType.IMPASSABLE
