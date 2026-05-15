"""Blob and line obstacle placement strategy.

This module implements the default :class:`BlobAndLineStrategy` which places
obstacles using a mix of filled ellipses and thick line segments.
"""

from __future__ import annotations

import random
from typing import ClassVar, Literal

from pydantic import Field

from hmls.core import CellType, GameMap
from hmls.mapgenerator.generators.base import (
    MapStrategy,
    StrategyConfigBase,
    register_strategy,
)

# ── Pydantic configuration model ─────────────────────────────────────


class BlobAndLineConfig(StrategyConfigBase, frozen=True, extra="forbid"):
    """Configuration for the Blob & Line map generation strategy.

    Serialisable Pydantic model that captures the parameters for
    :class:`BlobAndLineStrategy`.  The ``type`` literal serves as
    the discriminator in the :data:`~hmls.mapgenerator.generators.StrategyConfig`
    union.

    Attributes:
        type: Discriminator literal, always ``"blob_and_line"``.
        shape: Obstacle geometry blend.  0.0 = all linear walls,
            1.0 = all blobs (filled ellipses), 0.5 = mixed.
    """

    type: Literal["blob_and_line"] = "blob_and_line"
    shape: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Shape",
        description="0 = linear, 1 = blobs",
    )

    def create_strategy(self) -> BlobAndLineStrategy:
        """Create a :class:`BlobAndLineStrategy` with the configured shape."""
        return BlobAndLineStrategy(shape=self.shape)


# ── Strategy class ────────────────────────────────────────────────────


@register_strategy
class BlobAndLineStrategy(MapStrategy):
    """Default obstacle placement using a mix of ellipses and line segments.

    Algorithm
    ---------
    Obstacles are placed one at a time until the impassable cell count
    reaches ``floor(fraction × total_cells)``.  Since each shape adds
    multiple cells, the final count may slightly overshoot the target.
    For each obstacle:

    1. A random number ``r`` in ``[0, 1)`` is drawn.
    2. If ``r < shape``, a **blob** (filled ellipse) is placed at a
       random position with random radii.
    3. Otherwise, a **linear wall** (thick line segment) is placed between
       two random endpoints with a random thickness.

    Attributes:
        shape: Obstacle geometry blend.  0.0 = all linear walls,
            1.0 = all blobs (filled ellipses), 0.5 = mixed.
    """

    display_name = "Blob & Line"
    config_class: ClassVar[type[StrategyConfigBase]] = BlobAndLineConfig

    def __init__(self, shape: float = 0.5) -> None:
        """Create a blob-and-line strategy.

        Args:
            shape: Obstacle geometry blend (0.0–1.0).

        Raises:
            ValueError: If *shape* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= shape <= 1.0:
            raise ValueError(f"shape must be 0.0–1.0, got {shape}")  # noqa: EM102
        self.shape = shape

    def place_obstacles(
        self,
        game_map: GameMap,
        fraction: float,
        rng: random.Random,
    ) -> None:
        """Place impassable cells using a mix of blobs and lines.

        Args:
            game_map: An all-passable map to modify in place.
            fraction: Target fraction of impassable cells (0.0–1.0).
            rng: Seeded RNG for reproducibility.
        """
        target_count = int(fraction * game_map.total_cells)
        current_count = game_map.count_impassable()

        min_dim = min(game_map.width, game_map.height)

        max_zero_progress = 50
        zero_progress_streak = 0

        while current_count < target_count:
            remaining = target_count - current_count
            if remaining <= 0:
                break

            # Fallback: after too many failed attempts, fill random passable cells
            if zero_progress_streak >= max_zero_progress:
                passable_cells = [
                    (x, y)
                    for x, y in game_map.all_positions()
                    if game_map[x, y] == CellType.PASSABLE
                ]
                if not passable_cells:
                    break
                rng.shuffle(passable_cells)
                for x, y in passable_cells[:remaining]:
                    game_map[x, y] = CellType.IMPASSABLE
                    current_count += 1
                break

            if rng.random() < self.shape:
                added = self._place_ellipse(game_map, rng, min_dim)
            else:
                added = self._place_line(game_map, rng, min_dim)

            if added == 0:
                zero_progress_streak += 1
            else:
                zero_progress_streak = 0
                current_count += added

    def _place_ellipse(
        self,
        game_map: GameMap,
        rng: random.Random,
        min_dim: int,
    ) -> int:
        """Place a filled ellipse of impassable cells at a random position.

        Returns the number of newly impassable cells created.
        """
        cx = rng.randint(0, game_map.width - 1)
        cy = rng.randint(0, game_map.height - 1)

        max_r = max(1, min_dim // 4)
        rx = rng.randint(1, max_r)
        ry = rng.randint(1, max_r)

        count = 0
        for y in range(max(0, cy - ry), min(game_map.height, cy + ry + 1)):
            for x in range(max(0, cx - rx), min(game_map.width, cx + rx + 1)):
                dx = x - cx
                dy = y - cy
                if (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1.0:
                    if game_map[x, y] == CellType.PASSABLE:
                        game_map[x, y] = CellType.IMPASSABLE
                        count += 1

        return count

    def _place_line(
        self,
        game_map: GameMap,
        rng: random.Random,
        min_dim: int,
    ) -> int:
        """Place a thick line segment of impassable cells.

        Returns the number of newly impassable cells created.
        """
        x0 = rng.randint(0, game_map.width - 1)
        y0 = rng.randint(0, game_map.height - 1)
        x1 = rng.randint(0, game_map.width - 1)
        y1 = rng.randint(0, game_map.height - 1)

        thickness = rng.randint(1, max(1, min_dim // 10))
        line_cells = _bresenham(x0, y0, x1, y1)

        count = 0
        for lx, ly in line_cells:
            for dy in range(-thickness, thickness + 1):
                for dx in range(-thickness, thickness + 1):
                    if dx * dx + dy * dy <= thickness * thickness:
                        nx, ny = lx + dx, ly + dy
                        if game_map.in_bounds(nx, ny) and game_map[nx, ny] == CellType.PASSABLE:
                            game_map[nx, ny] = CellType.IMPASSABLE
                            count += 1

        return count


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm.  Returns a list of ``(x, y)`` cells."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return cells
