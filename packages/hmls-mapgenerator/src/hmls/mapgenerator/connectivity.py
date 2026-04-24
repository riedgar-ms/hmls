"""Connectivity utilities for :class:`~hmls.core.GameMap`.

This module provides algorithms for analysing and enforcing connectivity
of regions in a game map.  The two main guarantees it provides are:

1. **Passable connectivity**: all passable cells form a single connected
   component (using 4-connectivity: up/down/left/right).
2. **Impassable connectivity** (optional): all impassable cells form a
   single connected component, achieved by bridging disjoint regions.

Algorithm overview
------------------
- *Flood fill* uses BFS from a seed cell, collecting all cells that satisfy
  a predicate and are reachable via 4-connected steps.
- *Component finding* partitions all cells matching a predicate into their
  connected components by repeated flood fills.
- *Corridor carving* connects two cells by walking a Manhattan-distance
  (L-shaped) path between them, setting each cell along the path to the
  desired state.
- *Ensure connectivity* iteratively merges the two closest components
  until only one remains.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from hmls.core import CellType, GameMap


def flood_fill(
    game_map: GameMap,
    start: tuple[int, int],
    predicate: Callable[[GameMap, int, int], bool],
) -> set[tuple[int, int]]:
    """BFS flood fill from *start*, collecting all 4-connected cells
    that satisfy *predicate*.

    The predicate receives ``(game_map, x, y)`` and should return
    ``True`` if the cell should be included in the filled region.

    Args:
        game_map: The map to flood fill on.
        start: ``(x, y)`` seed position.  Must satisfy the predicate.
        predicate: Function that decides whether a cell belongs to the region.

    Returns:
        Set of ``(x, y)`` positions reachable from *start* via cells
        that satisfy the predicate.
    """
    sx, sy = start
    if not predicate(game_map, sx, sy):
        return set()

    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        x, y = queue.popleft()
        for nx, ny in game_map.neighbours(x, y):
            if (nx, ny) not in visited and predicate(game_map, nx, ny):
                visited.add((nx, ny))
                queue.append((nx, ny))

    return visited


def find_components(
    game_map: GameMap,
    predicate: Callable[[GameMap, int, int], bool],
) -> list[set[tuple[int, int]]]:
    """Find all connected components of cells satisfying *predicate*.

    Uses repeated flood fills.  Each component is a set of ``(x, y)``
    positions.  Components are returned in arbitrary order.

    Args:
        game_map: The map to analyse.
        predicate: Function ``(game_map, x, y) -> bool`` selecting
            which cells to consider.

    Returns:
        List of components (sets of positions).  Empty list if no cells
        satisfy the predicate.
    """
    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []

    for pos in game_map.all_positions():
        x, y = pos
        if pos not in seen and predicate(game_map, x, y):
            component = flood_fill(game_map, pos, predicate)
            seen |= component
            components.append(component)

    return components


def _closest_pair(
    comp_a: set[tuple[int, int]],
    comp_b: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the pair of cells (one from each component) with the smallest
    Manhattan distance.

    This is O(|comp_a| × |comp_b|).  For typical game map sizes this is fine.
    """
    best_dist = float("inf")
    best_a = next(iter(comp_a))
    best_b = next(iter(comp_b))

    for ax, ay in comp_a:
        for bx, by in comp_b:
            dist = abs(ax - bx) + abs(ay - by)
            if dist < best_dist:
                best_dist = dist
                best_a = (ax, ay)
                best_b = (bx, by)

    return best_a, best_b


def carve_corridor(
    game_map: GameMap,
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    *,
    cell_type: CellType = CellType.PASSABLE,
) -> list[tuple[int, int]]:
    """Carve an L-shaped corridor between two cells.

    Walks horizontally from *cell_a* to align with *cell_b*'s column,
    then vertically to reach *cell_b*.  Every cell along the path is set
    to *cell_type*.

    Args:
        game_map: The map to modify in place.
        cell_a: Starting ``(x, y)`` position.
        cell_b: Ending ``(x, y)`` position.
        cell_type: The cell type to assign along the corridor.

    Returns:
        List of ``(x, y)`` positions that were changed.
    """
    ax, ay = cell_a
    bx, by = cell_b
    changed: list[tuple[int, int]] = []

    # Step 1: walk horizontally from ax to bx at row ay
    step_x = 1 if bx >= ax else -1
    x = ax
    while x != bx:
        if game_map[x, ay] != cell_type:
            game_map[x, ay] = cell_type
            changed.append((x, ay))
        x += step_x

    # Step 2: walk vertically from ay to by at column bx
    step_y = 1 if by >= ay else -1
    y = ay
    while y != by:
        if game_map[bx, y] != cell_type:
            game_map[bx, y] = cell_type
            changed.append((bx, y))
        y += step_y

    # Ensure the endpoint itself is set
    if game_map[bx, by] != cell_type:
        game_map[bx, by] = cell_type
        changed.append((bx, by))

    return changed


def _ensure_connectivity(
    game_map: GameMap,
    predicate: Callable[[GameMap, int, int], bool],
    *,
    cell_type: CellType,
) -> int:
    """Merge all components matching *predicate* by carving corridors.

    Iteratively finds the two closest components (by Manhattan distance)
    and carves an L-shaped corridor between them, setting cells along the
    path to *cell_type*.  Repeats until a single component remains.

    After each corridor carve, components are recomputed from scratch so
    that incidental merges are captured automatically.

    Args:
        game_map: The map to modify in place.
        predicate: Function ``(game_map, x, y) -> bool`` selecting
            which cells belong to the regions to connect.
        cell_type: Value to assign to cells along carved corridors.

    Returns:
        Number of corridors carved (0 if already connected or fewer than
        two components exist).
    """
    components = find_components(game_map, predicate)

    if len(components) <= 1:
        return 0

    corridors_carved = 0

    while len(components) > 1:
        # Find the pair of components with the smallest gap
        best_a, best_b = _closest_pair(components[0], components[1])
        best_dist = abs(best_a[0] - best_b[0]) + abs(best_a[1] - best_b[1])

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if i == 0 and j == 1:
                    continue
                a, b = _closest_pair(components[i], components[j])
                dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if dist < best_dist:
                    best_dist = dist
                    best_a, best_b = a, b

        carve_corridor(game_map, best_a, best_b, cell_type=cell_type)
        corridors_carved += 1

        # Recompute so we capture incidental merges
        components = find_components(game_map, predicate)

    return corridors_carved


def ensure_passable_connectivity(game_map: GameMap) -> int:
    """Ensure all passable cells are 4-connected by carving corridors.

    Finds all connected components of passable terrain, then iteratively
    merges the two closest by carving a passable corridor between them.
    Repeats until a single component remains.

    This may reduce the impassable fraction slightly.

    Args:
        game_map: The map to modify in place.

    Returns:
        Number of corridors carved (0 if already connected).
    """
    return _ensure_connectivity(
        game_map,
        lambda gm, x, y: gm[x, y] == CellType.PASSABLE,
        cell_type=CellType.PASSABLE,
    )


def connect_impassable_regions(game_map: GameMap) -> int:
    """Connect disjoint impassable regions by carving impassable bridges.

    Operates on impassable cells, turning passable cells into impassable
    ones to bridge gaps between disjoint impassable regions.

    Warning: this may disconnect passable terrain.  Typically called
    *before* :func:`ensure_passable_connectivity` so the passable fix-up
    runs afterwards.

    Args:
        game_map: The map to modify in place.

    Returns:
        Number of bridges carved (0 if already connected or no impassable cells).
    """
    return _ensure_connectivity(
        game_map,
        lambda gm, x, y: gm[x, y] == CellType.IMPASSABLE,
        cell_type=CellType.IMPASSABLE,
    )
