"""Tests for connectivity utilities."""

from hmls.core import CellType, GameMap
from hmls.mapgenerator.connectivity import (
    carve_corridor,
    connect_impassable_regions,
    ensure_passable_connectivity,
    find_components,
    flood_fill,
)


def _is_passable(gm: GameMap, x: int, y: int) -> bool:
    return gm[x, y] == CellType.PASSABLE


def _is_impassable(gm: GameMap, x: int, y: int) -> bool:
    return gm[x, y] == CellType.IMPASSABLE


class TestFloodFill:
    """Tests for BFS flood fill."""

    def test_fill_all_passable(self) -> None:
        """Filling from any cell on an all-passable map reaches everything."""
        gm = GameMap(width=5, height=5)
        filled = flood_fill(gm, (0, 0), _is_passable)
        assert len(filled) == 25

    def test_fill_blocked(self) -> None:
        """A vertical wall at x=2 should split the grid."""
        gm = GameMap(width=5, height=5)
        for y in range(5):
            gm[2, y] = CellType.IMPASSABLE
        filled = flood_fill(gm, (0, 0), _is_passable)
        assert len(filled) == 10
        assert all(x < 2 for x, y in filled)

    def test_fill_with_non_matching_start(self) -> None:
        """Starting on a cell that fails the predicate returns empty."""
        gm = GameMap(width=3, height=3)
        gm[1, 1] = CellType.IMPASSABLE
        filled = flood_fill(gm, (1, 1), _is_passable)
        assert len(filled) == 0


class TestFindComponents:
    """Tests for connected component discovery."""

    def test_single_component(self) -> None:
        """An all-passable map has one component."""
        gm = GameMap(width=5, height=5)
        comps = find_components(gm, _is_passable)
        assert len(comps) == 1
        assert len(comps[0]) == 25

    def test_two_components(self) -> None:
        """A vertical wall splits passable terrain into two components."""
        gm = GameMap(width=5, height=5)
        for y in range(5):
            gm[2, y] = CellType.IMPASSABLE
        comps = find_components(gm, _is_passable)
        assert len(comps) == 2
        sizes = sorted(len(c) for c in comps)
        assert sizes == [10, 10]

    def test_no_matching_cells(self) -> None:
        """If no cells match, return an empty list."""
        gm = GameMap(width=3, height=3)
        comps = find_components(gm, _is_impassable)
        assert len(comps) == 0


class TestCarveCorridor:
    """Tests for L-shaped corridor carving."""

    def test_carve_horizontal(self) -> None:
        """Carving a horizontal corridor on an all-impassable 5×1 map."""
        gm = GameMap(width=5, height=1)
        for x in range(5):
            gm[x, 0] = CellType.IMPASSABLE
        changed = carve_corridor(gm, (0, 0), (4, 0), cell_type=CellType.PASSABLE)
        assert all(gm[x, 0] == CellType.PASSABLE for x in range(5))
        assert len(changed) == 5

    def test_carve_l_shape(self) -> None:
        """L-shaped corridor: horizontal then vertical."""
        gm = GameMap(width=5, height=5)
        for x, y in gm.all_positions():
            gm[x, y] = CellType.IMPASSABLE
        carve_corridor(gm, (0, 0), (4, 4))
        # Horizontal at y=0
        for x in range(5):
            assert gm[x, 0] == CellType.PASSABLE
        # Vertical at x=4
        for y in range(5):
            assert gm[4, y] == CellType.PASSABLE

    def test_carve_impassable(self) -> None:
        """Carving an impassable corridor on a passable map."""
        gm = GameMap(width=5, height=1)
        carve_corridor(gm, (1, 0), (3, 0), cell_type=CellType.IMPASSABLE)
        assert gm[0, 0] == CellType.PASSABLE
        assert gm[1, 0] == CellType.IMPASSABLE
        assert gm[2, 0] == CellType.IMPASSABLE
        assert gm[3, 0] == CellType.IMPASSABLE
        assert gm[4, 0] == CellType.PASSABLE


class TestEnsurePassableConnectivity:
    """Tests for passable connectivity enforcement."""

    def test_already_connected(self) -> None:
        """No corridors needed for an all-passable map."""
        gm = GameMap(width=5, height=5)
        result = ensure_passable_connectivity(gm)
        assert result == 0

    def test_splits_are_fixed(self) -> None:
        """A vertical wall is carved through to reconnect."""
        gm = GameMap(width=5, height=5)
        for y in range(5):
            gm[2, y] = CellType.IMPASSABLE
        comps_before = find_components(gm, _is_passable)
        assert len(comps_before) == 2

        ensure_passable_connectivity(gm)
        comps_after = find_components(gm, _is_passable)
        assert len(comps_after) == 1

    def test_multiple_islands(self) -> None:
        """Multiple isolated passable islands get connected."""
        gm = GameMap(width=9, height=9)
        for x, y in gm.all_positions():
            gm[x, y] = CellType.IMPASSABLE
        gm[1, 1] = CellType.PASSABLE
        gm[4, 4] = CellType.PASSABLE
        gm[7, 7] = CellType.PASSABLE

        ensure_passable_connectivity(gm)
        comps = find_components(gm, _is_passable)
        assert len(comps) == 1

    def test_fully_impassable_grid(self) -> None:
        """No passable cells — nothing to connect."""
        gm = GameMap(width=3, height=3)
        for x, y in gm.all_positions():
            gm[x, y] = CellType.IMPASSABLE
        result = ensure_passable_connectivity(gm)
        assert result == 0


class TestConnectImpassableRegions:
    """Tests for impassable bridging."""

    def test_disjoint_impassable_bridged(self) -> None:
        """Two disjoint impassable cells get bridged."""
        gm = GameMap(width=9, height=1)
        gm[0, 0] = CellType.IMPASSABLE
        gm[8, 0] = CellType.IMPASSABLE
        comps_before = find_components(gm, _is_impassable)
        assert len(comps_before) == 2

        connect_impassable_regions(gm)
        comps_after = find_components(gm, _is_impassable)
        assert len(comps_after) == 1
