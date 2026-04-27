"""Tests for the fog-of-war visibility system."""

from __future__ import annotations

import pytest

from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.core.visibility import (
    FogCell,
    PlayerView,
    VisibleCell,
    build_player_view,
    compute_visibility_mask,
    extract_patch,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_state(
    width: int = 9,
    height: int = 9,
    tanks: list[Tank] | None = None,
) -> GameState:
    """Create a simple game state for testing."""
    game_map = GameMap(width=width, height=height)
    tanks = tanks or []
    turn_order = [t.id for t in tanks]
    return GameState(game_map=game_map, tanks=tanks, turn_order=turn_order)


# ── compute_visibility_mask ───────────────────────────────────────────


class TestComputeVisibilityMask:
    """Tests for compute_visibility_mask."""

    def test_invalid_even_size(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            compute_visibility_mask(4)

    def test_invalid_too_small(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            compute_visibility_mask(1)

    def test_size_3_shape(self) -> None:
        mask = compute_visibility_mask(3)
        assert len(mask) == 3
        assert all(len(row) == 3 for row in mask)

    def test_size_3_all_visible(self) -> None:
        """A 3x3 patch is entirely the 8-neighbour ring, so all cells visible."""
        mask = compute_visibility_mask(3)
        for row in mask:
            for cell in row:
                assert cell is True

    def test_centre_always_visible(self) -> None:
        for n in (3, 5, 7):
            mask = compute_visibility_mask(n)
            half = n // 2
            assert mask[half][half] is True

    def test_eight_neighbours_visible(self) -> None:
        """All 8 neighbours of the centre should be visible."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                assert mask[half + dr][half + dc] is True

    def test_forward_cone_visible(self) -> None:
        """Cells straight ahead (dr > 1, dc = 0) should be visible."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        for row in range(half - 1):
            assert mask[row][half] is True, f"row {row} should be visible"

    def test_forward_cone_diagonal(self) -> None:
        """Cells at 45° from forward (|dc| == dr) should be visible."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        # 2 cells forward, 2 cells right (dr=2, dc=2): |dc| == dr → visible
        assert mask[half - 2][half + 2] is True
        assert mask[half - 2][half - 2] is True

    def test_rear_cells_not_visible(self) -> None:
        """Cells behind the tank (dr < -1) beyond the ring should be fogged."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        # 2 rows behind centre
        assert mask[half + 2][half] is False

    def test_side_cells_not_visible(self) -> None:
        """Cells to the side beyond the ring should be fogged."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        # 2 cells to the right, same row
        assert mask[half][half + 2] is False

    def test_outside_cone_not_visible(self) -> None:
        """Cells forward but outside 45° should be fogged."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        # 1 forward, 2 right → |dc| > dr → not visible
        assert mask[half - 1][half + 2] is False

    def test_mask_symmetry(self) -> None:
        """The forward cone should be symmetric left-right."""
        mask = compute_visibility_mask(7)
        half = 7 // 2
        for row in range(7):
            for col in range(half + 1):
                mirror_col = 7 - 1 - col
                assert mask[row][col] == mask[row][mirror_col], (
                    f"Asymmetry at row={row}, col={col} vs {mirror_col}"
                )


# ── extract_patch rotation ────────────────────────────────────────────


class TestExtractPatchRotation:
    """Test that patches are correctly rotated to egocentric space."""

    def _make_marked_state(self, direction: Direction) -> tuple[GameState, str]:
        """Create a state with a wall one cell ahead of the tank.

        Returns the state and the tank ID.
        """
        game_map = GameMap(width=9, height=9)
        # Place a wall one cell in the tank's forward direction.
        tank_pos = Position(4, 4)
        dx, dy = direction.forward_delta()
        wall_x, wall_y = tank_pos.x + dx, tank_pos.y + dy
        game_map[wall_x, wall_y] = CellType.IMPASSABLE

        tank = Tank(id="t0", team="a", position=tank_pos, direction=direction)
        state = _make_state(tanks=[tank])
        state = state.model_copy(update={"game_map": game_map})
        return state, "t0"

    @pytest.mark.parametrize("direction", list(Direction))
    def test_wall_ahead_appears_at_row_above_centre(self, direction: Direction) -> None:
        """Regardless of world direction, a wall ahead should appear one row above centre."""
        state, tid = self._make_marked_state(direction)
        patch = extract_patch(state, tid, 5)
        half = 5 // 2
        # One row above centre, same column = forward cell in ego space
        cell = patch.grid[half - 1][half]
        assert isinstance(cell, VisibleCell)
        assert cell.cell_type == CellType.IMPASSABLE

    @pytest.mark.parametrize("direction", list(Direction))
    def test_centre_is_tank(self, direction: Direction) -> None:
        """The centre cell should contain the tank itself."""
        state, tid = self._make_marked_state(direction)
        patch = extract_patch(state, tid, 5)
        half = 5 // 2
        cell = patch.grid[half][half]
        assert isinstance(cell, VisibleCell)
        assert cell.tank is not None
        assert cell.tank.id == tid

    def test_enemy_to_right_appears_right_of_centre(self) -> None:
        """An enemy one cell to the right should appear at (half, half+1)."""
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        # Enemy one cell to the east (right when facing north)
        enemy = Tank(id="e0", team="b", position=Position(5, 4), direction=Direction.SOUTH)
        state = _make_state(tanks=[tank, enemy])
        patch = extract_patch(state, "t0", 5)
        half = 5 // 2
        cell = patch.grid[half][half + 1]
        assert isinstance(cell, VisibleCell)
        assert cell.tank is not None
        assert cell.tank.id == "e0"

    def test_enemy_behind_is_visible_in_ring(self) -> None:
        """An enemy one cell behind (in the 8-ring) should be visible."""
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        enemy = Tank(id="e0", team="b", position=Position(4, 5), direction=Direction.SOUTH)
        state = _make_state(tanks=[tank, enemy])
        patch = extract_patch(state, "t0", 5)
        half = 5 // 2
        # Behind = one row below centre in ego space
        cell = patch.grid[half + 1][half]
        assert isinstance(cell, VisibleCell)
        assert cell.tank is not None
        assert cell.tank.id == "e0"


# ── extract_patch edge cases ─────────────────────────────────────────


class TestExtractPatchEdgeCases:
    """Test edge cases for patch extraction."""

    def test_map_edge_produces_fog(self) -> None:
        """Cells outside the map should be FogCell."""
        tank = Tank(id="t0", team="a", position=Position(0, 0), direction=Direction.NORTH)
        state = _make_state(width=5, height=5, tanks=[tank])
        patch = extract_patch(state, "t0", 5)
        # Top-left corner of patch: ego position (0, 0) for a 5x5 patch
        # centred at (0, 0) means 2 cells forward and 2 cells left,
        # which is out of bounds.
        cell = patch.grid[0][0]
        assert isinstance(cell, FogCell)

    def test_impassable_cell_shown_as_impassable(self) -> None:
        """A visible impassable cell should report its type."""
        game_map = GameMap(width=9, height=9)
        game_map[4, 3] = CellType.IMPASSABLE  # one cell north of (4,4)
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        state = GameState(game_map=game_map, tanks=[tank], turn_order=["t0"])
        patch = extract_patch(state, "t0", 5)
        half = 5 // 2
        cell = patch.grid[half - 1][half]
        assert isinstance(cell, VisibleCell)
        assert cell.cell_type == CellType.IMPASSABLE

    def test_enemy_in_fog_not_visible(self) -> None:
        """An enemy outside the visibility cone should be fogged."""
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        # Enemy 2 cells behind and 2 to the right (outside 8-ring, behind)
        enemy = Tank(id="e0", team="b", position=Position(6, 6), direction=Direction.SOUTH)
        state = _make_state(tanks=[tank, enemy])
        patch = extract_patch(state, "t0", 7)
        # This enemy should be fogged — it's behind and to the side
        # World offset: (2, 2) from tank at (4,4). Facing NORTH:
        # forward=(0,-1), right=(1,0)
        # forward_steps = 2*0 + 2*(-1) = -2 (behind)
        # right_steps = 2*1 + 2*0 = 2
        # ego_row = 3 - (-2) = 5, ego_col = 3 + 2 = 5
        cell = patch.grid[5][5]
        assert isinstance(cell, FogCell)


# ── build_player_view ─────────────────────────────────────────────────


class TestBuildPlayerView:
    """Tests for build_player_view."""

    def test_one_alive_tank_produces_one_patch(self) -> None:
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        state = _make_state(tanks=[tank])
        view = build_player_view(state, "a", 5)
        assert len(view.patches) == 1
        assert view.patches[0].tank_id == "t0"

    def test_dead_tank_produces_no_patch(self) -> None:
        tank = Tank(
            id="t0",
            team="a",
            position=Position(4, 4),
            direction=Direction.NORTH,
            alive=False,
        )
        state = _make_state(tanks=[tank])
        view = build_player_view(state, "a", 5)
        assert len(view.patches) == 0

    def test_dead_tank_still_in_tank_info(self) -> None:
        tank = Tank(
            id="t0",
            team="a",
            position=Position(4, 4),
            direction=Direction.NORTH,
            alive=False,
        )
        state = _make_state(tanks=[tank])
        view = build_player_view(state, "a", 5)
        assert len(view.tanks) == 1
        assert view.tanks[0].alive is False

    def test_enemy_tanks_not_in_tank_info(self) -> None:
        ally = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        enemy = Tank(id="e0", team="b", position=Position(6, 6), direction=Direction.SOUTH)
        state = _make_state(tanks=[ally, enemy])
        view = build_player_view(state, "a", 5)
        tank_ids = [t.tank_id for t in view.tanks]
        assert "e0" not in tank_ids

    def test_multiple_alive_tanks_produce_multiple_patches(self) -> None:
        t0 = Tank(id="t0", team="a", position=Position(2, 2), direction=Direction.NORTH)
        t1 = Tank(id="t1", team="a", position=Position(6, 6), direction=Direction.SOUTH)
        state = _make_state(tanks=[t0, t1])
        view = build_player_view(state, "a", 5)
        assert len(view.patches) == 2
        patch_ids = {p.tank_id for p in view.patches}
        assert patch_ids == {"t0", "t1"}

    def test_player_view_is_serialisable(self) -> None:
        """PlayerView must round-trip through JSON."""
        tank = Tank(id="t0", team="a", position=Position(4, 4), direction=Direction.NORTH)
        state = _make_state(tanks=[tank])
        view = build_player_view(state, "a", 5)
        json_str = view.model_dump_json()
        restored = PlayerView.model_validate_json(json_str)
        assert len(restored.patches) == len(view.patches)
        assert len(restored.tanks) == len(view.tanks)
