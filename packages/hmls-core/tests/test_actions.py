"""Tests for hmls.core.actions – validate_action and apply_action."""

import pytest

from hmls.core.actions import apply_action, validate_action
from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position

# ── Helpers ───────────────────────────────────────────────────────────


def _make_map(width: int = 5, height: int = 5) -> GameMap:
    """Build a default all-passable GameMap for testing."""
    return GameMap(width=width, height=height)


def _make_state(
    tanks: list[Tank] | None = None,
    current_tank_id: str | None = None,
) -> GameState:
    """Build a GameState for testing."""
    if tanks is None:
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(id="b1", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
    if current_tank_id is None and tanks:
        current_tank_id = tanks[0].id
    return GameState(
        tanks=tanks,
        current_tank_id=current_tank_id,
    )


# ── Validation tests ──────────────────────────────────────────────────


class TestValidateAction:
    """Tests for validate_action."""

    def test_move_forward_valid(self) -> None:
        """Moving forward into an open cell is valid."""
        state = _make_state()
        game_map = _make_map()
        result = validate_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert result.valid

    def test_move_forward_out_of_bounds(self) -> None:
        """Moving forward off the map is invalid."""
        tanks = [Tank(id="a1", team="alpha", position=Position(2, 0), direction=Direction.NORTH)]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        result = validate_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert not result.valid
        assert "out of bounds" in result.reason.lower()

    def test_move_forward_impassable(self) -> None:
        """Moving forward into an impassable cell is invalid."""
        state = _make_state()
        game_map = _make_map()
        # Place a wall directly north of a1 at (2, 2): north is (2, 1).
        game_map[2, 1] = CellType.IMPASSABLE
        result = validate_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert not result.valid
        assert "impassable" in result.reason.lower()

    def test_move_forward_occupied(self) -> None:
        """Moving forward into a cell occupied by another tank is invalid."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(id="b1", team="beta", position=Position(2, 1), direction=Direction.SOUTH),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        result = validate_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert not result.valid
        assert "occupied" in result.reason.lower()

    def test_move_forward_blocked_by_wreckage(self) -> None:
        """Moving forward into a cell occupied by destroyed wreckage is invalid."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(
                id="b1",
                team="beta",
                position=Position(2, 1),
                direction=Direction.SOUTH,
                alive=False,
            ),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        result = validate_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert not result.valid
        assert "occupied" in result.reason.lower() or "wreckage" in result.reason.lower()
        """Turning is always valid."""
        state = _make_state()
        assert validate_action(state, game_map, "a1", Action.TURN_LEFT).valid
        assert validate_action(state, game_map, "a1", Action.TURN_RIGHT).valid

    def test_fire_always_valid(self) -> None:
        """Firing is always valid (regardless of what's ahead)."""
        state = _make_state()
        game_map = _make_map()
        assert validate_action(state, game_map, "a1", Action.FIRE).valid

    def test_pass_always_valid(self) -> None:
        """Passing is always valid."""
        state = _make_state()
        game_map = _make_map()
        assert validate_action(state, game_map, "a1", Action.PASS).valid

    def test_dead_tank_invalid(self) -> None:
        """A dead tank cannot act."""
        tanks = [
            Tank(
                id="a1",
                team="alpha",
                position=Position(2, 2),
                direction=Direction.NORTH,
                alive=False,
            ),
            Tank(id="b1", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        result = validate_action(state, game_map, "a1", Action.PASS)
        assert not result.valid
        assert "not alive" in result.reason.lower()

    def test_wrong_turn_invalid(self) -> None:
        """A tank that is not the current turn cannot act."""
        state = _make_state()
        game_map = _make_map()
        result = validate_action(state, game_map, "b1", Action.PASS)
        assert not result.valid
        assert "not tank" in result.reason.lower() or "turn" in result.reason.lower()

    def test_nonexistent_tank_invalid(self) -> None:
        """An unknown tank ID is invalid."""
        state = _make_state()
        game_map = _make_map()
        result = validate_action(state, game_map, "zz", Action.PASS)
        assert not result.valid


# ── Apply action tests ────────────────────────────────────────────────


class TestApplyAction:
    """Tests for apply_action."""

    def test_move_forward(self) -> None:
        """Moving forward updates the tank's position."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.MOVE_FORWARD)
        tank = new.get_tank("a1")
        assert tank.position == Position(2, 1)

    def test_move_forward_invalid_loses_turn(self) -> None:
        """Moving into impassable terrain does nothing but the action still completes."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 0), direction=Direction.NORTH),
            Tank(id="b1", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.MOVE_FORWARD)
        # Position unchanged.
        assert new.get_tank("a1").position == Position(2, 0)
        # Turn is NOT advanced — the engine owns scheduling.
        assert new.current_tank_id == "a1"

    def test_turn_left(self) -> None:
        """Turning left rotates the tank 90° counter-clockwise."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.TURN_LEFT)
        assert new.get_tank("a1").direction == Direction.WEST

    def test_turn_right(self) -> None:
        """Turning right rotates the tank 90° clockwise."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.TURN_RIGHT)
        assert new.get_tank("a1").direction == Direction.EAST

    def test_fire_destroys_enemy(self) -> None:
        """Firing at an adjacent enemy tank destroys it."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(id="b1", team="beta", position=Position(2, 1), direction=Direction.SOUTH),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert not new.get_tank("b1").alive

    def test_fire_destroys_friendly(self) -> None:
        """Friendly fire also destroys the target."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(id="a2", team="alpha", position=Position(2, 1), direction=Direction.SOUTH),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert not new.get_tank("a2").alive

    def test_fire_into_empty_cell(self) -> None:
        """Firing into an empty cell has no effect on tanks."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert all(t.alive for t in new.tanks)

    def test_fire_into_wall(self) -> None:
        """Firing into an impassable cell does nothing harmful."""
        state = _make_state()
        game_map = _make_map()
        game_map[2, 1] = CellType.IMPASSABLE
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert all(t.alive for t in new.tanks)

    def test_fire_into_wreckage(self) -> None:
        """Firing into a destroyed tank (wreckage) has no additional effect."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(
                id="b1",
                team="beta",
                position=Position(2, 1),
                direction=Direction.SOUTH,
                alive=False,
            ),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.FIRE)
        # b1 was already dead — should remain dead, no error.
        assert not new.get_tank("b1").alive

    def test_pass_does_nothing(self) -> None:
        """Passing leaves the state unchanged (including current_tank_id)."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.PASS)
        assert new.get_tank("a1").position == Position(2, 2)
        assert new.get_tank("a1").direction == Direction.NORTH
        assert new.current_tank_id == "a1"

    def test_turn_not_advanced(self) -> None:
        """apply_action does not advance the turn — the engine owns scheduling."""
        state = _make_state()
        game_map = _make_map()
        new = apply_action(state, game_map, "a1", Action.PASS)
        assert new.current_tank_id == "a1"

    def test_apply_dead_tank_raises(self) -> None:
        """Applying an action for a dead tank raises ValueError."""
        tanks = [
            Tank(
                id="a1",
                team="alpha",
                position=Position(2, 2),
                direction=Direction.NORTH,
                alive=False,
            ),
            Tank(id="b1", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        with pytest.raises(ValueError, match="not alive"):
            apply_action(state, game_map, "a1", Action.PASS)

    def test_apply_wrong_turn_raises(self) -> None:
        """Applying an action out of turn raises ValueError."""
        state = _make_state()
        game_map = _make_map()
        with pytest.raises(ValueError, match="turn"):
            apply_action(state, game_map, "b1", Action.PASS)

    def test_original_state_unchanged(self) -> None:
        """apply_action does not mutate the original state."""
        state = _make_state()
        game_map = _make_map()
        original_pos = state.get_tank("a1").position
        _ = apply_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert state.get_tank("a1").position == original_pos
        assert state.current_tank_id == "a1"


# ── Tank list order invariant tests ──────────────────────────────────


def _tank_ids(state: GameState) -> list[str]:
    """Return the list of tank IDs in their current list order."""
    return [t.id for t in state.tanks]


class TestTankListOrderPreserved:
    """The tanks list order must never change during a game.

    The tank list order is a fundamental invariant; any reordering
    would silently break consumers that rely on stable ordering.
    """

    def test_move_preserves_order(self) -> None:
        """Moving a tank does not change the tank list order."""
        state = _make_state()
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.MOVE_FORWARD)
        assert _tank_ids(new) == original

    def test_turn_left_preserves_order(self) -> None:
        """Turning left does not change the tank list order."""
        state = _make_state()
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.TURN_LEFT)
        assert _tank_ids(new) == original

    def test_turn_right_preserves_order(self) -> None:
        """Turning right does not change the tank list order."""
        state = _make_state()
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.TURN_RIGHT)
        assert _tank_ids(new) == original

    def test_fire_kill_preserves_order(self) -> None:
        """Destroying a tank via fire does not change the tank list order."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(2, 2), direction=Direction.NORTH),
            Tank(id="b1", team="beta", position=Position(2, 1), direction=Direction.SOUTH),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert not new.get_tank("b1").alive
        assert _tank_ids(new) == original

    def test_fire_miss_preserves_order(self) -> None:
        """Firing into an empty cell does not change the tank list order."""
        state = _make_state()
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.FIRE)
        assert _tank_ids(new) == original

    def test_pass_preserves_order(self) -> None:
        """Passing does not change the tank list order."""
        state = _make_state()
        game_map = _make_map()
        original = _tank_ids(state)
        new = apply_action(state, game_map, "a1", Action.PASS)
        assert _tank_ids(new) == original

    def test_multi_turn_preserves_order(self) -> None:
        """Tank list order is stable across multiple turns."""
        tanks = [
            Tank(id="a1", team="alpha", position=Position(0, 2), direction=Direction.EAST),
            Tank(id="b1", team="beta", position=Position(4, 2), direction=Direction.WEST),
            Tank(id="a2", team="alpha", position=Position(0, 4), direction=Direction.EAST),
        ]
        state = _make_state(tanks=tanks)
        game_map = _make_map()
        original = _tank_ids(state)
        # Play several turns with varied actions, manually advancing turns.
        state = apply_action(state, game_map, "a1", Action.MOVE_FORWARD)
        state = state.model_copy(update={"current_tank_id": "b1"})
        state = apply_action(state, game_map, "b1", Action.TURN_LEFT)
        state = state.model_copy(update={"current_tank_id": "a2"})
        state = apply_action(state, game_map, "a2", Action.PASS)
        state = state.model_copy(update={"current_tank_id": "a1"})
        state = apply_action(state, game_map, "a1", Action.TURN_RIGHT)
        assert _tank_ids(state) == original
