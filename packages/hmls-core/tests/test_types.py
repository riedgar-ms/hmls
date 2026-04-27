"""Tests for hmls.core.types – Direction, Position, Action."""

from hmls.core.types import Action, Direction, Position


class TestDirection:
    """Tests for the Direction enum and its rotation methods."""

    def test_turn_right_cycle(self) -> None:
        """Turning right four times returns to the original direction."""
        d = Direction.NORTH
        for _ in range(4):
            d = d.turn_right()
        assert d == Direction.NORTH

    def test_turn_left_cycle(self) -> None:
        """Turning left four times returns to the original direction."""
        d = Direction.NORTH
        for _ in range(4):
            d = d.turn_left()
        assert d == Direction.NORTH

    def test_turn_right_sequence(self) -> None:
        """NORTH → EAST → SOUTH → WEST when turning right."""
        assert Direction.NORTH.turn_right() == Direction.EAST
        assert Direction.EAST.turn_right() == Direction.SOUTH
        assert Direction.SOUTH.turn_right() == Direction.WEST
        assert Direction.WEST.turn_right() == Direction.NORTH

    def test_turn_left_sequence(self) -> None:
        """NORTH → WEST → SOUTH → EAST when turning left."""
        assert Direction.NORTH.turn_left() == Direction.WEST
        assert Direction.WEST.turn_left() == Direction.SOUTH
        assert Direction.SOUTH.turn_left() == Direction.EAST
        assert Direction.EAST.turn_left() == Direction.NORTH

    def test_left_right_inverse(self) -> None:
        """Turning left then right (or vice versa) is identity."""
        for d in Direction:
            assert d.turn_left().turn_right() == d
            assert d.turn_right().turn_left() == d

    def test_forward_delta_north(self) -> None:
        """NORTH moves up (y decreases)."""
        assert Direction.NORTH.forward_delta() == (0, -1)

    def test_forward_delta_east(self) -> None:
        """EAST moves right (x increases)."""
        assert Direction.EAST.forward_delta() == (1, 0)

    def test_forward_delta_south(self) -> None:
        """SOUTH moves down (y increases)."""
        assert Direction.SOUTH.forward_delta() == (0, 1)

    def test_forward_delta_west(self) -> None:
        """WEST moves left (x decreases)."""
        assert Direction.WEST.forward_delta() == (-1, 0)


class TestPosition:
    """Tests for the Position named tuple."""

    def test_creation(self) -> None:
        """Position can be created with x and y."""
        p = Position(3, 7)
        assert p.x == 3
        assert p.y == 7

    def test_equality(self) -> None:
        """Two positions with the same coordinates are equal."""
        assert Position(1, 2) == Position(1, 2)

    def test_hashable(self) -> None:
        """Positions can be used as dictionary keys."""
        d: dict[Position, str] = {Position(0, 0): "origin"}
        assert d[Position(0, 0)] == "origin"

    def test_tuple_unpacking(self) -> None:
        """Position supports tuple unpacking."""
        x, y = Position(5, 10)
        assert x == 5
        assert y == 10


class TestAction:
    """Tests for the Action enum."""

    def test_all_actions_exist(self) -> None:
        """All five expected actions are defined."""
        names = {a.name for a in Action}
        assert names == {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "FIRE", "PASS"}

    def test_values_are_strings(self) -> None:
        """Action values are descriptive strings."""
        assert Action.MOVE_FORWARD.value == "move_forward"
        assert Action.PASS.value == "pass"
