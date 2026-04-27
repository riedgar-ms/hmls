"""Tests for hmls.core.tank – Tank model."""

import json

from hmls.core.tank import Tank
from hmls.core.types import Direction, Position


class TestTankConstruction:
    """Tests for creating Tank instances."""

    def test_basic_tank(self) -> None:
        """A tank can be created with required fields."""
        t = Tank(id="t1", team="alpha", position=Position(2, 3), direction=Direction.NORTH)
        assert t.id == "t1"
        assert t.team == "alpha"
        assert t.position == Position(2, 3)
        assert t.direction == Direction.NORTH
        assert t.alive is True

    def test_dead_tank(self) -> None:
        """A tank can be created as not alive."""
        t = Tank(
            id="t2", team="beta", position=Position(0, 0), direction=Direction.SOUTH, alive=False
        )
        assert t.alive is False


class TestTankSerialisation:
    """Tests for JSON round-trip of Tank."""

    def test_json_round_trip(self) -> None:
        """A Tank should survive a JSON serialise/deserialise cycle."""
        original = Tank(id="t1", team="alpha", position=Position(1, 2), direction=Direction.EAST)
        json_str = original.model_dump_json()
        restored = Tank.model_validate_json(json_str)
        assert restored == original

    def test_json_structure(self) -> None:
        """The JSON output should contain the expected keys."""
        t = Tank(id="t1", team="alpha", position=Position(0, 0), direction=Direction.NORTH)
        data = json.loads(t.model_dump_json())
        assert "id" in data
        assert "team" in data
        assert "position" in data
        assert "direction" in data
        assert "alive" in data
