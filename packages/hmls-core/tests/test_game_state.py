"""Tests for hmls.core.game_state – GameState model."""

import pytest

from hmls.core.game_state import GameState
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position


def _make_state(
    tanks: list[Tank] | None = None,
    current_tank_id: str | None = None,
) -> GameState:
    """Helper to build a simple GameState for testing."""
    if tanks is None:
        tanks = [
            Tank(id="a1", team="alpha", position=Position(0, 0), direction=Direction.EAST),
            Tank(id="b1", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
    if current_tank_id is None and tanks:
        current_tank_id = tanks[0].id
    return GameState(
        tanks=tanks,
        current_tank_id=current_tank_id,
    )


class TestGameStateConstruction:
    """Tests for creating GameState instances."""

    def test_basic_construction(self) -> None:
        """A game state can be created with tanks."""
        state = _make_state()
        assert len(state.tanks) == 2
        assert state.current_tank_id == "a1"

    def test_alive_tanks(self) -> None:
        """alive_tanks filters out dead tanks."""
        state = _make_state()
        assert len(state.alive_tanks) == 2
        # Kill one tank.
        state.tanks[1] = state.tanks[1].model_copy(update={"alive": False})
        assert len(state.alive_tanks) == 1
        assert state.alive_tanks[0].id == "a1"

    def test_current_tank_id_none_by_default(self) -> None:
        """current_tank_id defaults to None when not set."""
        state = GameState(tanks=[])
        assert state.current_tank_id is None


class TestGameStateLookup:
    """Tests for lookup helpers."""

    def test_get_tank(self) -> None:
        """get_tank returns the correct tank."""
        state = _make_state()
        t = state.get_tank("a1")
        assert t.id == "a1"

    def test_get_tank_missing(self) -> None:
        """get_tank raises KeyError for unknown IDs."""
        state = _make_state()
        with pytest.raises(KeyError, match="no_such"):
            state.get_tank("no_such")

    def test_tank_positions(self) -> None:
        """tank_positions maps position → tank ID for all tanks."""
        state = _make_state()
        positions = state.tank_positions
        assert positions[Position(0, 0)] == "a1"
        assert positions[Position(4, 4)] == "b1"

    def test_tank_positions_includes_dead(self) -> None:
        """Dead tanks (wreckage) are included in tank_positions."""
        state = _make_state()
        state.tanks[1] = state.tanks[1].model_copy(update={"alive": False})
        positions = state.tank_positions
        assert Position(4, 4) in positions

    def test_current_tank_id(self) -> None:
        """current_tank_id returns the assigned tank."""
        state = _make_state()
        assert state.current_tank_id == "a1"

    def test_current_tank_id_explicit(self) -> None:
        """current_tank_id can be set to any tank."""
        state = _make_state(current_tank_id="b1")
        assert state.current_tank_id == "b1"
