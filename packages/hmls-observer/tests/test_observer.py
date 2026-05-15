"""Tests for observer protocol message serialization and parsing."""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.protocol import (
    GameOverMessage,
    GameStartMessage,
    ObserveMessage,
    ServerMessage,
    StateUpdateMessage,
    TurnResultMessage,
)

_server_message_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def _make_simple_map() -> GameMap:
    """Create a small 5x5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_tanks() -> list[Tank]:
    """Create a minimal set of tanks for two teams."""
    return [
        Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
        Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
    ]


class TestObserveMessage:
    """Tests for the ObserveMessage client message."""

    def test_observe_message_serialization(self) -> None:
        """ObserveMessage should serialize with correct type discriminator."""
        msg = ObserveMessage(observer_name="MyObserver")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == "observe"
        assert data["observer_name"] == "MyObserver"

    def test_observe_message_default_name(self) -> None:
        """ObserveMessage should have a default observer name."""
        msg = ObserveMessage()
        assert msg.observer_name == "Observer"


class TestGameStartMessage:
    """Tests for the GameStartMessage parsing."""

    def test_game_start_roundtrip(self) -> None:
        """GameStartMessage should serialize and deserialize correctly."""
        game_map = _make_simple_map()
        tanks = _make_tanks()
        msg = GameStartMessage(
            game_map=game_map,
            tanks=tanks,
            player_names={"A": "Alice", "B": "Bob"},
            patch_size=7,
            max_turns=200,
        )

        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, GameStartMessage)
        assert parsed.game_map.width == 5
        assert parsed.game_map.height == 5
        assert len(parsed.tanks) == 2
        assert parsed.player_names == {"A": "Alice", "B": "Bob"}
        assert parsed.patch_size == 7
        assert parsed.max_turns == 200

    def test_game_start_includes_tank_data(self) -> None:
        """Tanks in GameStartMessage should preserve all fields."""
        tanks = _make_tanks()
        msg = GameStartMessage(
            game_map=_make_simple_map(),
            tanks=tanks,
            player_names={"A": "Alice", "B": "Bob"},
            patch_size=7,
            max_turns=100,
        )

        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)
        assert isinstance(parsed, GameStartMessage)

        tank_a = next(t for t in parsed.tanks if t.id == "A1")
        assert tank_a.team == "A"
        assert tank_a.position == Position(1, 1)
        assert tank_a.direction == Direction.EAST
        assert tank_a.alive is True


class TestStateUpdateMessage:
    """Tests for the StateUpdateMessage parsing."""

    def test_state_update_roundtrip(self) -> None:
        """StateUpdateMessage should serialize and deserialize correctly."""
        state = GameState(tanks=_make_tanks())
        msg = StateUpdateMessage(
            state=state,
            current_tank_id="A1",
            turns_taken=5,
        )

        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, StateUpdateMessage)
        assert parsed.current_tank_id == "A1"
        assert parsed.turns_taken == 5
        assert len(parsed.state.tanks) == 2

    def test_state_update_none_tank_id_when_game_over(self) -> None:
        """StateUpdateMessage with None current_tank_id for game over."""
        state = GameState(tanks=_make_tanks())
        msg = StateUpdateMessage(
            state=state,
            current_tank_id=None,
            turns_taken=10,
        )

        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, StateUpdateMessage)
        assert parsed.current_tank_id is None


class TestTurnResultMessage:
    """Tests for turn result parsing by the observer."""

    def test_turn_result_valid_action(self) -> None:
        """TurnResultMessage for a valid action parses correctly."""
        msg = TurnResultMessage(
            tank_id="A1",
            action=Action.MOVE_FORWARD,
            valid=True,
            reason="",
        )
        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, TurnResultMessage)
        assert parsed.tank_id == "A1"
        assert parsed.action == Action.MOVE_FORWARD
        assert parsed.valid is True

    def test_turn_result_invalid_action(self) -> None:
        """TurnResultMessage for an invalid action includes reason."""
        msg = TurnResultMessage(
            tank_id="B1",
            action=Action.MOVE_FORWARD,
            valid=False,
            reason="Blocked by wall",
        )
        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, TurnResultMessage)
        assert parsed.valid is False
        assert parsed.reason == "Blocked by wall"


class TestGameOverMessage:
    """Tests for game over parsing by the observer."""

    def test_game_over_with_winner(self) -> None:
        """GameOverMessage with a winner parses correctly."""
        msg = GameOverMessage(winner="A", reason="All tanks destroyed")
        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, GameOverMessage)
        assert parsed.winner == "A"
        assert parsed.reason == "All tanks destroyed"

    def test_game_over_draw(self) -> None:
        """GameOverMessage for a draw has None winner."""
        msg = GameOverMessage(winner=None, reason="Turn limit reached")
        json_str = msg.model_dump_json()
        parsed = _server_message_adapter.validate_json(json_str)

        assert isinstance(parsed, GameOverMessage)
        assert parsed.winner is None
