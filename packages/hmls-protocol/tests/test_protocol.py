"""Tests for wire protocol message serialisation/deserialisation."""

from __future__ import annotations

from pydantic import TypeAdapter

from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import TankInfo
from hmls.protocol import (
    ActionMessage,
    AssignMessage,
    ClientMessage,
    ErrorMessage,
    GameOverMessage,
    JoinMessage,
    ServerMessage,
    TurnResultMessage,
    WaitingMessage,
)

_server_adapter: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


class TestServerMessages:
    """Test serialisation round-trips for server messages."""

    def test_waiting_round_trip(self) -> None:
        """WaitingMessage should serialise and deserialise correctly."""
        msg = WaitingMessage(message="Please wait")
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, WaitingMessage)
        assert parsed.message == "Please wait"

    def test_assign_round_trip(self) -> None:
        """AssignMessage should serialise and deserialise correctly."""
        msg = AssignMessage(
            team="A",
            tanks=[
                TankInfo(
                    tank_id="A1",
                    position=Position(1, 2),
                    direction=Direction.NORTH,
                    alive=True,
                )
            ],
            map_width=10,
            map_height=8,
            patch_size=7,
        )
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, AssignMessage)
        assert parsed.team == "A"
        assert len(parsed.tanks) == 1
        assert parsed.tanks[0].tank_id == "A1"
        assert parsed.map_width == 10

    def test_turn_result_round_trip(self) -> None:
        """TurnResultMessage should serialise and deserialise correctly."""
        msg = TurnResultMessage(
            tank_id="B1",
            action=Action.FIRE,
            valid=True,
        )
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, TurnResultMessage)
        assert parsed.tank_id == "B1"
        assert parsed.action == Action.FIRE
        assert parsed.valid is True

    def test_game_over_round_trip(self) -> None:
        """GameOverMessage should serialise and deserialise correctly."""
        msg = GameOverMessage(winner="A", reason="All enemies destroyed")
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, GameOverMessage)
        assert parsed.winner == "A"

    def test_game_over_draw(self) -> None:
        """GameOverMessage with None winner (draw)."""
        msg = GameOverMessage(winner=None, reason="Turn limit")
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, GameOverMessage)
        assert parsed.winner is None

    def test_error_round_trip(self) -> None:
        """ErrorMessage should serialise and deserialise correctly."""
        msg = ErrorMessage(message="Bad request")
        raw = msg.model_dump_json()
        parsed = _server_adapter.validate_json(raw)
        assert isinstance(parsed, ErrorMessage)
        assert parsed.message == "Bad request"


class TestClientMessages:
    """Test serialisation round-trips for client messages."""

    def test_join_round_trip(self) -> None:
        """JoinMessage should serialise and deserialise correctly."""
        msg = JoinMessage(player_name="Alice")
        raw = msg.model_dump_json()
        parsed = _client_adapter.validate_json(raw)
        assert isinstance(parsed, JoinMessage)
        assert parsed.player_name == "Alice"

    def test_action_round_trip(self) -> None:
        """ActionMessage should serialise and deserialise correctly."""
        for action in Action:
            msg = ActionMessage(action=action)
            raw = msg.model_dump_json()
            parsed = _client_adapter.validate_json(raw)
            assert isinstance(parsed, ActionMessage)
            assert parsed.action == action
