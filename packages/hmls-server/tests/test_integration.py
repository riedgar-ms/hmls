"""Tests for server components: RemotePlayer and protocol round-tripping."""

from __future__ import annotations

import asyncio

import pytest

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import PlayerView, build_player_view
from hmls.server.remote_player import RemotePlayer


def _make_simple_map() -> GameMap:
    """Create a small 5x5 all-passable map for testing."""
    return GameMap(width=5, height=5)


def _make_view(game_map: GameMap, tanks: list[Tank], team: str) -> PlayerView:
    """Build a PlayerView for the given team."""
    state = GameState(tanks=tanks, current_tank_id=tanks[0].id)
    return build_player_view(state, game_map, team, patch_size=7)


class TestRemotePlayer:
    """Tests for the RemotePlayer class."""

    def test_basic_action_flow(self) -> None:
        """Test the request → submit → choose_action flow."""
        player = RemotePlayer("A")
        game_map = _make_simple_map()
        tanks = [
            Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
            Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
        ]
        view = _make_view(game_map, tanks, "A")
        loop = asyncio.new_event_loop()
        try:
            player.request_action("A1", view, loop)
            player.submit_action(Action.MOVE_FORWARD)
            result = loop.run_until_complete(player.wait_for_action())
            assert result == Action.MOVE_FORWARD

            chosen = player.choose_action("A1", view)
            assert chosen == Action.MOVE_FORWARD
        finally:
            loop.close()

    def test_submit_without_request_raises(self) -> None:
        """Submitting an action without a pending request should raise."""
        player = RemotePlayer("A")
        with pytest.raises(RuntimeError):
            player.submit_action(Action.PASS)

    def test_choose_without_submit_raises(self) -> None:
        """Calling choose_action without a prior submit should raise."""
        player = RemotePlayer("A")
        game_map = _make_simple_map()
        tanks = [
            Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
            Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
        ]
        view = _make_view(game_map, tanks, "A")
        with pytest.raises(RuntimeError):
            player.choose_action("A1", view)

    @pytest.mark.anyio
    async def test_async_action_flow(self) -> None:
        """Test the async request → submit → await flow."""
        player = RemotePlayer("A")
        game_map = _make_simple_map()
        tanks = [
            Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.EAST),
            Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.WEST),
        ]
        view = _make_view(game_map, tanks, "A")

        loop = asyncio.get_event_loop()
        player.request_action("A1", view, loop)

        async def submit_later() -> None:
            await asyncio.sleep(0.01)
            player.submit_action(Action.FIRE)

        submit_task = asyncio.create_task(submit_later())
        action = await player.wait_for_action()
        await submit_task

        assert action == Action.FIRE
