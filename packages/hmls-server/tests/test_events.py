"""Tests for the EventBus and event dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hmls.server.events import (
    EventBus,
    GameOverEvent,
    GameStartedEvent,
    PlayerDisconnectedEvent,
)


@dataclass
class _DummyEventA:
    """Test event A."""

    value: int


@dataclass
class _DummyEventB:
    """Test event B."""

    name: str


class TestEventBus:
    """Tests for the EventBus pub/sub mechanism."""

    @pytest.mark.anyio
    async def test_subscribe_and_emit(self) -> None:
        """A subscribed callback should receive emitted events."""
        bus = EventBus()
        received: list[_DummyEventA] = []

        async def handler(event: _DummyEventA) -> None:
            received.append(event)

        bus.subscribe(_DummyEventA, handler)
        await bus.emit(_DummyEventA(value=42))

        assert len(received) == 1
        assert received[0].value == 42

    @pytest.mark.anyio
    async def test_multiple_subscribers(self) -> None:
        """Multiple callbacks for the same event type all receive the event."""
        bus = EventBus()
        results: list[str] = []

        async def handler_one(event: _DummyEventA) -> None:
            results.append("one")

        async def handler_two(event: _DummyEventA) -> None:
            results.append("two")

        bus.subscribe(_DummyEventA, handler_one)
        bus.subscribe(_DummyEventA, handler_two)
        await bus.emit(_DummyEventA(value=1))

        assert results == ["one", "two"]

    @pytest.mark.anyio
    async def test_subscribers_called_in_registration_order(self) -> None:
        """Callbacks should be invoked in the order they were registered."""
        bus = EventBus()
        order: list[int] = []

        for i in range(5):

            async def handler(event: _DummyEventA, idx: int = i) -> None:
                order.append(idx)

            bus.subscribe(_DummyEventA, handler)

        await bus.emit(_DummyEventA(value=0))
        assert order == [0, 1, 2, 3, 4]

    @pytest.mark.anyio
    async def test_no_subscribers_is_noop(self) -> None:
        """Emitting an event with no subscribers should not raise."""
        bus = EventBus()
        await bus.emit(_DummyEventA(value=99))  # Should not raise.

    @pytest.mark.anyio
    async def test_different_event_types_are_independent(self) -> None:
        """Subscribers only receive events of the type they registered for."""
        bus = EventBus()
        a_received: list[_DummyEventA] = []
        b_received: list[_DummyEventB] = []

        async def handle_a(event: _DummyEventA) -> None:
            a_received.append(event)

        async def handle_b(event: _DummyEventB) -> None:
            b_received.append(event)

        bus.subscribe(_DummyEventA, handle_a)
        bus.subscribe(_DummyEventB, handle_b)

        await bus.emit(_DummyEventA(value=1))
        await bus.emit(_DummyEventB(name="hello"))

        assert len(a_received) == 1
        assert len(b_received) == 1
        assert a_received[0].value == 1
        assert b_received[0].name == "hello"

    @pytest.mark.anyio
    async def test_failing_callback_does_not_block_others(self) -> None:
        """If one callback raises, remaining callbacks still execute."""
        bus = EventBus()
        results: list[str] = []

        async def good_first(event: _DummyEventA) -> None:
            results.append("first")

        async def bad_handler(event: _DummyEventA) -> None:
            raise ValueError("boom")

        async def good_last(event: _DummyEventA) -> None:
            results.append("last")

        bus.subscribe(_DummyEventA, good_first)
        bus.subscribe(_DummyEventA, bad_handler)
        bus.subscribe(_DummyEventA, good_last)

        await bus.emit(_DummyEventA(value=0))

        assert results == ["first", "last"]

    @pytest.mark.anyio
    async def test_multiple_emits(self) -> None:
        """The same subscriber should receive every emitted event."""
        bus = EventBus()
        count: list[int] = []

        async def handler(event: _DummyEventA) -> None:
            count.append(event.value)

        bus.subscribe(_DummyEventA, handler)
        await bus.emit(_DummyEventA(value=1))
        await bus.emit(_DummyEventA(value=2))
        await bus.emit(_DummyEventA(value=3))

        assert count == [1, 2, 3]

    @pytest.mark.anyio
    async def test_event_subclass_not_dispatched_to_parent(self) -> None:
        """Events are dispatched by exact type, not by inheritance."""

        @dataclass
        class ChildEvent(_DummyEventA):
            """Subclass of _DummyEventA."""

            extra: str = ""

        bus = EventBus()
        parent_received: list[_DummyEventA] = []

        async def parent_handler(event: _DummyEventA) -> None:
            parent_received.append(event)

        bus.subscribe(_DummyEventA, parent_handler)
        await bus.emit(ChildEvent(value=10, extra="hi"))

        # The parent handler should NOT be called for a child event.
        assert len(parent_received) == 0


class TestEventDataclasses:
    """Smoke tests that event dataclasses are well-formed."""

    def test_game_started_event(self) -> None:
        """GameStartedEvent should hold all game setup fields."""
        from hmls.core.map import GameMap

        event = GameStartedEvent(
            game_map=GameMap(width=5, height=5),
            tanks=[],
            player_names={"A": "Alice", "B": "Bob"},
            patch_size=7,
            max_turns=100,
        )
        assert event.patch_size == 7
        assert event.max_turns == 100

    def test_game_over_event(self) -> None:
        """GameOverEvent should hold winner and reason."""
        event = GameOverEvent(winner="A", reason="Victory")
        assert event.winner == "A"
        assert event.reason == "Victory"

    def test_game_over_event_draw(self) -> None:
        """GameOverEvent with no winner represents a draw."""
        event = GameOverEvent(winner=None, reason="Draw")
        assert event.winner is None

    def test_player_disconnected_event(self) -> None:
        """PlayerDisconnectedEvent should hold team."""
        event = PlayerDisconnectedEvent(team="B")
        assert event.team == "B"
