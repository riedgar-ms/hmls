"""Tests for the InteractivePlayer: action lifecycle and invalid-action tracking."""

from __future__ import annotations

import pytest

from hmls.core.types import Action
from hmls.core.visibility import PlayerView
from hmls.testharness.interactive_player import InteractivePlayer


def _empty_view() -> PlayerView:
    """Return a minimal ``PlayerView`` with no patches."""
    return PlayerView(patches=[], tanks=[])


# ── set / choose lifecycle ────────────────────────────────────────────


class TestSetAndChoose:
    """Tests for the set_next_action → choose_action flow."""

    def test_choose_returns_preloaded_action(self) -> None:
        """``choose_action`` returns the action set via ``set_next_action``."""
        player = InteractivePlayer("A")
        player.set_next_action(Action.MOVE_FORWARD)
        result = player.choose_action("A1", _empty_view())
        assert result is Action.MOVE_FORWARD

    def test_choose_clears_pending(self) -> None:
        """A second ``choose_action`` without a new ``set`` raises RuntimeError."""
        player = InteractivePlayer("A")
        player.set_next_action(Action.FIRE)
        player.choose_action("A1", _empty_view())

        with pytest.raises(RuntimeError):
            player.choose_action("A1", _empty_view())

    def test_no_action_raises(self) -> None:
        """``choose_action`` with no pending action raises RuntimeError."""
        player = InteractivePlayer("A")
        with pytest.raises(RuntimeError):
            player.choose_action("A1", _empty_view())


# ── invalid-action tracking ──────────────────────────────────────────


class TestInvalidAction:
    """Tests for ``notify_invalid_action`` and ``last_invalid``."""

    def test_notify_stores_invalid(self) -> None:
        """``notify_invalid_action`` populates ``last_invalid``."""
        player = InteractivePlayer("A")
        player.notify_invalid_action("A1", Action.MOVE_FORWARD, "blocked by wall")
        assert player.last_invalid == ("A1", Action.MOVE_FORWARD, "blocked by wall")

    def test_choose_clears_invalid(self) -> None:
        """After ``choose_action``, ``last_invalid`` is reset to None."""
        player = InteractivePlayer("A")
        player.notify_invalid_action("A1", Action.FIRE, "no target")
        player.set_next_action(Action.PASS)
        player.choose_action("A1", _empty_view())
        assert player.last_invalid is None
