"""Lethargy detection policies for training games.

Provides pluggable policies that monitor tank actions during training
games and detect degenerate behaviours such as spinning in place.
When a lethargy policy triggers, the offending tank's team loses the
game immediately.

The :class:`LethargyPolicy` abstract base class defines the interface;
concrete implementations live alongside it in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hmls.core.tank import TankId
from hmls.core.types import Action


class LethargyPolicy(ABC):
    """Base class for lethargy detection during training games.

    Subclasses observe each action taken during a game and may declare
    a team as having lost due to lethargic (degenerate) play.  The
    game runner calls :meth:`observe_action` after every engine step
    and terminates the game early if a losing team is returned.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new game.

        Called before the game loop begins.
        """
        ...

    @abstractmethod
    def observe_action(self, tank_id: TankId, action: Action) -> str | None:
        """Observe an action and optionally declare a loser.

        Args:
            tank_id: The tank that just acted (e.g. ``"A1"``, ``"B2"``).
            action: The action that was actually applied (after
                validity resolution).

        Returns:
            The team name (e.g. ``"A"``) that should lose due to
            lethargy, or ``None`` if play is acceptable.
        """
        ...


class NoLethargyCheck(LethargyPolicy):
    """No-op lethargy policy — lethargy detection is disabled."""

    def reset(self) -> None:
        """No-op: nothing to reset."""

    def observe_action(self, tank_id: TankId, action: Action) -> str | None:
        """Always returns ``None`` — no lethargy detection."""
        return None


class ConsecutiveTurnLimit(LethargyPolicy):
    """Detect tanks that repeatedly turn without doing anything else.

    A tank is declared lethargic if it takes ``max_consecutive_turns``
    consecutive turn actions (``TURN_LEFT`` or ``TURN_RIGHT``) without
    any intervening non-turn action.  Each tank is tracked
    independently.

    Args:
        max_consecutive_turns: Number of consecutive turn actions
            that triggers a lethargy loss.  Must be at least 2.
    """

    _TURN_ACTIONS: frozenset[Action] = frozenset({Action.TURN_LEFT, Action.TURN_RIGHT})

    def __init__(self, max_consecutive_turns: int = 5) -> None:
        if max_consecutive_turns < 2:
            raise ValueError(f"max_consecutive_turns must be >= 2, got {max_consecutive_turns}")
        self._max: int = max_consecutive_turns
        self._streak: dict[TankId, int] = {}

    def reset(self) -> None:
        """Clear all per-tank turn counters."""
        self._streak.clear()

    def observe_action(self, tank_id: TankId, action: Action) -> str | None:
        """Track consecutive turns and trigger on threshold.

        Args:
            tank_id: The acting tank.
            action: The applied action.

        Returns:
            The losing team name if the tank exceeded the consecutive
            turn limit, otherwise ``None``.
        """
        if action in self._TURN_ACTIONS:
            count = self._streak.get(tank_id, 0) + 1
            self._streak[tank_id] = count
            if count >= self._max:
                # Team is the first character of the tank ID (e.g. "A1" → "A")
                return tank_id[0]
        else:
            self._streak[tank_id] = 0
        return None
