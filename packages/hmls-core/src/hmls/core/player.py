"""Abstract player interface for the tank game.

Concrete implementations must subclass :class:`Player` and provide
a :meth:`choose_action` method that returns an :class:`~hmls.core.types.Action`
for a given tank, based on the fog-of-war :class:`~hmls.core.visibility.PlayerView`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView


class Player(ABC):
    """Base class for a tank-game player.

    Each player controls one team.  On each turn the engine calls
    :meth:`choose_action` with the ID of the tank that must act and
    a fog-of-war :class:`PlayerView`.  If the returned action is
    invalid, :meth:`notify_invalid_action` is called (the tank still
    loses its turn).

    Args:
        team: The team identifier this player controls.
    """

    def __init__(self, team: str) -> None:
        self._team = team

    @property
    def team(self) -> str:
        """The team this player controls."""
        return self._team

    @abstractmethod
    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Choose an action for *tank_id* given the current *view*.

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The desired :class:`~hmls.core.types.Action`.
        """
        ...

    def notify_invalid_action(self, tank_id: TankId, action: Action, reason: str) -> None:
        """Called when a chosen action is invalid.

        The default implementation is a no-op.  Override to log or
        adapt strategy when an invalid action is attempted.

        Args:
            tank_id: The tank that attempted the action.
            action: The invalid action that was chosen.
            reason: Human-readable explanation of why the action is invalid.
        """
