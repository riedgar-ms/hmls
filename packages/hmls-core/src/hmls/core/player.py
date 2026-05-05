"""Abstract player interface for the tank game.

Concrete implementations must subclass :class:`Player` and provide
a :meth:`choose_action` method that returns an :class:`~hmls.core.types.Action`
for a given tank, based on the fog-of-war :class:`~hmls.core.visibility.PlayerView`.

For players whose action is pre-loaded before the engine step, subclass
:class:`PendingActionPlayer` which provides the guard-and-consume logic.
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


class PendingActionPlayer(Player):
    """Intermediate base for players whose action is pre-loaded before the engine step.

    Subclasses set :attr:`_pending_action` via their own mechanism
    (e.g. a TUI callback or an async WebSocket handler).  The engine
    then calls :meth:`choose_action` which guards against a missing
    action, consumes it, and invokes the :meth:`_on_action_consumed`
    hook so subclasses can reset additional state.

    Args:
        team: The team identifier this player controls.
    """

    def __init__(self, team: str) -> None:
        super().__init__(team)
        self._pending_action: Action | None = None

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        """Return and consume the pre-loaded action.

        Args:
            tank_id: The tank that must act this turn.
            view: Fog-of-war information for the player's team.

        Returns:
            The action previously stored in :attr:`_pending_action`.

        Raises:
            RuntimeError: If no action has been pre-loaded.
        """
        if self._pending_action is None:
            raise RuntimeError(self._no_action_message())
        action = self._pending_action
        self._pending_action = None
        self._on_action_consumed(tank_id, view, action)
        return action

    def _no_action_message(self) -> str:
        """Return the error message when no action is pending.

        Override to customise the message for a specific subclass.
        """
        return (
            f"No action set for {type(self).__name__} (team={self.team!r}). "
            "Pre-load an action before calling engine.step()."
        )

    def _on_action_consumed(self, tank_id: TankId, view: PlayerView, action: Action) -> None:
        """Hook called after the pending action is consumed.

        Override to reset additional state (e.g. async futures, UI flags).
        The default implementation is a no-op.

        Args:
            tank_id: The tank that acted.
            view: The fog-of-war view that was provided.
            action: The action that was consumed.
        """
