"""Interactive player for the TUI test harness.

Provides a concrete :class:`~hmls.core.player.Player` subclass that
the TUI drives by pre-loading an action before each engine step.
"""

from __future__ import annotations

from hmls.core.player import PendingActionPlayer
from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView


class InteractivePlayer(PendingActionPlayer):
    """A player whose actions are supplied by the TUI.

    Before each call to :meth:`GameEngine.step`, the TUI calls
    :meth:`set_next_action` to pre-load the action the user chose.
    The engine then calls :meth:`choose_action`, which returns and
    consumes the pre-loaded action.

    Args:
        team: The team identifier this player controls.
    """

    def __init__(self, team: str) -> None:
        super().__init__(team)
        self._last_invalid: tuple[TankId, Action, str] | None = None

    def set_next_action(self, action: Action) -> None:
        """Pre-load the action to return on the next :meth:`choose_action` call.

        Args:
            action: The action chosen by the TUI user.
        """
        self._pending_action = action

    @property
    def last_invalid(self) -> tuple[TankId, Action, str] | None:
        """The most recent invalid-action notification, or ``None``.

        Returns a ``(tank_id, action, reason)`` tuple if an invalid
        action was attempted, cleared on the next :meth:`choose_action`.
        """
        return self._last_invalid

    def _no_action_message(self) -> str:
        """Return error message specific to InteractivePlayer."""
        return (
            f"No action set for InteractivePlayer (team={self.team!r}). "
            "Call set_next_action() before engine.step()."
        )

    def _on_action_consumed(self, tank_id: TankId, view: PlayerView, action: Action) -> None:
        """Clear the last-invalid tracking when a new action is consumed."""
        self._last_invalid = None

    def notify_invalid_action(self, tank_id: TankId, action: Action, reason: str) -> None:
        """Record the invalid action for the TUI to display.

        Args:
            tank_id: The tank that attempted the action.
            action: The invalid action that was chosen.
            reason: Human-readable explanation of why the action is invalid.
        """
        self._last_invalid = (tank_id, action, reason)
