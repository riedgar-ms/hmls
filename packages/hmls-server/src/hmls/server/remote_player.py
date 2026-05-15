"""Remote player: bridges WebSocket clients with the game engine.

Extends :class:`~hmls.core.player.PendingActionPlayer` to coordinate
between the async game loop and the WebSocket layer.  This class does
not perform WebSocket I/O itself — the
:class:`~hmls.server.network_manager.NetworkManager` handles sending
and receiving protocol messages.
"""

from __future__ import annotations

import asyncio

from hmls.core.player import PendingActionPlayer
from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView


class RemotePlayer(PendingActionPlayer):
    """A player whose actions come from a remote WebSocket client.

    The game loop (async) sets up a pending turn by calling
    :meth:`request_action`, which stores the view and signals that a
    turn is active. The game loop then awaits :meth:`wait_for_action`
    to get the client's response.

    The WebSocket handler calls :meth:`submit_action` when it receives
    the client's action message.

    The synchronous :meth:`choose_action` (inherited from
    :class:`PendingActionPlayer`) returns the action that was already
    submitted — the game loop must ensure the action is available
    before calling ``engine.step()``.

    Args:
        team: The team identifier this player controls.
    """

    def __init__(self, team: str) -> None:
        super().__init__(team)
        self._action_future: asyncio.Future[Action] | None = None
        self._current_view: PlayerView | None = None
        self._current_tank_id: TankId | None = None

    @property
    def current_view(self) -> PlayerView | None:
        """The PlayerView for the current pending turn, if any."""
        return self._current_view

    @property
    def current_tank_id(self) -> TankId | None:
        """The tank ID for the current pending turn, if any."""
        return self._current_tank_id

    def request_action(
        self, tank_id: TankId, view: PlayerView, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Set up a pending turn request.

        Called by the async game loop before sending ``your_turn`` to the client.

        Args:
            tank_id: The tank that must act.
            view: The fog-of-war view to send to the client.
            loop: The event loop to create the future on.
        """
        self._current_tank_id = tank_id
        self._current_view = view
        self._action_future = loop.create_future()

    async def wait_for_action(self) -> Action:
        """Await the client's action response.

        Returns:
            The action submitted by the client.

        Raises:
            RuntimeError: If no turn is pending.
        """
        if self._action_future is None:
            raise RuntimeError("No pending turn to wait for")  # noqa: EM101
        action = await self._action_future
        self._pending_action = action
        return action

    def submit_action(self, action: Action) -> None:
        """Submit the client's chosen action (called by WebSocket handler).

        Args:
            action: The action the client chose.

        Raises:
            RuntimeError: If no turn is pending or action already submitted.
        """
        if self._action_future is None or self._action_future.done():
            msg = "No pending turn or action already submitted"
            raise RuntimeError(msg)
        self._action_future.set_result(action)

    def _no_action_message(self) -> str:
        """Return error message specific to RemotePlayer."""
        return (
            f"No action submitted for RemotePlayer (team={self.team!r}). "
            "The game loop must await wait_for_action() before engine.step()."
        )

    def _on_action_consumed(self, tank_id: TankId, view: PlayerView, action: Action) -> None:
        """Reset async turn state after action is consumed."""
        self._action_future = None
        self._current_view = None
        self._current_tank_id = None

    def notify_invalid_action(self, tank_id: TankId, action: Action, reason: str) -> None:
        """Record invalid action notification (no-op for remote players).

        The server will send the turn_result with valid=False to the client.

        Args:
            tank_id: The tank that attempted the action.
            action: The invalid action that was chosen.
            reason: Human-readable explanation.
        """
