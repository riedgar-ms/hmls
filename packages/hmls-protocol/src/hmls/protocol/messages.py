"""Pydantic models for all wire protocol messages.

Messages are exchanged as JSON over WebSocket. Each message has a ``type``
discriminator field used for dispatch. Server and client message unions
allow type-safe parsing of incoming frames.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from hmls.core.tank import TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView, TankInfo

# ── Server → Client messages ─────────────────────────────────────────


class WaitingMessage(BaseModel):
    """Sent to the first client while waiting for the second to connect.

    Attributes:
        type: Discriminator, always ``"waiting"``.
        message: Human-readable status message.
    """

    type: Literal["waiting"] = "waiting"
    message: str


class AssignMessage(BaseModel):
    """Sent to each client once both have connected, before the game starts.

    Provides the client with their team assignment and initial tank info,
    plus the map dimensions so the automapper can allocate its grid.

    Attributes:
        type: Discriminator, always ``"assign"``.
        team: The team letter assigned to this client (e.g. ``"A"``).
        tanks: Initial info for the client's tanks.
        map_width: Width of the game map in cells.
        map_height: Height of the game map in cells.
        patch_size: Visibility patch size used by the engine.
    """

    type: Literal["assign"] = "assign"
    team: str
    tanks: list[TankInfo]
    map_width: int
    map_height: int
    patch_size: int


class YourTurnMessage(BaseModel):
    """Sent to a client when one of their tanks must act.

    Attributes:
        type: Discriminator, always ``"your_turn"``.
        tank_id: The tank that must act this turn.
        view: The fog-of-war PlayerView for the client's team.
    """

    type: Literal["your_turn"] = "your_turn"
    tank_id: TankId
    view: PlayerView


class TurnResultMessage(BaseModel):
    """Sent to both clients after each turn resolves.

    Attributes:
        type: Discriminator, always ``"turn_result"``.
        tank_id: The tank that acted.
        action: The action that was applied.
        valid: Whether the requested action was legal.
        reason: Explanation if the action was invalid.
    """

    type: Literal["turn_result"] = "turn_result"
    tank_id: TankId
    action: Action
    valid: bool
    reason: str = ""


class GameOverMessage(BaseModel):
    """Sent to both clients when the game ends.

    Attributes:
        type: Discriminator, always ``"game_over"``.
        winner: Team name of the winner, or ``None`` for a draw.
        reason: Human-readable explanation (e.g. "all tanks destroyed").
    """

    type: Literal["game_over"] = "game_over"
    winner: str | None
    reason: str


class ErrorMessage(BaseModel):
    """Sent to a client on protocol errors.

    Attributes:
        type: Discriminator, always ``"error"``.
        message: Human-readable error description.
    """

    type: Literal["error"] = "error"
    message: str


ServerMessage = Annotated[
    WaitingMessage
    | AssignMessage
    | YourTurnMessage
    | TurnResultMessage
    | GameOverMessage
    | ErrorMessage,
    Field(discriminator="type"),
]
"""Union of all messages the server can send to a client."""


# ── Client → Server messages ─────────────────────────────────────────


class JoinMessage(BaseModel):
    """Sent by a client immediately after connecting.

    Attributes:
        type: Discriminator, always ``"join"``.
        player_name: Human-readable name for this player.
    """

    type: Literal["join"] = "join"
    player_name: str


class ActionMessage(BaseModel):
    """Sent by a client in response to a ``your_turn`` message.

    Attributes:
        type: Discriminator, always ``"action"``.
        action: The chosen action for the active tank.
    """

    type: Literal["action"] = "action"
    action: Action


ClientMessage = Annotated[
    JoinMessage | ActionMessage,
    Field(discriminator="type"),
]
"""Union of all messages a client can send to the server."""
