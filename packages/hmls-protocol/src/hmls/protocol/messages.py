"""Pydantic models for all wire protocol messages.

Messages are exchanged as JSON over WebSocket. Each message has a ``type``
discriminator field used for dispatch. Server and client message unions
allow type-safe parsing of incoming frames.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action
from hmls.core.visibility import PlayerView, TankInfo

# ── Server → Client messages ─────────────────────────────────────────


class WaitingMessage(BaseModel, extra="forbid"):
    """Sent to the first client while waiting for the second to connect.

    Attributes:
        type: Discriminator, always ``"waiting"``.
        message: Human-readable status message.
    """

    type: Literal["waiting"] = "waiting"
    message: str


class AssignMessage(BaseModel, extra="forbid"):
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


class YourTurnMessage(BaseModel, extra="forbid"):
    """Sent to a client when one of their tanks must act.

    Attributes:
        type: Discriminator, always ``"your_turn"``.
        tank_id: The tank that must act this turn.
        view: The fog-of-war PlayerView for the client's team.
    """

    type: Literal["your_turn"] = "your_turn"
    tank_id: TankId
    view: PlayerView


class TurnResultMessage(BaseModel, extra="forbid"):
    """Sent to the acting player's client after each turn resolves.

    Only the team whose tank acted receives this message; the opponent
    does not, to preserve fog-of-war.  Observers also receive a copy.

    Attributes:
        type: Discriminator, always ``"turn_result"``.
        tank_id: The tank that acted.
        action: The action that was applied.
        valid: Whether the requested action was legal.
        reason: Explanation if the action was invalid.
        hit: Whether a fire action hit an enemy tank.  ``True`` if a
            tank was destroyed, ``False`` if the shot missed, ``None``
            for non-fire actions.
    """

    type: Literal["turn_result"] = "turn_result"
    tank_id: TankId
    action: Action
    valid: bool
    reason: str = ""
    hit: bool | None = None


class GameOverMessage(BaseModel, extra="forbid"):
    """Sent to both clients when the game ends.

    Attributes:
        type: Discriminator, always ``"game_over"``.
        winner: Team name of the winner, or ``None`` for a draw.
        reason: Human-readable explanation (e.g. "all tanks destroyed").
    """

    type: Literal["game_over"] = "game_over"
    winner: str | None
    reason: str


class ErrorMessage(BaseModel, extra="forbid"):
    """Sent to a client on protocol errors.

    Attributes:
        type: Discriminator, always ``"error"``.
        message: Human-readable error description.
    """

    type: Literal["error"] = "error"
    message: str


# ── Server → Observer messages ────────────────────────────────────────


class GameStartMessage(BaseModel, extra="forbid"):
    """Sent to an observer once it connects (or once the game starts).

    Provides the full map, initial tank positions, player names, and
    configuration so the observer can render the complete game state.

    Attributes:
        type: Discriminator, always ``"game_start"``.
        game_map: The full game map.
        tanks: All tanks with initial positions.
        player_names: Mapping of team name to player display name.
        patch_size: Visibility patch size used by the engine.
        max_turns: Maximum number of individual turns.
    """

    type: Literal["game_start"] = "game_start"
    game_map: GameMap
    tanks: list[Tank]
    player_names: dict[str, str]
    patch_size: int
    max_turns: int


class StateUpdateMessage(BaseModel, extra="forbid"):
    """Sent to observers after each state change (turn resolution).

    Carries the full game state so observers can render the complete
    board without fog-of-war.

    Attributes:
        type: Discriminator, always ``"state_update"``.
        state: The current full game state.
        current_tank_id: The tank that will act next, or ``None`` if game over.
        turns_taken: Number of turns completed so far.
    """

    type: Literal["state_update"] = "state_update"
    state: GameState
    current_tank_id: TankId | None = None
    turns_taken: int = 0


ServerMessage = Annotated[
    WaitingMessage
    | AssignMessage
    | YourTurnMessage
    | TurnResultMessage
    | GameOverMessage
    | ErrorMessage
    | GameStartMessage
    | StateUpdateMessage,
    Field(discriminator="type"),
]
"""Union of all messages the server can send to a client or observer."""


# ── Client → Server messages ─────────────────────────────────────────


class JoinMessage(BaseModel, extra="forbid"):
    """Sent by a client immediately after connecting.

    Attributes:
        type: Discriminator, always ``"join"``.
        player_name: Human-readable name for this player.
    """

    type: Literal["join"] = "join"
    player_name: str


class ObserveMessage(BaseModel, extra="forbid"):
    """Sent by an observer client immediately after connecting.

    Identifies the connection as an observer rather than a player.

    Attributes:
        type: Discriminator, always ``"observe"``.
        observer_name: Optional human-readable name for this observer.
    """

    type: Literal["observe"] = "observe"
    observer_name: str = "Observer"


class ActionMessage(BaseModel, extra="forbid"):
    """Sent by a client in response to a ``your_turn`` message.

    Attributes:
        type: Discriminator, always ``"action"``.
        action: The chosen action for the active tank.
    """

    type: Literal["action"] = "action"
    action: Action


ClientMessage = Annotated[
    JoinMessage | ObserveMessage | ActionMessage,
    Field(discriminator="type"),
]
"""Union of all messages a client can send to the server."""
