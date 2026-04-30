"""Wire protocol models for HMLS tank game server/client communication.

All messages exchanged between server and client are defined here as
Pydantic models, ensuring a single source of truth for the wire format.
"""

from hmls.protocol.messages import (
    ActionMessage,
    AssignMessage,
    ClientMessage,
    ErrorMessage,
    GameOverMessage,
    GameStartMessage,
    JoinMessage,
    ObserveMessage,
    ServerMessage,
    StateUpdateMessage,
    TurnResultMessage,
    WaitingMessage,
    YourTurnMessage,
)

__all__ = [
    "ActionMessage",
    "AssignMessage",
    "ClientMessage",
    "ErrorMessage",
    "GameOverMessage",
    "GameStartMessage",
    "JoinMessage",
    "ObserveMessage",
    "ServerMessage",
    "StateUpdateMessage",
    "TurnResultMessage",
    "WaitingMessage",
    "YourTurnMessage",
]
