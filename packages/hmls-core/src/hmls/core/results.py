"""Game result models and winner-determination logic.

This module defines the data models that capture the outcome of a game:
:class:`HistoryEntry` for individual turns and :class:`GameResult` for
the complete game.  The private :func:`_determine_winner` helper encodes
the win-condition logic used by both the engine and result construction.
"""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import TankId
from hmls.core.types import Action

# ── Result models ─────────────────────────────────────────────────────


class HistoryEntry(BaseModel, extra="forbid"):
    """One step in the game history.

    Attributes:
        tank_id: The tank that acted.
        requested_action: The action the player asked for.
        applied_action: The action actually applied (may differ if the
            request was invalid — in which case :data:`Action.PASS` is
            substituted).
        valid: Whether the requested action was legal.
        reason: Explanation when the action is invalid (empty string
            when valid).
        hit: Whether a fire action hit another tank.  ``True`` if an
            alive tank was destroyed (friendly fire included),
            ``False`` if the shot missed, ``None`` for non-fire actions.
        state_after: The full game state *after* the action was applied.
    """

    tank_id: TankId
    requested_action: Action
    applied_action: Action
    valid: bool
    reason: str = ""
    hit: bool | None = None
    state_after: GameState


class GameResult(BaseModel, extra="forbid"):
    """Outcome of a complete game.

    Attributes:
        winner: Team name of the winning side, or ``None`` for a draw.
        game_map: The map the game was played on (stored once here
            rather than duplicated in every history entry).
        initial_state: The game state before any actions were taken.
        history: Ordered list of every action taken during the game.
        turns_played: Total number of individual turns taken.
    """

    winner: str | None
    game_map: GameMap
    initial_state: GameState
    history: list[HistoryEntry]
    turns_played: int

    @property
    def final_state(self) -> GameState:
        """The game state when the game ended.

        Returns the state after the last action, or the initial state
        if no actions were taken.
        """
        if self.history:
            return self.history[-1].state_after
        return self.initial_state


# ── Winner determination ──────────────────────────────────────────────


def _determine_winner(state: GameState) -> str | None:
    """Determine the winner from the current game state.

    Returns the team name of the winning side, or ``None`` for a draw.
    """
    alive_counts: dict[str, int] = {}
    for t in state.tanks:
        if t.alive:
            alive_counts[t.team] = alive_counts.get(t.team, 0) + 1
    if not alive_counts:
        return None
    if len(alive_counts) == 1:
        return next(iter(alive_counts))
    max_count = max(alive_counts.values())
    leaders = [t for t, c in alive_counts.items() if c == max_count]
    return leaders[0] if len(leaders) == 1 else None
