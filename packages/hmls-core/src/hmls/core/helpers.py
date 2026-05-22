"""Turn-management helper functions for the game engine.

These utilities handle tank cycling and state updates that the engine
uses internally to manage turn alternation.
"""

from __future__ import annotations

from hmls.core.game_state import GameState
from hmls.core.tank import TankId


def _count_alive_by_team(state: GameState) -> dict[str, int]:
    """Return a mapping from team name to number of alive tanks."""
    counts: dict[str, int] = {}
    for t in state.tanks:
        if t.alive:
            counts[t.team] = counts.get(t.team, 0) + 1
    return counts


def _next_alive_tank(state: GameState, team: str, cursor: int) -> tuple[TankId, int]:
    """Pick the next alive tank for *team*, cycling from *cursor*.

    The cursor indexes into the team's original tank list (alive or
    dead).  This function walks forward, wrapping around, until it
    finds an alive tank.

    Args:
        state: Current game state.
        team: Team whose tanks to cycle through.
        cursor: Starting index into the team's tank list.

    Returns:
        ``(tank_id, next_cursor)`` where *next_cursor* is the index
        after the chosen tank (for the next call).

    Raises:
        StopIteration: If the team has no alive tanks.
    """
    team_tanks = [t for t in state.tanks if t.team == team]
    n = len(team_tanks)
    for i in range(n):
        idx = (cursor + i) % n
        if team_tanks[idx].alive:
            return team_tanks[idx].id, (idx + 1) % n
    raise StopIteration(f"No alive tanks for team {team!r}")  # noqa: EM102


def _set_current_tank(state: GameState, tank_id: TankId) -> GameState:
    """Return a copy of *state* with ``current_tank_id`` set to *tank_id*.

    The tank must exist in ``state.tanks``.
    """
    state.get_tank(tank_id)  # Validate existence.
    return state.model_copy(update={"current_tank_id": tank_id})
