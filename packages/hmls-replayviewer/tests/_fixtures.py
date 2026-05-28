"""Shared test fixtures for the replay viewer package."""

from __future__ import annotations

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import GameResult, HistoryEntry
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position


def make_two_tank_game_result(
    *,
    history_len: int = 5,
    actions: list[Action] | None = None,
    winner: str | None = None,
    team_a: str = "Alpha",
    team_b: str = "Bravo",
) -> GameResult:
    """Build a GameResult with two tanks and alternating actions.

    Args:
        history_len: Number of history entries to generate.
        actions: Actions to cycle through. Defaults to ``[Action.PASS]``.
        winner: Winning team name (or ``None`` for a draw).
        team_a: Name for the first team.
        team_b: Name for the second team.

    Returns:
        A ``GameResult`` with tanks "A1" (team_a) and "B1" (team_b).
    """
    if actions is None:
        actions = [Action.PASS]

    tank_a = Tank(id="A1", team=team_a, position=Position(1, 1), direction=Direction.NORTH)
    tank_b = Tank(id="B1", team=team_b, position=Position(3, 3), direction=Direction.SOUTH)
    initial = GameState(tanks=[tank_a, tank_b], current_tank_id="A1")
    game_map = GameMap(width=5, height=5)

    tanks = [tank_a, tank_b]
    history: list[HistoryEntry] = []
    for i in range(history_len):
        acting = tanks[i % 2]
        action = actions[i % len(actions)]
        history.append(
            HistoryEntry(
                tank_id=acting.id,
                requested_action=action,
                applied_action=action,
                valid=True,
                state_after=initial.model_copy(deep=True),
            )
        )

    return GameResult(
        winner=winner,
        game_map=game_map,
        initial_state=initial,
        history=history,
        turns_played=history_len,
    )


def make_minimal_game_result(*, history_len: int = 0) -> GameResult:
    """Build a minimal GameResult with a single tank.

    Each history entry records a PASS action, and the state after is a copy
    of the initial state. Useful for CLI and timeline tests.

    Args:
        history_len: Number of history entries to generate.

    Returns:
        A ``GameResult`` with one tank "t1" on team "A".
    """
    tank = Tank(id="t1", team="A", position=Position(1, 1), direction=Direction.NORTH)
    initial = GameState(tanks=[tank], current_tank_id="t1")
    game_map = GameMap(width=3, height=3)

    history: list[HistoryEntry] = []
    for _ in range(history_len):
        history.append(
            HistoryEntry(
                tank_id="t1",
                requested_action=Action.PASS,
                applied_action=Action.PASS,
                valid=True,
                state_after=initial.model_copy(deep=True),
            )
        )

    return GameResult(
        winner=None,
        game_map=game_map,
        initial_state=initial,
        history=history,
        turns_played=history_len,
    )
