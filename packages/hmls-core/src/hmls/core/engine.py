"""Game engine: orchestrates a complete tank-game match.

The engine alternates between players on every turn.  Each player
independently cycles through their alive tanks, so the shorter team's
tanks are reused to match the longer team's count.  The engine manages
fog-of-war visibility, action validation, and history recording.  It
runs for a fixed number of rounds or until one side is fully destroyed.
"""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.actions import apply_action, validate_action
from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action
from hmls.core.visibility import build_player_view

# ── Result models ─────────────────────────────────────────────────────


class HistoryEntry(BaseModel):
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
        state_after: The full game state *after* the action was applied.
    """

    tank_id: TankId
    requested_action: Action
    applied_action: Action
    valid: bool
    reason: str = ""
    state_after: GameState


class GameResult(BaseModel):
    """Outcome of a complete game.

    Attributes:
        winner: Team name of the winning side, or ``None`` for a draw.
        final_state: The game state when the game ended.
        history: Ordered list of every action taken during the game.
        rounds_played: Number of full rounds completed.
    """

    winner: str | None
    final_state: GameState
    history: list[HistoryEntry]
    rounds_played: int


# ── Helpers ───────────────────────────────────────────────────────────


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
    raise StopIteration(f"No alive tanks for team {team!r}")


def _set_current_tank(state: GameState, tank_id: TankId) -> GameState:
    """Return a copy of *state* with ``current_turn_index`` pointing to *tank_id*.

    The tank must appear in ``state.turn_order``.
    """
    idx = state.turn_order.index(tank_id)
    return state.model_copy(update={"current_turn_index": idx})


# ── Engine ────────────────────────────────────────────────────────────


class GameEngine:
    """Orchestrates a turn-based tank-game match.

    Players always alternate turns.  On each turn the engine asks the
    current player for an action for their next alive tank, cycling
    through the player's tanks independently.  This means the shorter
    team's tanks are reused to match the longer team's count within
    each round.

    A *round* consists of ``max(alive_team_size)`` turns **per player**
    (recalculated at the start of each round), for a total of
    ``max(alive_team_size) * 2`` individual actions.

    Args:
        game_map: The map to play on.
        tanks: Starting tank list (must contain tanks for exactly 2 teams).
        players: Mapping from team name to the :class:`Player` controlling
            that team.  Must contain an entry for every team present in
            *tanks*.
        max_rounds: Maximum number of full rounds before the game ends.
        patch_size: Side length of the egocentric visibility patches
            (must be odd, ≥ 3).  Defaults to ``7``.

    Raises:
        ValueError: If the inputs are inconsistent (see validation).
    """

    def __init__(
        self,
        game_map: GameMap,
        tanks: list[Tank],
        players: dict[str, Player],
        max_rounds: int,
        patch_size: int = 7,
    ) -> None:
        self._validate_inputs(game_map, tanks, players, max_rounds, patch_size)

        # turn_order is a flat list of all unique tank IDs — the engine
        # manages alternation itself and points current_turn_index at
        # the correct tank before each apply_action call.
        turn_order = [t.id for t in tanks]
        self._state = GameState(
            game_map=game_map,
            tanks=tanks,
            turn_order=turn_order,
            current_turn_index=0,
        )
        self._players = players
        self._max_rounds = max_rounds
        self._patch_size = patch_size
        # Teams sorted alphabetically for deterministic alternation.
        self._team_order: list[str] = sorted({t.team for t in tanks})

    # ── Input validation ──────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        game_map: GameMap,
        tanks: list[Tank],
        players: dict[str, Player],
        max_rounds: int,
        patch_size: int,
    ) -> None:
        """Validate engine construction parameters.

        Raises:
            ValueError: On any invalid configuration.
        """
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}")

        if max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")

        if not tanks:
            raise ValueError("Must provide at least one tank")

        # Unique IDs.
        ids = [t.id for t in tanks]
        if len(set(ids)) != len(ids):
            raise ValueError("Tank IDs must be unique")

        # Exactly 2 teams.
        team_names = {t.team for t in tanks}
        if len(team_names) != 2:
            raise ValueError(f"Exactly 2 teams required, got {len(team_names)}: {team_names}")

        # Every team has a player.
        for team in team_names:
            if team not in players:
                raise ValueError(f"No player provided for team {team!r}")

        # Player teams match.
        for name, player in players.items():
            if player.team != name:
                raise ValueError(f"Player registered under {name!r} has team {player.team!r}")

        # No overlapping positions.
        positions = [t.position for t in tanks]
        if len(set(positions)) != len(positions):
            raise ValueError("Tanks must not share starting positions")

        # All tanks in bounds and on passable cells.
        for t in tanks:
            if not game_map.in_bounds(t.position.x, t.position.y):
                raise ValueError(f"Tank {t.id!r} is out of bounds at {t.position}")
            from hmls.core.map import CellType

            if game_map[t.position.x, t.position.y] != CellType.PASSABLE:
                raise ValueError(f"Tank {t.id!r} starts on an impassable cell at {t.position}")

    # ── Game loop ─────────────────────────────────────────────────

    def run(self) -> GameResult:
        """Execute the game and return the result.

        The game runs for at most *max_rounds* full rounds.  Each round
        has ``max(alive_team_size) * 2`` turns (players alternate, with
        the shorter team cycling its tanks).  The game ends early if all
        tanks of one team are destroyed.

        Returns:
            A :class:`GameResult` with the winner, final state, and
            full action history.
        """
        state = self._state
        history: list[HistoryEntry] = []

        # Per-team cursors into their tank lists (survives across rounds).
        cursors: dict[str, int] = {t: 0 for t in self._team_order}

        for round_num in range(self._max_rounds):
            # How many turns per player this round?
            alive_counts = _count_alive_by_team(state)
            if len(alive_counts) < 2:
                # One side already eliminated.
                break
            turns_per_player = max(alive_counts.values())

            for _slot in range(turns_per_player):
                for team in self._team_order:
                    # Skip if team is fully eliminated.
                    if team not in _count_alive_by_team(state):
                        continue

                    tank_id, cursors[team] = _next_alive_tank(state, team, cursors[team])

                    # Point GameState at the chosen tank so apply_action
                    # accepts it as the current turn.
                    state = _set_current_tank(state, tank_id)

                    player = self._players[team]
                    view = build_player_view(state, team, self._patch_size)
                    requested = player.choose_action(tank_id, view)
                    result = validate_action(state, tank_id, requested)

                    if result.valid:
                        applied = requested
                    else:
                        player.notify_invalid_action(tank_id, requested, result.reason)
                        applied = Action.PASS

                    state = apply_action(state, tank_id, applied)

                    history.append(
                        HistoryEntry(
                            tank_id=tank_id,
                            requested_action=requested,
                            applied_action=applied,
                            valid=result.valid,
                            reason=result.reason,
                            state_after=state,
                        )
                    )

                    # Early termination: one side fully destroyed.
                    if len(_count_alive_by_team(state)) < 2:
                        return self._make_result(state, history, round_num + 1)

        return self._make_result(state, history, self._max_rounds)

    @staticmethod
    def _make_result(
        state: GameState,
        history: list[HistoryEntry],
        rounds_played: int,
    ) -> GameResult:
        """Determine the winner and build the final :class:`GameResult`."""
        alive_counts = _count_alive_by_team(state)
        if not alive_counts:
            winner = None
        elif len(alive_counts) == 1:
            winner = next(iter(alive_counts))
        else:
            max_count = max(alive_counts.values())
            leaders = [t for t, c in alive_counts.items() if c == max_count]
            winner = leaders[0] if len(leaders) == 1 else None
        return GameResult(
            winner=winner,
            final_state=state,
            history=history,
            rounds_played=rounds_played,
        )
