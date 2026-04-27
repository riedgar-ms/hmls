"""Game engine: orchestrates a complete tank-game match.

The engine manages turn order (interleaved by team), fog-of-war
visibility, action validation, and history recording.  It runs for
a fixed number of rounds or until one side is fully destroyed.
"""

from __future__ import annotations

from itertools import zip_longest

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


def _build_interleaved_turn_order(tanks: list[Tank]) -> list[TankId]:
    """Build an interleaved turn order from a list of tanks.

    Tanks are grouped by team (preserving within-team order), then
    interleaved: first tank of team A, first tank of team B, second
    tank of team A, second tank of team B, etc.  Teams are sorted
    alphabetically for determinism.  If teams have unequal sizes, the
    shorter team's slots are simply skipped once exhausted.

    Args:
        tanks: All tanks in the game.

    Returns:
        Flat list of tank IDs in interleaved order.
    """
    teams: dict[str, list[TankId]] = {}
    for t in tanks:
        teams.setdefault(t.team, []).append(t.id)

    sorted_team_names = sorted(teams)
    team_lists = [teams[name] for name in sorted_team_names]

    order: list[TankId] = []
    for slot in zip_longest(*team_lists):
        for tid in slot:
            if tid is not None:
                order.append(tid)
    return order


def _count_alive_by_team(state: GameState) -> dict[str, int]:
    """Return a mapping from team name to number of alive tanks."""
    counts: dict[str, int] = {}
    for t in state.tanks:
        if t.alive:
            counts[t.team] = counts.get(t.team, 0) + 1
    return counts


# ── Engine ────────────────────────────────────────────────────────────


class GameEngine:
    """Orchestrates a turn-based tank-game match.

    The engine alternates between players, asking each for the desired
    action for each of their tanks in interleaved order.  Actions are
    validated; invalid actions trigger a notification to the player and
    a :data:`~hmls.core.types.Action.PASS` is substituted.

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

        turn_order = _build_interleaved_turn_order(tanks)
        self._state = GameState(
            game_map=game_map,
            tanks=tanks,
            turn_order=turn_order,
            current_turn_index=0,
        )
        self._players = players
        self._max_rounds = max_rounds
        self._patch_size = patch_size

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

        The game runs for at most :attr:`max_rounds` full rounds.  A
        round consists of every tank in the turn order acting once
        (dead tanks are skipped).  The game ends early if all tanks
        of one team are destroyed.

        Returns:
            A :class:`GameResult` with the winner, final state, and
            full action history.
        """
        state = self._state
        history: list[HistoryEntry] = []
        turn_order = state.turn_order
        num_tanks = len(turn_order)

        for round_num in range(self._max_rounds):
            for _slot in range(num_tanks):
                # The game state tracks whose turn it is.
                current_tid = state.current_tank_id
                tank = state.get_tank(current_tid)

                # Dead tanks are skipped by GameState.current_tank_id,
                # but if all tanks are dead we should stop.
                if not tank.alive:
                    break

                team = tank.team
                player = self._players[team]
                view = build_player_view(state, team, self._patch_size)

                requested = player.choose_action(current_tid, view)
                result = validate_action(state, current_tid, requested)

                if result.valid:
                    applied = requested
                else:
                    player.notify_invalid_action(current_tid, requested, result.reason)
                    applied = Action.PASS

                state = apply_action(state, current_tid, applied)

                history.append(
                    HistoryEntry(
                        tank_id=current_tid,
                        requested_action=requested,
                        applied_action=applied,
                        valid=result.valid,
                        reason=result.reason,
                        state_after=state,
                    )
                )

                # Early termination: check if one side is fully destroyed.
                alive_counts = _count_alive_by_team(state)
                if any(
                    team_name not in alive_counts for team_name in {t.team for t in state.tanks}
                ):
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
