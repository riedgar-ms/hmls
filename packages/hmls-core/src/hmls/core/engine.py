"""Game engine: orchestrates a complete tank-game match.

The engine alternates between players on every turn.  Each player
independently cycles through their alive tanks.  The engine manages
fog-of-war visibility, action validation, and history recording.  It
runs for a fixed number of turns or until one side is fully destroyed.

The engine supports two usage patterns:

* **Batch mode** — call :meth:`GameEngine.run` to play an entire game
  and receive a :class:`GameResult`.
* **Step mode** — call :meth:`GameEngine.step` repeatedly to advance
  one turn at a time, inspecting :attr:`~GameEngine.state`,
  :attr:`~GameEngine.game_over`, etc. between steps.
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
        hit: Whether a fire action hit an enemy tank.  ``True`` if a
            tank was destroyed, ``False`` if the shot missed, ``None``
            for non-fire actions.
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
    """Return a copy of *state* with ``current_tank_id`` set to *tank_id*.

    The tank must exist in ``state.tanks``.
    """
    state.get_tank(tank_id)  # Validate existence.
    return state.model_copy(update={"current_tank_id": tank_id})


def _determine_winner(state: GameState) -> str | None:
    """Determine the winner from the current game state.

    Returns the team name of the winning side, or ``None`` for a draw.
    """
    alive_counts = _count_alive_by_team(state)
    if not alive_counts:
        return None
    if len(alive_counts) == 1:
        return next(iter(alive_counts))
    max_count = max(alive_counts.values())
    leaders = [t for t, c in alive_counts.items() if c == max_count]
    return leaders[0] if len(leaders) == 1 else None


# ── Engine ────────────────────────────────────────────────────────────


class GameEngine:
    """Orchestrates a turn-based tank-game match.

    Players always alternate turns.  On each turn the engine asks the
    current player for an action for their next alive tank, cycling
    through the player's tanks independently.

    The game runs for at most *max_turns* individual turns (each turn
    is one player acting with one tank) or until one side is fully
    destroyed.

    The engine supports two usage patterns:

    * **Batch mode** — call :meth:`run` to play the entire game and
      receive a :class:`GameResult`.
    * **Step mode** — call :meth:`step` one turn at a time.  Between
      steps, inspect :attr:`state`, :attr:`game_over`,
      :attr:`current_tank_id`, :attr:`current_team`, etc.

    Args:
        game_map: The map to play on.
        tanks: Starting tank list (must contain tanks for exactly 2 teams).
        players: Mapping from team name to the :class:`Player` controlling
            that team.  Must contain an entry for every team present in
            *tanks*.
        max_turns: Maximum number of individual turns before the game ends.
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
        max_turns: int,
        patch_size: int = 9,
    ) -> None:
        self._validate_inputs(game_map, tanks, players, max_turns, patch_size)

        self._state = GameState(
            tanks=tanks,
        )
        self._game_map = game_map
        self._players = players
        self._max_turns = max_turns
        self._patch_size = patch_size

        # Teams sorted alphabetically for deterministic alternation.
        self._team_order: list[str] = sorted({t.team for t in tanks})
        # Per-team cursors for cycling through tanks.
        self._cursors: dict[str, int] = dict.fromkeys(self._team_order, 0)
        # Global turn counter (indexes into team_order for alternation).
        self._global_turn: int = 0
        self._turns_taken: int = 0
        self._history: list[HistoryEntry] = []

        # Advance to the first valid tank, then capture the initial state.
        self._advance_to_next_tank()
        self._initial_state = self._state

    # ── Public properties ─────────────────────────────────────────

    @property
    def state(self) -> GameState:
        """Current game state."""
        return self._state

    @property
    def game_map(self) -> GameMap:
        """The game map."""
        return self._game_map

    @property
    def players(self) -> dict[str, Player]:
        """Mapping from team name to player."""
        return self._players

    @property
    def history(self) -> list[HistoryEntry]:
        """All turns taken so far."""
        return list(self._history)

    @property
    def turns_taken(self) -> int:
        """Number of individual turns taken."""
        return self._turns_taken

    @property
    def max_turns(self) -> int:
        """Maximum number of turns allowed."""
        return self._max_turns

    @property
    def patch_size(self) -> int:
        """Visibility patch size."""
        return self._patch_size

    @property
    def current_tank_id(self) -> TankId:
        """ID of the tank that should act next.

        Only meaningful when :attr:`game_over` is ``False``.
        """
        tank_id = self._state.current_tank_id
        if tank_id is None:
            raise RuntimeError("No current tank (game may be over or not started)")
        return tank_id

    @property
    def current_team(self) -> str:
        """Team of the tank that should act next.

        Only meaningful when :attr:`game_over` is ``False``.
        """
        tank = self._state.get_tank(self.current_tank_id)
        return tank.team

    @property
    def game_over(self) -> bool:
        """Whether the game has ended."""
        if self._turns_taken >= self._max_turns:
            return True
        alive_teams = {t.team for t in self._state.alive_tanks}
        return len(alive_teams) < 2

    @property
    def winner(self) -> str | None:
        """The winning team, or ``None`` for a draw.

        Only meaningful when :attr:`game_over` is ``True``.
        """
        return _determine_winner(self._state)

    # ── Input validation ──────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        game_map: GameMap,
        tanks: list[Tank],
        players: dict[str, Player],
        max_turns: int,
        patch_size: int,
    ) -> None:
        """Validate engine construction parameters.

        Raises:
            ValueError: On any invalid configuration.
        """
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}")

        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

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

    # ── Turn management ───────────────────────────────────────────

    def _advance_to_next_tank(self) -> None:
        """Advance internal state to point at the next alive tank.

        Walks through the team alternation, skipping eliminated teams,
        and sets ``current_tank_id`` on the state.
        """
        team_count = len(self._team_order)
        for _ in range(team_count):
            team = self._team_order[self._global_turn % team_count]
            alive_counts = _count_alive_by_team(self._state)

            if team not in alive_counts:
                self._global_turn += 1
                continue

            tank_id, self._cursors[team] = _next_alive_tank(self._state, team, self._cursors[team])
            self._state = _set_current_tank(self._state, tank_id)
            return

    # ── Step-by-step execution ────────────────────────────────────

    def step(self) -> HistoryEntry:
        """Execute one turn and return the history entry.

        Determines whose turn it is, asks the player for an action
        (via :meth:`Player.choose_action`), validates it, applies it,
        records history, and advances to the next tank.

        Returns:
            A :class:`HistoryEntry` recording what happened.

        Raises:
            RuntimeError: If the game is already over.
        """
        if self.game_over:
            raise RuntimeError("Game is already over")

        tank_id = self.current_tank_id
        team = self.current_team

        player = self._players[team]
        view = build_player_view(self._state, self._game_map, team, self._patch_size)
        requested = player.choose_action(tank_id, view)
        result = validate_action(self._state, self._game_map, tank_id, requested)

        if result.valid:
            applied = requested
        else:
            player.notify_invalid_action(tank_id, requested, result.reason)
            applied = Action.PASS

        apply_result = apply_action(self._state, self._game_map, tank_id, applied)
        self._state = apply_result.state
        self._turns_taken += 1
        self._global_turn += 1

        # Advance to the next tank (if game not over) so the history
        # entry reflects the correct *next* active tank.
        if not self.game_over:
            self._advance_to_next_tank()

        entry = HistoryEntry(
            tank_id=tank_id,
            requested_action=requested,
            applied_action=applied,
            valid=result.valid,
            reason=result.reason,
            hit=apply_result.hit,
            state_after=self._state,
        )
        self._history.append(entry)

        return entry

    # ── Batch execution ───────────────────────────────────────────

    def run(self) -> GameResult:
        """Execute the game and return the result.

        Players alternate turns, each cycling through their alive tanks.
        The game runs for at most *max_turns* individual turns or until
        one side is fully destroyed.

        Returns:
            A :class:`GameResult` with the winner, final state, and
            full action history.
        """
        while not self.game_over:
            self.step()
        return self.make_result()

    def make_result(self) -> GameResult:
        """Build a :class:`GameResult` from the current engine state.

        This can be called at any point — during or after the game — to
        snapshot the result.  Useful for the TUI save dialog.

        Returns:
            A :class:`GameResult` reflecting the current state.
        """
        return GameResult(
            winner=_determine_winner(self._state),
            game_map=self._game_map,
            initial_state=self._initial_state,
            history=list(self._history),
            turns_played=self._turns_taken,
        )
