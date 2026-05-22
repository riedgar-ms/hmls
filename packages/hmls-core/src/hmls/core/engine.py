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

from hmls.core.actions import apply_action, validate_action
from hmls.core.game_state import GameState
from hmls.core.helpers import _count_alive_by_team, _next_alive_tank, _set_current_tank
from hmls.core.map import GameMap
from hmls.core.player import Player
from hmls.core.results import GameResult, HistoryEntry, _determine_winner
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action
from hmls.core.validation import _validate_basic_params, _validate_tanks, _validate_teams
from hmls.core.visibility import build_player_view


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
            (must be odd, ≥ 3).  Defaults to ``9``.

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
            msg = "No current tank (game may be over or not started)"
            raise RuntimeError(msg)
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
        _validate_basic_params(max_turns, patch_size)
        _validate_tanks(tanks, game_map)
        _validate_teams(tanks, players)

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
            raise RuntimeError("Game is already over")  # noqa: EM101

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
