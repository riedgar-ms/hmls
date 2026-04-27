"""Turn management logic for the interactive test harness.

Replicates the engine's team-alternation and per-team tank cycling,
but exposes a step-by-step interface so the TUI can drive one turn
at a time.
"""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.actions import ActionResult, apply_action, validate_action
from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import TankId
from hmls.core.types import Action


class HistoryEntry(BaseModel):
    """Record of a single turn in the interactive game.

    Attributes:
        tank_id: The tank that acted.
        requested_action: The action the user chose.
        applied_action: The action actually applied.
        valid: Whether the requested action was legal.
        reason: Explanation if the action was invalid.
        state_after: Game state after the action was applied.
    """

    tank_id: TankId
    requested_action: Action
    applied_action: Action
    valid: bool
    reason: str = ""
    state_after: GameState


class GameLoop:
    """Step-by-step game loop for the interactive TUI.

    This mirrors the engine's turn alternation logic but allows the TUI
    to call :meth:`step` one turn at a time.

    Args:
        game_map: The map being played on.
        initial_state: Starting game state.
        max_turns: Maximum number of individual turns.
        patch_size: Visibility patch size (odd, >= 3).
    """

    def __init__(
        self,
        game_map: GameMap,
        initial_state: GameState,
        max_turns: int = 200,
        patch_size: int = 7,
    ) -> None:
        self._game_map = game_map
        self._state = initial_state
        self._initial_state = initial_state
        self._max_turns = max_turns
        self._patch_size = patch_size
        self._history: list[HistoryEntry] = []
        self._turns_taken: int = 0

        # Teams sorted alphabetically for deterministic alternation.
        self._team_order: list[str] = sorted({t.team for t in initial_state.tanks})
        # Per-team cursors for cycling through tanks.
        self._cursors: dict[str, int] = {t: 0 for t in self._team_order}
        # Track the global turn number (indexes into team_order).
        self._global_turn: int = 0

        # Advance to the first valid tank.
        self._advance_to_next_tank()

    @property
    def state(self) -> GameState:
        """Current game state."""
        return self._state

    @property
    def game_map(self) -> GameMap:
        """The game map."""
        return self._game_map

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
        """ID of the tank that should act next."""
        return self._state.current_tank_id

    @property
    def current_team(self) -> str:
        """Team of the tank that should act next."""
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
        alive_counts: dict[str, int] = {}
        for t in self._state.alive_tanks:
            alive_counts[t.team] = alive_counts.get(t.team, 0) + 1
        if not alive_counts:
            return None
        if len(alive_counts) == 1:
            return next(iter(alive_counts))
        max_count = max(alive_counts.values())
        leaders = [team for team, count in alive_counts.items() if count == max_count]
        return leaders[0] if len(leaders) == 1 else None

    def _advance_to_next_tank(self) -> None:
        """Advance internal state to point at the next alive tank.

        Walks through the team alternation, skipping eliminated teams,
        and sets ``current_turn_index`` on the state.
        """
        team_count = len(self._team_order)
        # Try each team slot until we find one with alive tanks.
        for _ in range(team_count):
            team = self._team_order[self._global_turn % team_count]
            team_tanks = [t for t in self._state.tanks if t.team == team]
            alive_team_tanks = [t for t in team_tanks if t.alive]

            if not alive_team_tanks:
                self._global_turn += 1
                continue

            # Cycle to next alive tank for this team.
            cursor = self._cursors[team]
            n = len(team_tanks)
            for i in range(n):
                idx = (cursor + i) % n
                if team_tanks[idx].alive:
                    tank_id = team_tanks[idx].id
                    self._cursors[team] = (idx + 1) % n
                    break
            else:
                # Should not happen since we checked alive_team_tanks.
                self._global_turn += 1
                continue

            # Point the state at this tank.
            turn_idx = self._state.turn_order.index(tank_id)
            self._state = self._state.model_copy(update={"current_turn_index": turn_idx})
            return

    def step(self, action: Action) -> HistoryEntry:
        """Execute one turn with the given action.

        Args:
            action: The action chosen by the user for the current tank.

        Returns:
            A history entry recording what happened.

        Raises:
            RuntimeError: If the game is already over.
        """
        if self.game_over:
            raise RuntimeError("Game is already over")

        tank_id = self.current_tank_id
        result: ActionResult = validate_action(self._state, self._game_map, tank_id, action)

        if result.valid:
            applied = action
        else:
            applied = Action.PASS

        new_state = apply_action(self._state, self._game_map, tank_id, applied)
        self._turns_taken += 1

        entry = HistoryEntry(
            tank_id=tank_id,
            requested_action=action,
            applied_action=applied,
            valid=result.valid,
            reason=result.reason,
            state_after=new_state,
        )
        self._history.append(entry)

        self._state = new_state
        self._global_turn += 1

        # Advance to the next tank (if game not over).
        if not self.game_over:
            self._advance_to_next_tank()

        return entry
