"""Aggregate game state: map, tanks, and turn tracking."""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.map import GameMap
from hmls.core.tank import Tank, TankId
from hmls.core.types import Position


class GameState(BaseModel):
    """Complete snapshot of the game at a point in time.

    The game state is treated as immutable by convention: mutation
    functions (in :mod:`hmls.core.actions`) return a *new* ``GameState``
    rather than modifying in place, which makes undo/replay trivial.

    Attributes:
        game_map: The map on which the game is played.
        tanks: All tanks (alive and destroyed) in the game.
        turn_order: Tank IDs in the order they take turns.
        current_turn_index: Index into *turn_order* for the next tank to act.
    """

    game_map: GameMap
    tanks: list[Tank]
    turn_order: list[TankId]
    current_turn_index: int = 0

    # ── Lookup helpers ────────────────────────────────────────────────

    @property
    def alive_tanks(self) -> list[Tank]:
        """Return only the tanks that are still alive."""
        return [t for t in self.tanks if t.alive]

    @property
    def tank_positions(self) -> dict[Position, TankId]:
        """Build a mapping from position to tank ID for all tanks.

        Both alive tanks and destroyed wreckage occupy space on the map,
        so this includes every tank regardless of ``alive`` status.
        Useful for occupancy checks during move validation.
        """
        return {t.position: t.id for t in self.tanks}

    def get_tank(self, tank_id: TankId) -> Tank:
        """Look up a tank by its ID.

        Raises:
            KeyError: If no tank with the given ID exists.
        """
        for t in self.tanks:
            if t.id == tank_id:
                return t
        raise KeyError(f"No tank with id {tank_id!r}")

    @property
    def current_tank_id(self) -> TankId:
        """Return the ID of the tank whose turn it is.

        Skips over dead tanks in the turn order.  If all tanks are dead,
        returns the ID at the raw index (the caller should check for
        game-over conditions).
        """
        alive_ids = {t.id for t in self.tanks if t.alive}
        order_len = len(self.turn_order)
        # Walk forward from current_turn_index looking for an alive tank.
        for i in range(order_len):
            idx = (self.current_turn_index + i) % order_len
            tid = self.turn_order[idx]
            if tid in alive_ids:
                return tid
        # Fallback: no alive tanks — return raw index entry.
        return self.turn_order[self.current_turn_index % order_len]
