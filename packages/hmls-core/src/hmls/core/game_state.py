"""Aggregate game state: tanks and turn tracking.

The game map is intentionally *not* part of the state — it never
changes during a game and is supplied separately to functions that
need it (see :mod:`hmls.core.actions` and :mod:`hmls.core.visibility`).

Turn scheduling is the sole responsibility of the game engine
(:class:`~hmls.core.engine.GameEngine`).  ``GameState`` merely records
*which* tank is currently active via :attr:`current_tank_id`; it does
not contain scheduling logic.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from hmls.core.tank import Tank, TankId
from hmls.core.types import Position


class GameState(BaseModel):
    """Snapshot of the mutable game state at a point in time.

    The game state is treated as immutable by convention: mutation
    functions (in :mod:`hmls.core.actions`) return a *new* ``GameState``
    rather than modifying in place, which makes undo/replay trivial.

    The game map is *not* stored here because it never changes during
    a game.  It is passed separately to functions that need terrain
    information.

    Attributes:
        tanks: All tanks (alive and destroyed) in the game.  The list
            order is stable for the lifetime of a game.
        current_tank_id: ID of the tank whose turn it is, or ``None``
            when the state has no meaningful active turn (e.g. an
            empty tank list, or a state constructed outside the engine).
    """

    tanks: list[Tank]
    current_tank_id: TankId | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_current_turn_index(cls, data: dict) -> dict:  # type: ignore[type-arg]
        """Accept legacy ``current_turn_index`` and convert to ``current_tank_id``.

        This allows old serialised ``GameState`` payloads (which stored
        an integer index) to be deserialised transparently.
        """
        if not isinstance(data, dict):
            return data
        if "current_tank_id" not in data and "current_turn_index" in data:
            idx = data.pop("current_turn_index")
            tanks = data.get("tanks", [])
            if tanks and 0 <= idx < len(tanks):
                tank = tanks[idx]
                # Handle both raw dicts and Tank instances.
                data["current_tank_id"] = tank["id"] if isinstance(tank, dict) else tank.id
        return data

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
