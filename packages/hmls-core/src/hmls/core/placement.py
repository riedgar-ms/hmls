"""Tank placement: randomly assign tanks to passable cells on a map."""

from __future__ import annotations

import random

from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position


class InsufficientPassableCellsError(Exception):
    """Raised when the map has too few passable cells for the requested tanks.

    Attributes:
        needed: Number of passable cells required.
        available: Number of passable cells actually on the map.
    """

    def __init__(self, needed: int, available: int) -> None:
        self.needed = needed
        self.available = available
        super().__init__(f"Need {needed} passable cells but map only has {available}")


def place_tanks(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks randomly on passable cells for two teams.

    Teams are named ``"A"`` and ``"B"``.  Tanks are assigned IDs like
    ``"A1"``, ``"A2"``, ``"B1"``, ``"B2"``, etc.  Each tank gets a
    random direction.

    Args:
        game_map: The map to place tanks on.
        tanks_per_player: Number of tanks per team.
        seed: Optional random seed for reproducibility.

    Returns:
        List of all tanks for both teams.

    Raises:
        InsufficientPassableCellsError: If there are not enough passable
            cells on the map.
    """
    total_needed = tanks_per_player * 2
    passable_positions = [
        Position(x, y) for x, y in game_map.all_positions() if game_map[x, y] == CellType.PASSABLE
    ]
    if len(passable_positions) < total_needed:
        raise InsufficientPassableCellsError(needed=total_needed, available=len(passable_positions))

    rng = random.Random(seed)
    chosen = rng.sample(passable_positions, total_needed)
    directions = list(Direction)
    tanks: list[Tank] = []

    for team_idx, team_name in enumerate(["A", "B"]):
        for i in range(tanks_per_player):
            pos = chosen[team_idx * tanks_per_player + i]
            tanks.append(
                Tank(
                    id=f"{team_name}{i + 1}",
                    team=team_name,
                    position=pos,
                    direction=rng.choice(directions),
                )
            )

    return tanks
