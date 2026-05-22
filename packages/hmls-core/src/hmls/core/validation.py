"""Input-validation helpers for GameEngine construction.

These functions verify that the parameters passed to the engine are
internally consistent before any game logic runs.
"""

from __future__ import annotations

from hmls.core.map import CellType, GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank


def _validate_basic_params(max_turns: int, patch_size: int) -> None:
    """Check that *patch_size* and *max_turns* are sensible.

    Raises:
        ValueError: If *patch_size* is not odd or < 3, or *max_turns* < 1.
    """
    if patch_size < 3 or patch_size % 2 == 0:
        msg = f"patch_size must be odd and >= 3, got {patch_size}"
        raise ValueError(msg)
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")  # noqa: EM102


def _validate_tanks(tanks: list[Tank], game_map: GameMap) -> None:
    """Validate the tank list against the game map.

    Checks that *tanks* is non-empty, all IDs are unique, no two tanks
    share a position, and every tank is in bounds on a passable cell.

    Raises:
        ValueError: On any invalid tank configuration.
    """
    if not tanks:
        raise ValueError("Must provide at least one tank")  # noqa: EM101

    ids = [t.id for t in tanks]
    if len(set(ids)) != len(ids):
        raise ValueError("Tank IDs must be unique")  # noqa: EM101

    positions = [t.position for t in tanks]
    if len(set(positions)) != len(positions):
        raise ValueError("Tanks must not share starting positions")  # noqa: EM101

    for t in tanks:
        if not game_map.in_bounds(t.position.x, t.position.y):
            msg = f"Tank {t.id!r} is out of bounds at {t.position}"
            raise ValueError(msg)
        if game_map[t.position.x, t.position.y] != CellType.PASSABLE:
            msg = f"Tank {t.id!r} starts on an impassable cell at {t.position}"
            raise ValueError(msg)


def _validate_teams(tanks: list[Tank], players: dict[str, Player]) -> None:
    """Validate team/player consistency.

    Checks exactly 2 teams exist, every team has a corresponding player,
    and each player's ``team`` attribute matches its registration key.

    Raises:
        ValueError: On any team/player mismatch.
    """
    team_names = {t.team for t in tanks}
    if len(team_names) != 2:
        msg = f"Exactly 2 teams required, got {len(team_names)}: {team_names}"
        raise ValueError(msg)

    for team in team_names:
        if team not in players:
            raise ValueError(f"No player provided for team {team!r}")  # noqa: EM102

    for name, player in players.items():
        if player.team != name:
            msg = f"Player registered under {name!r} has team {player.team!r}"
            raise ValueError(msg)
