"""Fog-of-war visibility: egocentric patches for tank vision.

Each tank sees an NxN patch of the world centred on itself, rotated so
that the tank's forward direction points "up" (toward row 0).  Cells
within the 8-neighbour ring around the tank are always visible.  Beyond
that ring, only cells within a 45° forward cone are revealed.  All
other cells – and cells outside the map boundary – are fog.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank, TankId
from hmls.core.types import Direction, Position

# ── Patch cell types ──────────────────────────────────────────────────


class VisibleCell(BaseModel):
    """A cell whose contents are visible to the observing tank.

    Attributes:
        kind: Discriminator tag, always ``"visible"``.
        cell_type: Terrain type of the cell.
        tank: The tank occupying this cell, or ``None`` if empty.
    """

    kind: Literal["visible"] = "visible"
    cell_type: CellType
    tank: Tank | None = None


class FogCell(BaseModel):
    """A cell hidden by fog of war – reveals nothing.

    Attributes:
        kind: Discriminator tag, always ``"fog"``.
    """

    kind: Literal["fog"] = "fog"


PatchCell = Annotated[VisibleCell | FogCell, Field(discriminator="kind")]
"""A single cell in a visibility patch: either visible or fogged."""


# ── Aggregate models ──────────────────────────────────────────────────


class TankPatch(BaseModel):
    """Egocentric NxN visibility patch for one tank.

    The grid is oriented so that the tank's forward direction is "up"
    (row 0).  The tank itself sits at the centre cell
    ``grid[half][half]`` where ``half = len(grid) // 2``.

    Attributes:
        tank_id: ID of the tank this patch is centred on.
        position: World position of the tank.
        direction: World-space direction the tank is facing.
        grid: Row-major NxN list of :data:`PatchCell` values.
    """

    tank_id: TankId
    position: Position
    direction: Direction
    grid: list[list[PatchCell]]


class TankInfo(BaseModel):
    """Lightweight metadata for a friendly tank.

    This is always provided to the owning player regardless of
    fog-of-war, so the player always knows where their own tanks are.

    Attributes:
        tank_id: Unique identifier.
        position: Current world position.
        direction: Current facing direction.
        alive: Whether the tank is still in play.
    """

    tank_id: TankId
    position: Position
    direction: Direction
    alive: bool


class PlayerView(BaseModel):
    """The information provided to a player on their turn.

    Fog of war is enforced: enemy tanks are only visible when they
    fall within a friendly tank's visibility cone.  Friendly tank
    metadata is always available via :attr:`tanks`.

    Attributes:
        patches: One egocentric :class:`TankPatch` per alive friendly
            tank.
        tanks: Position, direction, and alive status for every
            friendly tank (alive or dead).
    """

    patches: list[TankPatch]
    tanks: list[TankInfo]


# ── Visibility mask ───────────────────────────────────────────────────


def compute_visibility_mask(n: int) -> list[list[bool]]:
    """Return an NxN boolean mask for the egocentric visibility cone.

    The mask is in egocentric space where "forward" is toward row 0.
    The centre of the patch is at ``(half, half)`` where ``half = n // 2``.

    A cell is visible if:

    * it lies within the 8-neighbour ring (Chebyshev distance ≤ 1 from
      the centre), **or**
    * it is in the forward direction (``dr > 0``, i.e. row < centre)
      **and** within 45° of straight ahead (``|dc| ≤ dr``).

    Here ``dr = half - row`` (positive means toward row 0 / forward)
    and ``dc = col - half`` (positive means rightward).

    Args:
        n: Patch side length.  Must be odd and ≥ 3.

    Returns:
        Row-major ``n × n`` list of booleans.

    Raises:
        ValueError: If *n* is even or less than 3.
    """
    if n < 3 or n % 2 == 0:
        raise ValueError(f"patch size must be odd and >= 3, got {n}")

    half = n // 2
    mask: list[list[bool]] = []
    for row in range(n):
        row_mask: list[bool] = []
        for col in range(n):
            dr = half - row  # positive = forward (toward row 0)
            dc = col - half  # positive = rightward
            # 8-neighbour ring (includes the centre cell)
            if abs(dr) <= 1 and abs(dc) <= 1:
                visible = True
            # Forward 45° cone beyond the ring
            elif dr > 0 and abs(dc) <= dr:
                visible = True
            else:
                visible = False
            row_mask.append(visible)
        mask.append(row_mask)
    return mask


# ── Patch extraction ──────────────────────────────────────────────────



def extract_patch(
    game_state: GameState,
    game_map: GameMap,
    tank_id: TankId,
    patch_size: int,
) -> TankPatch:
    """Build an egocentric visibility patch for a single tank.

    The patch is an ``patch_size × patch_size`` grid centred on the
    tank, rotated so the tank's forward direction is "up" (row 0).
    Cells outside the visibility mask or outside the map are
    :class:`FogCell`.

    Args:
        game_state: Current game state (tanks and turn info).
        game_map: The map on which the game is played.
        tank_id: The tank to build the patch for.
        patch_size: Side length of the square patch (must be odd, ≥ 3).

    Returns:
        A :class:`TankPatch` with the populated grid.

    Raises:
        KeyError: If *tank_id* does not exist in the game state.
        ValueError: If *patch_size* is invalid.
    """
    tank = game_state.get_tank(tank_id)
    mask = compute_visibility_mask(patch_size)
    half = patch_size // 2

    forward = tank.direction.forward_delta()
    right = tank.direction.turn_right().forward_delta()

    # Build a position→Tank lookup for the whole map.
    pos_to_tank: dict[Position, Tank] = {}
    for t in game_state.tanks:
        pos_to_tank[t.position] = t

    fog = FogCell()
    grid: list[list[PatchCell]] = []

    for ego_row in range(patch_size):
        row_cells: list[PatchCell] = []
        for ego_col in range(patch_size):
            if not mask[ego_row][ego_col]:
                row_cells.append(fog)
                continue

            # Map egocentric (row, col) back to world offset (ego→world).
            # forward_steps = half - ego_row
            # right_steps = ego_col - half
            # (dx, dy) = forward_steps * forward + right_steps * right
            fwd_steps = half - ego_row
            rgt_steps = ego_col - half
            fx, fy = forward
            rx, ry = right
            world_dx = fwd_steps * fx + rgt_steps * rx
            world_dy = fwd_steps * fy + rgt_steps * ry

            world_x = tank.position.x + world_dx
            world_y = tank.position.y + world_dy

            # Out-of-bounds cells are fog.
            if not game_map.in_bounds(world_x, world_y):
                row_cells.append(fog)
                continue

            cell_type = game_map[world_x, world_y]
            world_pos = Position(world_x, world_y)
            occupant = pos_to_tank.get(world_pos)
            row_cells.append(VisibleCell(cell_type=cell_type, tank=occupant))

        grid.append(row_cells)

    return TankPatch(
        tank_id=tank_id,
        position=tank.position,
        direction=tank.direction,
        grid=grid,
    )


def build_player_view(
    game_state: GameState,
    game_map: GameMap,
    team: str,
    patch_size: int,
) -> PlayerView:
    """Build the fog-of-war view for a team.

    Creates one :class:`TankPatch` per alive friendly tank and
    populates the friendly-tank metadata list with all tanks on
    the team (alive or dead).

    Args:
        game_state: Current game state (tanks and turn info).
        game_map: The map on which the game is played.
        team: Team identifier (e.g. ``"alpha"``).
        patch_size: Side length of each visibility patch.

    Returns:
        A :class:`PlayerView` ready to pass to a :class:`Player`.
    """
    team_tanks = [t for t in game_state.tanks if t.team == team]

    patches = [extract_patch(game_state, game_map, t.id, patch_size) for t in team_tanks if t.alive]

    tank_infos = [
        TankInfo(
            tank_id=t.id,
            position=t.position,
            direction=t.direction,
            alive=t.alive,
        )
        for t in team_tanks
    ]

    return PlayerView(patches=patches, tanks=tank_infos)
