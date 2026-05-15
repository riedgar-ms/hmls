"""Fog-of-war visibility: egocentric patches for tank vision.

Each tank sees an NxN patch of the world centred on itself, rotated so
that the tank's forward direction points "up" (toward row 0).  Cells
within the 8-neighbour ring around the tank are always visible.  Beyond
that ring, only cells within a 45° forward cone are revealed.  All
other cells are fog; cells outside the map boundary are marked as
boundary (definitively impassable).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import Tank, TankId
from hmls.core.types import Direction, Position

# ── Patch cell types ──────────────────────────────────────────────────


class VisibleCell(BaseModel, extra="forbid"):
    """A cell whose contents are visible to the observing tank.

    Attributes:
        kind: Discriminator tag, always ``"visible"``.
        cell_type: Terrain type of the cell.
        tank: The tank occupying this cell, or ``None`` if empty.
    """

    kind: Literal["visible"] = "visible"
    cell_type: CellType
    tank: Tank | None = None


class FogCell(BaseModel, extra="forbid"):
    """A cell hidden by fog of war – reveals nothing.

    Attributes:
        kind: Discriminator tag, always ``"fog"``.
    """

    kind: Literal["fog"] = "fog"


class BoundaryCell(BaseModel, extra="forbid"):
    """A cell outside the map boundary – always impassable.

    Unlike :class:`FogCell`, a boundary cell is definitively known to be
    impassable (it is the edge of the world, not merely hidden).

    Attributes:
        kind: Discriminator tag, always ``"boundary"``.
    """

    kind: Literal["boundary"] = "boundary"


PatchCell = Annotated[VisibleCell | FogCell | BoundaryCell, Field(discriminator="kind")]
"""A single cell in a visibility patch: visible, fogged, or boundary."""


# ── Aggregate models ──────────────────────────────────────────────────


class TankPatch(BaseModel, extra="forbid"):
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


class TankInfo(BaseModel, extra="forbid"):
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


class PlayerView(BaseModel, extra="forbid"):
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
        msg = f"patch size must be odd and >= 3, got {n}"
        raise ValueError(msg)

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


# ── Coordinate conversion ─────────────────────────────────────────────


def ego_to_world_position(patch: TankPatch, ego_row: int, ego_col: int) -> Position:
    """Convert egocentric grid coordinates to a world position.

    Given a :class:`TankPatch` and an (ego_row, ego_col) pair within its
    grid, compute the corresponding world-space :class:`Position`.

    The conversion uses the patch's direction to determine the forward
    and right axes, then applies the standard ego→world transform::

        fwd_steps = half - ego_row
        rgt_steps = ego_col - half
        world = patch.position + fwd_steps * forward + rgt_steps * right

    Args:
        patch: The egocentric visibility patch (provides position,
            direction, and grid size).
        ego_row: Row index in the egocentric grid (0 = furthest forward).
        ego_col: Column index in the egocentric grid.

    Returns:
        The world-space :class:`Position` corresponding to (ego_row, ego_col).
    """
    half = len(patch.grid) // 2
    return _ego_to_world(
        origin=patch.position,
        direction=patch.direction,
        half=half,
        ego_row=ego_row,
        ego_col=ego_col,
    )


def _ego_to_world(
    origin: Position,
    direction: Direction,
    half: int,
    ego_row: int,
    ego_col: int,
) -> Position:
    """Low-level ego→world conversion (shared by public API and patch builder)."""
    forward = direction.forward_delta()
    right = direction.turn_right().forward_delta()

    fwd_steps = half - ego_row
    rgt_steps = ego_col - half
    fx, fy = forward
    rx, ry = right

    world_x = origin.x + fwd_steps * fx + rgt_steps * rx
    world_y = origin.y + fwd_steps * fy + rgt_steps * ry
    return Position(world_x, world_y)


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
    :class:`FogCell`.  Cells within the mask but outside the map
    boundary are :class:`BoundaryCell`.

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

    # Build a position→Tank lookup for the whole map.
    pos_to_tank: dict[Position, Tank] = {}
    for t in game_state.tanks:
        pos_to_tank[t.position] = t

    fog = FogCell()
    boundary = BoundaryCell()
    grid: list[list[PatchCell]] = []

    for ego_row in range(patch_size):
        row_cells: list[PatchCell] = []
        for ego_col in range(patch_size):
            if not mask[ego_row][ego_col]:
                row_cells.append(fog)
                continue

            world_pos = _ego_to_world(
                origin=tank.position,
                direction=tank.direction,
                half=half,
                ego_row=ego_row,
                ego_col=ego_col,
            )

            # Out-of-bounds cells are boundary (definitively impassable).
            if not game_map.in_bounds(world_pos.x, world_pos.y):
                row_cells.append(boundary)
                continue

            cell_type = game_map[world_pos.x, world_pos.y]
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
