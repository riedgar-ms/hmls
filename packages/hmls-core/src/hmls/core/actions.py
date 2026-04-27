"""Action validation and execution for the tank game.

All functions are pure: they accept a :class:`~hmls.core.game_state.GameState`
and return either a validation result or a *new* ``GameState``.
"""

from __future__ import annotations

from pydantic import BaseModel

from hmls.core.game_state import GameState
from hmls.core.map import CellType
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action, Position


class ActionResult(BaseModel):
    """Outcome of validating a proposed action.

    Attributes:
        valid: Whether the action is legal.
        reason: Human-readable explanation when the action is invalid.
    """

    valid: bool
    reason: str = ""


# ── Validation ────────────────────────────────────────────────────────


def validate_action(state: GameState, tank_id: TankId, action: Action) -> ActionResult:
    """Check whether *action* is legal for *tank_id* in the given *state*.

    An action is **invalid** if:

    * The tank does not exist or is dead.
    * It is not the tank's turn.
    * The action is ``MOVE_FORWARD`` and the destination cell is out of
      bounds, impassable, or occupied by another alive tank.

    All other actions (``TURN_LEFT``, ``TURN_RIGHT``, ``FIRE``, ``PASS``)
    are always valid (assuming the tank is alive and it is their turn).
    """
    # --- Tank existence & liveness ---
    try:
        tank = state.get_tank(tank_id)
    except KeyError:
        return ActionResult(valid=False, reason=f"No tank with id {tank_id!r}")

    if not tank.alive:
        return ActionResult(valid=False, reason=f"Tank {tank_id!r} is not alive")

    # --- Turn check ---
    if state.current_tank_id != tank_id:
        return ActionResult(
            valid=False,
            reason=f"It is not tank {tank_id!r}'s turn (current: {state.current_tank_id!r})",
        )

    # --- Action-specific checks ---
    if action == Action.MOVE_FORWARD:
        dx, dy = tank.direction.forward_delta()
        dest = Position(tank.position.x + dx, tank.position.y + dy)

        if not state.game_map.in_bounds(dest.x, dest.y):
            return ActionResult(valid=False, reason="Destination is out of bounds")

        if state.game_map[dest.x, dest.y] == CellType.IMPASSABLE:
            return ActionResult(valid=False, reason="Destination cell is impassable")

        occupied = state.tank_positions
        if dest in occupied and occupied[dest] != tank_id:
            return ActionResult(
                valid=False, reason="Destination cell is occupied by a tank or wreckage"
            )

    return ActionResult(valid=True)


# ── Execution ─────────────────────────────────────────────────────────


def _advance_turn(state: GameState) -> GameState:
    """Return a new state with ``current_turn_index`` advanced to the next alive tank."""
    order_len = len(state.turn_order)
    alive_ids = {t.id for t in state.tanks if t.alive}

    next_idx = (state.current_turn_index + 1) % order_len
    # Walk forward to find the next alive tank.
    for _ in range(order_len):
        if state.turn_order[next_idx] in alive_ids:
            break
        next_idx = (next_idx + 1) % order_len

    return state.model_copy(update={"current_turn_index": next_idx})


def _replace_tank(tanks: list[Tank], updated: Tank) -> list[Tank]:
    """Return a new tank list with *updated* replacing the tank of the same ID."""
    return [updated if t.id == updated.id else t for t in tanks]


def apply_action(state: GameState, tank_id: TankId, action: Action) -> GameState:
    """Apply *action* for *tank_id* and return a new :class:`GameState`.

    **Move semantics:**

    * ``MOVE_FORWARD``: if the destination is valid the tank moves;
      otherwise the turn is silently lost (no error raised).
    * ``TURN_LEFT`` / ``TURN_RIGHT``: rotate the tank 90°.
    * ``FIRE``: check the single cell directly ahead.  If an alive tank
      occupies it, that tank is destroyed (friendly fire included).
      Firing into wreckage (a dead tank) has no additional effect.
    * ``PASS``: do nothing.

    After the action, ``current_turn_index`` advances to the next alive
    tank.

    Raises:
        KeyError: If *tank_id* does not exist.
        ValueError: If the tank is dead or it is not their turn.
    """
    tank = state.get_tank(tank_id)
    if not tank.alive:
        raise ValueError(f"Tank {tank_id!r} is not alive")
    if state.current_tank_id != tank_id:
        raise ValueError(f"It is not tank {tank_id!r}'s turn (current: {state.current_tank_id!r})")

    new_tanks = list(state.tanks)

    if action == Action.MOVE_FORWARD:
        result = validate_action(state, tank_id, action)
        if result.valid:
            dx, dy = tank.direction.forward_delta()
            moved = tank.model_copy(
                update={"position": Position(tank.position.x + dx, tank.position.y + dy)}
            )
            new_tanks = _replace_tank(new_tanks, moved)
        # else: turn lost silently

    elif action == Action.TURN_LEFT:
        turned = tank.model_copy(update={"direction": tank.direction.turn_left()})
        new_tanks = _replace_tank(new_tanks, turned)

    elif action == Action.TURN_RIGHT:
        turned = tank.model_copy(update={"direction": tank.direction.turn_right()})
        new_tanks = _replace_tank(new_tanks, turned)

    elif action == Action.FIRE:
        dx, dy = tank.direction.forward_delta()
        target_pos = Position(tank.position.x + dx, tank.position.y + dy)
        occupied = state.tank_positions
        if target_pos in occupied:
            target_id = occupied[target_pos]
            if target_id != tank_id:
                target_tank = state.get_tank(target_id)
                if target_tank.alive:
                    destroyed = target_tank.model_copy(update={"alive": False})
                    new_tanks = _replace_tank(new_tanks, destroyed)

    # PASS: nothing to do

    new_state = state.model_copy(update={"tanks": new_tanks})
    return _advance_turn(new_state)
