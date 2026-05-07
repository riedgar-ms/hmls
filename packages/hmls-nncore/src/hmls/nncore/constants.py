"""Shared constants for NN-based tank players.

Defines the canonical mapping between integer action indices and
:class:`~hmls.core.types.Action` enum values.  All modules that need
to convert between action indices and enum values should import from
here to ensure consistency.
"""

from __future__ import annotations

from hmls.core.types import Action

#: Ordered mapping from action index to Action enum (stable ordering).
ACTION_INDEX_TO_ACTION: list[Action] = [
    Action.MOVE_FORWARD,
    Action.TURN_LEFT,
    Action.TURN_RIGHT,
    Action.FIRE,
    Action.PASS,
]

#: Reverse mapping from Action enum to integer index.
ACTION_TO_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ACTION_INDEX_TO_ACTION)}

#: The number of discrete actions available to a tank (derived from the action mapping).
NUM_ACTIONS: int = len(ACTION_INDEX_TO_ACTION)
