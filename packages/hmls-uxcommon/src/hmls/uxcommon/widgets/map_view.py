"""God-view map renderer: shows the entire map with all tanks visible."""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from hmls.core.game_state import GameState
from hmls.core.map import CellType, GameMap
from hmls.core.tank import TankId
from hmls.core.types import Position
from hmls.uxcommon.styles import (
    ACTIVE_HIGHLIGHT_STYLE,
    CELL_CHARS,
    CELL_WIDTH,
    DEAD_MARKER,
    DEAD_STYLE,
    DIRECTION_ARROWS,
    IMPASSABLE_STYLE,
    PASSABLE_STYLE,
    TEAM_A_STYLE,
    TEAM_STYLES,
)


class MapView(Static):
    """Full god-view map renderer showing terrain and all tanks.

    The active tank is highlighted with a distinct style so the user
    can immediately see which tank they are controlling.
    """

    active_tank_id: reactive[str] = reactive("")
    """ID of the currently active tank (triggers re-render)."""

    def __init__(
        self,
        game_map: GameMap,
        state: GameState,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._game_map = game_map
        self._state = state

    def update_state(self, state: GameState) -> None:
        """Update the game state and re-render the map.

        Args:
            state: The new game state to display.
        """
        self._state = state
        self._render_map()

    def watch_active_tank_id(self, _old: str, _new: str) -> None:
        """Re-render when the active tank changes."""
        self._render_map()

    def on_mount(self) -> None:
        """Render the map when the widget is first mounted."""
        self._render_map()

    def _render_map(self) -> None:
        """Build a Rich Text representation of the map and update the widget."""
        game_map = self._game_map
        state = self._state

        # Build position → tank lookup.
        pos_to_tank: dict[Position, tuple[TankId, str, int, bool]] = {}
        for tank in state.tanks:
            pos_to_tank[tank.position] = (
                tank.id,
                tank.team,
                int(tank.direction),
                tank.alive,
            )

        text = Text()
        for y in range(game_map.height):
            for x in range(game_map.width):
                pos = Position(x, y)
                if pos in pos_to_tank:
                    tank_id, team, direction, alive = pos_to_tank[pos]
                    is_active = tank_id == self.active_tank_id

                    if not alive:
                        style = ACTIVE_HIGHLIGHT_STYLE if is_active else DEAD_STYLE
                        text.append(DEAD_MARKER, style=style)
                    else:
                        arrow = DIRECTION_ARROWS.get(direction, "? ")
                        if is_active:
                            style = ACTIVE_HIGHLIGHT_STYLE
                        else:
                            style = TEAM_STYLES.get(team, TEAM_A_STYLE)
                        text.append(arrow, style=style)
                elif game_map[x, y] == CellType.PASSABLE:
                    text.append(CELL_CHARS, style=PASSABLE_STYLE)
                else:
                    text.append(CELL_CHARS, style=IMPASSABLE_STYLE)
            text.append("\n")

        self.styles.min_width = game_map.width * CELL_WIDTH
        self.update(text)
