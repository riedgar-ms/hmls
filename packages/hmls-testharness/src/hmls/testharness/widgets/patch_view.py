"""Single tank patch renderer: shows one tank's egocentric visibility."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from hmls.core.tank import TankId
from hmls.core.visibility import FogCell, TankPatch, VisibleCell
from hmls.testharness.styles import (
    CELL_CHARS,
    CELL_WIDTH,
    DEAD_MARKER,
    DEAD_STYLE,
    DIRECTION_ARROWS,
    FOG_STYLE,
    IMPASSABLE_STYLE,
    PASSABLE_STYLE,
    TEAM_A_STYLE,
    TEAM_B_STYLE,
)

_TEAM_STYLES: dict[str, str] = {
    "A": TEAM_A_STYLE,
    "B": TEAM_B_STYLE,
}

_ACTIVE_BORDER_STYLE = "bold yellow"
"""Textual CSS border colour for the active tank's patch."""


class PatchView(Static):
    """Renders a single tank's egocentric visibility patch.

    When this patch belongs to the currently active tank, a highlighted
    border is shown around the widget.

    Args:
        tank_id: ID of the tank this patch represents.
        patch: The visibility patch data.
        is_active: Whether this tank is the currently active one.
    """

    DEFAULT_CSS = """
    PatchView {
        border: solid $surface-lighten-2;
        padding: 0 1;
        margin: 0 1 0 0;
        width: auto;
        height: auto;
    }
    PatchView.active-patch {
        border: heavy $warning;
    }
    """

    def __init__(
        self,
        tank_id: TankId,
        patch: TankPatch,
        *,
        is_active: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._tank_id = tank_id
        self._patch = patch
        self._is_active = is_active

    def on_mount(self) -> None:
        """Render the patch when the widget is first mounted."""
        self._render_patch()

    def update_patch(self, patch: TankPatch, *, is_active: bool = False) -> None:
        """Update the patch data and re-render.

        Args:
            patch: New visibility patch data.
            is_active: Whether this is the active tank.
        """
        self._patch = patch
        self._is_active = is_active
        self._render_patch()

    def _render_patch(self) -> None:
        """Build a Rich Text representation of the patch."""
        patch = self._patch
        grid = patch.grid

        # Toggle active border class.
        if self._is_active:
            self.add_class("active-patch")
        else:
            self.remove_class("active-patch")

        text = Text()
        text.append(f" {self._tank_id} ", style="bold underline")
        text.append("\n")

        patch_size = len(grid)
        for row in grid:
            for cell in row:
                if isinstance(cell, FogCell):
                    text.append(CELL_CHARS, style=FOG_STYLE)
                elif isinstance(cell, VisibleCell):
                    if cell.tank is not None:
                        tank = cell.tank
                        if not tank.alive:
                            text.append(DEAD_MARKER, style=DEAD_STYLE)
                        else:
                            arrow = DIRECTION_ARROWS.get(int(tank.direction), "? ")
                            style = _TEAM_STYLES.get(tank.team, TEAM_A_STYLE)
                            text.append(arrow, style=style)
                    elif cell.cell_type.value == 1:  # PASSABLE
                        text.append(CELL_CHARS, style=PASSABLE_STYLE)
                    else:
                        text.append(CELL_CHARS, style=IMPASSABLE_STYLE)
            text.append("\n")

        self.styles.min_width = patch_size * CELL_WIDTH + 4  # +4 for padding + border
        self.update(text)
