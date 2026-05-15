"""Horizontal team colour legend widget for HMLS TUI applications."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from hmls.uxcommon.styles import CELL_CHARS


class TeamLegend(Static):
    """Compact horizontal legend showing each team's colour swatch and name.

    Renders a single line such as ``██ Team A   ██ Team B`` where each
    swatch block is styled with the team's Rich style string.

    Args:
        team_styles: Mapping of team ID to Rich style string
            (e.g. ``{"A": "bold cyan", "B": "bold magenta"}``).
        id: Optional Textual widget ID.
    """

    DEFAULT_CSS = """
    TeamLegend {
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(
        self,
        team_styles: dict[str, str],
        *,
        id: str | None = None,  # noqa: A002 -- matches Textual Widget API
    ) -> None:
        super().__init__(id=id)
        self._team_styles = team_styles

    def on_mount(self) -> None:
        """Render the legend when the widget is first mounted."""
        self._render_legend()

    def _render_legend(self) -> None:
        """Build a Rich Text legend and update the widget content."""
        text = Text()
        for i, (team_id, style) in enumerate(self._team_styles.items()):
            if i > 0:
                text.append("   ")
            text.append(CELL_CHARS, style=style)
            text.append(f" Team {team_id}")
        self.update(text)
