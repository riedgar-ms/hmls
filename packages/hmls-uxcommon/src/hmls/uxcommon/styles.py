"""Shared style constants for HMLS TUI applications."""

from __future__ import annotations

# ── Team colours ──────────────────────────────────────────────────────

TEAM_A_STYLE = "bold cyan"
"""Rich style for Team A tanks."""

TEAM_B_STYLE = "bold magenta"
"""Rich style for Team B tanks."""

DEAD_STYLE = "dim red"
"""Rich style for destroyed tanks (wreckage)."""

ACTIVE_HIGHLIGHT_STYLE = "bold yellow on dark_green"
"""Rich style for the cell of the currently active tank."""

# ── Terrain ───────────────────────────────────────────────────────────

PASSABLE_STYLE = "green"
"""Rich style for passable terrain cells."""

IMPASSABLE_STYLE = "rgb(80,80,80)"
"""Rich style for impassable terrain cells."""

FOG_STYLE = "rgb(40,40,40)"
"""Rich style for fog-of-war cells."""

# ── Cell rendering ────────────────────────────────────────────────────

CELL_CHARS = "██"
"""Two-character block used to render a single map cell."""

CELL_WIDTH = 2
"""Number of characters per cell horizontally."""

# ── Direction arrows ──────────────────────────────────────────────────

DIRECTION_ARROWS: dict[int, str] = {
    0: "▲ ",  # NORTH
    1: "► ",  # EAST
    2: "▼ ",  # SOUTH
    3: "◄ ",  # WEST
}
"""Direction value → 2-char arrow string for rendering tanks."""

DEAD_MARKER = "✕ "
"""2-char marker for destroyed tanks."""
