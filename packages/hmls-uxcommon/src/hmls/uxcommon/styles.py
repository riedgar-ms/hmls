"""Shared style constants for HMLS TUI applications."""

from __future__ import annotations

# ── Team colours ──────────────────────────────────────────────────────

TEAM_A_STYLE = "bold cyan"
"""Rich style for Team A tanks."""

TEAM_B_STYLE = "bold magenta"
"""Rich style for Team B tanks."""

DEAD_STYLE = "dim red"
"""Rich style for destroyed tanks (wreckage)."""

ACTIVE_HIGHLIGHT_BG = "on dark_green"
"""Background portion of the active-tank highlight."""

ACTIVE_TEAM_STYLES: dict[str, str] = {
    "A": f"bold cyan {ACTIVE_HIGHLIGHT_BG}",
    "B": f"bold magenta {ACTIVE_HIGHLIGHT_BG}",
}
"""Per-team Rich styles for the active tank (team foreground + highlight background)."""

ACTIVE_DEAD_STYLE = f"dim red {ACTIVE_HIGHLIGHT_BG}"
"""Rich style for a destroyed but currently-active tank."""

# Kept for backward compatibility; prefer ACTIVE_TEAM_STYLES for new code.
ACTIVE_HIGHLIGHT_STYLE = f"bold yellow {ACTIVE_HIGHLIGHT_BG}"
"""Rich style for the cell of the currently active tank (legacy)."""

# ── Terrain ───────────────────────────────────────────────────────────

PASSABLE_STYLE = "green"
"""Rich style for passable terrain cells."""

IMPASSABLE_STYLE = "rgb(80,80,80)"
"""Rich style for impassable terrain cells."""

FOG_STYLE = "rgb(40,40,40)"
"""Rich style for fog-of-war cells."""

BOUNDARY_STYLE = "rgb(120,60,60)"
"""Rich style for boundary cells (outside the map edge)."""

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

# ── Team style mapping ────────────────────────────────────────────────

TEAM_STYLES: dict[str, str] = {
    "A": TEAM_A_STYLE,
    "B": TEAM_B_STYLE,
}
"""Mapping of team ID → Rich style string for rendering team-coloured elements."""
