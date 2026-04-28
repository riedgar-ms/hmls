"""Sanity checks for shared style constants in hmls.uxcommon.styles."""

from __future__ import annotations

from hmls.uxcommon.styles import (
    CELL_CHARS,
    CELL_WIDTH,
    DEAD_MARKER,
    DIRECTION_ARROWS,
)


class TestCellDimensions:
    """CELL_CHARS and CELL_WIDTH must be consistent."""

    def test_cell_chars_length_matches_width(self) -> None:
        """CELL_CHARS should have exactly CELL_WIDTH characters."""
        assert len(CELL_CHARS) == CELL_WIDTH

    def test_cell_width_positive(self) -> None:
        """CELL_WIDTH must be a positive integer."""
        assert CELL_WIDTH > 0


class TestDirectionArrows:
    """DIRECTION_ARROWS should cover all four cardinal directions."""

    def test_exactly_four_directions(self) -> None:
        """There must be exactly four direction entries (0–3)."""
        assert set(DIRECTION_ARROWS.keys()) == {0, 1, 2, 3}

    def test_arrow_widths_match_cell_width(self) -> None:
        """Each arrow string must be CELL_WIDTH characters wide."""
        for direction, arrow in DIRECTION_ARROWS.items():
            assert len(arrow) == CELL_WIDTH, (
                f"Direction {direction} arrow {arrow!r} has length {len(arrow)}, "
                f"expected {CELL_WIDTH}"
            )


class TestDeadMarker:
    """DEAD_MARKER must match cell width for alignment."""

    def test_dead_marker_width(self) -> None:
        """DEAD_MARKER should be CELL_WIDTH characters wide."""
        assert len(DEAD_MARKER) == CELL_WIDTH
