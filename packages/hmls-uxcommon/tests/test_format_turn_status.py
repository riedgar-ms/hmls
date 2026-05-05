"""Unit tests for format_turn_status and LogStatusMixin._log_turn_result."""

from __future__ import annotations

import pytest

from hmls.uxcommon.mixins import format_turn_status


class TestFormatTurnStatus:
    """Tests for the format_turn_status helper function."""

    def test_invalid_action(self) -> None:
        """Invalid action returns red status with reason."""
        result = format_turn_status(valid=False, reason="blocked by wall", hit=None)
        assert "✗" in result
        assert "blocked by wall" in result
        assert "[red]" in result

    def test_hit(self) -> None:
        """Hit action returns bold green HIT! status."""
        result = format_turn_status(valid=True, reason="", hit=True)
        assert "HIT!" in result
        assert "[bold green]" in result

    def test_miss(self) -> None:
        """Miss action returns dim miss status."""
        result = format_turn_status(valid=True, reason="", hit=False)
        assert "miss" in result
        assert "[dim]" in result

    def test_valid_non_fire(self) -> None:
        """Valid non-fire action returns checkmark."""
        result = format_turn_status(valid=True, reason="", hit=None)
        assert result == "✓"

    @pytest.mark.parametrize(
        "reason",
        ["cannot move into wall", "tank is dead", "already fired"],
    )
    def test_invalid_reason_included(self, reason: str) -> None:
        """The reason text is included in the invalid status string."""
        result = format_turn_status(valid=False, reason=reason, hit=None)
        assert reason in result
