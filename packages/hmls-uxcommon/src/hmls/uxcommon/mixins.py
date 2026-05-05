"""Mixin classes and helpers for common Textual app functionality."""

from __future__ import annotations

from textual.widgets import RichLog, Static


def format_turn_status(valid: bool, reason: str, hit: bool | None) -> str:
    """Return a Rich-markup string describing a turn's outcome.

    Args:
        valid: Whether the action was legal.
        reason: Explanation when the action is invalid (empty string if valid).
        hit: ``True`` if a fire action hit, ``False`` if it missed,
            ``None`` for non-fire actions.

    Returns:
        A Rich-markup fragment such as ``[bold green]HIT![/bold green]``.
    """
    if not valid:
        return f"[red]✗ ({reason})[/red]"
    elif hit is True:
        return "[bold green]HIT![/bold green]"
    elif hit is False:
        return "[dim]miss[/dim]"
    else:
        return "✓"


class LogStatusMixin:
    """Mixin providing ``_write_log`` and ``_update_status`` helpers.

    Intended for Textual ``App`` subclasses whose compose tree includes
    a ``RichLog`` with id ``#log-panel`` and a ``Static`` with id
    ``#status-bar``.
    """

    def _write_log(self, message: str) -> None:
        """Write a message to the log panel."""
        try:
            log_panel = self.query_one("#log-panel", RichLog)  # type: ignore[attr-defined]
            log_panel.write(message)
        except Exception:
            pass

    def _update_status(self, text: str) -> None:
        """Update the status bar."""
        try:
            status = self.query_one("#status-bar", Static)  # type: ignore[attr-defined]
            status.update(text)
        except Exception:
            pass

    def _log_turn_result(
        self,
        tank_id: str,
        action_value: str,
        valid: bool,
        reason: str,
        hit: bool | None,
    ) -> None:
        """Format and log a turn result to the log panel.

        Args:
            tank_id: Identifier of the tank that acted.
            action_value: The string value of the action (e.g. ``"fire"``).
            valid: Whether the action was legal.
            reason: Explanation if the action was invalid.
            hit: ``True`` if hit, ``False`` if miss, ``None`` for non-fire.
        """
        status = format_turn_status(valid, reason, hit)
        self._write_log(f"  {tank_id} → {action_value} — {status}")
