"""Mixin classes for common Textual app functionality."""

from __future__ import annotations

from textual.widgets import RichLog, Static


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
