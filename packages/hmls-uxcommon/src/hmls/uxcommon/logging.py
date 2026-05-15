"""Logging utilities for Textual TUI applications.

Provides :class:`TextualLogHandler`, a custom :class:`logging.Handler`
that routes Python log records to a Textual ``RichLog`` widget with
Rich-markup level colouring.
"""

from __future__ import annotations

import logging

from textual.widgets import RichLog


class TextualLogHandler(logging.Handler):
    """A :class:`logging.Handler` that writes formatted records to a Textual ``RichLog``.

    Records are formatted with Rich markup for level colouring:
    - DEBUG: dim
    - WARNING: yellow
    - ERROR/CRITICAL: red bold

    Args:
        widget: The ``RichLog`` widget to write log records to.
    """

    _LEVEL_STYLES: dict[int, tuple[str, str]] = {
        logging.DEBUG: ("[dim]", "[/dim]"),
        logging.INFO: ("", ""),
        logging.WARNING: ("[yellow]", "[/yellow]"),
        logging.ERROR: ("[red bold]", "[/red bold]"),
        logging.CRITICAL: ("[red bold]", "[/red bold]"),
    }

    def __init__(self, widget: RichLog) -> None:
        super().__init__()
        self._widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        """Format and write a log record to the RichLog widget.

        Args:
            record: The log record to emit.
        """
        try:
            msg = self.format(record)
            open_tag, close_tag = self._LEVEL_STYLES.get(record.levelno, ("", ""))
            self._widget.write(f"{open_tag}{msg}{close_tag}")
        except Exception:  # noqa: BLE001
            self.handleError(record)
