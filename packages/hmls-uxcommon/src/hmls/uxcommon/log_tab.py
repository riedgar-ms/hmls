"""Mixin providing a Log tab for Textual TUI applications.

The :class:`LogTabMixin` adds a ``TabbedContent`` structure with a
"Game" tab (for existing app content) and a "Log" tab that captures
all ``hmls.*`` logger output via :class:`~hmls.uxcommon.logging.TextualLogHandler`.
"""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.widgets import RichLog, TabPane

from hmls.uxcommon.logging import TextualLogHandler

# Logger namespace prefix to capture.
_HMLS_LOGGER_NAMESPACE = "hmls"


class LogTabMixin:
    """Mixin that sets up a Log tab handler for ``hmls.*`` loggers.

    Apps using this mixin should:

    1. Include a ``TabbedContent`` in their compose tree with a
       ``TabPane`` whose id is ``"log-tab"`` containing a ``RichLog``
       with id ``"internal-log"``.
    2. Call ``self._setup_log_tab()`` in their ``on_mount`` method.

    The mixin attaches a :class:`TextualLogHandler` to the ``hmls``
    namespace logger so all ``hmls.*`` log output appears in the Log tab.
    """

    _log_handler: TextualLogHandler | None = None

    def _compose_log_tab(self) -> ComposeResult:
        """Yield the Log TabPane for inclusion in a TabbedContent.

        Usage in compose::

            with TabbedContent(initial="game-tab"):
                with TabPane("Game", id="game-tab"):
                    yield from self._compose_game_content()
                yield from self._compose_log_tab()
        """
        with TabPane("Log", id="log-tab"):
            yield RichLog(id="internal-log", highlight=True, markup=True)

    def _setup_log_tab(self) -> None:
        """Attach the log handler to the ``hmls`` namespace logger.

        Call this in the app's ``on_mount`` method after the widget tree
        is composed.
        """
        log_widget = self.query_one("#internal-log", RichLog)  # type: ignore[attr-defined]
        handler = TextualLogHandler(log_widget)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
        )
        hmls_logger = logging.getLogger(_HMLS_LOGGER_NAMESPACE)
        hmls_logger.addHandler(handler)
        hmls_logger.setLevel(logging.DEBUG)
        self._log_handler = handler

    def _teardown_log_tab(self) -> None:
        """Remove the log handler on app shutdown to avoid dangling references."""
        if self._log_handler is not None:
            hmls_logger = logging.getLogger(_HMLS_LOGGER_NAMESPACE)
            hmls_logger.removeHandler(self._log_handler)
            self._log_handler = None
