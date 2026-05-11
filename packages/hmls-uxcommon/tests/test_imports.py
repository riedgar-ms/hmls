"""Smoke tests: verify that all uxcommon modules and public symbols import."""

from __future__ import annotations

from textual.widget import Widget


class TestPackageImport:
    """Importing the top-level package should succeed."""

    def test_import_package(self) -> None:
        """The hmls.uxcommon package is importable."""
        import hmls.uxcommon  # noqa: F401


class TestStylesImport:
    """All public names in styles should be importable."""

    def test_import_styles_module(self) -> None:
        """The styles module is importable."""
        from hmls.uxcommon import styles  # noqa: F401

    def test_import_style_constants(self) -> None:
        """Every documented style constant is importable."""
        from hmls.uxcommon.styles import (  # noqa: F401
            ACTIVE_DEAD_STYLE,
            ACTIVE_HIGHLIGHT_BG,
            ACTIVE_HIGHLIGHT_STYLE,
            ACTIVE_TEAM_STYLES,
            BOUNDARY_STYLE,
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
            TEAM_STYLES,
        )


class TestWidgetImports:
    """All widget classes should be importable from the widgets sub-package."""

    def test_import_widgets_package(self) -> None:
        """The widgets sub-package is importable."""
        from hmls.uxcommon import widgets  # noqa: F401

    def test_import_map_view(self) -> None:
        """MapView is importable and is a Textual Widget subclass."""
        from hmls.uxcommon.widgets import MapView

        assert issubclass(MapView, Widget)

    def test_import_patch_view(self) -> None:
        """PatchView is importable and is a Textual Widget subclass."""
        from hmls.uxcommon.widgets import PatchView

        assert issubclass(PatchView, Widget)

    def test_import_player_view_region(self) -> None:
        """PlayerViewRegion is importable and is a Textual Widget subclass."""
        from hmls.uxcommon.widgets import PlayerViewRegion

        assert issubclass(PlayerViewRegion, Widget)

    def test_import_team_legend(self) -> None:
        """TeamLegend is importable and is a Textual Widget subclass."""
        from hmls.uxcommon.widgets import TeamLegend

        assert issubclass(TeamLegend, Widget)
