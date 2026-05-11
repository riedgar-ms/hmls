"""Tests for the TeamLegend widget."""

from __future__ import annotations

from textual.widget import Widget

from hmls.uxcommon.styles import TEAM_STYLES
from hmls.uxcommon.widgets.team_legend import TeamLegend


class TestTeamLegendImport:
    """TeamLegend should be importable and is a Widget subclass."""

    def test_is_widget_subclass(self) -> None:
        """TeamLegend inherits from Textual Widget."""
        assert issubclass(TeamLegend, Widget)


class TestTeamLegendConstruction:
    """TeamLegend should accept various team style mappings."""

    def test_construct_with_default_styles(self) -> None:
        """Can construct with the shared TEAM_STYLES mapping."""
        legend = TeamLegend(TEAM_STYLES)
        assert legend._team_styles == TEAM_STYLES

    def test_construct_with_single_team(self) -> None:
        """Can construct with a single team entry."""
        styles: dict[str, str] = {"X": "bold red"}
        legend = TeamLegend(styles)
        assert legend._team_styles == styles

    def test_construct_with_empty_styles(self) -> None:
        """Can construct with no teams (edge case)."""
        legend = TeamLegend({})
        assert legend._team_styles == {}

    def test_construct_with_custom_id(self) -> None:
        """Widget ID is forwarded correctly."""
        legend = TeamLegend(TEAM_STYLES, id="my-legend")
        assert legend.id == "my-legend"
