"""Integration tests for ReplayViewerApp using Textual's pilot testing framework.

These tests mount the app headlessly and exercise actions, navigation,
and display updates via key presses.
"""

from __future__ import annotations

import pytest

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.results import GameResult, HistoryEntry
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.replayviewer.app import (
    ReplayViewerApp,
    _tank_log_id,
    _tank_panel_id,
)
from hmls.uxcommon.widgets.map_view import MapView

# ── Fixtures ──────────────────────────────────────────────────────────


def _two_tank_game_result(*, history_len: int = 5) -> GameResult:
    """Build a GameResult with two tanks and alternating actions."""
    tank_a = Tank(id="A1", team="A", position=Position(1, 1), direction=Direction.NORTH)
    tank_b = Tank(id="B1", team="B", position=Position(3, 3), direction=Direction.SOUTH)
    initial = GameState(tanks=[tank_a, tank_b], current_tank_id="A1")
    game_map = GameMap(width=5, height=5)

    tanks = [tank_a, tank_b]
    actions = [Action.MOVE_FORWARD, Action.FIRE, Action.PASS, Action.TURN_LEFT, Action.TURN_RIGHT]
    history: list[HistoryEntry] = []
    for i in range(history_len):
        acting = tanks[i % 2]
        history.append(
            HistoryEntry(
                tank_id=acting.id,
                requested_action=actions[i % len(actions)],
                applied_action=actions[i % len(actions)],
                valid=True,
                state_after=initial.model_copy(deep=True),
            )
        )

    return GameResult(
        winner="A",
        game_map=game_map,
        initial_state=initial,
        history=history,
        turns_played=history_len,
    )


def _make_app(history_len: int = 5) -> ReplayViewerApp:
    """Create a ReplayViewerApp for testing."""
    result = _two_tank_game_result(history_len=history_len)
    app = ReplayViewerApp(result)
    app.title = "Test Replay"
    return app


def _get_status_text(app: ReplayViewerApp) -> str:
    """Get the current status bar text content from the mounted app."""
    from textual.widgets import Static

    status = app.query_one("#status-bar", Static)
    content: str = status._Static__content  # type: ignore[attr-defined]
    return content


# ── Compose & Mount tests ─────────────────────────────────────────────


class TestPilotCompose:
    """Verify the app mounts successfully and has expected widgets."""

    @pytest.mark.asyncio
    async def test_app_mounts(self) -> None:
        """App should mount without errors."""
        app = _make_app()
        async with app.run_test() as pilot:
            assert pilot.app is app

    @pytest.mark.asyncio
    async def test_map_view_exists(self) -> None:
        """MapView widget should be present after mount."""
        app = _make_app()
        async with app.run_test():
            map_view = app.query_one("#map-view", MapView)
            assert map_view is not None

    @pytest.mark.asyncio
    async def test_status_bar_exists(self) -> None:
        """Status bar should be present and show initial state."""
        app = _make_app()
        async with app.run_test():
            text = _get_status_text(app)
            assert "Step 0/" in text
            assert "Paused" in text

    @pytest.mark.asyncio
    async def test_tank_log_panels_exist(self) -> None:
        """Per-tank log panels should be created for all tanks."""
        app = _make_app()
        async with app.run_test():
            from textual.widgets import RichLog

            for tid in ["A1", "B1"]:
                log = app.query_one(f"#{_tank_log_id(tid)}", RichLog)
                assert log is not None
                panel = app.query_one(f"#{_tank_panel_id(tid)}")
                assert panel is not None


# ── Navigation tests ──────────────────────────────────────────────────


class TestPilotNavigation:
    """Test navigation key bindings."""

    @pytest.mark.asyncio
    async def test_step_forward(self) -> None:
        """Right arrow steps forward."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            assert app._current_step == 0
            await pilot.press("right")
            assert app._current_step == 1

    @pytest.mark.asyncio
    async def test_step_back(self) -> None:
        """Left arrow steps back."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("right")
            assert app._current_step == 2
            await pilot.press("left")
            assert app._current_step == 1

    @pytest.mark.asyncio
    async def test_step_back_at_zero_stays(self) -> None:
        """Left arrow at step 0 stays at 0."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("left")
            assert app._current_step == 0

    @pytest.mark.asyncio
    async def test_step_forward_at_end_stays(self) -> None:
        """Right arrow at the last step stays put."""
        app = _make_app(history_len=3)
        async with app.run_test() as pilot:
            # Navigate to end
            for _ in range(10):
                await pilot.press("right")
            assert app._current_step == app._max_step

    @pytest.mark.asyncio
    async def test_jump_start(self) -> None:
        """Home key jumps to step 0."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("right")
            await pilot.press("home")
            assert app._current_step == 0

    @pytest.mark.asyncio
    async def test_jump_end(self) -> None:
        """End key jumps to max_step."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("end")
            assert app._current_step == app._max_step

    @pytest.mark.asyncio
    async def test_status_bar_updates_on_navigation(self) -> None:
        """Status bar reflects current step after navigation."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("right")
            text = _get_status_text(app)
            assert "Step 2/" in text


# ── Playback tests ────────────────────────────────────────────────────


class TestPilotPlayback:
    """Test playback control key bindings."""

    @pytest.mark.asyncio
    async def test_toggle_play_starts_playing(self) -> None:
        """Space toggles from paused to playing."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            assert app._playing is False
            await pilot.press("space")
            assert app._playing is True

    @pytest.mark.asyncio
    async def test_toggle_play_pauses(self) -> None:
        """Space again pauses playback."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("space")
            await pilot.press("space")
            assert app._playing is False

    @pytest.mark.asyncio
    async def test_speed_up(self) -> None:
        """Up arrow decreases delay."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            initial_delay = app._delay
            await pilot.press("up")
            assert app._delay < initial_delay

    @pytest.mark.asyncio
    async def test_slow_down(self) -> None:
        """Down arrow increases delay."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            initial_delay = app._delay
            await pilot.press("down")
            assert app._delay > initial_delay

    @pytest.mark.asyncio
    async def test_speed_up_status_updates(self) -> None:
        """Speed up changes the delay shown in status bar."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("up")
            text = _get_status_text(app)
            assert "0.4s" in text  # Default 0.5 - 0.1 = 0.4

    @pytest.mark.asyncio
    async def test_auto_advance(self) -> None:
        """Auto-play advances the step after a delay."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            app._delay = 0.1  # Fast for testing
            await pilot.press("space")  # Start playing
            await pilot.pause(0.35)  # Wait for a few ticks
            assert app._current_step > 0

    @pytest.mark.asyncio
    async def test_toggle_at_end_restarts(self) -> None:
        """Toggling play at the end restarts from beginning."""
        app = _make_app(history_len=3)
        async with app.run_test() as pilot:
            await pilot.press("end")  # Jump to end
            assert app._current_step == app._max_step
            await pilot.press("space")  # Toggle play → should restart
            assert app._current_step == 0
            assert app._playing is True

    @pytest.mark.asyncio
    async def test_navigation_stops_playback(self) -> None:
        """Stepping manually stops auto-play."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("space")  # Start playing
            assert app._playing is True
            await pilot.press("left")  # Manual step
            assert app._playing is False

    @pytest.mark.asyncio
    async def test_auto_play_stops_at_end(self) -> None:
        """Auto-play should stop when reaching the end."""
        app = _make_app(history_len=2)
        async with app.run_test() as pilot:
            app._delay = 0.05  # Very fast
            await pilot.press("space")  # Start playing
            await pilot.pause(0.5)  # Wait long enough to reach end
            assert app._current_step == app._max_step
            assert app._playing is False

    @pytest.mark.asyncio
    async def test_speed_up_while_playing(self) -> None:
        """Speed up while playing restarts timer with new delay."""
        app = _make_app(history_len=10)
        async with app.run_test() as pilot:
            await pilot.press("space")  # Start playing
            assert app._playing is True
            initial_delay = app._delay
            await pilot.press("up")  # Speed up
            assert app._delay < initial_delay
            assert app._playing is True  # Still playing

    @pytest.mark.asyncio
    async def test_slow_down_while_playing(self) -> None:
        """Slow down while playing restarts timer with new delay."""
        app = _make_app(history_len=10)
        async with app.run_test() as pilot:
            await pilot.press("space")  # Start playing
            assert app._playing is True
            initial_delay = app._delay
            await pilot.press("down")  # Slow down
            assert app._delay > initial_delay
            assert app._playing is True  # Still playing


# ── Log rebuild tests ─────────────────────────────────────────────────


class TestPilotLogRebuild:
    """Test that per-tank log content updates on navigation."""

    @pytest.mark.asyncio
    async def test_logs_empty_at_step_zero(self) -> None:
        """At step 0, all tank logs should be empty."""
        app = _make_app(history_len=5)
        async with app.run_test():
            from textual.widgets import RichLog

            log_a = app.query_one(f"#{_tank_log_id('A1')}", RichLog)
            # RichLog lines are internal; check lines list length
            assert len(log_a.lines) == 0

    @pytest.mark.asyncio
    async def test_logs_populated_after_step(self) -> None:
        """After stepping forward, the acting tank's log has content."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            from textual.widgets import RichLog

            await pilot.press("right")
            # First entry acts with A1 (index 0 % 2 = 0)
            log_a = app.query_one(f"#{_tank_log_id('A1')}", RichLog)
            assert len(log_a.lines) == 1

    @pytest.mark.asyncio
    async def test_active_tank_panel_highlighted(self) -> None:
        """The active tank's panel gets the 'active-tank' class."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("right")  # Step 1: A1 acts
            panel_a = app.query_one(f"#{_tank_panel_id('A1')}")
            panel_b = app.query_one(f"#{_tank_panel_id('B1')}")
            assert "active-tank" in panel_a.classes
            assert "active-tank" not in panel_b.classes

    @pytest.mark.asyncio
    async def test_active_tank_switches(self) -> None:
        """Active tank highlight switches between tanks."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            await pilot.press("right")
            await pilot.press("right")  # Step 2: B1 acts
            panel_a = app.query_one(f"#{_tank_panel_id('A1')}")
            panel_b = app.query_one(f"#{_tank_panel_id('B1')}")
            assert "active-tank" not in panel_a.classes
            assert "active-tank" in panel_b.classes

    @pytest.mark.asyncio
    async def test_logs_cleared_on_jump_start(self) -> None:
        """Jumping to start clears all logs."""
        app = _make_app(history_len=5)
        async with app.run_test() as pilot:
            from textual.widgets import RichLog

            await pilot.press("right")
            await pilot.press("right")
            await pilot.press("home")
            log_a = app.query_one(f"#{_tank_log_id('A1')}", RichLog)
            assert len(log_a.lines) == 0
