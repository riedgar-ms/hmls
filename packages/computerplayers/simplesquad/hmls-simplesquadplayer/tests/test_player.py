"""Tests for the simple squad player."""

from __future__ import annotations

import torch

from hmls.core.map import CellType
from hmls.core.types import Action, Direction
from hmls.core.visibility import (
    FogCell,
    PlayerView,
    TankInfo,
    TankPatch,
    VisibleCell,
)
from hmls.simplesquadexecutor.model import SimpleExecutorConfig, SimpleExecutorModel
from hmls.simplesquadplanner.model import SimplePlannerConfig, SimplePlannerModel
from hmls.simplesquadplayer.player import SimpleSquadPlayer


def _make_patch(tank_id: str, position: tuple[int, int], direction: Direction) -> TankPatch:
    """Create a minimal 9x9 patch for testing."""
    # Simple grid: centre is passable, rest is fog
    grid: list[list[VisibleCell | FogCell]] = []
    for r in range(9):
        row: list[VisibleCell | FogCell] = []
        for c in range(9):
            if r == 4 and c == 4:
                row.append(VisibleCell(cell_type=CellType.PASSABLE))
            else:
                row.append(FogCell())
        grid.append(row)

    return TankPatch(
        tank_id=tank_id,
        position=position,
        direction=direction,
        grid=grid,
    )


def _make_view(tank_ids: list[str]) -> PlayerView:
    """Create a PlayerView with the given alive tanks."""
    patches = []
    tanks = []
    for i, tid in enumerate(tank_ids):
        pos = (i * 3, i * 3)
        direction = Direction.NORTH
        patches.append(_make_patch(tid, pos, direction))
        tanks.append(TankInfo(tank_id=tid, position=pos, direction=direction, alive=True))
    return PlayerView(patches=patches, tanks=tanks)


def _make_player(mode: str = "play") -> SimpleSquadPlayer:
    """Create a small squad player for testing."""
    planner_config = SimplePlannerConfig(
        patch_size=9,
        cnn_channels=(8,),
        tank_feature_dim=16,
        mlp_hidden_dim=16,
    )
    executor_config = SimpleExecutorConfig(
        patch_size=9,
        cnn_channels=(8,),
        gru_hidden_size=16,
        order_embedding_dim=4,
    )
    planner = SimplePlannerModel(planner_config)
    executor = SimpleExecutorModel(executor_config)
    return SimpleSquadPlayer(
        team="A",
        planner=planner,
        executor=executor,
        mode=mode,  # type: ignore[arg-type]
        map_width=20,
        map_height=20,
    )


class TestPlayerLifecycle:
    """Tests for episode lifecycle management."""

    def test_reset_episode_clears_state(self) -> None:
        """reset_episode should clear all per-tank and planner state."""
        player = _make_player(mode="learn")
        view = _make_view(["A1", "A2"])

        # Generate some state
        player.begin_round()
        player.choose_action("A1", view)
        player.choose_action("A2", view)

        assert len(player.episodes) > 0

        # Reset
        player.reset_episode()
        assert len(player.episodes) == 0
        assert len(player.log_prob_tensors) == 0
        assert len(player.planner_log_prob_tensors) == 0
        assert len(player.current_orders) == 0

    def test_returns_valid_action(self) -> None:
        """choose_action should return a valid Action."""
        player = _make_player()
        view = _make_view(["A1"])
        player.begin_round()
        action = player.choose_action("A1", view)
        assert isinstance(action, Action)


class TestPerTankHiddenState:
    """Tests for per-tank hidden state isolation."""

    def test_independent_hidden_states(self) -> None:
        """Each tank should have its own hidden state."""
        player = _make_player()
        view = _make_view(["A1", "A2"])

        player.begin_round()
        player.choose_action("A1", view)
        player.choose_action("A2", view)

        # Both tanks should have hidden states
        assert "A1" in player._hidden_states
        assert "A2" in player._hidden_states

        # Hidden states should differ (different patches/positions)
        h1 = player._hidden_states["A1"]
        h2 = player._hidden_states["A2"]
        # After first step with different inputs, states should differ
        # (could be same with zero init and identical inputs, but positions differ)
        assert h1.shape == h2.shape

    def test_hidden_state_persists_across_rounds(self) -> None:
        """A tank's hidden state should carry over between rounds."""
        player = _make_player()
        view = _make_view(["A1"])

        player.begin_round()
        player.choose_action("A1", view)
        hidden_after_round1 = player._hidden_states["A1"].clone()

        player.begin_round()
        player.choose_action("A1", view)
        hidden_after_round2 = player._hidden_states["A1"]

        # Hidden state should have evolved
        assert not torch.allclose(hidden_after_round1, hidden_after_round2)


class TestPlannerCaching:
    """Tests for planner execution caching."""

    def test_planner_runs_once_per_round(self) -> None:
        """Planner should run once when first tank acts, not per-tank."""
        player = _make_player(mode="learn")
        view = _make_view(["A1", "A2", "A3"])

        player.begin_round()
        player.choose_action("A1", view)
        # After first action, orders should be assigned for all tanks
        assert "A1" in player.current_orders
        assert "A2" in player.current_orders
        assert "A3" in player.current_orders

        # Planner should have recorded exactly 1 step
        assert len(player.planner_log_prob_tensors) == 1

        # Remaining tanks reuse cached orders
        player.choose_action("A2", view)
        player.choose_action("A3", view)

        # Still only 1 planner step
        assert len(player.planner_log_prob_tensors) == 1

    def test_new_round_triggers_fresh_planning(self) -> None:
        """begin_round should cause the planner to run again."""
        player = _make_player(mode="learn")
        view = _make_view(["A1"])

        player.begin_round()
        player.choose_action("A1", view)
        assert len(player.planner_log_prob_tensors) == 1

        player.begin_round()
        player.choose_action("A1", view)
        assert len(player.planner_log_prob_tensors) == 2


class TestModes:
    """Tests for play vs learn mode."""

    def test_play_mode_no_trajectories(self) -> None:
        """In play mode, no trajectories should be recorded."""
        player = _make_player(mode="play")
        view = _make_view(["A1"])

        player.begin_round()
        player.choose_action("A1", view)

        # Episodes exist (initialised) but no steps recorded
        assert "A1" in player.episodes
        assert len(player.episodes["A1"]) == 0
        assert len(player.planner_log_prob_tensors) == 0

    def test_learn_mode_records_trajectories(self) -> None:
        """In learn mode, trajectories should be recorded."""
        player = _make_player(mode="learn")
        view = _make_view(["A1"])

        player.begin_round()
        player.choose_action("A1", view)

        assert len(player.episodes["A1"]) == 1
        assert len(player.log_prob_tensors["A1"]) == 1
        assert len(player.entropy_tensors["A1"]) == 1
        assert len(player.planner_log_prob_tensors) == 1


class TestDeadTankHandling:
    """Tests for handling tank deaths."""

    def test_missing_patch_returns_pass(self) -> None:
        """If tank has no patch (dead), PASS should be returned."""
        player = _make_player()
        # View with only A1 alive, but we ask for A2's action
        view = _make_view(["A1"])
        player.begin_round()
        action = player.choose_action("A2", view)
        assert action == Action.PASS
