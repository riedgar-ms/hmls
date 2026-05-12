"""Tests for RandomTankPlayer rule-based action selection."""

from __future__ import annotations

import random
from typing import Any

from hmls.core.map import CellType
from hmls.core.tank import Tank
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import (
    BoundaryCell,
    FogCell,
    PlayerView,
    TankInfo,
    TankPatch,
    VisibleCell,
)
from hmls.randomtank.model import RandomTankModel, RandomTankModelConfig
from hmls.randomtank.player import RandomTankPlayer

# ── Helpers ──────────────────────────────────────────────────────────


def _make_patch(
    front_cell: Any,
    patch_size: int = 9,
    tank_id: str = "A1",
) -> TankPatch:
    """Build a TankPatch with a specified cell directly in front.

    All cells are passable/visible except the one directly in front of
    the tank, which is set to ``front_cell``.
    """
    half = patch_size // 2
    grid: list[list[Any]] = []
    for r in range(patch_size):
        row: list[Any] = []
        for c in range(patch_size):
            if r == half and c == half:
                # The tank itself
                row.append(
                    VisibleCell(
                        cell_type=CellType.PASSABLE,
                        tank=Tank(
                            id=tank_id,
                            team="A",
                            position=Position(x=5, y=5),
                            direction=Direction.NORTH,
                        ),
                    )
                )
            elif r == half - 1 and c == half:
                # The cell directly in front
                row.append(front_cell)
            else:
                row.append(VisibleCell(cell_type=CellType.PASSABLE))
        grid.append(row)

    return TankPatch(
        tank_id=tank_id,
        position=Position(x=5, y=5),
        direction=Direction.NORTH,
        grid=grid,
    )


def _make_view(patch: TankPatch) -> PlayerView:
    """Wrap a patch into a PlayerView."""
    return PlayerView(
        patches=[patch],
        tanks=[
            TankInfo(
                tank_id=patch.tank_id,
                position=patch.position,
                direction=patch.direction,
                alive=True,
            )
        ],
    )


def _make_player(
    *,
    prob_forward: float = 0.7,
    prob_left_passable: float = 0.15,
    prob_left_blocked: float = 0.5,
    seed: int = 42,
) -> RandomTankPlayer:
    """Create a RandomTankPlayer with specified probabilities."""
    config = RandomTankModelConfig(
        prob_forward_on_passable=prob_forward,
        prob_turn_left_on_passable=prob_left_passable,
        prob_turn_left_on_blocked=prob_left_blocked,
    )
    model = RandomTankModel(config)
    rng = random.Random(seed)
    return RandomTankPlayer(team="A", model=model, mode="play", rng=rng)


# ── Tests ────────────────────────────────────────────────────────────


class TestFireRule:
    """Rule 1: alive enemy in front → FIRE."""

    def test_fire_on_alive_enemy(self) -> None:
        """Should fire when an alive enemy tank is directly ahead."""
        enemy = Tank(
            id="B1",
            team="B",
            position=Position(x=5, y=4),
            direction=Direction.SOUTH,
            alive=True,
        )
        front = VisibleCell(cell_type=CellType.PASSABLE, tank=enemy)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action == Action.FIRE

    def test_no_fire_on_dead_enemy(self) -> None:
        """Dead enemy wreckage should be treated as blocked, not fired at."""
        dead_enemy = Tank(
            id="B1",
            team="B",
            position=Position(x=5, y=4),
            direction=Direction.SOUTH,
            alive=False,
        )
        front = VisibleCell(cell_type=CellType.PASSABLE, tank=dead_enemy)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action in (Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_no_fire_on_alive_friendly(self) -> None:
        """An alive friendly tank should be treated as blocked, not fired at."""
        friendly = Tank(
            id="A2",
            team="A",
            position=Position(x=5, y=4),
            direction=Direction.NORTH,
            alive=True,
        )
        front = VisibleCell(cell_type=CellType.PASSABLE, tank=friendly)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action in (Action.TURN_LEFT, Action.TURN_RIGHT)


class TestBlockedRule:
    """Rule 2: blocked cell → turn left or right."""

    def test_boundary_cell_is_blocked(self) -> None:
        """A boundary cell should trigger a turn."""
        front = BoundaryCell()
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action in (Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_impassable_terrain_is_blocked(self) -> None:
        """An impassable terrain cell should trigger a turn."""
        front = VisibleCell(cell_type=CellType.IMPASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action in (Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_prob_left_on_blocked_all_left(self) -> None:
        """With prob_left_blocked=1.0, should always turn left."""
        front = BoundaryCell()
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_left_blocked=1.0)
        for _ in range(20):
            action = player.choose_action("A1", view)
            assert action == Action.TURN_LEFT

    def test_prob_left_on_blocked_all_right(self) -> None:
        """With prob_left_blocked=0.0, should always turn right."""
        front = BoundaryCell()
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_left_blocked=0.0)
        for _ in range(20):
            action = player.choose_action("A1", view)
            assert action == Action.TURN_RIGHT


class TestPassableRule:
    """Rule 3: passable cell → move forward, turn left, or turn right."""

    def test_passable_cell_actions(self) -> None:
        """Passable cell should produce forward, left, or right."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player()
        action = player.choose_action("A1", view)
        assert action in (Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_all_forward(self) -> None:
        """With prob_forward=1.0, should always move forward."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_forward=1.0, prob_left_passable=0.0)
        for _ in range(20):
            action = player.choose_action("A1", view)
            assert action == Action.MOVE_FORWARD

    def test_all_left(self) -> None:
        """With prob_forward=0.0 and prob_left_passable=1.0, should always turn left."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_forward=0.0, prob_left_passable=1.0)
        for _ in range(20):
            action = player.choose_action("A1", view)
            assert action == Action.TURN_LEFT

    def test_all_right(self) -> None:
        """With prob_forward=0.0 and prob_left_passable=0.0, should always turn right."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_forward=0.0, prob_left_passable=0.0)
        for _ in range(20):
            action = player.choose_action("A1", view)
            assert action == Action.TURN_RIGHT

    def test_distribution_over_many_trials(self) -> None:
        """Action distribution should roughly match configured probabilities."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        n_trials = 5000
        counts: dict[Action, int] = {
            Action.MOVE_FORWARD: 0,
            Action.TURN_LEFT: 0,
            Action.TURN_RIGHT: 0,
        }

        player = _make_player(
            prob_forward=0.5,
            prob_left_passable=0.3,
            seed=123,
        )
        for _ in range(n_trials):
            action = player.choose_action("A1", view)
            counts[action] += 1

        # Allow 5% tolerance
        assert abs(counts[Action.MOVE_FORWARD] / n_trials - 0.5) < 0.05
        assert abs(counts[Action.TURN_LEFT] / n_trials - 0.3) < 0.05
        assert abs(counts[Action.TURN_RIGHT] / n_trials - 0.2) < 0.05


class TestFogCell:
    """Fog cells should be treated as passable (defensive fallback)."""

    def test_fog_treated_as_passable(self) -> None:
        """A fog cell in front should trigger passable logic."""
        front = FogCell()
        patch = _make_patch(front)
        view = _make_view(patch)

        player = _make_player(prob_forward=1.0, prob_left_passable=0.0)
        action = player.choose_action("A1", view)
        assert action == Action.MOVE_FORWARD


class TestMissingPatch:
    """When no patch is found for the tank, PASS should be returned."""

    def test_no_patch_returns_pass(self) -> None:
        """If the tank's patch is missing, return PASS."""
        view = PlayerView(patches=[], tanks=[])
        player = _make_player()
        action = player.choose_action("A1", view)
        assert action == Action.PASS


class TestLearnMode:
    """In learn mode, dummy trajectory data should be recorded."""

    def test_learn_mode_records_trajectory(self) -> None:
        """Learn mode should add episode steps with dummy log_prob."""
        front = VisibleCell(cell_type=CellType.PASSABLE)
        patch = _make_patch(front)
        view = _make_view(patch)

        config = RandomTankModelConfig()
        model = RandomTankModel(config)
        player = RandomTankPlayer(team="A", model=model, mode="learn")

        player.choose_action("A1", view)
        assert len(player.episode) == 1
        assert player.episode.steps[0].log_prob == 0.0
        assert len(player.log_prob_tensors) == 1
        assert len(player.entropy_tensors) == 1
