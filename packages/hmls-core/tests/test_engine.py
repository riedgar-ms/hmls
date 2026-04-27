"""Tests for the game engine."""

from __future__ import annotations

import pytest

from hmls.core.engine import (
    GameEngine,
    GameResult,
    HistoryEntry,
    _build_interleaved_turn_order,
)
from hmls.core.map import CellType, GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank, TankId
from hmls.core.types import Action, Direction, Position
from hmls.core.visibility import PlayerView

# ── Test helpers ──────────────────────────────────────────────────────


class PassPlayer(Player):
    """A player that always passes."""

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        return Action.PASS


class ScriptedPlayer(Player):
    """A player that follows a pre-scripted sequence of actions.

    Loops back to the start if the script is exhausted.
    """

    def __init__(self, team: str, actions: list[Action]) -> None:
        super().__init__(team)
        self._actions = actions
        self._index = 0
        self.invalid_notifications: list[tuple[TankId, Action, str]] = []

    def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        return action

    def notify_invalid_action(self, tank_id: TankId, action: Action, reason: str) -> None:
        self.invalid_notifications.append((tank_id, action, reason))


def _default_map(width: int = 9, height: int = 9) -> GameMap:
    return GameMap(width=width, height=height)


def _two_tank_setup() -> tuple[GameMap, list[Tank], dict[str, Player]]:
    """Create a simple 1v1 setup on a 9x9 map."""
    game_map = _default_map()
    tanks = [
        Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
        Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
    ]
    players: dict[str, Player] = {
        "alpha": PassPlayer("alpha"),
        "beta": PassPlayer("beta"),
    }
    return game_map, tanks, players


# ── Turn order ────────────────────────────────────────────────────────


class TestInterleavedTurnOrder:
    """Tests for _build_interleaved_turn_order."""

    def test_equal_teams(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(0, 0), direction=Direction.NORTH),
            Tank(id="a1", team="alpha", position=Position(1, 0), direction=Direction.NORTH),
            Tank(id="b0", team="beta", position=Position(0, 1), direction=Direction.SOUTH),
            Tank(id="b1", team="beta", position=Position(1, 1), direction=Direction.SOUTH),
        ]
        order = _build_interleaved_turn_order(tanks)
        assert order == ["a0", "b0", "a1", "b1"]

    def test_unequal_teams(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(0, 0), direction=Direction.NORTH),
            Tank(id="a1", team="alpha", position=Position(1, 0), direction=Direction.NORTH),
            Tank(id="a2", team="alpha", position=Position(2, 0), direction=Direction.NORTH),
            Tank(id="b0", team="beta", position=Position(0, 1), direction=Direction.SOUTH),
        ]
        order = _build_interleaved_turn_order(tanks)
        # alpha: [a0, a1, a2], beta: [b0]
        # Shorter team cycles: a0, b0, a1, b0, a2, b0
        assert order == ["a0", "b0", "a1", "b0", "a2", "b0"]

    def test_teams_sorted_alphabetically(self) -> None:
        tanks = [
            Tank(id="z0", team="zulu", position=Position(0, 0), direction=Direction.NORTH),
            Tank(id="a0", team="alpha", position=Position(1, 0), direction=Direction.SOUTH),
        ]
        order = _build_interleaved_turn_order(tanks)
        assert order == ["a0", "z0"]


# ── Engine validation ─────────────────────────────────────────────────


class TestEngineValidation:
    """Tests for GameEngine construction validation."""

    def test_valid_setup_succeeds(self) -> None:
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=10)
        assert engine is not None

    def test_rejects_single_team(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
        ]
        with pytest.raises(ValueError, match="2 teams"):
            GameEngine(_default_map(), tanks, {"alpha": PassPlayer("alpha")}, max_rounds=10)

    def test_rejects_three_teams(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
            Tank(id="c0", team="gamma", position=Position(4, 6), direction=Direction.NORTH),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
            "gamma": PassPlayer("gamma"),
        }
        with pytest.raises(ValueError, match="2 teams"):
            GameEngine(_default_map(), tanks, players, max_rounds=10)

    def test_rejects_duplicate_tank_ids(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
            Tank(id="a0", team="beta", position=Position(6, 4), direction=Direction.WEST),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        with pytest.raises(ValueError, match="unique"):
            GameEngine(_default_map(), tanks, players, max_rounds=10)

    def test_rejects_overlapping_positions(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(4, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        with pytest.raises(ValueError, match="position"):
            GameEngine(_default_map(), tanks, players, max_rounds=10)

    def test_rejects_out_of_bounds_tank(self) -> None:
        tanks = [
            Tank(id="a0", team="alpha", position=Position(99, 99), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(4, 4), direction=Direction.WEST),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        with pytest.raises(ValueError, match="out of bounds"):
            GameEngine(_default_map(), tanks, players, max_rounds=10)

    def test_rejects_even_patch_size(self) -> None:
        game_map, tanks, players = _two_tank_setup()
        with pytest.raises(ValueError, match="odd"):
            GameEngine(game_map, tanks, players, max_rounds=10, patch_size=6)

    def test_rejects_missing_player(self) -> None:
        game_map, tanks, _ = _two_tank_setup()
        with pytest.raises(ValueError, match="No player"):
            GameEngine(game_map, tanks, {"alpha": PassPlayer("alpha")}, max_rounds=10)

    def test_rejects_tank_on_impassable(self) -> None:
        game_map = _default_map()
        game_map[2, 4] = CellType.IMPASSABLE
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        with pytest.raises(ValueError, match="impassable"):
            GameEngine(game_map, tanks, players, max_rounds=10)


# ── Game execution ────────────────────────────────────────────────────


class TestGameExecution:
    """Tests for GameEngine.run()."""

    def test_pass_game_runs_to_max_rounds(self) -> None:
        """When both players always pass, the game runs to max_rounds."""
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=5)
        result = engine.run()
        assert result.rounds_played == 5
        assert result.winner is None  # both still alive, equal count → draw

    def test_history_entries_recorded(self) -> None:
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()
        # 2 tanks, 1 round → 2 history entries
        assert len(result.history) == 2
        for entry in result.history:
            assert isinstance(entry, HistoryEntry)
            assert entry.valid is True
            assert entry.requested_action == Action.PASS
            assert entry.applied_action == Action.PASS

    def test_interleaved_turn_order_in_history(self) -> None:
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=2)
        result = engine.run()
        tank_ids = [e.tank_id for e in result.history]
        # alpha first (alphabetical), then beta, repeated
        assert tank_ids == ["a0", "b0", "a0", "b0"]

    def test_early_termination_on_destruction(self) -> None:
        """Game stops early when one side is fully destroyed."""
        game_map = _default_map()
        tanks = [
            Tank(id="a0", team="alpha", position=Position(4, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(5, 4), direction=Direction.WEST),
        ]
        # alpha fires (hitting b0), beta never gets to act
        alpha_player = ScriptedPlayer("alpha", [Action.FIRE])
        beta_player = ScriptedPlayer("beta", [Action.PASS])
        players: dict[str, Player] = {"alpha": alpha_player, "beta": beta_player}

        engine = GameEngine(game_map, tanks, players, max_rounds=100)
        result = engine.run()

        assert result.winner == "alpha"
        assert result.rounds_played == 1
        # Only one action should have been taken (alpha fires, game ends)
        assert len(result.history) == 1
        assert result.history[0].tank_id == "a0"
        assert result.history[0].applied_action == Action.FIRE

    def test_invalid_action_triggers_notification(self) -> None:
        """Invalid actions cause notification and PASS substitution."""
        game_map = _default_map()
        # Place tank facing the wall at the edge
        tanks = [
            Tank(id="a0", team="alpha", position=Position(0, 0), direction=Direction.NORTH),
            Tank(id="b0", team="beta", position=Position(8, 8), direction=Direction.SOUTH),
        ]
        alpha_player = ScriptedPlayer("alpha", [Action.MOVE_FORWARD])
        beta_player = PassPlayer("beta")
        players: dict[str, Player] = {"alpha": alpha_player, "beta": beta_player}

        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()

        # alpha's move was invalid (out of bounds)
        assert result.history[0].valid is False
        assert result.history[0].requested_action == Action.MOVE_FORWARD
        assert result.history[0].applied_action == Action.PASS
        assert result.history[0].reason != ""

        # Player was notified
        assert len(alpha_player.invalid_notifications) == 1
        tid, action, reason = alpha_player.invalid_notifications[0]
        assert tid == "a0"
        assert action == Action.MOVE_FORWARD

    def test_dead_tanks_skipped(self) -> None:
        """Dead tanks don't get turns."""
        game_map = _default_map()
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
            Tank(id="a1", team="alpha", position=Position(2, 6), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
            Tank(id="b1", team="beta", position=Position(6, 6), direction=Direction.WEST),
        ]
        # Kill a0 by marking it dead from the start
        tanks[0] = tanks[0].model_copy(update={"alive": False})

        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()

        acted_ids = [e.tank_id for e in result.history]
        assert "a0" not in acted_ids
        # a1, b0, b1 should have acted
        assert "a1" in acted_ids
        assert "b0" in acted_ids
        assert "b1" in acted_ids

    def test_draw_with_equal_survivors(self) -> None:
        """Equal alive counts result in a draw."""
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()
        assert result.winner is None

    def test_winner_by_alive_count(self) -> None:
        """The team with more alive tanks wins at max_rounds."""
        game_map = _default_map()
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 2), direction=Direction.EAST),
            Tank(id="a1", team="alpha", position=Position(2, 6), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
        ]
        players: dict[str, Player] = {
            "alpha": PassPlayer("alpha"),
            "beta": PassPlayer("beta"),
        }
        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()
        assert result.winner == "alpha"  # 2 alive vs 1

    def test_game_result_serialisable(self) -> None:
        """GameResult must round-trip through JSON."""
        game_map, tanks, players = _two_tank_setup()
        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()
        json_str = result.model_dump_json()
        restored = GameResult.model_validate_json(json_str)
        assert restored.winner == result.winner
        assert restored.rounds_played == result.rounds_played
        assert len(restored.history) == len(result.history)

    def test_fog_of_war_enforced(self) -> None:
        """Players receive PlayerView, not raw GameState."""

        class ViewCapturingPlayer(Player):
            def __init__(self, team: str) -> None:
                super().__init__(team)
                self.views: list[PlayerView] = []

            def choose_action(self, tank_id: TankId, view: PlayerView) -> Action:
                self.views.append(view)
                return Action.PASS

        game_map = _default_map()
        tanks = [
            Tank(id="a0", team="alpha", position=Position(2, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(6, 4), direction=Direction.WEST),
        ]
        alpha = ViewCapturingPlayer("alpha")
        beta = ViewCapturingPlayer("beta")
        players: dict[str, Player] = {"alpha": alpha, "beta": beta}

        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        engine.run()

        # Alpha got a view with only their own tank info
        assert len(alpha.views) == 1
        view = alpha.views[0]
        assert all(t.tank_id.startswith("a") for t in view.tanks)
        assert len(view.patches) == 1
        assert view.patches[0].tank_id == "a0"

    def test_movement_updates_state(self) -> None:
        """A valid move should update the tank's position in subsequent states."""
        game_map = _default_map()
        tanks = [
            Tank(id="a0", team="alpha", position=Position(4, 4), direction=Direction.EAST),
            Tank(id="b0", team="beta", position=Position(0, 0), direction=Direction.SOUTH),
        ]
        alpha = ScriptedPlayer("alpha", [Action.MOVE_FORWARD])
        beta = PassPlayer("beta")
        players: dict[str, Player] = {"alpha": alpha, "beta": beta}

        engine = GameEngine(game_map, tanks, players, max_rounds=1)
        result = engine.run()

        # After alpha moves east, position should be (5, 4)
        final_a0 = result.final_state.get_tank("a0")
        assert final_a0.position == Position(5, 4)
