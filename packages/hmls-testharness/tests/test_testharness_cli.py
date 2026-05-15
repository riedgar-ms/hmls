"""Tests for the CLI module: argument parsing, map loading, and tank placement."""

from __future__ import annotations

from pathlib import Path

import pytest

from hmls.core.map import GameMap, MapLoadError
from hmls.core.placement import InsufficientPassableCellsError
from hmls.testharness.cli import build_initial_state, load_map, parse_args, place_tanks

# ── Helpers ───────────────────────────────────────────────────────────


def _small_map(width: int = 5, height: int = 5) -> GameMap:
    """Return a small fully-passable map."""
    return GameMap(width=width, height=height)


# ── parse_args ────────────────────────────────────────────────────────


class TestParseArgs:
    """Tests for ``parse_args``."""

    def test_valid_args(self) -> None:
        """Positional arguments are parsed correctly."""
        ns = parse_args(["map.json", "3"])
        assert ns.map_file == Path("map.json")
        assert ns.tanks_per_player == 3

    def test_defaults(self) -> None:
        """Optional arguments have sensible defaults."""
        ns = parse_args(["map.json", "2"])
        assert ns.patch_size == 9
        assert ns.max_turns == 200
        assert ns.seed is None

    def test_missing_args_exits(self) -> None:
        """Missing required arguments should cause SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


# ── load_map ──────────────────────────────────────────────────────────


class TestLoadMap:
    """Tests for ``load_map``."""

    def test_loads_valid_map(self, tmp_path: Path) -> None:
        """A valid GameMap JSON file loads successfully."""
        gm = _small_map()
        path = tmp_path / "map.json"
        path.write_text(gm.model_dump_json(), encoding="utf-8")

        loaded = load_map(path)
        assert loaded.width == gm.width
        assert loaded.height == gm.height

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_map(tmp_path / "no-such-file.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Invalid JSON content should raise MapLoadError."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid", encoding="utf-8")
        with pytest.raises(MapLoadError):
            load_map(path)


# ── place_tanks ───────────────────────────────────────────────────────


class TestPlaceTanks:
    """Tests for ``place_tanks``."""

    def test_correct_count(self) -> None:
        """Two teams × N tanks_per_player = 2N tanks total."""
        tanks = place_tanks(_small_map(), 3, seed=42)
        assert len(tanks) == 6

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical placement."""
        gm = _small_map()
        first = place_tanks(gm, 2, seed=99)
        second = place_tanks(gm, 2, seed=99)
        assert [(t.id, t.position, t.direction) for t in first] == [
            (t.id, t.position, t.direction) for t in second
        ]

    def test_unique_positions(self) -> None:
        """Every tank should occupy a distinct position."""
        tanks = place_tanks(_small_map(), 3, seed=7)
        positions = [t.position for t in tanks]
        assert len(positions) == len(set(positions))

    def test_insufficient_cells_raises(self) -> None:
        """A 1×1 map cannot hold 2 tanks → InsufficientPassableCellsError."""
        tiny = GameMap(width=1, height=1)
        with pytest.raises(InsufficientPassableCellsError):
            place_tanks(tiny, 1)


# ── build_initial_state ──────────────────────────────────────────────


class TestBuildInitialState:
    """Tests for ``build_initial_state``."""

    def test_state_has_all_tanks(self) -> None:
        """The game state should contain every tank passed in."""
        tanks = place_tanks(_small_map(), 2, seed=1)
        state = build_initial_state(tanks)
        assert len(state.tanks) == len(tanks)

    def test_current_tank_is_first(self) -> None:
        """``current_tank_id`` should be the first tank's id."""
        tanks = place_tanks(_small_map(), 2, seed=1)
        state = build_initial_state(tanks)
        assert state.current_tank_id == tanks[0].id
