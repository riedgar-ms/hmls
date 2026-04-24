"""Tests for the map generation pipeline (generate_map and base classes)."""

import warnings

import pytest

from hmls.core import CellType, GameMap
from hmls.mapgenerator.connectivity import find_components
from hmls.mapgenerator.generators import generate_map
from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy


def _is_passable(gm: GameMap, x: int, y: int) -> bool:
    return gm[x, y] == CellType.PASSABLE


class TestGenerateMap:
    """Tests for the generate_map pipeline."""

    def test_basic_generation(self) -> None:
        """Generate a map and check dimensions."""
        gm = generate_map(20, 20, seed=42)
        assert gm.width == 20
        assert gm.height == 20

    def test_passable_always_connected(self) -> None:
        """The key invariant: passable terrain is always a single component."""
        for seed in range(10):
            gm = generate_map(20, 15, impassable_fraction=0.4, seed=seed)
            comps = find_components(gm, _is_passable)
            assert len(comps) == 1, f"Disconnected passable terrain with seed={seed}"

    def test_seed_reproducibility(self) -> None:
        """Same seed and strategy produce identical maps."""
        gm1 = generate_map(
            15,
            15,
            impassable_fraction=0.3,
            strategy=BlobAndLineStrategy(shape=0.7),
            seed=123,
        )
        gm2 = generate_map(
            15,
            15,
            impassable_fraction=0.3,
            strategy=BlobAndLineStrategy(shape=0.7),
            seed=123,
        )
        for pos in gm1.all_positions():
            assert gm1[pos] == gm2[pos], f"Mismatch at {pos}"

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different maps."""
        gm1 = generate_map(20, 20, seed=1)
        gm2 = generate_map(20, 20, seed=2)
        diffs = sum(1 for pos in gm1.all_positions() if gm1[pos] != gm2[pos])
        assert diffs > 0

    def test_fraction_approximate(self) -> None:
        """Impassable fraction should be roughly correct."""
        gm = generate_map(30, 30, impassable_fraction=0.3, seed=42)
        actual = gm.count_impassable() / gm.total_cells
        assert 0.1 <= actual <= 0.5, f"Actual fraction {actual} too far from 0.3"

    def test_fully_passable(self) -> None:
        """fraction=0.0 should produce an all-passable map."""
        gm = generate_map(10, 10, impassable_fraction=0.0, seed=42)
        assert gm.count_impassable() == 0

    def test_connected_obstacles(self) -> None:
        """connected_obstacles=True should produce fewer impassable components."""
        gm_connected = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            connected_obstacles=True,
            seed=42,
        )
        gm_disjoint = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            connected_obstacles=False,
            seed=42,
        )

        def is_imp(gm: GameMap, x: int, y: int) -> bool:
            return gm[x, y] == CellType.IMPASSABLE

        comps_connected = find_components(gm_connected, is_imp)
        comps_disjoint = find_components(gm_disjoint, is_imp)
        assert len(comps_connected) <= len(comps_disjoint)

    def test_small_grid(self) -> None:
        """Small grids should not crash."""
        gm = generate_map(3, 3, impassable_fraction=0.3, seed=42)
        assert gm.width == 3 and gm.height == 3

    def test_1x1_grid(self) -> None:
        """A 1×1 grid should not crash."""
        gm = generate_map(1, 1, seed=42)
        assert gm.total_cells == 1


class TestInvalidParameters:
    """Tests for parameter validation."""

    def test_invalid_fraction_low(self) -> None:
        """Negative fraction raises ValueError."""
        with pytest.raises(ValueError):
            generate_map(10, 10, impassable_fraction=-0.1)

    def test_invalid_fraction_high(self) -> None:
        """Fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            generate_map(10, 10, impassable_fraction=1.1)

    def test_shape_and_strategy_conflict(self) -> None:
        """Passing both shape and strategy raises TypeError."""
        with pytest.raises(TypeError):
            generate_map(10, 10, shape=0.5, strategy=BlobAndLineStrategy())


class TestDeprecatedShape:
    """Tests for the deprecated shape parameter."""

    def test_shape_param_emits_warning(self) -> None:
        """Using the deprecated shape parameter emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gm = generate_map(10, 10, shape=0.5, seed=42)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert gm.width == 10

    def test_shape_param_uses_blob_and_line(self) -> None:
        """Deprecated shape param produces same result as explicit strategy."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            gm1 = generate_map(15, 15, shape=0.7, seed=99)
        gm2 = generate_map(15, 15, strategy=BlobAndLineStrategy(shape=0.7), seed=99)
        for pos in gm1.all_positions():
            assert gm1[pos] == gm2[pos], f"Mismatch at {pos}"


class TestCustomStrategy:
    """Tests for plugging in custom strategies."""

    def test_custom_strategy_is_used(self) -> None:
        """A custom no-op strategy should produce an all-passable map."""

        class NoOpStrategy:
            """Strategy that places no obstacles."""

            def place_obstacles(
                self,
                game_map: GameMap,
                fraction: float,
                rng: object,
            ) -> None:
                pass

        gm = generate_map(10, 10, strategy=NoOpStrategy(), seed=42)
        assert gm.count_impassable() == 0
