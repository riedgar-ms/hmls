"""Tests for the Perlin noise map generation strategy."""

import pytest

from hmls.core import CellType, GameMap
from hmls.mapgenerator.connectivity import find_components
from hmls.mapgenerator.generators import STRATEGY_REGISTRY, StrategyParam, generate_map
from hmls.mapgenerator.generators.perlin import PerlinNoiseStrategy


def _is_passable(gm: GameMap, x: int, y: int) -> bool:
    return gm[x, y] == CellType.PASSABLE


class TestPerlinNoiseStrategy:
    """Tests for PerlinNoiseStrategy obstacle placement."""

    def test_basic_generation(self) -> None:
        """Perlin strategy should produce a valid grid without crashing."""
        strategy = PerlinNoiseStrategy()
        gm = generate_map(20, 20, strategy=strategy, seed=42)
        assert gm.width == 20
        assert gm.height == 20

    def test_passable_always_connected(self) -> None:
        """Passable terrain must be a single connected component."""
        for seed in range(10):
            gm = generate_map(
                20,
                15,
                impassable_fraction=0.4,
                strategy=PerlinNoiseStrategy(),
                seed=seed,
            )
            comps = find_components(gm, _is_passable)
            assert len(comps) == 1, f"Disconnected passable terrain with seed={seed}"

    def test_seed_reproducibility(self) -> None:
        """Same seed and parameters should produce identical maps."""
        kwargs: dict[str, object] = {
            "impassable_fraction": 0.3,
            "strategy": PerlinNoiseStrategy(scale=0.08, octaves=3),
            "seed": 123,
        }
        gm1 = generate_map(15, 15, **kwargs)  # type: ignore[arg-type]
        gm2 = generate_map(15, 15, **kwargs)  # type: ignore[arg-type]
        for pos in gm1.all_positions():
            assert gm1[pos] == gm2[pos], f"Mismatch at {pos}"

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different maps."""
        gm1 = generate_map(20, 20, strategy=PerlinNoiseStrategy(), seed=1)
        gm2 = generate_map(20, 20, strategy=PerlinNoiseStrategy(), seed=2)
        diffs = sum(1 for pos in gm1.all_positions() if gm1[pos] != gm2[pos])
        assert diffs > 0

    def test_fraction_approximate(self) -> None:
        """Impassable fraction should be close to the target."""
        gm = generate_map(
            30,
            30,
            impassable_fraction=0.35,
            strategy=PerlinNoiseStrategy(),
            seed=42,
        )
        actual = gm.count_impassable() / gm.total_cells
        assert 0.15 <= actual <= 0.55, f"Actual fraction {actual} too far from 0.35"

    def test_fully_passable(self) -> None:
        """fraction=0.0 should leave the grid all-passable."""
        gm = generate_map(
            10,
            10,
            impassable_fraction=0.0,
            strategy=PerlinNoiseStrategy(),
            seed=42,
        )
        assert gm.count_impassable() == 0

    def test_small_grid(self) -> None:
        """Small grids should not crash."""
        gm = generate_map(
            3,
            3,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(),
            seed=42,
        )
        assert gm.width == 3
        assert gm.height == 3

    def test_1x1_grid(self) -> None:
        """A 1×1 grid should not crash."""
        gm = generate_map(1, 1, strategy=PerlinNoiseStrategy(), seed=42)
        assert gm.total_cells == 1

    def test_different_scale_produces_different_maps(self) -> None:
        """Different scale values should produce different terrain."""
        gm1 = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(scale=0.03),
            seed=42,
        )
        gm2 = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(scale=0.15),
            seed=42,
        )
        diffs = sum(1 for pos in gm1.all_positions() if gm1[pos] != gm2[pos])
        assert diffs > 0

    def test_different_octaves_produces_different_maps(self) -> None:
        """Different octave counts should produce different terrain."""
        gm1 = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(octaves=1),
            seed=42,
        )
        gm2 = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(octaves=6),
            seed=42,
        )
        diffs = sum(1 for pos in gm1.all_positions() if gm1[pos] != gm2[pos])
        assert diffs > 0

    def test_connected_obstacles_with_perlin(self) -> None:
        """connected_obstacles should work with Perlin strategy."""
        gm = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=PerlinNoiseStrategy(),
            connected_obstacles=True,
            seed=42,
        )
        comps = find_components(gm, _is_passable)
        assert len(comps) == 1


class TestPerlinStrategyValidation:
    """Tests for constructor parameter validation."""

    def test_invalid_scale_zero(self) -> None:
        """Scale of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"scale must be positive"):
            PerlinNoiseStrategy(scale=0.0)

    def test_invalid_scale_negative(self) -> None:
        """Negative scale raises ValueError."""
        with pytest.raises(ValueError, match=r"scale must be positive"):
            PerlinNoiseStrategy(scale=-0.1)

    def test_invalid_octaves_zero(self) -> None:
        """Zero octaves raises ValueError."""
        with pytest.raises(ValueError, match=r"octaves must be >= 1"):
            PerlinNoiseStrategy(octaves=0)

    def test_invalid_octaves_negative(self) -> None:
        """Negative octaves raises ValueError."""
        with pytest.raises(ValueError, match=r"octaves must be >= 1"):
            PerlinNoiseStrategy(octaves=-1)


class TestPerlinStrategyParams:
    """Tests for strategy parameter metadata."""

    def test_has_params(self) -> None:
        """PerlinNoiseStrategy declares its configurable parameters."""
        assert hasattr(PerlinNoiseStrategy, "get_params")
        assert len(PerlinNoiseStrategy.get_params()) == 2

    def test_param_types(self) -> None:
        """All params should be StrategyParam instances."""
        for param in PerlinNoiseStrategy.get_params():
            assert isinstance(param, StrategyParam)

    def test_scale_param(self) -> None:
        """Scale parameter has correct metadata."""
        param = PerlinNoiseStrategy.get_params()[0]
        assert param.name == "scale"
        assert param.param_type is float
        assert param.default == 0.05

    def test_octaves_param(self) -> None:
        """Octaves parameter has correct metadata."""
        param = PerlinNoiseStrategy.get_params()[1]
        assert param.name == "octaves"
        assert param.param_type is int
        assert param.default == 4

    def test_registered_in_strategy_registry(self) -> None:
        """Perlin strategy should be registered for TUI discovery."""
        assert "Perlin Noise" in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY["Perlin Noise"] is PerlinNoiseStrategy
