"""Tests for the BlobAndLineStrategy."""

import pytest

from hmls.core import CellType, GameMap
from hmls.mapgenerator.connectivity import find_components
from hmls.mapgenerator.generators import StrategyParam, generate_map
from hmls.mapgenerator.generators.blob_and_line import BlobAndLineStrategy


def _is_passable(gm: GameMap, x: int, y: int) -> bool:
    return gm[x, y] == CellType.PASSABLE


class TestBlobAndLineStrategy:
    """Tests for the BlobAndLineStrategy obstacle placement."""

    def test_shape_zero_linear(self) -> None:
        """shape=0.0 should produce valid, connected maps."""
        gm = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=BlobAndLineStrategy(shape=0.0),
            seed=42,
        )
        comps = find_components(gm, _is_passable)
        assert len(comps) <= 1 or gm.count_passable() == 0

    def test_shape_one_circular(self) -> None:
        """shape=1.0 should produce valid, connected maps."""
        gm = generate_map(
            20,
            20,
            impassable_fraction=0.3,
            strategy=BlobAndLineStrategy(shape=1.0),
            seed=42,
        )
        comps = find_components(gm, _is_passable)
        assert len(comps) <= 1 or gm.count_passable() == 0

    def test_invalid_shape(self) -> None:
        """Invalid shape values raise ValueError."""
        with pytest.raises(ValueError, match=r"shape must be 0\.0–1\.0"):
            BlobAndLineStrategy(shape=-0.1)
        with pytest.raises(ValueError, match=r"shape must be 0\.0–1\.0"):
            BlobAndLineStrategy(shape=1.1)


class TestStrategyParam:
    """Tests for BlobAndLineStrategy parameter metadata."""

    def test_blob_and_line_has_params(self) -> None:
        """BlobAndLineStrategy declares its configurable parameters."""
        assert hasattr(BlobAndLineStrategy, "get_params")
        assert len(BlobAndLineStrategy.get_params()) == 1
        param = BlobAndLineStrategy.get_params()[0]
        assert isinstance(param, StrategyParam)
        assert param.name == "shape"
        assert param.param_type is float
