"""Tests for strategy Pydantic config models and discriminated union."""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from hmls.core import CellType, GameMap
from hmls.mapgenerator.connectivity import find_components
from hmls.mapgenerator.generators import (
    StrategyConfig,
    StrategyParam,
    generate_map_from_config,
)
from hmls.mapgenerator.generators.blob_and_line import (
    BlobAndLineConfig,
    BlobAndLineStrategy,
)
from hmls.mapgenerator.generators.perlin import (
    PerlinNoiseConfig,
    PerlinNoiseStrategy,
)

# TypeAdapter for parsing the discriminated union from JSON.
_strategy_adapter: TypeAdapter[StrategyConfig] = TypeAdapter(StrategyConfig)  # type: ignore[type-var]


# ── BlobAndLineConfig ─────────────────────────────────────────────────


class TestBlobAndLineConfig:
    """Tests for BlobAndLineConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default config uses expected values."""
        cfg = BlobAndLineConfig()
        assert cfg.type == "blob_and_line"
        assert cfg.shape == 0.5

    def test_custom_shape(self) -> None:
        """Custom shape value is accepted."""
        cfg = BlobAndLineConfig(shape=0.8)
        assert cfg.shape == 0.8

    def test_shape_at_bounds(self) -> None:
        """Boundary values 0.0 and 1.0 are accepted."""
        assert BlobAndLineConfig(shape=0.0).shape == 0.0
        assert BlobAndLineConfig(shape=1.0).shape == 1.0

    def test_shape_out_of_range_low(self) -> None:
        """Negative shape raises ValidationError."""
        with pytest.raises(ValidationError):
            BlobAndLineConfig(shape=-0.1)

    def test_shape_out_of_range_high(self) -> None:
        """Shape > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            BlobAndLineConfig(shape=1.1)

    def test_create_strategy(self) -> None:
        """create_strategy() returns a BlobAndLineStrategy with matching shape."""
        cfg = BlobAndLineConfig(shape=0.7)
        strategy = cfg.create_strategy()
        assert isinstance(strategy, BlobAndLineStrategy)
        assert strategy.shape == 0.7

    def test_frozen(self) -> None:
        """Config is immutable after creation."""
        cfg = BlobAndLineConfig()
        with pytest.raises(ValidationError):
            cfg.shape = 0.9  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            BlobAndLineConfig(shape=0.5, unknown_param=42)  # type: ignore[call-arg]


# ── PerlinNoiseConfig ─────────────────────────────────────────────────


class TestPerlinNoiseConfig:
    """Tests for PerlinNoiseConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default config uses expected values."""
        cfg = PerlinNoiseConfig()
        assert cfg.type == "perlin_noise"
        assert cfg.scale == 0.05
        assert cfg.octaves == 4

    def test_custom_values(self) -> None:
        """Custom parameter values are accepted."""
        cfg = PerlinNoiseConfig(scale=0.1, octaves=6)
        assert cfg.scale == 0.1
        assert cfg.octaves == 6

    def test_scale_must_be_positive(self) -> None:
        """Zero or negative scale raises ValidationError."""
        with pytest.raises(ValidationError):
            PerlinNoiseConfig(scale=0.0)
        with pytest.raises(ValidationError):
            PerlinNoiseConfig(scale=-0.1)

    def test_octaves_must_be_at_least_one(self) -> None:
        """Octaves < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            PerlinNoiseConfig(octaves=0)

    def test_create_strategy(self) -> None:
        """create_strategy() returns a PerlinNoiseStrategy with matching params."""
        cfg = PerlinNoiseConfig(scale=0.1, octaves=6)
        strategy = cfg.create_strategy()
        assert isinstance(strategy, PerlinNoiseStrategy)
        assert strategy.scale == 0.1
        assert strategy.octaves == 6

    def test_frozen(self) -> None:
        """Config is immutable after creation."""
        cfg = PerlinNoiseConfig()
        with pytest.raises(ValidationError):
            cfg.scale = 0.2  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            PerlinNoiseConfig(scale=0.05, persistence=0.5)  # type: ignore[call-arg]


# ── Discriminated union parsing ───────────────────────────────────────


class TestStrategyConfigUnion:
    """Tests for StrategyConfig discriminated union JSON parsing."""

    def test_parse_blob_and_line(self) -> None:
        """JSON with type='blob_and_line' parses to BlobAndLineConfig."""
        data = json.dumps({"type": "blob_and_line", "shape": 0.7})
        cfg = _strategy_adapter.validate_json(data)
        assert isinstance(cfg, BlobAndLineConfig)
        assert cfg.shape == 0.7

    def test_parse_blob_and_line_defaults(self) -> None:
        """JSON with only type field uses defaults."""
        data = json.dumps({"type": "blob_and_line"})
        cfg = _strategy_adapter.validate_json(data)
        assert isinstance(cfg, BlobAndLineConfig)
        assert cfg.shape == 0.5

    def test_parse_perlin_noise(self) -> None:
        """JSON with type='perlin_noise' parses to PerlinNoiseConfig."""
        data = json.dumps({"type": "perlin_noise", "scale": 0.1, "octaves": 6})
        cfg = _strategy_adapter.validate_json(data)
        assert isinstance(cfg, PerlinNoiseConfig)
        assert cfg.scale == 0.1
        assert cfg.octaves == 6

    def test_parse_perlin_noise_defaults(self) -> None:
        """JSON with only type field uses defaults."""
        data = json.dumps({"type": "perlin_noise"})
        cfg = _strategy_adapter.validate_json(data)
        assert isinstance(cfg, PerlinNoiseConfig)
        assert cfg.scale == 0.05
        assert cfg.octaves == 4

    def test_unknown_type_raises(self) -> None:
        """Unknown type value raises ValidationError."""
        data = json.dumps({"type": "unknown_strategy"})
        with pytest.raises(ValidationError):
            _strategy_adapter.validate_json(data)

    def test_missing_type_raises(self) -> None:
        """Missing type field raises ValidationError."""
        data = json.dumps({"shape": 0.5})
        with pytest.raises(ValidationError):
            _strategy_adapter.validate_json(data)

    def test_extra_field_raises(self) -> None:
        """Extra fields on a union member raise ValidationError."""
        data = json.dumps({"type": "blob_and_line", "shape": 0.5, "extra": 1})
        with pytest.raises(ValidationError):
            _strategy_adapter.validate_json(data)

    def test_wrong_field_for_type_raises(self) -> None:
        """Passing a field from another strategy type raises ValidationError."""
        data = json.dumps({"type": "blob_and_line", "octaves": 4})
        with pytest.raises(ValidationError):
            _strategy_adapter.validate_json(data)

    def test_python_dict_validation(self) -> None:
        """validate_python works with plain dicts."""
        cfg = _strategy_adapter.validate_python({"type": "perlin_noise", "scale": 0.08})
        assert isinstance(cfg, PerlinNoiseConfig)
        assert cfg.scale == 0.08


# ── generate_map_from_config ──────────────────────────────────────────


def _is_passable(gm: object, x: int, y: int) -> bool:
    assert isinstance(gm, GameMap)
    return gm[x, y] == CellType.PASSABLE


class TestGenerateMapFromConfig:
    """Tests for generate_map_from_config entrypoint."""

    def test_default_config(self) -> None:
        """Calling with no strategy_config generates a valid map."""
        gm = generate_map_from_config(20, 15, seed=42)
        assert gm.width == 20
        assert gm.height == 15

    def test_blob_and_line_config(self) -> None:
        """BlobAndLineConfig produces a map with correct dimensions."""
        cfg = BlobAndLineConfig(shape=0.8)
        gm = generate_map_from_config(15, 15, strategy_config=cfg, seed=42)
        assert gm.width == 15
        assert gm.height == 15

    def test_perlin_noise_config(self) -> None:
        """PerlinNoiseConfig produces a map with correct dimensions."""
        cfg = PerlinNoiseConfig(scale=0.1, octaves=2)
        gm = generate_map_from_config(15, 15, strategy_config=cfg, seed=42)
        assert gm.width == 15
        assert gm.height == 15

    def test_passable_connectivity(self) -> None:
        """Maps from config always have connected passable terrain."""
        for cfg in [BlobAndLineConfig(shape=0.3), PerlinNoiseConfig(octaves=2)]:
            gm = generate_map_from_config(
                20, 20, impassable_fraction=0.4, strategy_config=cfg, seed=7
            )
            comps = find_components(gm, _is_passable)
            assert len(comps) == 1

    def test_config_affects_output(self) -> None:
        """Different configs with the same seed produce different maps."""
        gm1 = generate_map_from_config(
            20, 20, strategy_config=BlobAndLineConfig(shape=0.0), seed=42
        )
        gm2 = generate_map_from_config(20, 20, strategy_config=PerlinNoiseConfig(), seed=42)
        diffs = sum(1 for pos in gm1.all_positions() if gm1[pos] != gm2[pos])
        assert diffs > 0

    def test_seed_reproducibility(self) -> None:
        """Same config and seed produce identical maps."""
        cfg = PerlinNoiseConfig(scale=0.08, octaves=3)
        gm1 = generate_map_from_config(15, 15, strategy_config=cfg, seed=99)
        gm2 = generate_map_from_config(15, 15, strategy_config=cfg, seed=99)
        for pos in gm1.all_positions():
            assert gm1[pos] == gm2[pos], f"Mismatch at {pos}"


# ── get_params() derivation from Pydantic config ─────────────────────


class TestBlobAndLineConfigGetParams:
    """Tests that BlobAndLineConfig.get_params() derives correct StrategyParam values."""

    def test_returns_one_param(self) -> None:
        """BlobAndLineConfig has exactly one configurable parameter."""
        params = BlobAndLineConfig.get_params()
        assert len(params) == 1

    def test_shape_param_name(self) -> None:
        """The parameter name matches the Pydantic field name."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.name == "shape"

    def test_shape_param_label(self) -> None:
        """Label comes from Field(title=...)."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.label == "Shape"

    def test_shape_param_hint(self) -> None:
        """Hint comes from Field(description=...)."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.hint == "0 = linear, 1 = blobs"

    def test_shape_param_bounds(self) -> None:
        """Bounds are derived from Field(ge=, le=)."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.min_val == 0.0
        assert param.max_val == 1.0

    def test_shape_param_default(self) -> None:
        """Default is taken from the Field default."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.default == 0.5

    def test_shape_param_type(self) -> None:
        """Type annotation is float."""
        param = BlobAndLineConfig.get_params()[0]
        assert param.param_type is float


class TestPerlinNoiseConfigGetParams:
    """Tests that PerlinNoiseConfig.get_params() derives correct StrategyParam values."""

    def test_returns_two_params(self) -> None:
        """PerlinNoiseConfig has two configurable parameters."""
        params = PerlinNoiseConfig.get_params()
        assert len(params) == 2

    def test_scale_param(self) -> None:
        """Scale parameter has correct metadata."""
        param = PerlinNoiseConfig.get_params()[0]
        assert param.name == "scale"
        assert param.label == "Noise scale"
        assert param.param_type is float
        assert param.default == 0.05
        assert param.min_val == 0.02
        assert param.max_val == 0.2

    def test_scale_param_hint(self) -> None:
        """Scale has a description-based hint."""
        param = PerlinNoiseConfig.get_params()[0]
        assert param.hint is not None
        assert "feature" in param.hint.lower() or "scale" in param.hint.lower()

    def test_octaves_param(self) -> None:
        """Octaves parameter has correct metadata."""
        param = PerlinNoiseConfig.get_params()[1]
        assert param.name == "octaves"
        assert param.label == "Octaves"
        assert param.param_type is int
        assert param.default == 4
        assert param.min_val == 1
        assert param.max_val == 8

    def test_octaves_param_hint(self) -> None:
        """Octaves has a description-based hint."""
        param = PerlinNoiseConfig.get_params()[1]
        assert param.hint is not None


class TestStrategyGetParamsDelegation:
    """Tests that MapStrategy.get_params() delegates to config_class."""

    def test_blob_and_line_delegation(self) -> None:
        """BlobAndLineStrategy.get_params() returns same as BlobAndLineConfig.get_params()."""
        strategy_params = BlobAndLineStrategy.get_params()
        config_params = BlobAndLineConfig.get_params()
        assert strategy_params == config_params

    def test_perlin_delegation(self) -> None:
        """PerlinNoiseStrategy.get_params() returns same as PerlinNoiseConfig.get_params()."""
        strategy_params = PerlinNoiseStrategy.get_params()
        config_params = PerlinNoiseConfig.get_params()
        assert strategy_params == config_params

    def test_strategy_params_are_strategy_param_instances(self) -> None:
        """All params from strategy delegation are StrategyParam instances."""
        for param in BlobAndLineStrategy.get_params():
            assert isinstance(param, StrategyParam)
        for param in PerlinNoiseStrategy.get_params():
            assert isinstance(param, StrategyParam)
