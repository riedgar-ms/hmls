"""Tests for the persistence classes in hmls.nncore.persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.persistence import NNPlayerModelPersistence
from hmls.nncore.reward import BasicRewardConfig

# ── Minimal stub model for testing ────────────────────────────────────


class _StubConfig(TankModelConfig, frozen=True, extra="forbid"):
    """Minimal config for the stub model."""

    model_package: str = "test.stub"
    hidden_size: int = 4


class _StubModel(TankModelBase):
    """Minimal TankModelBase with a single linear layer."""

    def __init__(self, config: _StubConfig | None = None) -> None:
        super().__init__()
        self.config: _StubConfig = config or _StubConfig()
        self.linear = torch.nn.Linear(2, 2)

    def forward(
        self, patch_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dummy forward pass."""
        return patch_tensor, hidden

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return zero hidden state."""
        return torch.zeros(batch_size, self.config.hidden_size)

    @property
    def total_hidden_size(self) -> int:
        """Hidden size."""
        return self.config.hidden_size


# ── Shared fixture ────────────────────────────────────────────────────


@pytest.fixture()
def persistence() -> NNPlayerModelPersistence[_StubConfig, _StubModel]:
    """Return an NNPlayerModelPersistence for the stub types."""
    return NNPlayerModelPersistence(_StubConfig, _StubModel)


# ── Tests: model data persistence ─────────────────────────────────────


class TestModelData:
    """Tests for NNPlayerModelPersistence save/load model methods."""

    def test_roundtrip(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Model can be saved and loaded with identical weights."""
        model = _StubModel()
        model.eval()
        path = tmp_path / "model.pt"

        persistence.save_model(model, path, metadata={"episodes": 42})
        loaded, metadata = persistence.load_model(path)
        loaded.eval()

        assert metadata == {"episodes": 42}
        assert loaded.config.hidden_size == model.config.hidden_size

        # Check weights match
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_roundtrip_with_reward_config(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Saved reward_config appears in metadata on load."""
        model = _StubModel()
        path = tmp_path / "model.pt"
        reward = BasicRewardConfig(fire_hit_reward=5.0)

        persistence.save_model(model, path, reward_config=reward)
        _, metadata = persistence.load_model(path)

        assert "reward_config" in metadata
        assert isinstance(metadata["reward_config"], BasicRewardConfig)
        assert metadata["reward_config"].fire_hit_reward == 5.0

    def test_no_metadata(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Saving without metadata stores empty dict."""
        model = _StubModel()
        path = tmp_path / "model.pt"
        persistence.save_model(model, path)
        _, metadata = persistence.load_model(path)
        assert metadata == {}

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """save_model creates parent directories."""
        model = _StubModel()
        deep_path = tmp_path / "a" / "b" / "model.pt"
        persistence.save_model(model, deep_path)
        assert deep_path.exists()

    def test_load_missing_raises(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            persistence.load_model(tmp_path / "missing.pt")


# ── Tests: model config persistence ───────────────────────────────────


class TestModelConfig:
    """Tests for NNPlayerModelPersistence config methods."""

    def test_roundtrip(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Config can be saved and loaded from JSON."""
        config = _StubConfig(patch_size=7, hidden_size=16)
        persistence.save_model_config(config, tmp_path)
        loaded = persistence.load_model_config(tmp_path)
        assert loaded.patch_size == 7
        assert loaded.hidden_size == 16
        assert loaded.model_package == "test.stub"

    def test_load_missing_raises(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """Loading from a directory without model_config.json raises."""
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            persistence.load_model_config(tmp_path)

    def test_save_creates_directory(
        self,
        tmp_path: Path,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """save_model_config creates the directory if needed."""
        deep_dir = tmp_path / "x" / "y"
        persistence.save_model_config(_StubConfig(), deep_dir)
        assert (deep_dir / "model_config.json").exists()


# ── Tests: create_model ───────────────────────────────────────────────


class TestCreateModel:
    """Tests for NNPlayerModelPersistence.create_model."""

    def test_creates_model(
        self,
        persistence: NNPlayerModelPersistence[_StubConfig, _StubModel],
    ) -> None:
        """create_model returns a model of the correct type."""
        config = _StubConfig(hidden_size=8)
        model = persistence.create_model(config)
        assert isinstance(model, _StubModel)
        assert model.config.hidden_size == 8
