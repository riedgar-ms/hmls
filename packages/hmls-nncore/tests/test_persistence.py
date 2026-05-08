"""Tests for the shared persistence helpers in hmls.nncore.persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.persistence import (
    load_model_config_data,
    load_model_data,
    load_reward_config,
    save_model_config_data,
    save_model_data,
    save_reward_config,
)
from hmls.nncore.reward import DefaultRewardConfig

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


# ── Tests: reward config persistence ──────────────────────────────────


class TestRewardConfig:
    """Tests for save/load_reward_config."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """DefaultRewardConfig can be saved and loaded from JSON."""
        config = DefaultRewardConfig(fire_hit_reward=1.0, death_reward=-2.0)
        save_reward_config(config, tmp_path)
        loaded = load_reward_config(tmp_path)
        assert loaded.fire_hit_reward == 1.0
        assert loaded.death_reward == -2.0

    def test_default_roundtrip(self, tmp_path: Path) -> None:
        """Default config round-trips correctly."""
        config = DefaultRewardConfig()
        save_reward_config(config, tmp_path)
        loaded = load_reward_config(tmp_path)
        assert loaded == config

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without reward_config.json raises."""
        with pytest.raises(FileNotFoundError, match="reward_config.json"):
            load_reward_config(tmp_path)

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_reward_config creates the directory if needed."""
        deep_dir = tmp_path / "a" / "b"
        save_reward_config(DefaultRewardConfig(), deep_dir)
        assert (deep_dir / "reward_config.json").exists()


# ── Tests: model data persistence ─────────────────────────────────────


class TestModelData:
    """Tests for save_model_data / load_model_data."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Model can be saved and loaded with identical weights."""
        model = _StubModel()
        model.eval()
        path = tmp_path / "model.pt"

        save_model_data(model, path, metadata={"episodes": 42})
        loaded, metadata = load_model_data(path, _StubConfig, _StubModel)
        loaded.eval()

        assert metadata == {"episodes": 42}
        assert loaded.config.hidden_size == model.config.hidden_size

        # Check weights match
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
            assert k1 == k2
            assert torch.equal(v1, v2)

    def test_roundtrip_with_reward_config(self, tmp_path: Path) -> None:
        """Saved reward_config appears in metadata on load."""
        model = _StubModel()
        path = tmp_path / "model.pt"
        reward = DefaultRewardConfig(fire_hit_reward=5.0)

        save_model_data(model, path, reward_config=reward)
        _, metadata = load_model_data(path, _StubConfig, _StubModel)

        assert "reward_config" in metadata
        assert isinstance(metadata["reward_config"], DefaultRewardConfig)
        assert metadata["reward_config"].fire_hit_reward == 5.0

    def test_no_metadata(self, tmp_path: Path) -> None:
        """Saving without metadata stores empty dict."""
        model = _StubModel()
        path = tmp_path / "model.pt"
        save_model_data(model, path)
        _, metadata = load_model_data(path, _StubConfig, _StubModel)
        assert metadata == {}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_model_data creates parent directories."""
        model = _StubModel()
        deep_path = tmp_path / "a" / "b" / "model.pt"
        save_model_data(model, deep_path)
        assert deep_path.exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model_data(tmp_path / "missing.pt", _StubConfig, _StubModel)


# ── Tests: model config persistence ───────────────────────────────────


class TestModelConfig:
    """Tests for save_model_config_data / load_model_config_data."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Config can be saved and loaded from JSON."""
        config = _StubConfig(patch_size=7, hidden_size=16)
        save_model_config_data(config, tmp_path)
        loaded = load_model_config_data(tmp_path, _StubConfig)
        assert loaded.patch_size == 7
        assert loaded.hidden_size == 16
        assert loaded.model_package == "test.stub"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without model_config.json raises."""
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            load_model_config_data(tmp_path, _StubConfig)

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_model_config_data creates the directory if needed."""
        deep_dir = tmp_path / "x" / "y"
        save_model_config_data(_StubConfig(), deep_dir)
        assert (deep_dir / "model_config.json").exists()
