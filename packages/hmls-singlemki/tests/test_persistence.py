"""Tests for model persistence (save/load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hmls.nncore.reward import DefaultRewardConfig
from hmls.singlemki.model import MkIModelConfig, MkITankPolicyNetwork
from hmls.singlemki.persistence import PERSISTENCE


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A saved model can be loaded and produces identical output."""
    config = MkIModelConfig(patch_size=9, cnn_channels=(16, 32), gru_hidden_size=64)
    model = MkITankPolicyNetwork(config)
    model.eval()

    # Save
    model_path = tmp_path / "test_model.pt"
    PERSISTENCE.save_model(model, model_path, metadata={"episodes": 100})

    # Load
    loaded_model, metadata = PERSISTENCE.load_model(model_path)
    loaded_model.eval()

    assert metadata == {"episodes": 100}
    assert loaded_model.config.patch_size == 9
    assert loaded_model.config.cnn_channels == (16, 32)
    assert loaded_model.config.gru_hidden_size == 64

    # Verify identical outputs
    x = torch.randn(5, 9, 9)
    h = model.initial_hidden().squeeze(0)
    with torch.no_grad():
        out1, h1 = model(x, h)
        out2, h2 = loaded_model(x, h)
    assert torch.allclose(out1, out2)
    assert torch.allclose(h1, h2)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """save_model creates parent directories if needed."""
    config = MkIModelConfig(patch_size=5)
    model = MkITankPolicyNetwork(config)
    deep_path = tmp_path / "a" / "b" / "c" / "model.pt"
    PERSISTENCE.save_model(model, deep_path)
    assert deep_path.exists()


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PERSISTENCE.load_model(tmp_path / "no_such_file.pt")


def test_save_without_metadata(tmp_path: Path) -> None:
    """Saving without metadata stores empty dict."""
    config = MkIModelConfig(patch_size=7)
    model = MkITankPolicyNetwork(config)
    model_path = tmp_path / "model.pt"
    PERSISTENCE.save_model(model, model_path)

    _, metadata = PERSISTENCE.load_model(model_path)
    assert metadata == {}


class TestModelConfigJson:
    """Tests for model_config.json save/load utilities."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """MkIModelConfig can be saved and loaded from JSON."""
        config = MkIModelConfig(patch_size=7, cnn_channels=(16, 32, 64), gru_hidden_size=256)
        PERSISTENCE.save_model_config(config, tmp_path)

        loaded = PERSISTENCE.load_model_config(tmp_path)
        assert loaded.patch_size == 7
        assert loaded.cnn_channels == (16, 32, 64)
        assert loaded.gru_hidden_size == 256

    def test_default_config_roundtrip(self, tmp_path: Path) -> None:
        """Default MkIModelConfig round-trips correctly."""
        config = MkIModelConfig()
        PERSISTENCE.save_model_config(config, tmp_path)

        loaded = PERSISTENCE.load_model_config(tmp_path)
        assert loaded == config

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without model_config.json raises."""
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            PERSISTENCE.load_model_config(tmp_path)

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_model_config creates the directory if needed."""
        deep_dir = tmp_path / "a" / "b" / "c"
        PERSISTENCE.save_model_config(MkIModelConfig(), deep_dir)
        assert (deep_dir / "model_config.json").exists()


class TestRewardConfigJson:
    """Tests for reward_config.json save/load utilities."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """DefaultRewardConfig can be saved and loaded from JSON."""
        config = DefaultRewardConfig(
            fire_hit_reward=1.0,
            death_reward=-2.0,
            exploration_reward=0.05,
        )
        PERSISTENCE.save_reward_config(config, tmp_path)

        loaded = PERSISTENCE.load_reward_config(tmp_path)
        assert loaded.fire_hit_reward == 1.0
        assert loaded.death_reward == -2.0
        assert loaded.exploration_reward == 0.05

    def test_default_config_roundtrip(self, tmp_path: Path) -> None:
        """Default DefaultRewardConfig round-trips correctly."""
        config = DefaultRewardConfig()
        PERSISTENCE.save_reward_config(config, tmp_path)

        loaded = PERSISTENCE.load_reward_config(tmp_path)
        assert loaded == config

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without reward_config.json raises."""
        with pytest.raises(FileNotFoundError, match="reward_config.json"):
            PERSISTENCE.load_reward_config(tmp_path)

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """save_reward_config creates the directory if needed."""
        deep_dir = tmp_path / "a" / "b" / "c"
        PERSISTENCE.save_reward_config(DefaultRewardConfig(), deep_dir)
        assert (deep_dir / "reward_config.json").exists()
