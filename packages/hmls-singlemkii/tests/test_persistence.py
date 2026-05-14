"""Tests for Mk-II model persistence (save/load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork
from hmls.singlemkii.persistence import PERSISTENCE


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A saved Mk-II model can be loaded and produces identical output."""
    config = MkIIModelConfig(
        patch_size=9,
        cnn_channels=(16, 32),
        gru1_hidden_size=64,
        gru2_hidden_size=32,
    )
    model = MkIITankPolicyNetwork(config)
    model.eval()

    model_path = tmp_path / "test_model.pt"
    PERSISTENCE.save_model(model, model_path, metadata={"episodes": 100})

    loaded_model, metadata = PERSISTENCE.load_model(model_path)
    loaded_model.eval()

    assert metadata == {"episodes": 100}
    assert loaded_model.config.gru1_hidden_size == 64
    assert loaded_model.config.gru2_hidden_size == 32

    # Verify identical outputs
    x = torch.randn(FiveChannelPatchEncoder.NUM_CHANNELS, 9, 9)
    h = model.initial_hidden().squeeze(0)
    with torch.no_grad():
        out1, h1 = model(x, h)
        out2, h2 = loaded_model(x, h)
    assert torch.allclose(out1, out2)
    assert torch.allclose(h1, h2)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """save_model creates parent directories if needed."""
    model = MkIITankPolicyNetwork(MkIIModelConfig(patch_size=5))
    deep_path = tmp_path / "a" / "b" / "c" / "model.pt"
    PERSISTENCE.save_model(model, deep_path)
    assert deep_path.exists()


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PERSISTENCE.load_model(tmp_path / "no_such_file.pt")


def test_save_without_metadata(tmp_path: Path) -> None:
    """Saving without metadata stores empty dict."""
    model = MkIITankPolicyNetwork(MkIIModelConfig(patch_size=7))
    model_path = tmp_path / "model.pt"
    PERSISTENCE.save_model(model, model_path)

    _, metadata = PERSISTENCE.load_model(model_path)
    assert metadata == {}


class TestModelConfigJson:
    """Tests for model_config.json save/load utilities."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """MkIIModelConfig can be saved and loaded from JSON."""
        config = MkIIModelConfig(
            patch_size=7,
            cnn_channels=(16, 32, 64),
            gru1_hidden_size=256,
            gru2_hidden_size=128,
        )
        PERSISTENCE.save_model_config(config, tmp_path)

        loaded = PERSISTENCE.load_model_config(tmp_path)
        assert loaded.patch_size == 7
        assert loaded.gru1_hidden_size == 256
        assert loaded.gru2_hidden_size == 128
        assert loaded.model_id == "hmls.singlemkii"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without model_config.json raises."""
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            PERSISTENCE.load_model_config(tmp_path)
