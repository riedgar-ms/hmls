"""Tests for Mk-III model persistence (save/load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hmls.nncore.encoding import FiveChannelPatchEncoder
from hmls.singlemkiii.model import MkIIIModelConfig, MkIIITankPolicyNetwork
from hmls.singlemkiii.persistence import PERSISTENCE


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A saved Mk-III model can be loaded and produces identical output."""
    config = MkIIIModelConfig(
        patch_size=9,
        gru_hidden_size=64,
    )
    model = MkIIITankPolicyNetwork(config)
    model.eval()

    model_path = tmp_path / "test_model.pt"
    PERSISTENCE.save_model(model, model_path, metadata={"episodes": 100})

    loaded_model, metadata = PERSISTENCE.load_model(model_path)
    loaded_model.eval()

    assert metadata == {"episodes": 100}
    assert loaded_model.config.gru_hidden_size == 64

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
    model = MkIIITankPolicyNetwork(MkIIIModelConfig(patch_size=5))
    deep_path = tmp_path / "a" / "b" / "c" / "model.pt"
    PERSISTENCE.save_model(model, deep_path)
    assert deep_path.exists()


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PERSISTENCE.load_model(tmp_path / "no_such_file.pt")


def test_save_without_metadata(tmp_path: Path) -> None:
    """Saving without metadata stores empty dict."""
    model = MkIIITankPolicyNetwork(MkIIIModelConfig(patch_size=7))
    model_path = tmp_path / "model.pt"
    PERSISTENCE.save_model(model, model_path)

    _, metadata = PERSISTENCE.load_model(model_path)
    assert metadata == {}


class TestModelConfigJson:
    """Tests for model_config.json save/load utilities."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """MkIIIModelConfig can be saved and loaded from JSON."""
        config = MkIIIModelConfig(
            patch_size=7,
            gru_hidden_size=256,
        )
        PERSISTENCE.save_model_config(config, tmp_path)

        loaded = PERSISTENCE.load_model_config(tmp_path)
        assert loaded.patch_size == 7
        assert loaded.gru_hidden_size == 256
        assert loaded.model_package == "hmls.singlemkiii"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        """Loading from a directory without model_config.json raises."""
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            PERSISTENCE.load_model_config(tmp_path)
