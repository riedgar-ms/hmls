"""Tests for model persistence (save/load)."""

from __future__ import annotations

from pathlib import Path

import torch

from hmls.singletanknn.model import ModelConfig, TankPolicyNetwork
from hmls.singletanknn.persistence import load_model, save_model


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A saved model can be loaded and produces identical output."""
    config = ModelConfig(patch_size=9, cnn_channels=(16, 32), gru_hidden_size=64)
    model = TankPolicyNetwork(config)
    model.eval()

    # Save
    model_path = tmp_path / "test_model.pt"
    save_model(model, model_path, metadata={"episodes": 100})

    # Load
    loaded_model, metadata = load_model(model_path)
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
    config = ModelConfig(patch_size=5)
    model = TankPolicyNetwork(config)
    deep_path = tmp_path / "a" / "b" / "c" / "model.pt"
    save_model(model, deep_path)
    assert deep_path.exists()


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "no_such_file.pt")


def test_save_without_metadata(tmp_path: Path) -> None:
    """Saving without metadata stores empty dict."""
    config = ModelConfig(patch_size=7)
    model = TankPolicyNetwork(config)
    model_path = tmp_path / "model.pt"
    save_model(model, model_path)

    _, metadata = load_model(model_path)
    assert metadata == {}
