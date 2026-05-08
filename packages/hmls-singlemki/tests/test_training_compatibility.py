"""Tests for Mk-I model config compatibility with training validation."""

from __future__ import annotations

from hmls.reinforcetrainer.training_loop import _validate_model_configs
from hmls.singlemki.model import MkIModelConfig


class TestMkIConfigCompatibility:
    """Tests that Mk-I configs with different architectures pass validation."""

    def test_different_cnn_channels_allowed(self) -> None:
        """Configs with different cnn_channels are valid (only patch_size matters)."""
        config_a = MkIModelConfig(cnn_channels=(32, 64))
        config_b = MkIModelConfig(cnn_channels=(16, 32, 64, 128))
        _validate_model_configs(config_a, config_b)  # Should not raise

    def test_different_gru_hidden_size_allowed(self) -> None:
        """Configs with different gru_hidden_size are valid."""
        config_a = MkIModelConfig(gru_hidden_size=128)
        config_b = MkIModelConfig(gru_hidden_size=256)
        _validate_model_configs(config_a, config_b)  # Should not raise

    def test_different_architecture_same_patch_size(self) -> None:
        """Completely different architectures with same patch_size are valid."""
        config_a = MkIModelConfig(patch_size=9, cnn_channels=(32, 64), gru_hidden_size=128)
        config_b = MkIModelConfig(patch_size=9, cnn_channels=(16, 32, 64, 128), gru_hidden_size=256)
        _validate_model_configs(config_a, config_b)  # Should not raise
