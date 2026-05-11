"""Integration tests verifying that sample configurations are loadable.

Discovers every ``config.json`` under the repository's ``sample_configs/``
directory and checks that:

1. The trainer configuration can be parsed and validated by
   :func:`hmls.reinforcetrainer.cli.load_config`.
2. Each model directory referenced in the config contains a valid
   ``model_config.json`` loadable by
   :func:`hmls.nncore.persistence.load_model_config`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hmls.nncore.persistence import load_model_config
from hmls.reinforcetrainer.cli import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CONFIGS_DIR = REPO_ROOT / "sample_configs"

# Discover all config.json files under sample_configs/.
_config_files = sorted(SAMPLE_CONFIGS_DIR.rglob("config.json"))


def _label(path: Path) -> str:
    """Return a human-readable label for parametrised test IDs."""
    return str(path.relative_to(SAMPLE_CONFIGS_DIR))


@pytest.mark.parametrize(
    "config_path",
    _config_files,
    ids=[_label(p) for p in _config_files],
)
class TestSampleConfigs:
    """Verify that every sample config is loadable."""

    def test_trainer_config_loads(self, config_path: Path) -> None:
        """The trainer config JSON parses and validates without error."""
        config = load_config(config_path)
        # Smoke-check: the two model refs must be present.
        assert config.model_a is not None
        assert config.model_b is not None

    def test_model_configs_load(self, config_path: Path) -> None:
        """Each model directory's model_config.json is loadable."""
        config = load_config(config_path)
        for model_ref in (config.model_a, config.model_b):
            model_cfg = load_model_config(model_ref.dir)
            assert model_cfg is not None
