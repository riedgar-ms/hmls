"""Persistence helpers for the simple squad executor model.

Provides save/load utilities for the executor model component
of a squad directory.  These are consumed by the squad player
package's :class:`SquadPersistence` — the executor is not
independently registered as an entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hmls.simplesquadexecutor.model import SimpleExecutorConfig, SimpleExecutorModel

MODEL_CONFIG_FILENAME = "model_config.json"
MODEL_WEIGHTS_FILENAME = "model.pt"


def save_executor(
    model: SimpleExecutorModel,
    directory: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save executor model config and weights to a directory.

    Args:
        model: The executor model to save.
        directory: Target directory (created if needed).
        metadata: Optional extra metadata to store in the checkpoint.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = directory / MODEL_CONFIG_FILENAME
    config_path.write_text(model.config.model_dump_json(indent=2))

    # Save weights
    save_data: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": model.config.model_dump(),
    }
    if metadata is not None:
        save_data["metadata"] = metadata
    torch.save(save_data, directory / MODEL_WEIGHTS_FILENAME)


def load_executor(directory: Path) -> tuple[SimpleExecutorModel, dict[str, Any]]:
    """Load an executor model from a directory.

    Args:
        directory: Directory containing ``model_config.json`` and
            ``model.pt``.

    Returns:
        A tuple of ``(model, metadata)``.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    weights_path = directory / MODEL_WEIGHTS_FILENAME
    if not weights_path.exists():
        msg = f"Executor weights not found: {weights_path}"
        raise FileNotFoundError(msg)

    save_data: dict[str, Any] = torch.load(weights_path, weights_only=True)
    config = SimpleExecutorConfig.model_validate(save_data["config"])
    model = SimpleExecutorModel(config)
    model.load_state_dict(save_data["state_dict"])
    metadata: dict[str, Any] = save_data.get("metadata", {})
    return model, metadata


def load_executor_config(directory: Path) -> SimpleExecutorConfig:
    """Load just the executor config from a directory.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        The parsed executor config.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    config_path = directory / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        msg = f"Executor config not found: {config_path}"
        raise FileNotFoundError(msg)

    config_text = config_path.read_text()
    return SimpleExecutorConfig.model_validate_json(config_text)


def create_executor(directory: Path) -> SimpleExecutorModel:
    """Create a fresh executor model from config and save initial weights.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        A freshly initialised executor model.
    """
    config = load_executor_config(directory)
    model = SimpleExecutorModel(config)
    save_executor(model, directory)
    return model


def load_or_create_executor(directory: Path) -> SimpleExecutorModel:
    """Load existing executor weights or create fresh ones.

    Args:
        directory: Directory that should contain the executor files.

    Returns:
        The executor model (loaded or freshly created).
    """
    weights_path = directory / MODEL_WEIGHTS_FILENAME
    if weights_path.exists():
        model, _ = load_executor(directory)
        return model
    return create_executor(directory)
