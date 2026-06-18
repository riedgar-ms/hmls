"""Persistence helpers for the simple squad planner model.

Provides save/load utilities for the planner model component
of a squad directory.  Consumed by the squad player package's
:class:`SquadPersistence` — the planner is not independently
registered as an entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hmls.simplesquadplanner.model import SimplePlannerConfig, SimplePlannerModel

MODEL_CONFIG_FILENAME = "model_config.json"
MODEL_WEIGHTS_FILENAME = "model.pt"


def save_planner(
    model: SimplePlannerModel,
    directory: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save planner model config and weights to a directory.

    Args:
        model: The planner model to save.
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


def load_planner(directory: Path) -> tuple[SimplePlannerModel, dict[str, Any]]:
    """Load a planner model from a directory.

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
        msg = f"Planner weights not found: {weights_path}"
        raise FileNotFoundError(msg)

    save_data: dict[str, Any] = torch.load(weights_path, weights_only=True)
    config = SimplePlannerConfig.model_validate(save_data["config"])
    model = SimplePlannerModel(config)
    model.load_state_dict(save_data["state_dict"])
    metadata: dict[str, Any] = save_data.get("metadata", {})
    return model, metadata


def load_planner_config(directory: Path) -> SimplePlannerConfig:
    """Load just the planner config from a directory.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        The parsed planner config.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    config_path = directory / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        msg = f"Planner config not found: {config_path}"
        raise FileNotFoundError(msg)

    config_text = config_path.read_text()
    return SimplePlannerConfig.model_validate_json(config_text)


def create_planner(directory: Path) -> SimplePlannerModel:
    """Create a fresh planner model from config and save initial weights.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        A freshly initialised planner model.
    """
    config = load_planner_config(directory)
    model = SimplePlannerModel(config)
    save_planner(model, directory)
    return model


def load_or_create_planner(directory: Path) -> SimplePlannerModel:
    """Load existing planner weights or create fresh ones.

    Args:
        directory: Directory that should contain the planner files.

    Returns:
        The planner model (loaded or freshly created).
    """
    weights_path = directory / MODEL_WEIGHTS_FILENAME
    if weights_path.exists():
        model, _ = load_planner(directory)
        return model
    return create_planner(directory)
