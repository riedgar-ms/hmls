"""Stub persistence module for dynamic dispatch.

This module provides the interface that
:mod:`hmls.nncore.persistence` expects from a model package's
``persistence`` submodule.  It enables the reinforcetrainer tests
to use :func:`~hmls.nncore.persistence.load_or_create_model` and
:func:`~hmls.nncore.persistence.create_player` without depending
on any concrete model package.

The ``model_package`` for stubs is ``"hmls.reinforcetrainer._testing"``
and this module is importable as
``hmls.reinforcetrainer._testing.persistence``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from hmls.nncore.persistence import MODEL_CONFIG_FILENAME, REWARD_CONFIG_FILENAME
from hmls.nncore.player import NNPlayerBase
from hmls.nncore.reward import DefaultRewardConfig
from hmls.reinforcetrainer._testing.stub_model import StubModelConfig, StubTankModel
from hmls.reinforcetrainer._testing.stub_player import StubNNPlayer

# --- Model factory ---


def create_model(config: StubModelConfig) -> StubTankModel:
    """Create a new :class:`StubTankModel` from configuration.

    Args:
        config: The stub model configuration.

    Returns:
        A freshly initialised StubTankModel.
    """
    return StubTankModel(config)


# --- Model persistence ---


def save_model(
    model: StubTankModel,
    path: Path,
    reward_config: DefaultRewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a stub model to disk.

    Args:
        model: The model to save.
        path: Destination file path.
        reward_config: Optional reward configuration.
        metadata: Optional metadata dictionary.
    """
    save_data: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": model.config.model_dump(),
    }
    if reward_config is not None:
        save_data["reward_config"] = reward_config.model_dump()
    if metadata is not None:
        save_data["metadata"] = metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, path)


def load_model(path: Path) -> tuple[StubTankModel, dict[str, Any]]:
    """Load a stub model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        A tuple of ``(model, metadata)``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    save_data: dict[str, Any] = torch.load(path, weights_only=False)
    config = StubModelConfig.model_validate(save_data["config"])
    model = StubTankModel(config)
    model.load_state_dict(save_data["state_dict"])

    metadata: dict[str, Any] = save_data.get("metadata", {})
    if "reward_config" in save_data:
        metadata["reward_config"] = DefaultRewardConfig.model_validate(save_data["reward_config"])

    return model, metadata


# --- Config file utilities ---


def save_model_config(config: StubModelConfig, directory: Path) -> None:
    """Save a :class:`StubModelConfig` as JSON to a directory.

    Args:
        config: The model configuration to save.
        directory: Target directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_model_config(directory: Path) -> StubModelConfig:
    """Load a :class:`StubModelConfig` from a directory.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        The loaded StubModelConfig.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    path = directory / MODEL_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}. "
            f"Each model directory must contain a '{MODEL_CONFIG_FILENAME}'."
        )
    return StubModelConfig.model_validate_json(path.read_text())


def save_reward_config(config: DefaultRewardConfig, directory: Path) -> None:
    """Save a :class:`DefaultRewardConfig` as JSON to a directory.

    Args:
        config: The reward configuration to save.
        directory: Target directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REWARD_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_reward_config(directory: Path) -> DefaultRewardConfig:
    """Load a :class:`DefaultRewardConfig` from a directory.

    Args:
        directory: Directory containing ``reward_config.json``.

    Returns:
        The loaded DefaultRewardConfig.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    path = directory / REWARD_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Reward configuration file not found: {path}. "
            f"Each model directory must contain a '{REWARD_CONFIG_FILENAME}'."
        )
    return DefaultRewardConfig.model_validate_json(path.read_text())


# --- Player factory ---


def create_player(
    team: str,
    model: StubTankModel,
    mode: Literal["play", "learn"] = "play",
) -> NNPlayerBase:
    """Create a :class:`StubNNPlayer` for this model.

    Args:
        team: The team this player controls.
        model: The stub model to use.
        mode: Operating mode.

    Returns:
        A StubNNPlayer instance.
    """
    return StubNNPlayer(team=team, model=model, mode=mode)
