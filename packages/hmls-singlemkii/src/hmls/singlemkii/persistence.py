"""Model persistence: save and load Mk-II trained networks.

This module serves as the dynamic dispatch entry point: the generic
loader in :mod:`hmls.nncore.persistence` imports
``hmls.singlemkii.persistence`` and calls the standard functions
defined here.

Most of the heavy lifting is delegated to the model-agnostic helpers
in :mod:`hmls.nncore.persistence`.  This module provides thin,
type-specific wrappers so that the dynamic dispatch protocol has
concrete type signatures to work with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from hmls.nncore.persistence import (
    load_model_config_data,
    load_model_data,
    load_reward_config,
    save_model_config_data,
    save_model_data,
    save_reward_config,
)
from hmls.nncore.player import NNPlayerBase
from hmls.nncore.reward import DefaultRewardConfig
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork

# Re-export reward config helpers (model-agnostic, identical for all packages)
__all__ = [
    "save_model",
    "load_model",
    "save_model_config",
    "load_model_config",
    "save_reward_config",
    "load_reward_config",
    "create_model",
    "create_player",
]


def save_model(
    model: MkIITankPolicyNetwork,
    path: Path,
    reward_config: DefaultRewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trained Mk-II model to disk.

    Args:
        model: The model to save.
        path: Destination file path (typically ``.pt`` extension).
        reward_config: Optional reward configuration used during training.
        metadata: Optional dictionary of extra information.
    """
    save_model_data(model, path, reward_config=reward_config, metadata=metadata)


def load_model(
    path: Path,
) -> tuple[MkIITankPolicyNetwork, dict[str, Any]]:
    """Load a Mk-II model from disk.

    Args:
        path: Path to the saved model file.

    Returns:
        A tuple of ``(model, metadata)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the saved file is missing required keys.
    """
    return load_model_data(path, MkIIModelConfig, MkIITankPolicyNetwork)


def save_model_config(config: MkIIModelConfig, directory: Path) -> None:
    """Save a :class:`MkIIModelConfig` as JSON to a model directory.

    Args:
        config: The model configuration to save.
        directory: Target directory (created if it does not exist).
    """
    save_model_config_data(config, directory)


def load_model_config(directory: Path) -> MkIIModelConfig:
    """Load a :class:`MkIIModelConfig` from a model directory.

    Args:
        directory: Directory containing the config file.

    Returns:
        The loaded MkIIModelConfig.

    Raises:
        FileNotFoundError: If ``model_config.json`` is not present.
    """
    return load_model_config_data(directory, MkIIModelConfig)


# --- Factory functions for dynamic dispatch ---


def create_model(config: MkIIModelConfig) -> MkIITankPolicyNetwork:
    """Create a new :class:`MkIITankPolicyNetwork` from a configuration.

    Args:
        config: The model configuration.

    Returns:
        A freshly initialised MkIITankPolicyNetwork.
    """
    return MkIITankPolicyNetwork(config)


def create_player(
    team: str,
    model: MkIITankPolicyNetwork,
    mode: Literal["play", "learn"] = "play",
) -> NNPlayerBase:
    """Create an :class:`NNPlayer` for the Mk-II model.

    Args:
        team: The team this player controls.
        model: The :class:`MkIITankPolicyNetwork` to use.
        mode: ``"play"`` or ``"learn"``.

    Returns:
        An NNPlayer instance wrapping the model.
    """
    from hmls.nncore.player import NNPlayer

    return NNPlayer(team=team, model=model, mode=mode)
