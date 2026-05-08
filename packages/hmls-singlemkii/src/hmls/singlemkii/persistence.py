"""Model persistence: save and load Mk-II trained networks.

Mirrors the persistence interface of ``hmls.singlemki.persistence``
for the Mk-II model architecture.  This module serves as the dynamic
dispatch entry point: the generic loader in :mod:`hmls.nncore.persistence`
imports ``hmls.singlemkii.persistence`` and calls the standard functions
defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from hmls.nncore.persistence import MODEL_CONFIG_FILENAME, REWARD_CONFIG_FILENAME
from hmls.nncore.player import NNPlayerBase
from hmls.nncore.reward import DefaultRewardConfig
from hmls.singlemkii.model import MkIIModelConfig, MkIITankPolicyNetwork


def save_model(
    model: MkIITankPolicyNetwork,
    path: Path,
    reward_config: DefaultRewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trained Mk-II model to disk.

    The saved file contains:
    - ``"state_dict"``: The model's learnable parameters.
    - ``"config"``: The :class:`MkIIModelConfig` as a dict.
    - ``"reward_config"``: Optional reward configuration dict.
    - ``"metadata"``: Optional user-supplied metadata.

    Args:
        model: The model to save.
        path: Destination file path (typically ``.pt`` extension).
        reward_config: Optional reward configuration used during training.
        metadata: Optional dictionary of extra information.
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


def load_model(
    path: Path,
) -> tuple[MkIITankPolicyNetwork, dict[str, Any]]:
    """Load a Mk-II model from disk.

    Reconstructs the :class:`MkIITankPolicyNetwork` from the saved
    config and loads the trained weights.

    Args:
        path: Path to the saved model file.

    Returns:
        A tuple of ``(model, metadata)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the saved file is missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    save_data: dict[str, Any] = torch.load(path, weights_only=False)

    config_dict = save_data["config"]
    config = MkIIModelConfig.model_validate(config_dict)

    model = MkIITankPolicyNetwork(config)
    model.load_state_dict(save_data["state_dict"])

    metadata: dict[str, Any] = save_data.get("metadata", {})

    if "reward_config" in save_data:
        metadata["reward_config"] = DefaultRewardConfig.model_validate(save_data["reward_config"])

    return model, metadata


# --- Standalone JSON config file utilities ---


def save_model_config(config: MkIIModelConfig, directory: Path) -> None:
    """Save a :class:`MkIIModelConfig` as JSON to a model directory.

    Writes ``model_config.json`` in the given directory.

    Args:
        config: The model configuration to save.
        directory: Target directory (created if it does not exist).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_model_config(directory: Path) -> MkIIModelConfig:
    """Load a :class:`MkIIModelConfig` from a model directory.

    Reads ``model_config.json`` from the given directory.

    Args:
        directory: Directory containing the config file.

    Returns:
        The loaded MkIIModelConfig.

    Raises:
        FileNotFoundError: If ``model_config.json`` is not present.
    """
    path = directory / MODEL_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}. "
            f"Each model directory must contain a '{MODEL_CONFIG_FILENAME}'."
        )
    return MkIIModelConfig.model_validate_json(path.read_text())


def save_reward_config(config: DefaultRewardConfig, directory: Path) -> None:
    """Save a :class:`DefaultRewardConfig` as JSON to a model directory.

    Writes ``reward_config.json`` in the given directory.

    Args:
        config: The reward configuration to save.
        directory: Target directory (created if it does not exist).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REWARD_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_reward_config(directory: Path) -> DefaultRewardConfig:
    """Load a :class:`DefaultRewardConfig` from a model directory.

    Reads ``reward_config.json`` from the given directory.

    Args:
        directory: Directory containing the config file.

    Returns:
        The loaded DefaultRewardConfig.

    Raises:
        FileNotFoundError: If ``reward_config.json`` is not present.
    """
    path = directory / REWARD_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Reward configuration file not found: {path}. "
            f"Each model directory must contain a '{REWARD_CONFIG_FILENAME}'."
        )
    return DefaultRewardConfig.model_validate_json(path.read_text())


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
    from hmls.singlemkii.player import NNPlayer

    return NNPlayer(team=team, model=model, mode=mode)
