"""Model persistence: save and load trained networks.

Models are saved as a single file containing both the network weights
(state_dict) and the :class:`~hmls.singlemki.model.ModelConfig` that
defines the architecture.  This ensures that a loaded model can be
reconstructed without knowing the original hyperparameters.

Standalone JSON configuration files (``model_config.json`` and
``reward_config.json``) are also supported for use by the training
framework, where config must exist before any weights are produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hmls.nncore.reward import DefaultRewardConfig
from hmls.singlemki.model import ModelConfig, TankPolicyNetwork

MODEL_CONFIG_FILENAME = "model_config.json"
REWARD_CONFIG_FILENAME = "reward_config.json"


def save_model(
    model: TankPolicyNetwork,
    path: Path,
    reward_config: DefaultRewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trained model to disk.

    The saved file contains:
    - ``"state_dict"``: The model's learnable parameters.
    - ``"config"``: The :class:`ModelConfig` as a dict (for reconstruction).
    - ``"reward_config"``: The :class:`DefaultRewardConfig` as a dict
      (optional, for reproducing training configuration).
    - ``"metadata"``: Optional user-supplied metadata (e.g. training stats).

    Args:
        model: The model to save.
        path: Destination file path (typically ``.pt`` extension).
        reward_config: Optional reward configuration used during training.
        metadata: Optional dictionary of extra information to store
            alongside the model (e.g. training episode count, reward
            history).
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
) -> tuple[TankPolicyNetwork, dict[str, Any]]:
    """Load a model from disk.

    Reconstructs the :class:`TankPolicyNetwork` from the saved config
    and loads the trained weights.

    Args:
        path: Path to the saved model file.

    Returns:
        A tuple of ``(model, metadata)`` where *metadata* is the dict
        stored at save time (empty dict if none was provided).
        If a ``reward_config`` was saved, it will appear in metadata
        under the key ``"reward_config"`` as a :class:`DefaultRewardConfig`
        instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the saved file is missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    save_data: dict[str, Any] = torch.load(path, weights_only=False)

    config_dict = save_data["config"]
    config = ModelConfig.model_validate(config_dict)

    model = TankPolicyNetwork(config)
    model.load_state_dict(save_data["state_dict"])

    metadata: dict[str, Any] = save_data.get("metadata", {})

    # Restore reward config if present
    if "reward_config" in save_data:
        metadata["reward_config"] = DefaultRewardConfig.model_validate(save_data["reward_config"])

    return model, metadata


# --- Standalone JSON config file utilities ---


def save_model_config(config: ModelConfig, directory: Path) -> None:
    """Save a :class:`ModelConfig` as JSON to a model directory.

    Writes ``model_config.json`` in the given directory.

    Args:
        config: The model configuration to save.
        directory: Target directory (created if it does not exist).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_model_config(directory: Path) -> ModelConfig:
    """Load a :class:`ModelConfig` from a model directory.

    Reads ``model_config.json`` from the given directory.

    Args:
        directory: Directory containing the config file.

    Returns:
        The loaded ModelConfig.

    Raises:
        FileNotFoundError: If ``model_config.json`` is not present.
    """
    path = directory / MODEL_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}. "
            f"Each model directory must contain a '{MODEL_CONFIG_FILENAME}'."
        )
    return ModelConfig.model_validate_json(path.read_text())


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
