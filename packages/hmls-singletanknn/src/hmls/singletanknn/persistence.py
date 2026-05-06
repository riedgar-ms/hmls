"""Model persistence: save and load trained networks.

Models are saved as a single file containing both the network weights
(state_dict) and the :class:`~hmls.singletanknn.model.ModelConfig` that
defines the architecture.  This ensures that a loaded model can be
reconstructed without knowing the original hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hmls.singletanknn.model import ModelConfig, TankPolicyNetwork
from hmls.singletanknn.reward import DefaultRewardConfig


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
        "config": {
            "patch_size": model.config.patch_size,
            "cnn_channels": list(model.config.cnn_channels),
            "gru_hidden_size": model.config.gru_hidden_size,
            "num_actions": model.config.num_actions,
        },
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
    config = ModelConfig(
        patch_size=config_dict["patch_size"],
        cnn_channels=tuple(config_dict["cnn_channels"]),
        gru_hidden_size=config_dict["gru_hidden_size"],
        num_actions=config_dict["num_actions"],
    )

    model = TankPolicyNetwork(config)
    model.load_state_dict(save_data["state_dict"])

    metadata: dict[str, Any] = save_data.get("metadata", {})

    # Restore reward config if present
    if "reward_config" in save_data:
        metadata["reward_config"] = DefaultRewardConfig.model_validate(save_data["reward_config"])

    return model, metadata
