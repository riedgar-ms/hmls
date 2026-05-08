"""Generic model persistence with dynamic package dispatch.

Provides model-agnostic save/load infrastructure that discovers the
correct concrete persistence module at runtime by reading the
``model_package`` field from ``model_config.json``.

This module also provides reusable helper functions
(:func:`save_model_data`, :func:`load_model_data`,
:func:`save_model_config_data`, :func:`load_model_config_data`,
:func:`save_reward_config`, :func:`load_reward_config`) that
encapsulate the common persistence logic shared by all model packages.

Each model package (e.g. ``hmls.singlemki``, ``hmls.singlemkii``) must
expose a ``persistence`` submodule with the following functions:

- ``save_model(model, path, reward_config=None, metadata=None) -> None``
- ``load_model(path) -> tuple[TankModelBase, dict[str, Any]]``
- ``save_model_config(config, directory) -> None``
- ``load_model_config(directory) -> TankModelConfig``  (concrete subclass)
- ``create_player(team, model, mode) -> NNPlayerBase``
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

import torch

from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.reward import DefaultRewardConfig

MODEL_CONFIG_FILENAME = "model_config.json"
REWARD_CONFIG_FILENAME = "reward_config.json"

ModelT = TypeVar("ModelT", bound=TankModelBase)
ConfigT = TypeVar("ConfigT", bound=TankModelConfig)


def _import_persistence_module(model_package: str) -> ModuleType:
    """Import the persistence submodule for a model package.

    Args:
        model_package: Fully-qualified package name
            (e.g. ``"hmls.singlemki"``).

    Returns:
        The imported ``persistence`` module.

    Raises:
        ModuleNotFoundError: If the package or its ``persistence``
            submodule cannot be imported.
    """
    module_name = f"{model_package}.persistence"
    return importlib.import_module(module_name)


# ── Reusable persistence helpers ──────────────────────────────────────
#
# These are model-agnostic implementations that concrete model packages
# can delegate to, avoiding boilerplate duplication.


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


def save_model_data(
    model: TankModelBase,
    path: Path,
    reward_config: DefaultRewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trained model to disk (model-agnostic implementation).

    The saved file contains:
    - ``"state_dict"``: The model's learnable parameters.
    - ``"config"``: The model config as a dict (for reconstruction).
    - ``"reward_config"``: Optional reward configuration dict.
    - ``"metadata"``: Optional user-supplied metadata.

    Args:
        model: The model to save (any :class:`TankModelBase` subclass).
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


def load_model_data(
    path: Path,
    config_cls: type[ConfigT],
    model_cls: type[ModelT],
) -> tuple[ModelT, dict[str, Any]]:
    """Load a model from disk (model-agnostic implementation).

    Reconstructs a model from the saved config and loads the trained
    weights.

    Args:
        path: Path to the saved model file.
        config_cls: The concrete config class to deserialise into.
        model_cls: The concrete model class to instantiate.

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
    config = config_cls.model_validate(config_dict)

    model = model_cls(config)
    model.load_state_dict(save_data["state_dict"])

    metadata: dict[str, Any] = save_data.get("metadata", {})

    if "reward_config" in save_data:
        metadata["reward_config"] = DefaultRewardConfig.model_validate(save_data["reward_config"])

    return model, metadata


def save_model_config_data(config: TankModelConfig, directory: Path) -> None:
    """Save a model config as JSON to a model directory.

    Writes ``model_config.json`` in the given directory.

    Args:
        config: The model configuration to save.
        directory: Target directory (created if it does not exist).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2))


def load_model_config_data(
    directory: Path,
    config_cls: type[ConfigT],
) -> ConfigT:
    """Load a model config from a model directory.

    Reads ``model_config.json`` from the given directory.

    Args:
        directory: Directory containing the config file.
        config_cls: The concrete config class to deserialise into.

    Returns:
        The loaded config instance.

    Raises:
        FileNotFoundError: If ``model_config.json`` is not present.
    """
    path = directory / MODEL_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}. "
            f"Each model directory must contain a '{MODEL_CONFIG_FILENAME}'."
        )
    return config_cls.model_validate_json(path.read_text())


def read_model_package(directory: Path) -> str:
    """Read the ``model_package`` field from a model config JSON file.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        The ``model_package`` string.

    Raises:
        FileNotFoundError: If ``model_config.json`` is missing.
        KeyError: If ``model_package`` is not present in the JSON.
    """
    config_path = directory / MODEL_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {config_path}. "
            f"Each model directory must contain a '{MODEL_CONFIG_FILENAME}'."
        )
    data = json.loads(config_path.read_text())
    if "model_package" not in data:
        raise KeyError(
            f"'model_package' field missing from {config_path}. "
            f"Each model_config.json must specify the package that defines the model."
        )
    model_package: str = data["model_package"]
    return model_package


def load_model_config(directory: Path) -> TankModelConfig:
    """Load a model config using dynamic package dispatch.

    Reads ``model_config.json``, discovers the ``model_package``,
    imports the correct persistence module, and delegates to its
    ``load_model_config`` function.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        A concrete :class:`TankModelConfig` subclass instance.
    """
    model_package = read_model_package(directory)
    persistence = _import_persistence_module(model_package)
    config: TankModelConfig = persistence.load_model_config(directory)
    return config


def load_model(path: Path) -> tuple[TankModelBase, dict[str, Any]]:
    """Load a model using dynamic package dispatch.

    Reads the model config from the parent directory to discover the
    ``model_package``, then delegates to the package's ``load_model``.

    Args:
        path: Path to the saved model file (e.g. ``model.pt``).

    Returns:
        A tuple of ``(model, metadata)``.
    """
    model_dir = path.parent
    model_package = read_model_package(model_dir)
    persistence = _import_persistence_module(model_package)
    result: tuple[TankModelBase, dict[str, Any]] = persistence.load_model(path)
    return result


def save_model(
    model: TankModelBase,
    path: Path,
    reward_config: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model using dynamic package dispatch.

    Uses ``model.config.model_package`` to discover the correct
    persistence module.

    Args:
        model: The model to save.
        path: Destination file path.
        reward_config: Optional reward configuration.
        metadata: Optional metadata dictionary.
    """
    persistence = _import_persistence_module(model.config.model_package)
    persistence.save_model(model, path, reward_config=reward_config, metadata=metadata)


def load_or_create_model(model_dir: Path) -> TankModelBase:
    """Load an existing model or create a fresh one from config.

    Reads ``model_config.json`` (must exist) and the ``model_package``
    field to determine which package handles the model.  If ``model.pt``
    exists, loads trained weights; otherwise creates a new model from
    the configuration.

    Args:
        model_dir: Directory containing ``model_config.json`` and
            optionally ``model.pt``.

    Returns:
        A :class:`TankModelBase` instance (loaded or freshly initialised).
    """
    model_package = read_model_package(model_dir)
    persistence = _import_persistence_module(model_package)

    model_path = model_dir / "model.pt"
    if model_path.exists():
        model: TankModelBase
        model, _metadata = persistence.load_model(model_path)
        return model

    config = persistence.load_model_config(model_dir)
    result: TankModelBase = persistence.create_model(config)
    return result


def create_player(
    model_package: str,
    team: str,
    model: TankModelBase,
    mode: str,
) -> Any:
    """Create a player instance using dynamic package dispatch.

    Each model package's persistence module must expose a
    ``create_player(team, model, mode)`` function that returns an
    :class:`~hmls.nncore.player.NNPlayerBase` subclass.

    Args:
        model_package: Fully-qualified model package name.
        team: The team this player controls.
        model: The tank model to use.
        mode: ``"play"`` or ``"learn"``.

    Returns:
        An :class:`~hmls.nncore.player.NNPlayerBase` instance.
    """
    persistence = _import_persistence_module(model_package)
    return persistence.create_player(team=team, model=model, mode=mode)
