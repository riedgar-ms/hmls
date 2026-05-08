"""Generic model persistence with dynamic package dispatch.

Provides model-agnostic save/load infrastructure that discovers the
correct concrete persistence module at runtime by reading the
``model_package`` field from ``model_config.json``.

Each model package (e.g. ``hmls.singlemki``, ``hmls.singlemkii``) must
expose a ``persistence`` submodule with the following functions:

- ``save_model(model, path, reward_config=None, metadata=None) -> None``
- ``load_model(path) -> tuple[TankModelBase, dict[str, Any]]``
- ``save_model_config(config, directory) -> None``
- ``load_model_config(directory) -> TankModelConfig``  (concrete subclass)
- ``save_reward_config(config, directory) -> None``
- ``load_reward_config(directory) -> DefaultRewardConfig``
- ``create_player(team, model, mode) -> NNPlayerBase``
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from hmls.nncore.model import TankModelBase, TankModelConfig

MODEL_CONFIG_FILENAME = "model_config.json"
REWARD_CONFIG_FILENAME = "reward_config.json"


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
