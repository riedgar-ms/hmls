"""Generic model persistence with dynamic package dispatch.

Provides model-agnostic save/load infrastructure that discovers the
correct concrete persistence implementation at runtime by reading the
``model_package`` field from ``model_config.json``.

Architecture
~~~~~~~~~~~~

:class:`ModelPersistence`
    Abstract base class defining the full persistence + factory contract
    that every model package must satisfy.

:class:`NNPlayerModelPersistence`
    A generic concrete implementation for models that use the standard
    torch checkpoint format (``model.pt``), JSON configuration files,
    and :class:`~hmls.nncore.player.NNPlayer` instances.  Parameterised
    by the concrete config and model classes.

Each model package (e.g. ``hmls.singlemki``) must expose a
``persistence`` submodule with a ``PERSISTENCE`` attribute that is
an instance of :class:`ModelPersistence`.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import torch

from hmls.nncore.model import TankModelBase, TankModelConfig
from hmls.nncore.reward import RewardConfig

MODEL_CONFIG_FILENAME = "model_config.json"

ModelT = TypeVar("ModelT", bound=TankModelBase)
ConfigT = TypeVar("ConfigT", bound=TankModelConfig)


# ── Persistence ABC ──────────────────────────────────────────────────


class ModelPersistence(ABC, Generic[ConfigT, ModelT]):
    """Abstract base class for model persistence and factory operations.

    Every model package must provide a concrete implementation of this
    class and expose it as ``PERSISTENCE`` in its ``persistence``
    submodule.  The dynamic dispatch functions in this module
    (:func:`load_model`, :func:`save_model`, etc.) look up the
    ``PERSISTENCE`` attribute and delegate to its methods.

    Type parameters:
        ConfigT: The concrete :class:`TankModelConfig` subclass.
        ModelT: The concrete :class:`TankModelBase` subclass.
    """

    @abstractmethod
    def save_model(
        self,
        model: ModelT,
        path: Path,
        reward_config: RewardConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a trained model to disk.

        Args:
            model: The model to save.
            path: Destination file path (typically ``.pt`` extension).
            reward_config: Optional reward configuration to store as
                **informational metadata only**.  This records which
                reward shaping was in effect when the weights were
                produced, but it is never read back or used by any
                trainer on reload.  Trainers always derive their active
                reward configuration from their own run config (e.g.
                :class:`~hmls.reinforcetrainer.config.TrainerConfig`).
            metadata: Optional dictionary of extra information.
        """
        ...

    @abstractmethod
    def load_model(self, path: Path) -> tuple[ModelT, dict[str, Any]]:
        """Load a model from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            A tuple of ``(model, metadata)`` where *metadata* is the
            dict stored at save time (empty dict if none was provided).
            If a ``reward_config`` was saved alongside the weights, it
            is included in the metadata dict under the key
            ``"reward_config"`` for **informational/auditing purposes
            only** — trainers never consume it from here.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        ...

    @abstractmethod
    def save_model_config(self, config: ConfigT, directory: Path) -> None:
        """Save a model configuration to a directory.

        Args:
            config: The model configuration to save.
            directory: Target directory (created if it does not exist).
        """
        ...

    @abstractmethod
    def load_model_config(self, directory: Path) -> ConfigT:
        """Load a model configuration from a directory.

        Args:
            directory: Directory containing the config file.

        Returns:
            The loaded config instance.

        Raises:
            FileNotFoundError: If the config file is missing.
        """
        ...

    @abstractmethod
    def create_model(self, config: ConfigT) -> ModelT:
        """Create a new model instance from configuration.

        Args:
            config: The model configuration.

        Returns:
            A freshly initialised model.
        """
        ...

    @abstractmethod
    def create_player(
        self,
        team: str,
        model: ModelT,
        mode: Literal["play", "learn"],
    ) -> Any:
        """Create a player instance for the given model.

        Args:
            team: The team this player controls.
            model: The model to use for action selection.
            mode: ``"play"`` for deterministic inference, ``"learn"``
                for stochastic sampling with trajectory recording.

        Returns:
            A player instance (typically an
            :class:`~hmls.nncore.player.NNPlayerBase` subclass).
        """
        ...


# ── NNPlayer-based concrete implementation ────────────────────────────


class NNPlayerModelPersistence(ModelPersistence[ConfigT, ModelT]):
    """Persistence implementation for NNPlayer-based models.

    Handles the common pattern shared by all models that use:

    - ``model.pt`` (torch checkpoint with state_dict + config dict)
    - ``model_config.json`` (Pydantic model config as JSON)
    - :class:`~hmls.nncore.player.NNPlayer` for game play

    To use, simply instantiate with the concrete config and model
    classes::

        PERSISTENCE = NNPlayerModelPersistence(MkIModelConfig, MkITankPolicyNetwork)

    For models that need a custom player subclass, pass a
    ``player_factory``::

        PERSISTENCE = NNPlayerModelPersistence(
            StubModelConfig,
            StubTankModel,
            player_factory=lambda team, model, mode: StubNNPlayer(
                team=team, model=model, mode=mode,
            ),
        )

    Args:
        config_cls: The concrete config class (must subclass
            :class:`TankModelConfig`).
        model_cls: The concrete model class (must subclass
            :class:`TankModelBase`).
        player_factory: Optional callable ``(team, model, mode) ->
            NNPlayerBase``.  Defaults to creating an
            :class:`~hmls.nncore.player.NNPlayer`.
    """

    def __init__(
        self,
        config_cls: type[ConfigT],
        model_cls: type[ModelT],
        player_factory: Any | None = None,
    ) -> None:
        self._config_cls = config_cls
        self._model_cls = model_cls
        self._player_factory = player_factory

    # ── Model save / load ─────────────────────────────────────────────

    def save_model(
        self,
        model: ModelT,
        path: Path,
        reward_config: RewardConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a trained model as a torch checkpoint.

        The checkpoint contains ``state_dict``, ``config``, and
        optionally ``reward_config`` and ``metadata``.

        Note: ``reward_config`` is stored as a provenance record only —
        it documents which reward shaping produced these weights.  No
        trainer reads it back from the checkpoint; the active reward
        configuration always comes from the trainer's own run config.
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

    def load_model(self, path: Path) -> tuple[ModelT, dict[str, Any]]:
        """Load a model from a torch checkpoint.

        Reconstructs the model from the saved config dict and loads
        trained weights.  If a ``reward_config`` was saved, it is
        rehydrated into a :class:`BasicRewardConfig` and placed in
        ``metadata["reward_config"]`` for provenance/auditing purposes.
        Trainers do **not** consume this value — they always use the
        reward configuration from their own run config.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data: dict[str, Any] = torch.load(path, weights_only=False)

        config = self._config_cls.model_validate(save_data["config"])
        model = self._model_cls(config)
        model.load_state_dict(save_data["state_dict"])

        metadata: dict[str, Any] = save_data.get("metadata", {})
        if "reward_config" in save_data:
            metadata["reward_config"] = RewardConfig.model_validate(save_data["reward_config"])

        return model, metadata

    # ── Config file save / load ───────────────────────────────────────

    def save_model_config(self, config: ConfigT, directory: Path) -> None:
        """Save a model config as ``model_config.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MODEL_CONFIG_FILENAME
        path.write_text(config.model_dump_json(indent=2))

    def load_model_config(self, directory: Path) -> ConfigT:
        """Load a model config from ``model_config.json``."""
        path = directory / MODEL_CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found: {path}. "
                f"Each model directory must contain a "
                f"'{MODEL_CONFIG_FILENAME}'."
            )
        return self._config_cls.model_validate_json(path.read_text())

    # ── Factory methods ───────────────────────────────────────────────

    def create_model(self, config: ConfigT) -> ModelT:
        """Instantiate a new model from configuration."""
        return self._model_cls(config)

    def create_player(
        self,
        team: str,
        model: ModelT,
        mode: Literal["play", "learn"],
    ) -> Any:
        """Create an NNPlayer (or custom player) for the given model."""
        if self._player_factory is not None:
            return self._player_factory(team, model, mode)

        from hmls.nncore.player import NNPlayer

        return NNPlayer(team=team, model=model, mode=mode)


# ── Dynamic dispatch helpers ──────────────────────────────────────────


def _get_persistence(model_package: str) -> ModelPersistence[Any, Any]:
    """Look up the ``PERSISTENCE`` instance for a model package.

    Imports ``{model_package}.persistence`` and returns its
    ``PERSISTENCE`` attribute, which must be a :class:`ModelPersistence`
    instance.

    Args:
        model_package: Fully-qualified package name
            (e.g. ``"hmls.singlemki"``).

    Returns:
        The model package's persistence instance.

    Raises:
        ModuleNotFoundError: If the package or its ``persistence``
            submodule cannot be imported.
        AttributeError: If the module lacks a ``PERSISTENCE`` attribute.
    """
    module_name = f"{model_package}.persistence"
    module = importlib.import_module(module_name)
    persistence: ModelPersistence[Any, Any] = module.PERSISTENCE
    return persistence


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


# ── Public dispatch functions ─────────────────────────────────────────
#
# These are the primary API for callers (e.g. the training loop) that
# need to work with models without knowing the concrete type.  They
# discover the correct ModelPersistence instance via model_package and
# delegate.


def load_model_config(directory: Path) -> TankModelConfig:
    """Load a model config using dynamic package dispatch.

    Reads ``model_config.json``, discovers the ``model_package``,
    imports the correct persistence instance, and delegates to its
    :meth:`~ModelPersistence.load_model_config` method.

    Args:
        directory: Directory containing ``model_config.json``.

    Returns:
        A concrete :class:`TankModelConfig` subclass instance.
    """
    model_package = read_model_package(directory)
    persistence = _get_persistence(model_package)
    config: TankModelConfig = persistence.load_model_config(directory)
    return config


def load_model(path: Path) -> tuple[TankModelBase, dict[str, Any]]:
    """Load a model using dynamic package dispatch.

    Reads the model config from the parent directory to discover the
    ``model_package``, then delegates to the package's persistence
    instance.

    Args:
        path: Path to the saved model file (e.g. ``model.pt``).

    Returns:
        A tuple of ``(model, metadata)``.
    """
    model_dir = path.parent
    model_package = read_model_package(model_dir)
    persistence = _get_persistence(model_package)
    return persistence.load_model(path)


def save_model(
    model: TankModelBase,
    path: Path,
    reward_config: RewardConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a model using dynamic package dispatch.

    Uses ``model.config.model_package`` to discover the correct
    persistence instance.

    Args:
        model: The model to save.
        path: Destination file path.
        reward_config: Optional reward configuration.
        metadata: Optional metadata dictionary.
    """
    persistence = _get_persistence(model.config.model_package)
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
    persistence = _get_persistence(model_package)

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
    mode: Literal["play", "learn"],
) -> Any:
    """Create a player instance using dynamic package dispatch.

    Each model package's ``PERSISTENCE`` instance must implement
    :meth:`~ModelPersistence.create_player`.

    Args:
        model_package: Fully-qualified model package name.
        team: The team this player controls.
        model: The tank model to use.
        mode: ``"play"`` or ``"learn"``.

    Returns:
        An :class:`~hmls.nncore.player.NNPlayerBase` instance.
    """
    persistence = _get_persistence(model_package)
    return persistence.create_player(team=team, model=model, mode=mode)
