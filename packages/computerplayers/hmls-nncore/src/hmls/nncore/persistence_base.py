"""Abstract base class for model persistence.

Defines :class:`ModelPersistence`, the contract that every model package
must satisfy to participate in the entry-point-based registry
(:mod:`hmls.nncore.registry`).  Extracted into its own module so that
both ``persistence`` and ``registry`` can import it without circular
dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from hmls.nncore.model import TankModelBase, TankModelConfig

if TYPE_CHECKING:
    from hmls.nncore.player_base import NNPlayerBase
    from hmls.nncore.reward_config import RewardConfig

MODEL_CONFIG_FILENAME = "model_config.json"


class ModelPersistence[ConfigT: TankModelConfig, ModelT: TankModelBase](ABC):
    """Abstract base class for model persistence and factory operations.

    Every model package must provide a concrete implementation of this
    class and expose it as ``PERSISTENCE``, either in a ``persistence``
    submodule or via an entry point registered under the ``hmls.models``
    group.  The registry-based dispatch functions in
    :mod:`hmls.nncore.persistence` look up the ``PERSISTENCE`` instance
    and delegate to its methods.

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
    ) -> NNPlayerBase:
        """Create a player instance for the given model.

        Args:
            team: The team this player controls.
            model: The model to use for action selection.
            mode: ``"play"`` for deterministic inference, ``"learn"``
                for stochastic sampling with trajectory recording.

        Returns:
            An :class:`~hmls.nncore.player.NNPlayerBase` instance.
        """
        ...
