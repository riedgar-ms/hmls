"""Abstract base class for squad model persistence.

Defines :class:`SquadPersistenceBase`, the contract that squad packages
must satisfy to participate in the ``hmls.squads`` entry-point registry.
A squad persistence implementation handles saving and loading a paired
planner + executor model from a single squad directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from hmls.nncore.squad.executor_base import ExecutorModelBase
from hmls.nncore.squad.planner_base import PlannerModelBase


class SquadPersistenceBase(ABC):
    """Abstract base class for squad model persistence operations.

    Every squad package must provide a concrete implementation and
    expose it as ``PERSISTENCE``, registered via an entry point under
    the ``hmls.squads`` group in its ``pyproject.toml``::

        [project.entry-points."hmls.squads"]
        simplesquad = "hmls.simplesquadplayer.persistence:PERSISTENCE"

    A squad directory has the following layout::

        squad_dir/
        ├── planner/
        │   ├── model_config.json
        │   └── model.pt
        └── executor/
            ├── model_config.json
            └── model.pt
    """

    @abstractmethod
    def save_squad(
        self,
        planner: PlannerModelBase,
        executor: ExecutorModelBase,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a planner + executor pair to a squad directory.

        Args:
            planner: The planner model to save.
            executor: The executor model to save.
            path: Root directory of the squad.
            metadata: Optional extra metadata to store.
        """
        ...

    @abstractmethod
    def load_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase, dict[str, Any]]:
        """Load a planner + executor pair from a squad directory.

        Args:
            path: Root directory of the squad.

        Returns:
            A tuple of ``(planner, executor, metadata)``.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        ...

    @abstractmethod
    def create_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase]:
        """Create a new squad from config files and save initial weights.

        Reads ``model_config.json`` from the planner and executor
        subdirectories, instantiates fresh models with random weights,
        and saves them.

        Args:
            path: Root directory of the squad (must contain
                ``planner/model_config.json`` and
                ``executor/model_config.json``).

        Returns:
            A tuple of ``(planner, executor)`` with freshly initialised
            weights.

        Raises:
            FileNotFoundError: If config files are missing.
        """
        ...

    @abstractmethod
    def load_or_create_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase]:
        """Load existing squad weights or create fresh ones.

        If ``model.pt`` files exist, loads them.  Otherwise creates
        fresh models from the config files and saves initial weights.

        Args:
            path: Root directory of the squad.

        Returns:
            A tuple of ``(planner, executor)``.
        """
        ...

    @abstractmethod
    def create_player(
        self,
        path: Path,
        team: str,
        mode: Literal["play", "learn"] = "play",
    ) -> Any:  # noqa: ANN401
        """Create a player instance from a saved squad directory.

        Args:
            path: Root directory of the squad.
            team: Team identifier for the player.
            mode: Operating mode.

        Returns:
            A :class:`~hmls.core.player.Player` instance.
        """
        ...
