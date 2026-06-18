"""Squad persistence: save/load planner + executor as a paired unit.

Provides :data:`PERSISTENCE`, a :class:`SimpleSquadPersistence`
instance registered via the ``hmls.squads`` entry-point group.

Squad directory layout::

    squad_dir/
    ├── planner/
    │   ├── model_config.json
    │   └── model.pt
    └── executor/
        ├── model_config.json
        └── model.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from hmls.nncore.squad.executor_base import ExecutorModelBase
from hmls.nncore.squad.persistence_base import SquadPersistenceBase
from hmls.nncore.squad.planner_base import PlannerModelBase
from hmls.simplesquadexecutor.model import SimpleExecutorModel
from hmls.simplesquadexecutor.persistence import (
    load_executor,
    load_or_create_executor,
    save_executor,
)
from hmls.simplesquadplanner.model import SimplePlannerModel
from hmls.simplesquadplanner.persistence import (
    load_or_create_planner,
    load_planner,
    save_planner,
)
from hmls.simplesquadplayer.player import SimpleSquadPlayer

PLANNER_SUBDIR = "planner"
EXECUTOR_SUBDIR = "executor"


class SimpleSquadPersistence(SquadPersistenceBase):
    """Concrete persistence for the simple squad (planner + executor pair).

    Manages a squad directory with ``planner/`` and ``executor/``
    subdirectories, each containing model config and weights.
    """

    def save_squad(
        self,
        planner: PlannerModelBase,
        executor: ExecutorModelBase,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save both models to the squad directory.

        Args:
            planner: The planner model to save.
            executor: The executor model to save.
            path: Root directory of the squad.
            metadata: Optional extra metadata.
        """
        if not isinstance(planner, SimplePlannerModel):
            msg = f"Expected SimplePlannerModel, got {type(planner).__name__}"
            raise TypeError(msg)
        if not isinstance(executor, SimpleExecutorModel):
            msg = f"Expected SimpleExecutorModel, got {type(executor).__name__}"
            raise TypeError(msg)

        save_planner(planner, path / PLANNER_SUBDIR, metadata=metadata)
        save_executor(executor, path / EXECUTOR_SUBDIR, metadata=metadata)

    def load_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase, dict[str, Any]]:
        """Load both models from the squad directory.

        Args:
            path: Root directory of the squad.

        Returns:
            A tuple of ``(planner, executor, metadata)``.
        """
        planner, planner_meta = load_planner(path / PLANNER_SUBDIR)
        executor, executor_meta = load_executor(path / EXECUTOR_SUBDIR)
        metadata: dict[str, Any] = {
            "planner": planner_meta,
            "executor": executor_meta,
        }
        return planner, executor, metadata

    def create_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase]:
        """Create fresh models from config files and save initial weights.

        Args:
            path: Root directory containing ``planner/model_config.json``
                and ``executor/model_config.json``.

        Returns:
            Freshly initialised ``(planner, executor)`` pair.
        """
        from hmls.simplesquadexecutor.persistence import create_executor
        from hmls.simplesquadplanner.persistence import create_planner

        planner = create_planner(path / PLANNER_SUBDIR)
        executor = create_executor(path / EXECUTOR_SUBDIR)
        return planner, executor

    def load_or_create_squad(
        self,
        path: Path,
    ) -> tuple[PlannerModelBase, ExecutorModelBase]:
        """Load existing weights or create fresh ones.

        Args:
            path: Root directory of the squad.

        Returns:
            ``(planner, executor)`` pair.
        """
        planner = load_or_create_planner(path / PLANNER_SUBDIR)
        executor = load_or_create_executor(path / EXECUTOR_SUBDIR)
        return planner, executor

    def create_player(
        self,
        path: Path,
        team: str,
        mode: Literal["play", "learn"] = "play",
    ) -> SimpleSquadPlayer:
        """Create a player from a saved squad directory.

        Args:
            path: Root directory of the squad.
            team: Team identifier.
            mode: Operating mode.

        Returns:
            A :class:`SimpleSquadPlayer` instance.
        """
        planner = load_or_create_planner(path / PLANNER_SUBDIR)
        executor = load_or_create_executor(path / EXECUTOR_SUBDIR)
        return SimpleSquadPlayer(
            team=team,
            planner=planner,
            executor=executor,
            mode=mode,
        )


PERSISTENCE = SimpleSquadPersistence()
"""Singleton persistence instance registered via the ``hmls.squads`` entry point."""

__all__ = ["PERSISTENCE", "SimpleSquadPersistence"]
