"""Entry-point-based squad package registry.

Discovers squad packages registered under the ``hmls.squads`` entry-point
group and provides validated lookup and listing functions.  This mirrors
the pattern used by :mod:`hmls.nncore.registry` for single-tank models.

Third-party squad packages can register themselves by declaring an entry
point in their ``pyproject.toml``::

    [project.entry-points."hmls.squads"]
    my_squad = "my_package.persistence:PERSISTENCE"

Once installed, they are automatically discoverable by the training
infrastructure and game runners without any code changes.
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import Any

from hmls.nncore.squad.persistence_base import SquadPersistenceBase

logger = logging.getLogger(__name__)

SQUAD_ENTRY_POINT_GROUP = "hmls.squads"


class SquadRegistryError(Exception):
    """Raised when a squad package cannot be found or is invalid."""


def discover_squads() -> dict[str, SquadPersistenceBase]:
    """Discover all registered squad packages via entry points.

    Scans installed packages for entry points in the ``hmls.squads``
    group and loads each one, validating that it provides a
    :class:`SquadPersistenceBase` instance.

    Returns:
        Dict mapping short entry-point name to validated
        ``SquadPersistenceBase`` instance.

    Raises:
        SquadRegistryError: If duplicate entry-point names are detected.
    """
    eps = entry_points(group=SQUAD_ENTRY_POINT_GROUP)
    registry: dict[str, SquadPersistenceBase] = {}

    # Check for duplicate names
    names = [ep.name for ep in eps]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            msg = (
                f"Duplicate squad entry-point name '{name}' — "
                f"two installed packages register the same name under "
                f"the '{SQUAD_ENTRY_POINT_GROUP}' group. Uninstall one of the "
                f"conflicting packages to resolve this."
            )
            raise SquadRegistryError(msg)
        seen.add(name)

    for ep in eps:
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load squad entry point '%s' (%s): %s",
                ep.name,
                ep.value,
                exc,
            )
            continue

        if not isinstance(obj, SquadPersistenceBase):
            logger.warning(
                "Squad entry point '%s' does not provide a SquadPersistenceBase "
                "instance (got %s from %s); skipping.",
                ep.name,
                type(obj).__name__,
                ep.value,
            )
            continue
        registry[ep.name] = obj

    return registry


def list_available_squads() -> dict[str, SquadPersistenceBase]:
    """Return all installed squad packages.

    Returns:
        Dict mapping short name to ``SquadPersistenceBase`` instance.
    """
    return discover_squads()


def resolve_squad_id(squad_id: str) -> SquadPersistenceBase:
    """Resolve a ``squad_id`` string to a ``SquadPersistenceBase`` instance.

    Resolution order:

    1. **Short name match**: Try ``squad_id`` as an entry-point name
       (e.g. ``"simplesquad"``).
    2. **Module-path match**: If ``squad_id`` looks like a dotted
       import path, check whether any entry-point's value starts with
       that path.
    3. **Fallback import**: Use ``importlib.import_module()`` to load
       ``{squad_id}.persistence`` directly.

    Args:
        squad_id: A short entry-point name (e.g. ``"simplesquad"``)
            or a full Python package path.

    Returns:
        A validated :class:`SquadPersistenceBase` instance.

    Raises:
        SquadRegistryError: If the squad cannot be resolved.
    """
    # Step 1: Try as a short entry-point name
    registry = discover_squads()
    if squad_id in registry:
        return registry[squad_id]

    # Step 2: Try as a module path
    for _name, persistence in registry.items():
        ep_module = _get_squad_entry_point_module_path(_name)
        if ep_module is not None and ep_module.startswith(squad_id + "."):
            return persistence

    # Step 3: Fallback to direct import
    module_name = f"{squad_id}.persistence"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        available = sorted(registry.keys())
        available_str = ", ".join(available) if available else "(none)"
        msg = (
            f"Squad '{squad_id}' could not be resolved. "
            f"It is not a registered entry-point name, and the module "
            f"'{module_name}' could not be imported.\n"
            f"Available registered squads: {available_str}\n"
            f"Ensure the squad package is installed and declares an entry "
            f"point under the '{SQUAD_ENTRY_POINT_GROUP}' group."
        )
        raise SquadRegistryError(msg) from None

    if not hasattr(module, "PERSISTENCE"):
        msg = (
            f"Module '{module_name}' was imported successfully but does "
            f"not expose a 'PERSISTENCE' attribute. Every squad package "
            f"must define: PERSISTENCE = <SquadPersistenceBase subclass>()"
        )
        raise SquadRegistryError(msg)

    obj: Any = module.PERSISTENCE
    if not isinstance(obj, SquadPersistenceBase):
        msg = (
            f"'{module_name}.PERSISTENCE' is not a SquadPersistenceBase "
            f"instance (got {type(obj).__name__})."
        )
        raise SquadRegistryError(msg)

    return obj


def _get_squad_entry_point_module_path(name: str) -> str | None:
    """Get the module path for a registered squad entry-point name."""
    eps = entry_points(group=SQUAD_ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name == name:
            module_path = ep.value.split(":")[0]
            return module_path
    return None
