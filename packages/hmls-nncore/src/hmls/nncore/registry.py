"""Entry-point-based model package registry.

Discovers model packages registered under the ``hmls.models`` entry-point
group and provides validated lookup and listing functions.  This replaces
raw ``importlib.import_module()`` dispatch with a structured registry that
gives clear error messages and supports both short entry-point names and
full Python import paths.

Third-party packages can register themselves by declaring an entry point
in their ``pyproject.toml``::

    [project.entry-points."hmls.models"]
    my_model = "my_package.persistence:PERSISTENCE"

Once installed, they are automatically discoverable by the training
infrastructure without any changes to this repository.
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hmls.nncore.persistence import ModelPersistence

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "hmls.models"


class ModelRegistryError(Exception):
    """Raised when a model package cannot be found or is invalid."""


def discover_models() -> dict[str, ModelPersistence[Any, Any]]:
    """Discover all registered model packages via entry points.

    Scans installed packages for entry points in the ``hmls.models``
    group and loads each one, validating that it provides a
    :class:`~hmls.nncore.persistence.ModelPersistence` instance.

    Returns:
        Dict mapping short entry-point name to validated
        ``ModelPersistence`` instance.

    Raises:
        ModelRegistryError: If duplicate entry-point names are detected
            among installed packages.
    """
    # Lazy import: circular dependency with persistence.py
    from hmls.nncore.persistence import ModelPersistence

    eps = entry_points(group=ENTRY_POINT_GROUP)
    registry: dict[str, ModelPersistence[Any, Any]] = {}

    # Check for duplicate names before loading
    names = [ep.name for ep in eps]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            msg = (
                f"Duplicate model entry-point name '{name}' — "
                f"two installed packages register the same name under "
                f"the '{ENTRY_POINT_GROUP}' group. Uninstall one of the "
                f"conflicting packages to resolve this."
            )
            raise ModelRegistryError(msg)
        seen.add(name)

    for ep in eps:
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load entry point '%s' (%s): %s",
                ep.name,
                ep.value,
                exc,
            )
            continue

        if not isinstance(obj, ModelPersistence):
            logger.warning(
                "Entry point '%s' does not provide a ModelPersistence "
                "instance (got %s from %s); skipping.",
                ep.name,
                type(obj).__name__,
                ep.value,
            )
            continue
        registry[ep.name] = obj

    return registry


def list_available_models() -> dict[str, ModelPersistence[Any, Any]]:
    """Return all installed model packages.

    This is useful for CLI help text, validation error messages, and
    testing that all registered models provide valid persistence
    instances.

    Returns:
        Dict mapping short name to ``ModelPersistence`` instance.
    """
    return discover_models()


def resolve_model_id(model_id: str) -> ModelPersistence[Any, Any]:
    """Resolve a ``model_id`` string to a ``ModelPersistence`` instance.

    Resolution order:

    1. **Short name match**: Try ``model_id`` as an entry-point name
       (e.g. ``"singlemki"``).
    2. **Module-path match**: If ``model_id`` looks like a dotted
       import path (e.g. ``"hmls.singlemki"``), check whether any
       entry-point's value starts with that path.
    3. **Fallback import**: Use ``importlib.import_module()`` to load
       ``{model_id}.persistence`` directly — for backwards
       compatibility with unregistered or development-only packages.

    Args:
        model_id: A short entry-point name (e.g. ``"singlemki"``)
            or a full Python package path (e.g. ``"hmls.singlemki"``).

    Returns:
        A validated :class:`~hmls.nncore.persistence.ModelPersistence`
        instance.

    Raises:
        ModelRegistryError: If the model cannot be resolved by any
            method, or if the resolved object is not a valid
            ``ModelPersistence`` instance.
    """
    # Lazy import: circular dependency with persistence.py
    from hmls.nncore.persistence import ModelPersistence

    # Step 1: Try as a short entry-point name
    registry = discover_models()
    if model_id in registry:
        return registry[model_id]

    # Step 2: Try as a full module path — match against entry-point values
    for name, persistence in registry.items():
        ep_module = _get_entry_point_module_path(name)
        if ep_module is not None and ep_module.startswith(model_id + "."):
            logger.debug(
                "Resolved '%s' via entry-point '%s' (module: %s)",
                model_id,
                name,
                ep_module,
            )
            return persistence

    # Step 3: Fallback to direct import
    module_name = f"{model_id}.persistence"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        available = sorted(registry.keys())
        available_str = ", ".join(available) if available else "(none)"
        msg = (
            f"Model '{model_id}' could not be resolved. "
            f"It is not a registered entry-point name, and the module "
            f"'{module_name}' could not be imported.\n"
            f"Available registered models: {available_str}\n"
            f"Ensure the model is installed in the current environment "
            f"and declares an entry point under the "
            f"'{ENTRY_POINT_GROUP}' group, or verify the 'model_id' "
            f"field in model_config.json is correct."
        )
        raise ModelRegistryError(msg) from None

    if not hasattr(module, "PERSISTENCE"):
        msg = (
            f"Module '{module_name}' was imported successfully but does "
            f"not expose a 'PERSISTENCE' attribute. Every model package "
            f"must define: PERSISTENCE = NNPlayerModelPersistence(...)"
        )
        raise ModelRegistryError(msg)

    obj = module.PERSISTENCE
    if not isinstance(obj, ModelPersistence):
        msg = (
            f"'{module_name}.PERSISTENCE' is not a ModelPersistence "
            f"instance (got {type(obj).__name__}). It must be an "
            f"instance of hmls.nncore.persistence.ModelPersistence."
        )
        raise ModelRegistryError(msg)

    return obj


def _get_entry_point_module_path(name: str) -> str | None:
    """Get the module path for a registered entry-point name.

    Returns the dotted module path (without the attribute part) for
    the named entry point, or ``None`` if not found.
    """
    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name == name:
            # Entry point value format: "package.module:attribute"
            module_path = ep.value.split(":")[0]
            return module_path
    return None
