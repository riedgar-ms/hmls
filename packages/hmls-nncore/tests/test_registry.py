"""Tests for the entry-point-based model registry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hmls.nncore.persistence import ModelPersistence, NNPlayerModelPersistence
from hmls.nncore.registry import (
    ModelRegistryError,
    discover_models,
    list_available_models,
    resolve_model_package,
)

# ── Tests: discover_models ────────────────────────────────────────────


class TestDiscoverModels:
    """Tests for discover_models() entry-point scanning."""

    def test_discovers_installed_models(self) -> None:
        """Discover finds at least the models from this workspace."""
        registry = discover_models()
        # These should be registered via pyproject.toml entry points
        assert "singlemki" in registry
        assert "singlemkii" in registry
        assert "singlemkiii" in registry
        assert "randomtank" in registry

    def test_all_values_are_model_persistence(self) -> None:
        """Every discovered entry is a ModelPersistence instance."""
        registry = discover_models()
        for name, persistence in registry.items():
            assert isinstance(persistence, ModelPersistence), (
                f"Entry point '{name}' is {type(persistence).__name__}, expected ModelPersistence"
            )

    def test_duplicate_names_raises(self) -> None:
        """Duplicate entry-point names raise ModelRegistryError."""
        # Create mock entry points with duplicate names
        ep1 = MagicMock()
        ep1.name = "duplicate"
        ep1.value = "pkg1.persistence:PERSISTENCE"
        ep2 = MagicMock()
        ep2.name = "duplicate"
        ep2.value = "pkg2.persistence:PERSISTENCE"

        with patch(
            "hmls.nncore.registry.entry_points",
            return_value=[ep1, ep2],
        ):
            with pytest.raises(ModelRegistryError, match="Duplicate model entry-point name"):
                discover_models()

    def test_invalid_entry_point_skipped(self) -> None:
        """Entry points that don't provide ModelPersistence are skipped."""
        ep = MagicMock()
        ep.name = "bad_model"
        ep.value = "fake.module:NOT_PERSISTENCE"
        ep.load.return_value = "not a persistence object"

        with patch(
            "hmls.nncore.registry.entry_points",
            return_value=[ep],
        ):
            registry = discover_models()
            assert "bad_model" not in registry

    def test_entry_point_load_failure_skipped(self) -> None:
        """Entry points that fail to load are skipped with a warning."""
        ep = MagicMock()
        ep.name = "broken_model"
        ep.value = "broken.module:PERSISTENCE"
        ep.load.side_effect = ImportError("no such module")

        with patch(
            "hmls.nncore.registry.entry_points",
            return_value=[ep],
        ):
            registry = discover_models()
            assert "broken_model" not in registry


# ── Tests: list_available_models ──────────────────────────────────────


class TestListAvailableModels:
    """Tests for list_available_models()."""

    def test_returns_same_as_discover(self) -> None:
        """list_available_models returns the same result as discover."""
        assert list_available_models() == discover_models()


# ── Tests: resolve_model_package ──────────────────────────────────────


class TestResolveModelPackage:
    """Tests for resolve_model_package() resolution logic."""

    def test_resolve_short_name(self) -> None:
        """Short entry-point name resolves to correct persistence."""
        result = resolve_model_package("singlemki")
        assert isinstance(result, NNPlayerModelPersistence)

    def test_resolve_full_path(self) -> None:
        """Full module path resolves via entry-point module matching."""
        result = resolve_model_package("hmls.singlemki")
        assert isinstance(result, NNPlayerModelPersistence)

    def test_resolve_full_path_mkii(self) -> None:
        """Full path works for mkii as well."""
        result = resolve_model_package("hmls.singlemkii")
        assert isinstance(result, NNPlayerModelPersistence)

    def test_resolve_full_path_mkiii(self) -> None:
        """Full path works for mkiii."""
        result = resolve_model_package("hmls.singlemkiii")
        assert isinstance(result, NNPlayerModelPersistence)

    def test_resolve_full_path_randomtank(self) -> None:
        """Full path works for randomtank."""
        result = resolve_model_package("hmls.randomtank")
        assert isinstance(result, NNPlayerModelPersistence)

    def test_resolve_unknown_raises_with_suggestions(self) -> None:
        """Unknown model_package gives clear error listing available models."""
        with pytest.raises(ModelRegistryError, match="could not be resolved") as exc_info:
            resolve_model_package("hmls.nonexistent")
        # Error message should list available models
        assert "Available registered models" in str(exc_info.value)

    def test_resolve_fallback_to_import(self) -> None:
        """Unregistered packages fall back to importlib import.

        The reinforcetrainer._testing stub is registered via entry point,
        so we test the fallback by using the full module path which
        triggers Step 2 or Step 3 resolution.
        """
        # This resolves via entry-point module matching (step 2)
        result = resolve_model_package("hmls.reinforcetrainer._testing")
        assert isinstance(result, ModelPersistence)

    def test_resolve_missing_persistence_attribute(self) -> None:
        """Module without PERSISTENCE attribute gives clear error."""
        # Mock a module that exists but has no PERSISTENCE
        mock_module = MagicMock(spec=[])  # Empty spec = no attributes
        del mock_module.PERSISTENCE  # Ensure it truly doesn't exist

        with (
            patch(
                "hmls.nncore.registry.discover_models",
                return_value={},
            ),
            patch(
                "hmls.nncore.registry.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            with pytest.raises(ModelRegistryError, match="does not expose a 'PERSISTENCE'"):
                resolve_model_package("fake.package")

    def test_resolve_wrong_type_persistence(self) -> None:
        """Module with non-ModelPersistence PERSISTENCE gives clear error."""
        mock_module = MagicMock()
        mock_module.PERSISTENCE = "not a persistence instance"

        with (
            patch(
                "hmls.nncore.registry.discover_models",
                return_value={},
            ),
            patch(
                "hmls.nncore.registry.importlib.import_module",
                return_value=mock_module,
            ),
        ):
            with pytest.raises(ModelRegistryError, match="is not a ModelPersistence"):
                resolve_model_package("fake.package")
