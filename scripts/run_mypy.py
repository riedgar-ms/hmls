"""Run mypy with auto-discovered source roots for all workspace packages.

Why this script exists
---------------------
mypy has no native monorepo/workspace mode. In a uv workspace where multiple
packages each have their own tests/ directory, running ``mypy .`` globally
fails with "Duplicate module named 'tests'" (and similarly for identically-named
test files like test_player.py across packages). This is a long-standing mypy
limitation (https://github.com/python/mypy/issues/4008, open since 2018) with
no upstream fix planned.

The standard workaround — used by projects like Apache Airflow — is to invoke
mypy separately per package. This script automates that:

1. Auto-discovers all packages under packages/ (no manual list to maintain).
2. Builds MYPYPATH from all packages' src/ directories so cross-package
   hmls.* imports resolve correctly.
3. Runs mypy once per package (src + tests together) to avoid name collisions.

Alternatives considered
-----------------------
- ``files = ["packages/*/src"]`` in pyproject.toml: The ``files`` option supports
  globs and works for source code, but cannot include test directories without
  hitting the duplicate-module problem.
- ``packages = ["hmls.core", ...]`` in pyproject.toml: Works for installed
  source packages, but cannot type-check test files (they aren't installed).
- **pyright**: Does not have this limitation and handles monorepos natively.
  It may be worth evaluating as a replacement or complement to mypy in future.

Usage
-----
    uv run python scripts/run_mypy.py [mypy-args...]

If no arguments are given, checks all packages. Pass specific paths or
mypy flags to customise behaviour.
"""

import os
import subprocess
import sys
from pathlib import Path


def discover_packages(workspace_root: Path) -> list[Path]:
    """Return sorted list of package directories under packages/."""
    packages_dir = workspace_root / "packages"
    return sorted(d for d in packages_dir.iterdir() if d.is_dir())


def build_mypy_path(package_dirs: list[Path]) -> str:
    """Build MYPYPATH from all workspace packages.

    Includes each package's src/ directory so that cross-package
    hmls.* imports resolve correctly.
    """
    paths: list[str] = []
    for pkg_dir in package_dirs:
        src_dir = pkg_dir / "src"
        if src_dir.is_dir():
            paths.append(str(src_dir))
    return os.pathsep.join(paths)


def main() -> int:
    """Run mypy per-package with auto-discovered MYPYPATH."""
    workspace_root = Path(__file__).resolve().parent.parent
    package_dirs = discover_packages(workspace_root)
    mypy_path = build_mypy_path(package_dirs)
    env = {**os.environ, "MYPYPATH": mypy_path}

    extra_args = sys.argv[1:]
    overall_returncode = 0

    if extra_args:
        # If user passes explicit args/paths, run mypy once with those
        result = subprocess.run(
            ["mypy", *extra_args],
            env=env,
            cwd=str(workspace_root),
        )
        return result.returncode

    # Default: run per-package to avoid duplicate module name conflicts
    # across packages (each package has its own tests/ directory).
    workspace_tests = workspace_root / "tests"
    targets: list[tuple[str, list[str]]] = []

    for pkg_dir in package_dirs:
        src_dir = pkg_dir / "src"
        tests_dir = pkg_dir / "tests"
        pkg_targets: list[str] = []
        if src_dir.is_dir():
            pkg_targets.append(str(src_dir))
        if tests_dir.is_dir():
            pkg_targets.append(str(tests_dir))
        if pkg_targets:
            targets.append((pkg_dir.name, pkg_targets))

    # Also include workspace-level tests
    if workspace_tests.is_dir():
        targets.append(("workspace tests", [str(workspace_tests)]))

    failed: list[str] = []

    for name, paths in targets:
        result = subprocess.run(
            ["mypy", "--explicit-package-bases", *paths],
            env=env,
            cwd=str(workspace_root),
        )
        if result.returncode != 0:
            failed.append(name)
            overall_returncode = 1

    if failed:
        print(f"\nmypy failed for: {', '.join(failed)}")  # noqa: T201
    else:
        print(f"\nmypy passed for all {len(targets)} targets")  # noqa: T201

    return overall_returncode


if __name__ == "__main__":
    sys.exit(main())
