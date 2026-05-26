# Copilot Instructions

A tank game and AI training framework, written in Python.

## Project structure

This is a `uv` workspace (see root `pyproject.toml`). All packages live under `packages/` and share a single lockfile. The namespace is `hmls.*` (e.g. `hmls.core`, `hmls.protocol`).

Key package groups:
- **Core**: `hmls-core` — game types, map, engine, visibility
- **Neural networks**: `hmls-nncore` (base classes), `hmls-singlemki/ii/iii` (architectures), `hmls-randomtank` (rule-based), `hmls-reinforcetrainer`
- **Networking**: `hmls-protocol`, `hmls-networking`, `hmls-server`, `hmls-client`, `hmls-observer`
- **UX (Textual TUI)**: `hmls-uxcommon`, `hmls-mapgenerator`, `hmls-testharness`, `hmls-replayviewer`

When adding a new package, register it in the root `pyproject.toml` under both `[dependency-groups] dev` and `[tool.uv.sources]`. Also add the package to the matrix list in `.github/workflows/coverage.yml`. The mypy configuration auto-discovers packages via `scripts/run_mypy.py`, so no manual path updates are needed.

Neural network tank packages provide 4 semantic components (config, model, player, persistence) and register via `[project.entry-points."hmls.models"]` in their `pyproject.toml`. In practice, the config class and model class live together in `model.py`, and the player is typically the generic `NNPlayer` from `hmls-nncore` (a custom `player.py` is only needed for non-standard action logic). See `docs/reinforcement_learning.md` for details.

## Conventions

- **Type annotations**: All function signatures must be fully type-annotated (parameters and return types). Variables with non-obvious types should also be annotated.
- **Documentation**: All modules, classes, and public functions must have docstrings. Prefer clarity over brevity.
- **Code clarity over micro-optimisation**: Prioritise readable, well-documented code. Always choose efficient algorithms, but do not sacrifice clarity for low-level performance tricks.
- **Serialisation**: Use Pydantic models when serialisation/deserialisation is needed. When a Pydantic class has both a Python docstring and Pydantic annotations (e.g. `name`, `description`), they must be kept consistent.
- **File paths**: Prefer `pathlib.Path` over `os.path`.
- **Tooling**: Use `uv` for project and dependency management. Use `pyproject.toml` (not `requirements.txt` or `setup.py`). When providing commands to run code, use `uv run` (e.g., `uv run python script.py`, `uv run pytest`) rather than plain `python` invocations.
- **TUI**: Use the `textual` package for all terminal user interface implementations.

## Git

- **Do not commit** unless the user has explicitly asked for a commit.
- **Do not amend, rebase, or rewrite git history** without explicit user permission.

## Linting and type checking

All code must pass the following before merging:

```shell
ruff format --check .
ruff check .
uv run python scripts/run_mypy.py
```

To auto-fix formatting: `ruff format .`

### Lint suppression patterns

When a ruff rule fires but the code is intentional, use inline `# noqa` comments rather than disabling rules globally. Follow these conventions:

- **EM101/EM102** (string/f-string in exception): If the `raise` statement (from `raise` to closing paren) is ≤60 characters, suppress with `# noqa: EM101` or `# noqa: EM102`. If longer, extract the message to a `msg` variable on the preceding line.
- **BLE001** (blind `except Exception`): Suppress with `# noqa: BLE001` only when the caught exception is logged or otherwise reported to the user. Never silently swallow exceptions.
- **ANN401** (`Any` type): Suppress with `# noqa: ANN401` only where `Any` is genuinely unavoidable (e.g. external library types, generic factory callables).

## Testing

Tests live in `packages/*/tests/` (per-package) and `tests/` (workspace-level). Run all tests:

```shell
uv run pytest
```

Run a specific package's tests:

```shell
uv run pytest packages/hmls-core/tests/
```

## Common commands

| Task | Command |
|------|---------|
| Install all packages | `uv sync --all-packages` |
| Format code | `uv run ruff format .` |
| Check formatting | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Type check | `uv run python scripts/run_mypy.py` |
| Run tests | `uv run pytest` |
| Run map generator | `uv run hmls-mapgenerator` |
| Run test harness | `uv run hmls-testharness <map.json> <tanks>` |
| Run replay viewer | `uv run hmls-replayviewer <history.json>` |
| Train models | `uv run hmls-reinforcetrainer <config.json>` |
| Start game server | `uv run hmls-server <map.json> <tanks>` |
| Connect client | `uv run hmls-client ws://localhost:8765/ws --name <Name>` |
