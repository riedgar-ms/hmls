# Copilot Instructions

This repository contains assorted experiments in AI. It is Python-oriented (see `.gitignore`).

## Project structure

This is an early-stage experimental repo. Keep experiments self-contained in their own directories with their own `pyproject.toml`.

## Conventions

- **Type annotations**: All function signatures must be fully type-annotated (parameters and return types). Variables with non-obvious types should also be annotated.
- **Documentation**: All modules, classes, and public functions must have docstrings. Prefer clarity over brevity.
- **Code clarity over micro-optimisation**: Prioritise readable, well-documented code. Always choose efficient algorithms, but do not sacrifice clarity for low-level performance tricks.
- **Serialisation**: Use Pydantic models when serialisation/deserialisation is needed.
- **File paths**: Prefer `pathlib.Path` over `os.path`.
- **Tooling**: Use `uv` for project and dependency management. Use `pyproject.toml` (not `requirements.txt` or `setup.py`). When providing commands to run code, use `uv run` (e.g., `uv run python script.py`, `uv run mypy .`) rather than plain `python` invocations.

## Git

- **Do not commit** unless the user has explicitly asked for a commit.

## Linting and type checking

All code must pass the following before merging:

```shell
ruff format --check .
ruff check .
mypy .
```

To auto-fix formatting: `ruff format .`
