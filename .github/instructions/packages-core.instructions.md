---
applyTo: "packages/hmls-core/**"
---

# hmls-core — Foundational Game Package

This is the foundational package upon which all others depend. Changes here
affect the entire codebase; be conservative but backwards compatibility is
not essential if code simplification is achieved.

## Key Types

| Module | Exports |
|--------|---------|
| `types.py` | `Direction`, `Position`, `Action` (enum of 5 actions) |
| `map.py` | `CellType`, `GameMap` (grid-based terrain) |
| `tank.py` | `Tank`, `TankId` |
| `game_state.py` | `GameState` (mutable game world) |
| `engine.py` | `GameEngine` (orchestrates a match), `GameResult`, `HistoryEntry` |
| `visibility.py` | `build_player_view` (fog-of-war computation) |
| `player.py` | `Player` protocol (interface for all tank controllers) |
| `actions.py` | `apply_action`, `validate_action` |
| `placement.py` | Tank placement logic |
| `cli_args.py` | Shared CLI argument parsing |

## Coordinate System

- x increases rightward, y increases downward.
- `Direction.NORTH` is (0, −1), `Direction.EAST` is (+1, 0).

## Testing

Tests are in `packages/hmls-core/tests/`. Run with:

```shell
uv run pytest packages/hmls-core/tests/
```
