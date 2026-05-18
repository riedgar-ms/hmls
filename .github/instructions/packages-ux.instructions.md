---
applyTo: "packages/hmls-uxcommon/**,packages/hmls-mapgenerator/**,packages/hmls-testharness/**,packages/hmls-replayviewer/**"
---

# UX Packages (Textual TUI)

## Package Roles

| Package | Purpose |
|---------|---------|
| `hmls-uxcommon` | Shared Textual widgets (map rendering, status panels) |
| `hmls-mapgenerator` | Interactive map creation TUI |
| `hmls-testharness` | Local two-player game (shows full map + per-tank fog views) |
| `hmls-replayviewer` | Replay previously saved game histories |

## Conventions

- All TUI apps use the [Textual](https://textual.textualize.io/) framework.
- Shared widgets live in `hmls-uxcommon`; app-specific screens live in their own package.
- Map rendering should use `hmls-uxcommon` widgets for consistency.
- Textual CSS is co-located with the Python widget files.

## Running

```shell
uv run hmls-mapgenerator
uv run hmls-testharness path/to/map.json 3
uv run hmls-replayviewer path/to/history.json
```
