"""Textual TUI for the map generator.

This module provides an interactive terminal UI for configuring and
generating randomised grid maps.  The sidebar offers controls for
grid dimensions, impassable fraction, seed, obstacle connectivity,
and strategy selection with dynamically-generated parameter widgets.

Run with::

    uv run hmls-mapgenerator
    # or
    uv run python -m hmls.mapgenerator
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, Switch

# Ensure all strategies are registered by importing the generators subpackage
from hmls.core import CellType, GameMap
from hmls.mapgenerator.generators import (
    STRATEGY_REGISTRY,
    BlobAndLineStrategy,
    MapStrategy,
    generate_map,
)

# Each grid cell is rendered as two characters wide ("██") and one row tall.
_CELL_WIDTH_CHARS = 2

logger = logging.getLogger("hmls.mapgenerator")


# ── Map display widget ───────────────────────────────────────────────


class MapDisplay(Static):
    """Widget that renders a :class:`~hmls.core.GameMap` as colour-coded cells."""

    def __init__(
        self,
        renderable: str = "",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(renderable, id=id)
        self._game_map: GameMap | None = None

    def update_map(self, game_map: GameMap) -> None:
        """Render a :class:`~hmls.core.GameMap` as Rich Text with coloured blocks.

        Args:
            game_map: The map to render.
        """
        self._game_map = game_map
        text = Text()
        for y in range(game_map.height):
            for x in range(game_map.width):
                if game_map[x, y] == CellType.PASSABLE:
                    text.append("██", style="green")
                else:
                    text.append("██", style="rgb(80,80,80)")
            text.append("\n")

        self.styles.min_width = game_map.width * _CELL_WIDTH_CHARS
        self.update(text)


# ── Save dialog ──────────────────────────────────────────────────────


class SaveDialog(ModalScreen[str | None]):
    """Modal dialog that prompts for a file path and returns it on confirm.

    Dismisses with the path string on Save, or ``None`` on Cancel / Escape.
    """

    CSS = """
    SaveDialog {
        align: center middle;
    }

    #save-dialog {
        width: 80;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #save-dialog Label {
        margin-bottom: 1;
    }

    #save-dialog Input {
        margin-bottom: 1;
    }

    #save-buttons {
        height: 3;
        align: right middle;
    }

    #save-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, default_path: str = "map.json") -> None:
        super().__init__()
        self._default_path = default_path

    def compose(self) -> ComposeResult:
        """Build the save dialog layout."""
        with Vertical(id="save-dialog"):
            yield Label("Save map as JSON (full or relative path)")
            yield Input(
                value=self._default_path,
                id="save-path-input",
            )
            with Horizontal(id="save-buttons"):
                yield Button("Save", id="save-confirm", variant="primary")
                yield Button("Cancel", id="save-cancel")

    def on_mount(self) -> None:
        """Focus the path input when the dialog opens."""
        self.query_one("#save-path-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Save / Cancel button clicks."""
        if event.button.id == "save-confirm":
            path = self.query_one("#save-path-input", Input).value.strip()
            self.dismiss(path or None)
        elif event.button.id == "save-cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Allow Enter in the input to confirm the save."""
        if event.input.id == "save-path-input":
            path = event.value.strip()
            self.dismiss(path or None)

    def action_cancel(self) -> None:
        """Dismiss on Escape."""
        self.dismiss(None)


# ── Main application ─────────────────────────────────────────────────


class MapGeneratorApp(App[None]):
    """Textual application for interactive map generation.

    Use the sidebar to configure parameters and press G (or click
    the Generate button) to create a new randomised map.
    """

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 36;
        min-width: 36;
        padding: 1 2;
        background: $surface;
        border-right: thick $primary;
    }

    #sidebar Label {
        margin-top: 1;
        color: $text-muted;
    }

    #sidebar Input {
        margin-bottom: 0;
    }

    #switch-row {
        height: 3;
        align: left middle;
    }

    #switch-row Label {
        margin-right: 1;
    }

    #generate-btn {
        margin-top: 2;
        width: 100%;
    }

    #strategy-params {
        height: auto;
    }

    #strategy-params Label {
        margin-top: 1;
        color: $text-muted;
    }

    #map-area {
        padding: 1 2;
    }

    #stats {
        dock: bottom;
        height: 3;
        padding: 0 2;
        background: $surface;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("g", "generate", "Generate"),
        ("s", "save", "Save"),
        ("q", "quit", "Quit"),
    ]

    TITLE = "Map Generator"

    def __init__(self) -> None:
        super().__init__()
        self._game_map: GameMap | None = None

    # ── Layout ────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        """Build the two-pane layout: sidebar | map."""
        yield Header()
        with Horizontal():
            with ScrollableContainer(id="sidebar"):
                yield Label("Width")
                yield Input(value="40", id="width", type="integer")
                yield Label("Height")
                yield Input(value="25", id="height", type="integer")
                yield Label("Impassable fraction (0.0–1.0)")
                yield Input(value="0.35", id="fraction")
                yield Label("Seed (blank=random)")
                yield Input(value="", id="seed", placeholder="random")
                with Horizontal(id="switch-row"):
                    yield Label("Connected obstacles")
                    yield Switch(id="connected", value=False)
                yield Label("Strategy")
                strategy_options = [(name, name) for name in STRATEGY_REGISTRY]
                yield Select(
                    strategy_options,
                    value=next(iter(STRATEGY_REGISTRY)),
                    id="strategy-select",
                    allow_blank=False,
                )
                # Dynamic strategy-specific parameter widgets
                default_cls = STRATEGY_REGISTRY[next(iter(STRATEGY_REGISTRY))]
                default_params = default_cls.get_params()
                with Vertical(id="strategy-params"):
                    for param in default_params:
                        yield Label(param.label)
                        yield Input(
                            value=str(param.default),
                            id=f"sp-{param.name}",
                        )
                yield Button("Generate (G)", id="generate-btn", variant="primary")

            with ScrollableContainer(id="map-area"):
                yield MapDisplay(
                    "Press Generate (G) to create a map.",
                    id="map-display",
                )

        yield Static("Ready — press G or click Generate", id="stats")
        yield Footer()

    # ── Strategy parameter widgets ────────────────────────────────

    async def _rebuild_strategy_params(self) -> None:
        """Rebuild the dynamic strategy parameter widgets."""
        container = self.query_one("#strategy-params", Vertical)

        strategy_name = self.query_one("#strategy-select", Select).value
        if strategy_name is Select.BLANK:
            await container.remove_children()
            return
        strategy_cls = STRATEGY_REGISTRY.get(str(strategy_name))
        if strategy_cls is None:
            await container.remove_children()
            return

        params = strategy_cls.get_params()

        await container.remove_children()

        for param in params:
            await container.mount(Label(param.label))
            await container.mount(Input(value=str(param.default), id=f"sp-{param.name}"))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Rebuild strategy parameter widgets when the selection changes."""
        if event.select.id == "strategy-select":
            self.run_worker(self._rebuild_strategy_params())

    # ── Parameter reading ─────────────────────────────────────────

    def _get_params(self) -> dict[str, object]:
        """Read parameter values from the sidebar input widgets."""
        width = int(self.query_one("#width", Input).value or "40")
        height = int(self.query_one("#height", Input).value or "25")
        fraction = float(self.query_one("#fraction", Input).value or "0.35")
        seed_str = self.query_one("#seed", Input).value.strip()
        seed = int(seed_str) if seed_str else None
        connected = self.query_one("#connected", Switch).value

        width = max(1, min(200, width))
        height = max(1, min(200, height))
        fraction = max(0.0, min(1.0, fraction))

        strategy = self._build_strategy()

        return {
            "width": width,
            "height": height,
            "impassable_fraction": fraction,
            "strategy": strategy,
            "seed": seed,
            "connected_obstacles": connected,
        }

    def _build_strategy(self) -> MapStrategy:
        """Construct the currently selected strategy with parameter values."""
        strategy_name = self.query_one("#strategy-select", Select).value
        strategy_cls = STRATEGY_REGISTRY.get(str(strategy_name), BlobAndLineStrategy)

        params = strategy_cls.get_params()
        kwargs: dict[str, float | int] = {}

        for param in params:
            try:
                widget = self.query_one(f"#sp-{param.name}", Input)
                raw_val = widget.value or str(param.default)
                val = param.param_type(raw_val)

                if param.min_val is not None:
                    val = max(param.min_val, val)
                if param.max_val is not None:
                    val = min(param.max_val, val)

                kwargs[param.name] = val
            except Exception:
                logger.debug(
                    "Failed to parse parameter '%s', using default %r",
                    param.name,
                    param.default,
                    exc_info=True,
                )
                kwargs[param.name] = param.default

        return strategy_cls(**kwargs)

    # ── Map generation ────────────────────────────────────────────

    def action_generate(self) -> None:
        """Generate a new map with current parameters (bound to 'G')."""
        self._do_generate()

    def action_save(self) -> None:
        """Open the save dialog (bound to 'S')."""
        if self._game_map is None:
            self.query_one("#stats", Static).update("Nothing to save — generate a map first")
            return
        self.push_screen(SaveDialog(), callback=self._on_save_dialog_result)

    def _on_save_dialog_result(self, path_str: str | None) -> None:
        """Handle the result from the save dialog."""
        status = self.query_one("#stats", Static)
        if path_str is None:
            status.update("Save cancelled")
            return
        if self._game_map is None:
            return

        path = Path(path_str)
        try:
            path.write_text(
                self._game_map.model_dump_json(indent=2),
                encoding="utf-8",
            )
            status.update(f"Saved to {path.resolve()}")
        except Exception as e:
            status.update(f"Save error: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "generate-btn":
            self._do_generate()

    def _do_generate(self) -> None:
        """Generate a new map, store it, and refresh the display."""
        try:
            params = self._get_params()
            game_map = generate_map(**params)  # type: ignore[arg-type]
            self._game_map = game_map

            display = self.query_one("#map-display", MapDisplay)
            display.update_map(game_map)

            total = game_map.total_cells
            imp = game_map.count_impassable()
            pct = imp / total * 100 if total > 0 else 0
            stats = (
                f"{game_map.width}×{game_map.height} | "
                f"Impassable: {imp}/{total} ({pct:.1f}%) | "
                f"Seed: {params['seed'] or 'random'}"
            )
            self.query_one("#stats", Static).update(stats)
        except Exception as e:
            self.query_one("#stats", Static).update(f"Error: {e}")


def main() -> None:
    """Entry point for the TUI application."""
    app = MapGeneratorApp()
    app.run()


if __name__ == "__main__":
    main()
