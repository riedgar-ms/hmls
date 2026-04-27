"""Player view region: fog-of-war patches for one team's tanks."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label

from hmls.core.game_state import GameState
from hmls.core.map import GameMap
from hmls.core.tank import TankId
from hmls.core.visibility import build_player_view
from hmls.testharness.widgets.patch_view import PatchView


class PlayerViewRegion(Horizontal):
    """Container showing fog-of-war patches for all alive tanks of one team.

    Patches are arranged horizontally.  The active tank's patch gets a
    highlighted border.

    Args:
        team: Team name (e.g. ``"A"`` or ``"B"``).
        game_map: The game map.
        state: Current game state.
        patch_size: Visibility patch side length.
        active_tank_id: ID of the currently active tank.
    """

    DEFAULT_CSS = """
    PlayerViewRegion {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        team: str,
        game_map: GameMap,
        state: GameState,
        patch_size: int = 7,
        active_tank_id: TankId = "",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._team = team
        self._game_map = game_map
        self._state = state
        self._patch_size = patch_size
        self._active_tank_id = active_tank_id
        self._label = Label(f" Team {team} ", classes="team-label")

    def compose(self) -> ComposeResult:
        """Compose the initial layout with a team label."""

        yield self._label
        yield from self._build_patches()

    def _build_patches(self) -> list[PatchView]:
        """Build patch widgets for all alive tanks of this team."""
        view = build_player_view(self._state, self._game_map, self._team, self._patch_size)
        patches: list[PatchView] = []
        for patch in view.patches:
            is_active = patch.tank_id == self._active_tank_id
            patches.append(
                PatchView(
                    patch.tank_id,
                    patch,
                    is_active=is_active,
                    id=f"patch-{patch.tank_id}",
                )
            )
        return patches

    async def refresh_patches(
        self,
        state: GameState,
        active_tank_id: TankId,
    ) -> None:
        """Rebuild patches with updated state.

        Removes all existing patch widgets and replaces them with fresh
        ones based on the new game state.

        Args:
            state: Updated game state.
            active_tank_id: ID of the currently active tank.
        """
        self._state = state
        self._active_tank_id = active_tank_id

        # Remove existing patch widgets.
        for child in list(self.children):
            if isinstance(child, PatchView):
                await child.remove()

        # Mount new patches.
        new_patches = self._build_patches()
        for patch_widget in new_patches:
            await self.mount(patch_widget)
