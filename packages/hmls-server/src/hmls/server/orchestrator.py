"""Game orchestrator: drives the engine loop and emits events.

The :class:`GameOrchestrator` owns the :class:`~hmls.core.engine.GameEngine`
and the turn-by-turn game loop.  It communicates exclusively through the
:class:`~hmls.server.events.EventBus` — it never touches WebSocket
connections directly.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from hmls.core.engine import GameEngine
from hmls.core.map import GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank
from hmls.core.visibility import build_player_view
from hmls.server.events import (
    EventBus,
    GameOverEvent,
    GameStartedEvent,
    PlayerDisconnectedEvent,
    StateUpdatedEvent,
    TurnCompletedEvent,
    YourTurnEvent,
)
from hmls.server.remote_player import RemotePlayer

logger = logging.getLogger("hmls.server.orchestrator")


class GameOrchestrator:
    """Drives the game engine loop, emitting events for each phase.

    The orchestrator waits for both players to connect (signalled via
    :attr:`both_connected`), then runs the turn loop until the game ends
    or a player disconnects.

    Attributes:
        game_map: The map being played on.
        tanks: All tanks in the game.
        max_turns: Maximum number of turns before a draw.
        patch_size: Fog-of-war patch size.
        history_file: Path to save game history JSON, or ``None``.
        engine: The game engine (created when the game starts).
        game_over: Whether the game has ended.
    """

    def __init__(
        self,
        game_map: GameMap,
        tanks: list[Tank],
        players: dict[str, RemotePlayer],
        event_bus: EventBus,
        *,
        max_turns: int,
        patch_size: int,
        history_file: Path | None = None,
    ) -> None:
        self.game_map = game_map
        self.tanks = tanks
        self.players = players
        self.max_turns = max_turns
        self.patch_size = patch_size
        self.history_file = history_file

        self._event_bus = event_bus
        self.engine: GameEngine | None = None
        self.game_over: bool = False
        self.both_connected: asyncio.Event = asyncio.Event()
        self.player_names: dict[str, str] = {}

        # Subscribe to disconnection events.
        self._event_bus.subscribe(PlayerDisconnectedEvent, self._on_player_disconnected)

    async def _on_player_disconnected(self, event: PlayerDisconnectedEvent) -> None:
        """Handle a player disconnecting mid-game."""
        if not self.game_over:
            self.game_over = True
            other_team = "B" if event.team == "A" else "A"
            await self._event_bus.emit(
                GameOverEvent(
                    winner=other_team,
                    reason="Opponent disconnected",
                )
            )

    async def run_game(self) -> None:
        """Run the game loop once both players are connected.

        Blocks until :attr:`both_connected` is set, then drives the
        engine turn-by-turn, emitting events at each stage.
        """
        await self.both_connected.wait()

        logger.info("Both players connected. Starting game!")

        # Emit game_started event.
        await self._event_bus.emit(
            GameStartedEvent(
                game_map=self.game_map,
                tanks=self.tanks,
                player_names=self.player_names,
                patch_size=self.patch_size,
                max_turns=self.max_turns,
            )
        )

        # Create the engine.
        engine_players: dict[str, Player] = dict(self.players)
        self.engine = GameEngine(
            self.game_map,
            self.tanks,
            engine_players,
            max_turns=self.max_turns,
            patch_size=self.patch_size,
        )

        # Emit initial state.
        await self._event_bus.emit(
            StateUpdatedEvent(
                state=self.engine.state,
                current_tank_id=self.engine.current_tank_id,
                turns_taken=0,
            )
        )

        loop = asyncio.get_event_loop()

        # Turn loop.
        while not self.engine.game_over and not self.game_over:
            tank_id = self.engine.current_tank_id
            team = self.engine.current_team
            player = self.players[team]

            # Build the view and request action.
            view = build_player_view(self.engine.state, self.game_map, team, self.patch_size)
            player.request_action(tank_id, view, loop)

            # Emit your_turn event (NetworkManager will send to client).
            await self._event_bus.emit(YourTurnEvent(tank_id=tank_id, team=team, view=view))

            # Wait for the client's response.
            try:
                await asyncio.wait_for(player.wait_for_action(), timeout=300.0)
            except asyncio.TimeoutError:
                logger.warning("Team %s timed out. Ending game.", team)
                self.game_over = True
                break
            except RuntimeError:
                logger.warning("Team %s action error. Ending game.", team)
                self.game_over = True
                break

            # Execute the step.
            entry = self.engine.step()
            if not entry.valid:
                result_info = f" (INVALID: {entry.reason})"
            elif entry.hit is True:
                result_info = " — HIT"
            elif entry.hit is False:
                result_info = " — miss"
            else:
                result_info = ""
            logger.info(
                "Turn %d: %s → %s%s",
                self.engine.turns_taken,
                entry.tank_id,
                entry.applied_action.value,
                result_info,
            )

            # Emit state update.
            await self._event_bus.emit(
                StateUpdatedEvent(
                    state=self.engine.state,
                    current_tank_id=(
                        self.engine.current_tank_id if not self.engine.game_over else None
                    ),
                    turns_taken=self.engine.turns_taken,
                )
            )

            # Emit turn completed.
            await self._event_bus.emit(TurnCompletedEvent(entry=entry, acting_team=team))

        # Game over.
        self.game_over = True
        winner = self.engine.winner if self.engine else None
        reason = "Game complete"
        if self.engine and self.engine.game_over:
            if winner:
                reason = f"Team {winner} wins!"
            else:
                reason = "Draw — turn limit reached"

        logger.info("Game over: %s", reason)
        await self._event_bus.emit(GameOverEvent(winner=winner, reason=reason))

        # Save game history.
        self._save_history()

    def _save_history(self) -> None:
        """Write game history to disk if configured."""
        if self.history_file is not None and self.engine is not None:
            try:
                result = self.engine.make_result()
                self.history_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                logger.info("Game history saved to %s", self.history_file.resolve())
            except Exception as exc:
                logger.error("Failed to save game history: %s", exc)
