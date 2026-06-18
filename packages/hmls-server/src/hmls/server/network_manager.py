"""Network manager: handles all WebSocket connections for the game server.

The :class:`NetworkManager` owns the WebSocket connections for both players
and observers.  It routes incoming connections, manages the client message
loop, and subscribes to :class:`~hmls.server.events.EventBus` events to
deliver protocol messages to the appropriate recipients.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter

from hmls.core.map import GameMap
from hmls.core.tank import Tank
from hmls.core.visibility import TankInfo
from hmls.protocol import (
    ActionMessage,
    AssignMessage,
    ClientMessage,
    ErrorMessage,
    GameOverMessage,
    GameStartMessage,
    JoinMessage,
    ObserveMessage,
    StateUpdateMessage,
    TurnResultMessage,
    WaitingMessage,
    YourTurnMessage,
)
from hmls.server.event_bus import EventBus
from hmls.server.event_types import (
    GameOverEvent,
    GameStartedEvent,
    PlayerDisconnectedEvent,
    StateUpdatedEvent,
    TurnCompletedEvent,
    YourTurnEvent,
)
from hmls.server.remote_player import RemotePlayer

logger = logging.getLogger("hmls.server.network_manager")

_client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


class NetworkManager:
    """Manages all WebSocket connections and protocol message routing.

    Subscribes to :class:`~hmls.server.events.EventBus` events and
    translates them into the appropriate protocol messages for each
    connected client.

    Attributes:
        players: Mapping of team name to :class:`RemotePlayer`.
        websockets: Mapping of team name to WebSocket connection.
        player_names: Mapping of team name to human-readable player name.
        observers: List of connected observer WebSockets.
        game_over: Whether the game has ended (used to terminate loops).
    """

    def __init__(
        self,
        game_map: GameMap,
        tanks: list[Tank],
        players: dict[str, RemotePlayer],
        event_bus: EventBus,
        *,
        patch_size: int,
        max_turns: int,
    ) -> None:
        self._game_map = game_map
        self._tanks = tanks
        self.players = players
        self._event_bus = event_bus
        self._patch_size = patch_size
        self._max_turns = max_turns

        self.websockets: dict[str, WebSocket] = {}
        self.player_names: dict[str, str] = {}
        self.observers: list[WebSocket] = []
        self.game_over: bool = False
        self._game_started: bool = False
        self._last_state: StateUpdatedEvent | None = None

        # Set by this manager when both players join; shared with the
        # orchestrator in app.py so it can await the same event.
        self.both_connected: asyncio.Event = asyncio.Event()

        # Subscribe to events.
        self._event_bus.subscribe(GameStartedEvent, self._on_game_started)
        self._event_bus.subscribe(YourTurnEvent, self._on_your_turn)
        self._event_bus.subscribe(StateUpdatedEvent, self._on_state_updated)
        self._event_bus.subscribe(TurnCompletedEvent, self._on_turn_completed)
        self._event_bus.subscribe(GameOverEvent, self._on_game_over)

    # ── WebSocket connection handling ─────────────────────────────────

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection from a client.

        The first message determines whether this is a player
        (:class:`JoinMessage`) or an observer (:class:`ObserveMessage`).
        """
        await websocket.accept()

        try:
            raw = await websocket.receive_text()
            msg = _client_message_adapter.validate_json(raw)
        except (WebSocketDisconnect, Exception) as exc:  # noqa: BLE001
            logger.warning("Connection error before identification: %s", exc)
            return

        if isinstance(msg, ObserveMessage):
            await self._handle_observer(websocket, msg)
        elif isinstance(msg, JoinMessage):
            await self._handle_player_join(websocket, msg)
        else:
            await websocket.send_text(
                ErrorMessage(message="Expected 'join' or 'observe' message").model_dump_json()
            )
            await websocket.close()

    async def _handle_observer(self, websocket: WebSocket, msg: ObserveMessage) -> None:
        """Register an observer and stream game state to it."""
        logger.info("Observer '%s' connected", msg.observer_name)
        self.observers.append(websocket)

        # Always send the map so the observer can render immediately.
        game_start_msg = GameStartMessage(
            game_map=self._game_map,
            tanks=self._tanks,
            player_names=self.player_names,
            patch_size=self._patch_size,
            max_turns=self._max_turns,
        )
        try:
            await websocket.send_text(game_start_msg.model_dump_json())
        except Exception:  # noqa: BLE001
            logger.debug(
                "Observer disconnected during initial game_start send",
                exc_info=True,
            )
            self.observers.remove(websocket)
            return

        # If the game is already in progress, also send current state.
        if self._game_started and self._last_state is not None:
            state_msg = StateUpdateMessage(
                state=self._last_state.state,
                current_tank_id=self._last_state.current_tank_id,
                turns_taken=self._last_state.turns_taken,
            )
            try:
                await websocket.send_text(state_msg.model_dump_json())
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Observer disconnected during state_update send",
                    exc_info=True,
                )
                self.observers.remove(websocket)
                return

        # Keep connection alive until game ends or disconnect.
        try:
            while not self.game_over:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except TimeoutError:
                    continue
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in self.observers:
                self.observers.remove(websocket)
            logger.info("Observer '%s' disconnected", msg.observer_name)

    async def _handle_player_join(self, websocket: WebSocket, msg: JoinMessage) -> None:
        """Handle a player joining the game."""
        if "A" not in self.websockets:
            team = "A"
        elif "B" not in self.websockets:
            team = "B"
        else:
            await websocket.send_text(ErrorMessage(message="Game is full").model_dump_json())
            await websocket.close()
            return

        self.websockets[team] = websocket
        self.player_names[team] = msg.player_name
        logger.info("Player '%s' joined as Team %s", msg.player_name, team)

        if len(self.websockets) == 1:
            await websocket.send_text(
                WaitingMessage(message="Waiting for opponent to connect...").model_dump_json()
            )
        else:
            self.both_connected.set()

        # Listen for actions until game ends or disconnect.
        try:
            await self._client_loop(team, websocket)
        except WebSocketDisconnect:
            logger.info(
                "Team %s (%s) disconnected",
                team,
                self.player_names.get(team, "?"),
            )
            if not self.game_over:
                await self._event_bus.emit(PlayerDisconnectedEvent(team=team))

    async def _client_loop(self, team: str, websocket: WebSocket) -> None:
        """Listen for action messages from a connected client."""
        while not self.game_over:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except TimeoutError:
                continue
            except WebSocketDisconnect:
                raise

            try:
                msg = _client_message_adapter.validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_text(
                    ErrorMessage(message=f"Invalid message: {exc}").model_dump_json()
                )
                continue

            if isinstance(msg, ActionMessage):
                player = self.players[team]
                try:
                    player.submit_action(msg.action)
                except RuntimeError as exc:
                    await websocket.send_text(ErrorMessage(message=str(exc)).model_dump_json())
            else:
                await websocket.send_text(
                    ErrorMessage(
                        message=f"Unexpected message type: {getattr(msg, 'type', 'unknown')}"
                    ).model_dump_json()
                )

    # ── Broadcasting ──────────────────────────────────────────────────

    async def broadcast_to_observers(self, message: str) -> None:
        """Send a message to all connected observers.

        Disconnected observers are automatically removed.
        """
        disconnected: list[WebSocket] = []
        for ws in self.observers:
            try:
                await ws.send_text(message)
            except Exception:  # noqa: BLE001
                logger.debug("Observer disconnected during broadcast", exc_info=True)
                disconnected.append(ws)
        for ws in disconnected:
            self.observers.remove(ws)

    async def send_to_player(self, team: str, message: str) -> bool:
        """Send a message to a specific player.

        Args:
            team: The team identifier ("A" or "B").
            message: The JSON-encoded message string.

        Returns:
            ``True`` if sent successfully, ``False`` on failure.
        """
        if team not in self.websockets:
            return False
        try:
            await self.websockets[team].send_text(message)
            return True
        except Exception:  # noqa: BLE001
            logger.debug("Failed to send to Team %s", team, exc_info=True)
            return False

    # ── Event handlers ────────────────────────────────────────────────

    async def _on_game_started(self, event: GameStartedEvent) -> None:
        """Handle game start: send AssignMessages to players, broadcast to observers."""
        self._game_started = True

        # Send assign messages to both players.
        for team in ["A", "B"]:
            team_tanks = [t for t in self._tanks if t.team == team]
            tank_infos = [
                TankInfo(
                    tank_id=t.id,
                    position=t.position,
                    direction=t.direction,
                    alive=t.alive,
                )
                for t in team_tanks
            ]
            assign_msg = AssignMessage(
                team=team,
                tanks=tank_infos,
                map_width=self._game_map.width,
                map_height=self._game_map.height,
                patch_size=self._patch_size,
            )
            await self.send_to_player(team, assign_msg.model_dump_json())

        # Broadcast game_start to observers.
        game_start_msg = GameStartMessage(
            game_map=event.game_map,
            tanks=event.tanks,
            player_names=event.player_names,
            patch_size=event.patch_size,
            max_turns=event.max_turns,
        )
        await self.broadcast_to_observers(game_start_msg.model_dump_json())

    async def _on_your_turn(self, event: YourTurnEvent) -> None:
        """Handle your_turn: send YourTurnMessage to the acting player."""
        your_turn_msg = YourTurnMessage(tank_id=event.tank_id, view=event.view)
        success = await self.send_to_player(event.team, your_turn_msg.model_dump_json())
        if not success:
            logger.warning("Failed to send your_turn to Team %s", event.team)

    async def _on_state_updated(self, event: StateUpdatedEvent) -> None:
        """Handle state update: cache and broadcast to observers."""
        self._last_state = event
        state_msg = StateUpdateMessage(
            state=event.state,
            current_tank_id=event.current_tank_id,
            turns_taken=event.turns_taken,
        )
        await self.broadcast_to_observers(state_msg.model_dump_json())

    async def _on_turn_completed(self, event: TurnCompletedEvent) -> None:
        """Handle turn completed: send result to acting player and observers."""
        result_msg = TurnResultMessage(
            tank_id=event.entry.tank_id,
            action=event.entry.applied_action,
            valid=event.entry.valid,
            reason=event.entry.reason,
            hit=event.entry.hit,
        )
        result_json = result_msg.model_dump_json()

        # Send to acting player only.
        await self.send_to_player(event.acting_team, result_json)

        # Also broadcast to observers.
        await self.broadcast_to_observers(result_json)

    async def _on_game_over(self, event: GameOverEvent) -> None:
        """Handle game over: notify all players and observers."""
        self.game_over = True
        game_over_msg = GameOverMessage(winner=event.winner, reason=event.reason)
        game_over_json = game_over_msg.model_dump_json()

        for team in ["A", "B"]:
            await self.send_to_player(team, game_over_json)

        await self.broadcast_to_observers(game_over_json)
