"""Main server application: headless FastAPI WebSocket game server.

The server runs Uvicorn directly and manages game sessions. It accepts
player connections (via ``JoinMessage``) and observer connections (via
``ObserveMessage``). Observers receive the full game state after each turn.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter

from hmls.core.engine import GameEngine
from hmls.core.map import CellType, GameMap
from hmls.core.player import Player
from hmls.core.tank import Tank
from hmls.core.types import Direction, Position
from hmls.core.visibility import TankInfo, build_player_view
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
from hmls.server.cli import parse_args
from hmls.server.remote_player import RemotePlayer

logger = logging.getLogger("hmls.server")

# ── Type adapters for protocol parsing ────────────────────────────────

_client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)

# ── Helpers ───────────────────────────────────────────────────────────


def _load_map(path: Path) -> GameMap:
    """Load a GameMap from a JSON file."""
    if not path.exists():
        print(f"Error: map file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        text = path.read_text(encoding="utf-8")
        return GameMap.model_validate_json(text)
    except Exception as exc:
        print(f"Error loading map: {exc}", file=sys.stderr)
        sys.exit(1)


def _place_tanks(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks randomly on passable cells for two teams."""
    total_needed = tanks_per_player * 2
    passable_positions = [
        Position(x, y) for x, y in game_map.all_positions() if game_map[x, y] == CellType.PASSABLE
    ]
    if len(passable_positions) < total_needed:
        print(
            f"Error: need {total_needed} passable cells but map only has {len(passable_positions)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = random.Random(seed)
    chosen = rng.sample(passable_positions, total_needed)
    directions = list(Direction)
    tanks: list[Tank] = []

    for team_idx, team_name in enumerate(["A", "B"]):
        for i in range(tanks_per_player):
            pos = chosen[team_idx * tanks_per_player + i]
            tanks.append(
                Tank(
                    id=f"{team_name}{i + 1}",
                    team=team_name,
                    position=pos,
                    direction=rng.choice(directions),
                )
            )

    return tanks


# ── Game session state ────────────────────────────────────────────────


class GameSession:
    """Manages the game state and communication with connected clients.

    Attributes:
        game_map: The map being played on.
        tanks: All tanks in the game.
        engine: The game engine (created once both players connect).
        players: Mapping of team name to RemotePlayer.
        websockets: Mapping of team name to WebSocket connection.
        history_file: Path to save game history JSON after game over, or
            ``None`` to disable saving.
    """

    def __init__(
        self,
        game_map: GameMap,
        tanks: list[Tank],
        max_turns: int,
        patch_size: int,
        history_file: Path | None = None,
    ) -> None:
        self.game_map = game_map
        self.tanks = tanks
        self.max_turns = max_turns
        self.patch_size = patch_size
        self.history_file = history_file

        self.players: dict[str, RemotePlayer] = {
            "A": RemotePlayer("A"),
            "B": RemotePlayer("B"),
        }
        self.websockets: dict[str, WebSocket] = {}
        self.player_names: dict[str, str] = {}
        self._observers: list[WebSocket] = []
        self._both_connected: asyncio.Event = asyncio.Event()
        self._game_over: bool = False
        self._game_started: bool = False
        self.engine: GameEngine | None = None

    @property
    def game_over(self) -> bool:
        """Whether the game has ended."""
        return self._game_over

    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection from a client.

        The first message determines whether this is a player (JoinMessage)
        or an observer (ObserveMessage).
        """
        await websocket.accept()

        # Wait for identification message.
        try:
            raw = await websocket.receive_text()
            msg = _client_message_adapter.validate_json(raw)
        except (WebSocketDisconnect, Exception) as exc:
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
        self._observers.append(websocket)

        # Always send the map so the observer can render immediately,
        # even before all players have joined.
        game_start_msg = GameStartMessage(
            game_map=self.game_map,
            tanks=self.tanks,
            player_names=self.player_names,
            patch_size=self.patch_size,
            max_turns=self.max_turns,
        )
        try:
            await websocket.send_text(game_start_msg.model_dump_json())
        except Exception:
            self._observers.remove(websocket)
            return

        # If the game is already in progress, also send the current state.
        if self._game_started and self.engine:
            state_msg = StateUpdateMessage(
                state=self.engine.state,
                current_tank_id=self.engine.current_tank_id if not self.engine.game_over else "",
                turns_taken=self.engine.turns_taken,
            )
            try:
                await websocket.send_text(state_msg.model_dump_json())
            except Exception:
                self._observers.remove(websocket)
                return

        # Keep connection alive until game ends or disconnect.
        try:
            while not self._game_over:
                # Observers don't send meaningful messages, but we need to
                # keep the connection alive and detect disconnects.
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in self._observers:
                self._observers.remove(websocket)
            logger.info("Observer '%s' disconnected", msg.observer_name)

    async def _handle_player_join(self, websocket: WebSocket, msg: JoinMessage) -> None:
        """Handle a player joining the game."""
        # Determine which team slot is free.
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

        # If first player, tell them to wait.
        if len(self.websockets) == 1:
            await websocket.send_text(
                WaitingMessage(message="Waiting for opponent to connect...").model_dump_json()
            )
        else:
            # Both connected — start the game.
            self._both_connected.set()

        # Keep connection alive until game ends or disconnect.
        try:
            await self._client_loop(team, websocket)
        except WebSocketDisconnect:
            logger.info("Team %s (%s) disconnected", team, self.player_names.get(team, "?"))
            if not self._game_over:
                self._game_over = True
                other_team = "B" if team == "A" else "A"
                if other_team in self.websockets:
                    try:
                        await self.websockets[other_team].send_text(
                            GameOverMessage(
                                winner=other_team,
                                reason="Opponent disconnected",
                            ).model_dump_json()
                        )
                    except Exception:
                        pass

    async def _client_loop(self, team: str, websocket: WebSocket) -> None:
        """Listen for action messages from a connected client."""
        while not self._game_over:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                raise

            try:
                msg = _client_message_adapter.validate_json(raw)
            except Exception as exc:
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

    async def _broadcast_to_observers(self, message: str) -> None:
        """Send a message to all connected observers."""
        disconnected: list[WebSocket] = []
        for ws in self._observers:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._observers.remove(ws)

    async def run_game(self) -> None:
        """Run the game loop once both players are connected."""
        await self._both_connected.wait()

        # Send assign messages to both clients.
        for team in ["A", "B"]:
            team_tanks = [t for t in self.tanks if t.team == team]
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
                map_width=self.game_map.width,
                map_height=self.game_map.height,
                patch_size=self.patch_size,
            )
            await self.websockets[team].send_text(assign_msg.model_dump_json())

        logger.info("Both players connected. Starting game!")
        self._game_started = True

        # Broadcast game_start to observers.
        game_start_msg = GameStartMessage(
            game_map=self.game_map,
            tanks=self.tanks,
            player_names=self.player_names,
            patch_size=self.patch_size,
            max_turns=self.max_turns,
        )
        await self._broadcast_to_observers(game_start_msg.model_dump_json())

        # Create the engine.
        players: dict[str, Player] = {
            "A": self.players["A"],
            "B": self.players["B"],
        }
        self.engine = GameEngine(
            self.game_map,
            self.tanks,
            players,
            max_turns=self.max_turns,
            patch_size=self.patch_size,
        )

        # Send initial state to observers.
        state_msg = StateUpdateMessage(
            state=self.engine.state,
            current_tank_id=self.engine.current_tank_id,
            turns_taken=0,
        )
        await self._broadcast_to_observers(state_msg.model_dump_json())

        loop = asyncio.get_event_loop()

        # Game loop.
        while not self.engine.game_over and not self._game_over:
            tank_id = self.engine.current_tank_id
            team = self.engine.current_team
            player = self.players[team]

            # Build the view and request action.
            view = build_player_view(self.engine.state, self.game_map, team, self.patch_size)
            player.request_action(tank_id, view, loop)

            # Send your_turn to the client.
            your_turn_msg = YourTurnMessage(tank_id=tank_id, view=view)
            try:
                await self.websockets[team].send_text(your_turn_msg.model_dump_json())
            except Exception:
                logger.warning("Failed to send your_turn to Team %s", team)
                self._game_over = True
                break

            # Wait for the client's response.
            try:
                await asyncio.wait_for(player.wait_for_action(), timeout=300.0)
            except asyncio.TimeoutError:
                logger.warning("Team %s timed out. Ending game.", team)
                self._game_over = True
                break
            except RuntimeError:
                logger.warning("Team %s action error. Ending game.", team)
                self._game_over = True
                break

            # Execute the step.
            entry = self.engine.step()
            logger.info(
                "Turn %d: %s → %s%s",
                self.engine.turns_taken,
                entry.tank_id,
                entry.applied_action.value,
                "" if entry.valid else f" (INVALID: {entry.reason})",
            )

            # Broadcast updated state to observers.
            state_msg = StateUpdateMessage(
                state=self.engine.state,
                current_tank_id=self.engine.current_tank_id if not self.engine.game_over else "",
                turns_taken=self.engine.turns_taken,
            )
            await self._broadcast_to_observers(state_msg.model_dump_json())

            # Send turn_result only to the acting player's team.
            result_msg = TurnResultMessage(
                tank_id=entry.tank_id,
                action=entry.applied_action,
                valid=entry.valid,
                reason=entry.reason,
            )
            if team in self.websockets:
                try:
                    await self.websockets[team].send_text(result_msg.model_dump_json())
                except Exception:
                    pass

            # Also send turn_result to observers so they get the log info.
            await self._broadcast_to_observers(result_msg.model_dump_json())

        # Game over.
        self._game_over = True
        winner = self.engine.winner if self.engine else None
        reason = "Game complete"
        if self.engine and self.engine.game_over:
            if winner:
                reason = f"Team {winner} wins!"
            else:
                reason = "Draw — turn limit reached"

        logger.info("Game over: %s", reason)

        game_over_msg = GameOverMessage(winner=winner, reason=reason)
        game_over_json = game_over_msg.model_dump_json()
        for t in ["A", "B"]:
            if t in self.websockets:
                try:
                    await self.websockets[t].send_text(game_over_json)
                except Exception:
                    pass

        # Broadcast game over to observers.
        await self._broadcast_to_observers(game_over_json)

        # Save game history to file if configured.
        if self.history_file is not None and self.engine is not None:
            try:
                result = self.engine.make_result()
                self.history_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                logger.info("Game history saved to %s", self.history_file.resolve())
            except Exception as exc:
                logger.error("Failed to save game history: %s", exc)


# ── FastAPI application ───────────────────────────────────────────────


def create_fastapi_app(session: GameSession) -> FastAPI:
    """Create the FastAPI application with WebSocket endpoint.

    Args:
        session: The game session to attach.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="HMLS Game Server")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await session.handle_connection(websocket)

    return app


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the server application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    game_map = _load_map(args.map_file)
    tanks = _place_tanks(game_map, args.tanks_per_player, seed=args.seed)

    session = GameSession(
        game_map=game_map,
        tanks=tanks,
        max_turns=args.max_turns,
        patch_size=args.patch_size,
        history_file=args.history_file,
    )

    # Create FastAPI app.
    fastapi_app = create_fastapi_app(session)

    # Start the game loop as a background task once uvicorn is running.
    @fastapi_app.on_event("startup")
    async def start_game() -> None:
        asyncio.create_task(session.run_game())

    logger.info("Starting HMLS Game Server on port %d", args.port)

    # Run uvicorn directly (single-threaded async — no TUI).
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
