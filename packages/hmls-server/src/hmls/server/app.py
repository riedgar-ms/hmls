"""Main server application: headless FastAPI WebSocket game server.

The server runs Uvicorn directly and wires together the
:class:`~hmls.server.events.EventBus`,
:class:`~hmls.server.network_manager.NetworkManager`, and
:class:`~hmls.server.orchestrator.GameOrchestrator` to manage a single
game session.  It accepts player connections (via ``JoinMessage``) and
observer connections (via ``ObserveMessage``).
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn
from fastapi import FastAPI, WebSocket

from hmls.core.map import GameMap, MapLoadError, load_map
from hmls.core.placement import InsufficientPassableCellsError, place_tanks
from hmls.core.tank import Tank
from hmls.server.cli import parse_args
from hmls.server.events import EventBus
from hmls.server.network_manager import NetworkManager
from hmls.server.orchestrator import GameOrchestrator
from hmls.server.remote_player import RemotePlayer

logger = logging.getLogger("hmls.server")

# ── Helpers ───────────────────────────────────────────────────────────


def _place_tanks_or_exit(
    game_map: GameMap,
    tanks_per_player: int,
    *,
    seed: int | None = None,
) -> list[Tank]:
    """Place tanks, exiting the process on failure."""
    try:
        return place_tanks(game_map, tanks_per_player, seed=seed)
    except InsufficientPassableCellsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


# ── FastAPI application ───────────────────────────────────────────────


def create_fastapi_app(network_manager: NetworkManager) -> FastAPI:
    """Create the FastAPI application with WebSocket endpoint.

    Args:
        network_manager: The network manager that handles all connections.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(title="HMLS Game Server")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await network_manager.handle_connection(websocket)

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
    try:
        game_map = load_map(args.map_file)
    except (FileNotFoundError, MapLoadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    tanks = _place_tanks_or_exit(game_map, args.tanks_per_player, seed=args.seed)

    # Wire up components.
    event_bus = EventBus()
    players: dict[str, RemotePlayer] = {
        "A": RemotePlayer("A"),
        "B": RemotePlayer("B"),
    }

    network_manager = NetworkManager(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        patch_size=args.patch_size,
        max_turns=args.max_turns,
    )

    orchestrator = GameOrchestrator(
        game_map=game_map,
        tanks=tanks,
        players=players,
        event_bus=event_bus,
        max_turns=args.max_turns,
        patch_size=args.patch_size,
        history_file=args.history_file,
    )

    # Share the "both connected" event between network manager and orchestrator.
    orchestrator.both_connected = network_manager.both_connected

    # Share player names so the orchestrator can include them in GameStartedEvent.
    orchestrator.player_names = network_manager.player_names

    # Create FastAPI app.
    fastapi_app = create_fastapi_app(network_manager)

    @fastapi_app.on_event("startup")
    async def start_game() -> None:
        asyncio.create_task(orchestrator.run_game())

    logger.info("Starting HMLS Game Server on port %d", args.port)

    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
