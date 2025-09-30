"""
WebSocket bridge for real-time spectator mode.
Uses GameManager's spectator_callback for event-driven updates.

Author: Samuel Chakwera (stchakdev)
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from .websocket_server import GameWebSocketServer

logger = logging.getLogger(__name__)


class SpectatorWebSocketBridge:
    """Bridges GameManager events to WebSocket broadcasts."""

    def __init__(self, websocket_server, game_id: str):
        """
        Initialize spectator bridge.

        Args:
            websocket_server: GameWebSocketServer instance
            game_id: Unique game identifier
        """
        self.ws_server = websocket_server
        self.game_id = game_id
        self.loop = None

    def set_event_loop(self, loop):
        """Set event loop for scheduling async broadcasts."""
        self.loop = loop

    def handle_game_event(self, event: Dict[str, Any]):
        """
        Synchronous callback for GameManager spectator events.
        Schedules async WebSocket broadcast.

        Args:
            event: Game event dictionary from GameManager
        """
        print(f"🔔 Spectator event received: {event.get('type', 'unknown')}")

        if not self.loop:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No event loop available for spectator broadcast")
                print("⚠️  No event loop found")
                return

        # Add timestamp if not present
        if 'timestamp' not in event:
            event['timestamp'] = datetime.now().isoformat()

        # Schedule async broadcast
        try:
            asyncio.ensure_future(self._broadcast_event(event), loop=self.loop)
            print(f"✅ Scheduled broadcast for event: {event.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to schedule spectator broadcast: {e}")
            print(f"❌ Failed to schedule: {e}")

    async def _broadcast_event(self, event: Dict[str, Any]):
        """
        Broadcast event to all connected spectators.

        Args:
            event: Event data to broadcast
        """
        try:
            await self.ws_server.broadcast_game_event(self.game_id, event)
        except Exception as e:
            logger.error(f"Error broadcasting spectator event: {e}")


class SpectatorWebSocketServer:
    """
    Simplified WebSocket server wrapper for Phase 1 testing.
    Wraps GameWebSocketServer for easy testing and integration.
    """

    def __init__(self, host: str = "localhost", port: int = 8765, game_id: str = "test-game"):
        """
        Initialize spectator WebSocket server.

        Args:
            host: Server host
            port: Server port
            game_id: Game identifier for broadcasts
        """
        self.host = host
        self.port = port
        self.game_id = game_id
        self.server = GameWebSocketServer(host, port)
        self.server_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the WebSocket server in the background."""
        if self._running:
            logger.warning("Server already running")
            return

        # Start server in background task
        self.server_task = asyncio.create_task(self.server.start_server())
        self._running = True

        # Give server time to start
        await asyncio.sleep(0.5)

        logger.info(f"SpectatorWebSocketServer started on {self.host}:{self.port}")

    def broadcast(self, event: Dict[str, Any]):
        """
        Broadcast event to all connected clients (synchronous wrapper).

        Args:
            event: Event data to broadcast
        """
        if not self._running:
            logger.warning("Cannot broadcast - server not running")
            return

        # Schedule async broadcast
        try:
            asyncio.create_task(
                self.server.broadcast_game_event(self.game_id, event)
            )
        except Exception as e:
            logger.error(f"Error scheduling broadcast: {e}")

    async def stop(self):
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Cancel server task
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

        logger.info("SpectatorWebSocketServer stopped")