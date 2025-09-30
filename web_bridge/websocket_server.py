"""
WebSocket server for real-time communication between Python game engine and web frontend.
Bridges the gap between the LLM game engine and the Node.js Secret Hitler web app.
"""
import asyncio
import json
import websockets
from typing import Dict, Set, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GameWebSocketServer:
    """WebSocket server for real-time game communication."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.game_rooms: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.running = False
    
    async def start_server(self):
        """Start the WebSocket server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        self.running = True
        async with websockets.serve(self.handle_connection, self.host, self.port) as server:
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever

    async def handle_connection(self, websocket):
        """Handle new WebSocket connection."""
        logger.info(f"New connection from {websocket.remote_address}")
        self.connections.add(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            self.connections.discard(websocket)
            self._remove_from_rooms(websocket)
    
    async def handle_message(self, websocket, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            event_type = data.get("type")
            payload = data.get("payload", {})
            
            if event_type == "join_game":
                await self._handle_join_game(websocket, payload)
            elif event_type == "leave_game":
                await self._handle_leave_game(websocket, payload)
            elif event_type == "get_game_state":
                await self._handle_get_game_state(websocket, payload)
            elif event_type == "subscribe_logs":
                await self._handle_subscribe_logs(websocket, payload)
            else:
                await self._send_error(websocket, f"Unknown event type: {event_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error(websocket, f"Server error: {str(e)}")
    
    async def _handle_join_game(self, websocket, payload):
        """Handle client joining a game room."""
        game_id = payload.get("game_id")
        if not game_id:
            await self._send_error(websocket, "game_id required")
            return
        
        if game_id not in self.game_rooms:
            self.game_rooms[game_id] = set()
        
        self.game_rooms[game_id].add(websocket)
        
        await self._send_response(websocket, {
            "type": "joined_game",
            "payload": {"game_id": game_id}
        })
        
        logger.info(f"Client joined game room: {game_id}")
    
    async def _handle_leave_game(self, websocket, payload):
        """Handle client leaving a game room."""
        game_id = payload.get("game_id")
        if game_id and game_id in self.game_rooms:
            self.game_rooms[game_id].discard(websocket)
            if not self.game_rooms[game_id]:
                del self.game_rooms[game_id]
        
        await self._send_response(websocket, {
            "type": "left_game",
            "payload": {"game_id": game_id}
        })
    
    async def _handle_get_game_state(self, websocket, payload):
        """Handle request for current game state."""
        game_id = payload.get("game_id")
        if not game_id:
            await self._send_error(websocket, "game_id required")
            return
        
        # Try to load game state from logs
        game_state = await self._load_game_state(game_id)
        
        await self._send_response(websocket, {
            "type": "game_state",
            "payload": {
                "game_id": game_id,
                "state": game_state
            }
        })
    
    async def _handle_subscribe_logs(self, websocket, payload):
        """Handle log subscription request."""
        game_id = payload.get("game_id")
        log_type = payload.get("log_type", "public")  # public, game, player
        
        if not game_id:
            await self._send_error(websocket, "game_id required")
            return
        
        # Send recent log entries
        recent_logs = await self._get_recent_logs(game_id, log_type)
        
        await self._send_response(websocket, {
            "type": "log_history",
            "payload": {
                "game_id": game_id,
                "log_type": log_type,
                "logs": recent_logs
            }
        })
    
    async def broadcast_game_event(self, game_id: str, event_data: Dict[str, Any]):
        """Broadcast event to all clients in a game room."""
        if game_id not in self.game_rooms:
            return
        
        message = json.dumps({
            "type": "game_event",
            "payload": {
                "game_id": game_id,
                "timestamp": datetime.now().isoformat(),
                **event_data
            }
        })
        
        disconnected = set()
        for websocket in self.game_rooms[game_id]:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.game_rooms[game_id].discard(websocket)
    
    async def broadcast_log_entry(self, game_id: str, log_type: str, log_data: Dict[str, Any]):
        """Broadcast new log entry to subscribed clients."""
        if game_id not in self.game_rooms:
            return
        
        message = json.dumps({
            "type": "log_entry",
            "payload": {
                "game_id": game_id,
                "log_type": log_type,
                "timestamp": datetime.now().isoformat(),
                **log_data
            }
        })
        
        # Broadcast to all clients in the game room
        disconnected = set()
        for websocket in self.game_rooms[game_id]:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting log to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.game_rooms[game_id].discard(websocket)
    
    async def _load_game_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Load current game state from logs."""
        try:
            log_dir = Path(f"logs/{game_id}")
            if not log_dir.exists():
                return None
            
            # Try to load from metrics.json if available
            metrics_file = log_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            
            # Fallback: parse game.log for latest state
            game_log = log_dir / "game.log"
            if game_log.exists():
                with open(game_log, 'r') as f:
                    lines = f.readlines()
                    
                # Find the most recent game state
                for line in reversed(lines):
                    if '"event": "game_start"' in line or '"event": "state_transition"' in line:
                        try:
                            # Extract JSON from log line
                            json_start = line.find('{')
                            if json_start >= 0:
                                log_data = json.loads(line[json_start:])
                                return log_data.get("initial_state") or log_data.get("game_state")
                        except json.JSONDecodeError:
                            continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading game state for {game_id}: {e}")
            return None
    
    async def _get_recent_logs(self, game_id: str, log_type: str, limit: int = 50) -> list:
        """Get recent log entries."""
        try:
            log_dir = Path(f"logs/{game_id}")
            if not log_dir.exists():
                return []
            
            log_file = log_dir / f"{log_type}.log"
            if not log_file.exists():
                return []
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Return the last 'limit' lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            # Parse log entries
            logs = []
            for line in recent_lines:
                line = line.strip()
                if line:
                    # Try to extract timestamp and message
                    parts = line.split(' ', 2)
                    if len(parts) >= 3:
                        timestamp = f"{parts[0]} {parts[1]}"
                        level_and_message = parts[2]
                        logs.append({
                            "timestamp": timestamp,
                            "message": level_and_message
                        })
                    else:
                        logs.append({
                            "timestamp": "",
                            "message": line
                        })
            
            return logs
            
        except Exception as e:
            logger.error(f"Error getting recent logs for {game_id}: {e}")
            return []
    
    def _remove_from_rooms(self, websocket):
        """Remove websocket from all game rooms."""
        for room_connections in self.game_rooms.values():
            room_connections.discard(websocket)
        
        # Clean up empty rooms
        empty_rooms = [game_id for game_id, connections in self.game_rooms.items() if not connections]
        for game_id in empty_rooms:
            del self.game_rooms[game_id]
    
    async def _send_response(self, websocket, data: Dict[str, Any]):
        """Send response to client."""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending response: {e}")
    
    async def _send_error(self, websocket, message: str):
        """Send error message to client."""
        await self._send_response(websocket, {
            "type": "error",
            "payload": {"message": message}
        })

# Singleton instance
_server_instance = None

def get_server_instance(host: str = "localhost", port: int = 8765) -> GameWebSocketServer:
    """Get or create WebSocket server instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = GameWebSocketServer(host, port)
    return _server_instance

async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server."""
    server = get_server_instance(host, port)
    await server.start_server()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start server
    asyncio.run(start_websocket_server())