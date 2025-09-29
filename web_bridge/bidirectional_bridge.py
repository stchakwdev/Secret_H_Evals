"""
Bidirectional bridge server for hybrid human-AI Secret Hitler games.
Extends the existing web bridge to support human player interactions.
"""
import asyncio
import json
import websockets
from typing import Dict, Set, Optional, Any, Callable, List
from datetime import datetime
import logging
from pathlib import Path

from .websocket_server import GameWebSocketServer
from .game_adapter import LLMGameAdapter

logger = logging.getLogger(__name__)

class HybridGameBridgeServer(GameWebSocketServer):
    """Enhanced WebSocket server that supports bidirectional communication for human players."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        super().__init__(host, port)
        
        # Human player management
        self.human_players: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.player_games: Dict[str, str] = {}  # player_id -> game_id
        self.pending_actions: Dict[str, Dict[str, Any]] = {}  # player_id -> action_data
        self.action_callbacks: Dict[str, Callable] = {}  # game_id -> callback
        
        # Authentication tokens for human players
        self.player_tokens: Dict[str, str] = {}  # token -> player_id
        
        # Auto-discovery system
        self.active_games: Dict[str, Dict[str, Any]] = {}  # game_id -> game_info
        self.latest_game_id: Optional[str] = None
    
    async def handle_message(self, websocket, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            event_type = data.get("type")
            payload = data.get("payload", {})
            
            # Handle existing events
            if event_type in ["join_game", "leave_game", "get_game_state", "subscribe_logs"]:
                await super().handle_message(websocket, message)
                return
            
            # Handle human player events
            if event_type == "authenticate_player":
                await self._handle_authenticate_player(websocket, payload)
            elif event_type == "submit_action":
                await self._handle_submit_action(websocket, payload)
            elif event_type == "get_action_request":
                await self._handle_get_action_request(websocket, payload)
            elif event_type == "heartbeat":
                await self._handle_heartbeat(websocket, payload)
            elif event_type == "discover_games":
                await self._handle_discover_games(websocket, payload)
            elif event_type == "auto_connect":
                await self._handle_auto_connect(websocket, payload)
            else:
                await self._send_error(websocket, f"Unknown event type: {event_type}")
                
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error(websocket, f"Server error: {str(e)}")
    
    async def _handle_authenticate_player(self, websocket, payload):
        """Handle human player authentication."""
        player_id = payload.get("player_id")
        game_id = payload.get("game_id")
        token = payload.get("token")  # Optional authentication token
        
        if not player_id or not game_id:
            await self._send_error(websocket, "player_id and game_id required")
            return
        
        # Store player connection
        self.human_players[player_id] = websocket
        self.player_games[player_id] = game_id
        
        if token:
            self.player_tokens[token] = player_id
        
        # Join the game room
        if game_id not in self.game_rooms:
            self.game_rooms[game_id] = set()
        self.game_rooms[game_id].add(websocket)
        
        await self._send_response(websocket, {
            "type": "authenticated",
            "payload": {
                "player_id": player_id,
                "game_id": game_id,
                "status": "connected"
            }
        })
        
        # Send any pending action requests
        if player_id in self.pending_actions:
            await self._send_action_request(player_id, self.pending_actions[player_id])
        
        logger.info(f"Human player {player_id} authenticated for game {game_id}")
        
        # Notify any registered callbacks about the new connection
        if hasattr(self, 'on_player_connect') and callable(self.on_player_connect):
            await self.on_player_connect(player_id, game_id)
    
    async def _handle_submit_action(self, websocket, payload):
        """Handle action submission from human player."""
        player_id = payload.get("player_id")
        action_type = payload.get("action_type")
        action_data = payload.get("action_data")
        request_id = payload.get("request_id")
        
        if not all([player_id, action_type, action_data]):
            await self._send_error(websocket, "player_id, action_type, and action_data required")
            return
        
        if player_id not in self.human_players:
            await self._send_error(websocket, "Player not authenticated")
            return
        
        game_id = self.player_games.get(player_id)
        if not game_id:
            await self._send_error(websocket, "Player not in a game")
            return
        
        # Clear pending action for this player
        if player_id in self.pending_actions:
            del self.pending_actions[player_id]
        
        # Call the action callback if registered
        if game_id in self.action_callbacks:
            try:
                await self.action_callbacks[game_id]({
                    'type': 'action_response',
                    'player_id': player_id,
                    'action_type': action_type,
                    'action_data': action_data,
                    'request_id': request_id,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error calling action callback: {e}")
        
        await self._send_response(websocket, {
            "type": "action_submitted",
            "payload": {
                "player_id": player_id,
                "action_type": action_type,
                "request_id": request_id,
                "status": "received"
            }
        })
        
        logger.info(f"Received action from {player_id}: {action_type}")
    
    async def _handle_get_action_request(self, websocket, payload):
        """Handle request for current action from human player."""
        player_id = payload.get("player_id")
        
        if not player_id:
            await self._send_error(websocket, "player_id required")
            return
        
        if player_id in self.pending_actions:
            await self._send_action_request(player_id, self.pending_actions[player_id])
        else:
            await self._send_response(websocket, {
                "type": "no_action_pending",
                "payload": {"player_id": player_id}
            })
    
    async def _handle_heartbeat(self, websocket, payload):
        """Handle heartbeat from client."""
        player_id = payload.get("player_id")
        
        await self._send_response(websocket, {
            "type": "heartbeat_ack",
            "payload": {
                "player_id": player_id,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_action_request(self, player_id: str, action_request: Dict[str, Any]):
        """Send action request to a human player."""
        if player_id not in self.human_players:
            logger.warning(f"Player {player_id} not connected")
            return False
        
        # Store the pending action
        self.pending_actions[player_id] = action_request
        
        # Send to the player
        return await self._send_action_request(player_id, action_request)
    
    async def _send_action_request(self, player_id: str, action_request: Dict[str, Any]) -> bool:
        """Internal method to send action request to player."""
        if player_id not in self.human_players:
            return False
        
        websocket = self.human_players[player_id]
        
        try:
            await websocket.send(json.dumps({
                "type": "action_request",
                "payload": action_request
            }))
            
            logger.info(f"Sent action request to {player_id}: {action_request.get('decision_type')}")
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for player {player_id}")
            self._cleanup_player(player_id)
            return False
        except Exception as e:
            logger.error(f"Error sending action request to {player_id}: {e}")
            return False
    
    async def send_to_player(self, player_id: str, data: Dict[str, Any]) -> bool:
        """Send data directly to a specific player."""
        if player_id not in self.human_players:
            logger.warning(f"Player {player_id} not connected for message")
            return False
        
        websocket = self.human_players[player_id]
        
        try:
            # Determine message type based on data structure
            message_type = "game_event"
            if "decision_type" in data:
                message_type = "action_request"
            elif "role" in data:
                message_type = "role_assignment"
            elif "error" in data or "message" in data:
                message_type = "error"
            
            await websocket.send(json.dumps({
                "type": message_type,
                "payload": data
            }))
            
            logger.info(f"Sent {message_type} to {player_id}")
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed for player {player_id}")
            self._cleanup_player(player_id)
            return False
        except Exception as e:
            logger.error(f"Error sending data to {player_id}: {e}")
            return False

    def register_action_callback(self, game_id: str, callback: Callable):
        """Register callback for handling human player actions."""
        self.action_callbacks[game_id] = callback
        logger.info(f"Registered action callback for game {game_id}")
    
    def unregister_action_callback(self, game_id: str):
        """Unregister action callback for a game."""
        if game_id in self.action_callbacks:
            del self.action_callbacks[game_id]
            logger.info(f"Unregistered action callback for game {game_id}")
    
    def _cleanup_player(self, player_id: str):
        """Clean up disconnected player."""
        if player_id in self.human_players:
            del self.human_players[player_id]
        
        if player_id in self.player_games:
            del self.player_games[player_id]
        
        if player_id in self.pending_actions:
            del self.pending_actions[player_id]
        
        # Remove from token mapping
        tokens_to_remove = [token for token, pid in self.player_tokens.items() if pid == player_id]
        for token in tokens_to_remove:
            del self.player_tokens[token]
    
    async def handle_connection(self, websocket, path=None):
        """Handle new WebSocket connection with cleanup."""
        logger.info(f"New connection from {websocket.remote_address}")
        self.connections.add(websocket)
        
        # Track which player this connection belongs to
        connection_player_id = None
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
                
                # Try to identify the player for this connection
                if connection_player_id is None:
                    for player_id, player_ws in self.human_players.items():
                        if player_ws == websocket:
                            connection_player_id = player_id
                            break
                            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            self.connections.discard(websocket)
            self._remove_from_rooms(websocket)
            
            # Clean up player data if this was a player connection
            if connection_player_id:
                self._cleanup_player(connection_player_id)
    
    def get_connected_players(self, game_id: str) -> List[str]:
        """Get list of connected human players for a game."""
        return [
            player_id for player_id, player_game_id in self.player_games.items()
            if player_game_id == game_id and player_id in self.human_players
        ]
    
    def is_player_connected(self, player_id: str) -> bool:
        """Check if a human player is connected."""
        return player_id in self.human_players
    
    async def start_game_room(self, game_id: str):
        """Initialize a game room for the specified game."""
        if game_id not in self.game_rooms:
            self.game_rooms[game_id] = set()
        logger.info(f"Game room initialized for game: {game_id}")
    
    async def broadcast_to_game(self, game_id: str, data: Dict[str, Any]):
        """Broadcast data to all clients in the specified game room."""
        await self.broadcast_game_event(game_id, data)
    
    async def _handle_discover_games(self, websocket, payload):
        """Handle game discovery request."""
        games_list = []
        for game_id, game_info in self.active_games.items():
            games_list.append({
                "game_id": game_id,
                "status": game_info.get("status", "waiting"),
                "players": game_info.get("players", {}),
                "created": game_info.get("created"),
                "needs_humans": game_info.get("needs_humans", 0)
            })
        
        await self._send_response(websocket, {
            "type": "games_discovered",
            "payload": {
                "games": games_list,
                "latest_game": self.latest_game_id
            }
        })
    
    async def _handle_auto_connect(self, websocket, payload):
        """Handle auto-connection to latest game."""
        player_name = payload.get("player_name", "Human Player")
        
        if not self.latest_game_id:
            await self._send_error(websocket, "No active games available")
            return
        
        # Find next available human player slot
        game_info = self.active_games.get(self.latest_game_id, {})
        needed_players = game_info.get("needed_human_players", [])
        
        if not needed_players:
            await self._send_error(websocket, "No human player slots available")
            return
        
        # Assign first available slot
        player_id = needed_players[0]
        
        # Auto-authenticate this player
        await self._handle_authenticate_player(websocket, {
            "player_id": player_id,
            "game_id": self.latest_game_id,
            "player_name": player_name
        })
    
    def register_game(self, game_id: str, game_info: Dict[str, Any]):
        """Register a new game for auto-discovery."""
        self.active_games[game_id] = game_info
        self.latest_game_id = game_id
        logger.info(f"Registered game for auto-discovery: {game_id}")
    
    def unregister_game(self, game_id: str):
        """Unregister a game from auto-discovery."""
        if game_id in self.active_games:
            del self.active_games[game_id]
            if self.latest_game_id == game_id:
                # Find next most recent game or set to None
                if self.active_games:
                    self.latest_game_id = max(self.active_games.keys())
                else:
                    self.latest_game_id = None
            logger.info(f"Unregistered game from auto-discovery: {game_id}")
    
    def update_game_status(self, game_id: str, status: str, additional_info: Dict[str, Any] = None):
        """Update game status for auto-discovery."""
        if game_id in self.active_games:
            self.active_games[game_id]["status"] = status
            if additional_info:
                self.active_games[game_id].update(additional_info)


class HybridGameAdapter(LLMGameAdapter):
    """Enhanced game adapter that supports human players."""
    
    def __init__(self, game_manager, bridge_server: HybridGameBridgeServer):
        # Use the hybrid bridge server instead of the basic websocket server
        super().__init__(game_manager, bridge_server)
        self.bridge_server = bridge_server
        self.human_action_futures: Dict[str, asyncio.Future] = {}
        
        # Register callback for human actions
        self.bridge_server.register_action_callback(
            game_manager.game_id, 
            self._handle_human_action_response
        )
    
    async def _handle_human_action_response(self, action_response: Dict[str, Any]):
        """Handle action response from human player."""
        player_id = action_response.get('player_id')
        action_data = action_response.get('action_data')
        
        if player_id in self.human_action_futures:
            future = self.human_action_futures[player_id]
            if not future.done():
                # Extract the actual response text from action_data
                response_text = action_data.get('response', str(action_data))
                future.set_result(response_text)
            del self.human_action_futures[player_id]
    
    async def send_human_action_request(self, action_request: Dict[str, Any]) -> str:
        """Send action request to human player and wait for response."""
        player_id = action_request['player_id']
        
        # Create future for this action
        action_future = asyncio.Future()
        self.human_action_futures[player_id] = action_future
        
        # Send request
        success = await self.bridge_server.send_action_request(player_id, action_request)
        if not success:
            del self.human_action_futures[player_id]
            raise ValueError(f"Could not send action request to player {player_id}")
        
        try:
            # Wait for response with timeout
            timeout = action_request.get('timeout', 60.0)
            response = await asyncio.wait_for(action_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            if player_id in self.human_action_futures:
                del self.human_action_futures[player_id]
            raise asyncio.TimeoutError(f"Human player {player_id} timed out")
    
    async def stop_monitoring(self):
        """Stop monitoring and clean up."""
        await super().stop_monitoring()
        
        # Unregister callback
        self.bridge_server.unregister_action_callback(self.game_manager.game_id)
        
        # Cancel any pending human actions
        for future in self.human_action_futures.values():
            if not future.done():
                future.cancel()
        self.human_action_futures.clear()


# Singleton instance for the hybrid bridge
_hybrid_bridge_instance = None

def get_hybrid_bridge_instance(host: str = "localhost", port: int = 8765) -> HybridGameBridgeServer:
    """Get or create hybrid bridge server instance."""
    global _hybrid_bridge_instance
    if _hybrid_bridge_instance is None:
        _hybrid_bridge_instance = HybridGameBridgeServer(host, port)
    return _hybrid_bridge_instance

def create_hybrid_adapter(game_manager) -> HybridGameAdapter:
    """Create a hybrid game adapter with bidirectional communication support."""
    bridge_server = get_hybrid_bridge_instance()
    return HybridGameAdapter(game_manager, bridge_server)

async def start_hybrid_bridge_server(host: str = "localhost", port: int = 8765):
    """Start the hybrid bridge server."""
    server = get_hybrid_bridge_instance(host, port)
    await server.start_server()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start hybrid bridge server
    asyncio.run(start_hybrid_bridge_server())