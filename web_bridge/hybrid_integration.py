"""
Integration module that connects the hybrid game coordinator with the bidirectional bridge.
This module provides the interface between the game engine and web UI for human players.
"""
import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging

from .bidirectional_bridge import HybridGameBridgeServer, HybridGameAdapter, get_hybrid_bridge_instance
from .http_server import IntegratedWebServer

logger = logging.getLogger(__name__)

class HumanPlayerInterface:
    """Interface for managing human player interactions through the web bridge."""
    
    def __init__(self, bridge_server: HybridGameBridgeServer):
        self.bridge_server = bridge_server
        self.active_games: Dict[str, Any] = {}  # game_id -> game_manager
        
    def register_game(self, game_manager):
        """Register a game manager with the interface."""
        game_id = game_manager.game_id
        self.active_games[game_id] = game_manager
        
        # Set up the human action callback for the game manager
        if hasattr(game_manager, 'human_action_callback') and game_manager.human_action_callback is None:
            game_manager.human_action_callback = self._create_human_action_callback(game_id)
        
        logger.info(f"Registered game {game_id} with human player interface")
    
    def unregister_game(self, game_id: str):
        """Unregister a game from the interface."""
        if game_id in self.active_games:
            del self.active_games[game_id]
            self.bridge_server.unregister_action_callback(game_id)
            logger.info(f"Unregistered game {game_id} from human player interface")
    
    def _create_human_action_callback(self, game_id: str) -> Callable:
        """Create a human action callback for a specific game."""
        
        async def human_action_callback(action_request: Dict[str, Any]) -> str:
            """Handle human action requests from the game manager."""
            try:
                player_id = action_request['player_id']
                decision_type = action_request['decision_type']
                
                # Add game context and request ID
                enhanced_request = {
                    'game_id': game_id,
                    'player_id': player_id,
                    'decision_type': decision_type,
                    'prompt': action_request['prompt'],
                    'game_state': action_request['game_state'],
                    'private_info': action_request['private_info'],
                    'request_id': f"{game_id}_{player_id}_{datetime.now().isoformat()}",
                    'timestamp': datetime.now().isoformat(),
                    'timeout': 60.0  # Default timeout
                }
                
                # Check if player is connected
                if not self.bridge_server.is_player_connected(player_id):
                    logger.warning(f"Player {player_id} not connected, using default response")
                    return self._get_default_response(decision_type)
                
                # Send request to human player
                success = await self.bridge_server.send_action_request(player_id, enhanced_request)
                if not success:
                    logger.warning(f"Failed to send action request to {player_id}")
                    return self._get_default_response(decision_type)
                
                # Wait for response (the bridge server will call the registered callback)
                # For now, return a placeholder - this will be handled by the GameManager's Future mechanism
                return "HUMAN_ACTION_PENDING"
                
            except Exception as e:
                logger.error(f"Error in human action callback: {e}")
                return self._get_default_response(action_request.get('decision_type', 'unknown'))
        
        return human_action_callback
    
    def _get_default_response(self, decision_type: str) -> str:
        """Get default response for disconnected/timed-out human players."""
        defaults = {
            "acknowledge_role": "I understand my role.",
            "nominate_chancellor": "I nominate the first available player.",
            "vote_on_government": "Nein",
            "choose_policies_as_president": "I choose the first two policies.",
            "choose_policies_as_chancellor": "I choose the first policy.",
            "investigate_player": "I investigate the first available player.",
            "special_election": "I choose the first available player.",
            "execute_player": "I execute the first available player."
        }
        return defaults.get(decision_type, "I take the default action.")
    
    async def authenticate_human_player(self, player_id: str, game_id: str, websocket) -> bool:
        """Authenticate a human player for a game."""
        try:
            # The authentication is handled by the bridge server
            # We just need to verify the game exists
            if game_id not in self.active_games:
                logger.warning(f"Attempted to authenticate player {player_id} for unknown game {game_id}")
                return False
            
            logger.info(f"Authenticated human player {player_id} for game {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating player {player_id}: {e}")
            return False
    
    def get_game_state_for_player(self, player_id: str, game_id: str) -> Optional[Dict[str, Any]]:
        """Get current game state for a specific human player."""
        if game_id not in self.active_games:
            return None
        
        game_manager = self.active_games[game_id]
        
        try:
            # Get public game state
            public_state = game_manager._get_public_game_state()
            
            # Get private info for this player
            private_info = game_manager._get_private_info_for_player(player_id)
            
            return {
                'public_state': public_state,
                'private_info': private_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting game state for player {player_id}: {e}")
            return None
    
    def get_connected_human_players(self, game_id: str) -> List[str]:
        """Get list of connected human players for a game."""
        return self.bridge_server.get_connected_players(game_id)


class HybridGameIntegration:
    """Main integration class that sets up the complete hybrid game system."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, http_port: int = 8080):
        self.host = host
        self.port = port
        self.http_port = http_port
        self.bridge_server = get_hybrid_bridge_instance(host, port)
        self.human_interface = HumanPlayerInterface(self.bridge_server)
        self.web_server = IntegratedWebServer(port, http_port)
        self.server_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """Start the hybrid integration system."""
        if self.running:
            return
        
        logger.info("Starting hybrid game integration system")
        
        # Start the bridge server in the background
        self.server_task = asyncio.create_task(self.bridge_server.start_server())
        
        # Start the integrated web server
        await self.web_server.start_integrated_server(self.bridge_server)
        
        self.running = True
        
        # Give the server a moment to start
        await asyncio.sleep(0.5)
        
        logger.info(f"Hybrid integration system started on {self.host}:{self.port}")
        logger.info(f"Web interface available at http://{self.host}:{self.http_port}")
    
    async def stop(self):
        """Stop the hybrid integration system."""
        if not self.running:
            return
        
        logger.info("Stopping hybrid integration system")
        
        self.running = False
        
        # Stop the web server
        await self.web_server.stop_integrated_server()
        
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Hybrid integration system stopped")
    
    def create_hybrid_game_manager(self, player_configs, openrouter_api_key, game_id=None):
        """Create a GameManager configured for hybrid gameplay."""
        from core.game_manager import GameManager
        
        # Create enhanced human action callback
        def enhanced_human_callback(action_request):
            return self.human_interface._create_human_action_callback(
                action_request.get('game_id', game_id or 'unknown')
            )(action_request)
        
        # Create the game manager with human support
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=openrouter_api_key,
            game_id=game_id,
            human_action_callback=enhanced_human_callback,
            human_timeout=60.0
        )
        
        # Register with the human interface
        self.human_interface.register_game(game_manager)
        
        return game_manager
    
    def create_hybrid_adapter(self, game_manager) -> HybridGameAdapter:
        """Create a hybrid adapter for monitoring and visualization."""
        return HybridGameAdapter(game_manager, self.bridge_server)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Global integration instance
_integration_instance = None

def get_integration_instance(host: str = "localhost", port: int = 8765) -> HybridGameIntegration:
    """Get or create the global hybrid integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = HybridGameIntegration(host, port)
    return _integration_instance

async def start_hybrid_system(host: str = "localhost", port: int = 8765) -> HybridGameIntegration:
    """Start the complete hybrid game system."""
    integration = get_integration_instance(host, port)
    await integration.start()
    return integration

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        integration = await start_hybrid_system()
        
        try:
            logger.info("Hybrid system running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await integration.stop()
    
    asyncio.run(main())