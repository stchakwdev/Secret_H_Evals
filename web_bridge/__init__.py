"""
Web bridge module for connecting LLM game engine to web interface.
Provides real-time monitoring, visualization, and bidirectional communication for hybrid games.
"""

from .websocket_server import GameWebSocketServer, get_server_instance, start_websocket_server
from .event_converter import GameEventConverter, EventType, get_converter
from .game_adapter import LLMGameAdapter, create_adapter
from .bidirectional_bridge import (
    HybridGameBridgeServer, 
    HybridGameAdapter, 
    get_hybrid_bridge_instance, 
    create_hybrid_adapter,
    start_hybrid_bridge_server
)
from .hybrid_integration import (
    HumanPlayerInterface,
    HybridGameIntegration,
    get_integration_instance,
    start_hybrid_system
)

__all__ = [
    # Original components
    'GameWebSocketServer',
    'get_server_instance', 
    'start_websocket_server',
    'GameEventConverter',
    'EventType',
    'get_converter',
    'LLMGameAdapter',
    'create_adapter',
    
    # Hybrid game components
    'HybridGameBridgeServer',
    'HybridGameAdapter',
    'get_hybrid_bridge_instance',
    'create_hybrid_adapter',
    'start_hybrid_bridge_server',
    
    # Integration components
    'HumanPlayerInterface',
    'HybridGameIntegration',
    'get_integration_instance',
    'start_hybrid_system'
]