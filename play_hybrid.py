#!/usr/bin/env python3
"""
Simplified Secret Hitler Hybrid Game Launcher

A one-command solution for playing Secret Hitler with AI opponents.
This script handles everything automatically - no manual setup required!

Usage:
    python play_hybrid.py                 # 1 human, 4 AI players (default)
    python play_hybrid.py --humans 2      # 2 humans, 3 AI players  
    python play_hybrid.py --ai-only       # Watch AI-only game (5 AI players)
    python play_hybrid.py --browser       # Open browser interface
    python play_hybrid.py --mode solo     # Preset: 1 human, 4 AI
    python play_hybrid.py --mode team     # Preset: 2 humans, 3 AI
"""

import asyncio
import argparse
import sys
import os
import webbrowser
import threading
import time
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_game_coordinator import HybridGameCoordinator, HybridPlayerConfig, PlayerType
from web_bridge import start_hybrid_system
from ui.terminal_interface import SecretHitlerTerminalUI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up clean logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced noise
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SimpleHybridLauncher:
    """Simplified launcher that handles everything automatically."""
    
    def __init__(self):
        self.integration = None
        self.game_task = None
        self.human_interface_task = None
        self.current_game_id = None
        
    async def play_game(self, humans: int = 1, ais: int = 4, browser: bool = False, 
                       ai_only: bool = False, auto_connect: bool = True):
        """Launch and play a complete hybrid game with minimal user interaction."""
        
        if ai_only:
            humans, ais = 0, 5
            
        total_players = humans + ais
        if total_players < 5 or total_players > 10:
            print("❌ Total players must be between 5 and 10")
            return
        
        print("🎮 " + "="*50)
        print("🎮 SECRET HITLER - HYBRID GAME LAUNCHER")
        print("🎮 " + "="*50)
        print(f"👥 Players: {humans} human{'s' if humans != 1 else ''}, {ais} AI")
        print(f"🤖 AI Model: {os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')}")
        print()
        
        try:
            # Start the hybrid system
            print("🚀 Starting game server...")
            self.integration = await start_hybrid_system()
            print("✅ Server started on localhost:8765")
            
            # Generate game ID
            self.current_game_id = f"hybrid_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create player configurations
            player_configs = []
            
            # Add human players
            for i in range(humans):
                player_configs.append(HybridPlayerConfig(
                    id=f"human_{i+1}",
                    name=f"Human Player {i+1}",
                    type=PlayerType.HUMAN
                ))
            
            # Add AI players
            for i in range(ais):
                player_configs.append(HybridPlayerConfig(
                    id=f"ai_{i+1}",
                    name=f"AI Player {i+1}",
                    type=PlayerType.AI,
                    model=os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
                ))
            
            # Register game for auto-discovery
            human_player_ids = [f"human_{i+1}" for i in range(humans)]
            game_info = {
                "status": "waiting",
                "created": datetime.now().isoformat(),
                "needs_humans": humans,
                "needed_human_players": human_player_ids,
                "total_players": len(player_configs),
                "players": {p.id: {"name": p.name, "type": p.type.value} for p in player_configs}
            }
            self.integration.bridge_server.register_game(self.current_game_id, game_info)
            
            # Create coordinator  
            coordinator = HybridGameCoordinator(
                hybrid_players=player_configs,
                openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
                bridge_server=self.integration.bridge_server,
                game_id=self.current_game_id
            )
            
            # Start human interface if needed
            if humans > 0:
                if not browser:
                    print("🖥️  Starting terminal interface for human players...")
                    self.human_interface_task = asyncio.create_task(
                        self._run_terminal_interface()
                    )
                else:
                    print("🌐 Opening browser interface...")
                    threading.Timer(2.0, lambda: webbrowser.open("http://localhost:8080")).start()
                
                print("💡 Human players can now connect automatically!")
                print("   No need to specify game IDs - the system will handle it!")
                await asyncio.sleep(1)  # Brief pause for setup
            
            print("🎲 Starting Secret Hitler game...")
            print()
            
            # Start the game
            self.game_task = asyncio.create_task(coordinator.start_hybrid_game())
            
            # Wait for game completion
            result = await self.game_task
            
            print()
            print("🏁 Game Complete!")
            if 'error' not in result:
                print("🎉 Thanks for playing Secret Hitler!")
            else:
                print(f"⚠️  Game ended with issue: {result.get('error')}")
            
            return result
            
        except KeyboardInterrupt:
            print("\n🛑 Game interrupted by user")
            return {"cancelled": True}
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return {"error": str(e)}
        finally:
            await self._cleanup()
    
    async def _auto_connect_humans(self, coordinator, num_humans):
        """Automatically connect human players without manual intervention."""
        for i in range(num_humans):
            player_id = f"human_{i+1}"
            
            # Simulate auto-authentication
            await self.integration.bridge_server.start_game_room(self.current_game_id)
            
            # Add player to authenticated list
            self.integration.bridge_server.human_players[player_id] = "auto_connected"
            self.integration.bridge_server.player_games[player_id] = self.current_game_id
            
            print(f"✅ Auto-connected {player_id}")
    
    async def _run_terminal_interface(self):
        """Run Rich terminal interface for human players."""
        try:
            # Create Rich terminal UI
            terminal_ui = SecretHitlerTerminalUI(
                player_name="Human Player",
                websocket_uri=f"ws://localhost:8765"
            )
            
            # Run the terminal interface
            await terminal_ui.run()
            
        except Exception as e:
            print(f"❌ Terminal interface error: {e}")
        except asyncio.CancelledError:
            print("🛑 Terminal interface stopped")
    
    async def _cleanup(self):
        """Clean up resources."""
        if self.game_task and not self.game_task.done():
            self.game_task.cancel()
            
        if self.human_interface_task and not self.human_interface_task.done():
            self.human_interface_task.cancel()
            
        if self.integration:
            await self.integration.stop()

def get_preset_config(mode: str) -> Dict[str, int]:
    """Get preset configurations."""
    presets = {
        'solo': {'humans': 1, 'ais': 4},
        'team': {'humans': 2, 'ais': 3}, 
        'duo': {'humans': 2, 'ais': 3},
        'spectate': {'humans': 0, 'ais': 5},
        'ai-only': {'humans': 0, 'ais': 5}
    }
    return presets.get(mode, {'humans': 1, 'ais': 4})

def main():
    """Main entry point - handles all command line arguments."""
    parser = argparse.ArgumentParser(
        description="🎮 Secret Hitler Hybrid Game - One Command, Ready to Play!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_hybrid.py              # Default: 1 human vs 4 AI
  python play_hybrid.py --humans 2   # 2 humans vs 3 AI
  python play_hybrid.py --ai-only    # Watch AI-only game
  python play_hybrid.py --browser    # Use browser interface
  python play_hybrid.py --mode team  # Preset for team play
        """
    )
    
    parser.add_argument(
        '--humans', 
        type=int, 
        default=1,
        help='Number of human players (default: 1)'
    )
    
    parser.add_argument(
        '--ais',
        type=int,
        default=4, 
        help='Number of AI players (default: 4)'
    )
    
    parser.add_argument(
        '--ai-only',
        action='store_true',
        help='Watch AI-only game (overrides human/ai counts)'
    )
    
    parser.add_argument(
        '--browser',
        action='store_true', 
        help='Open browser interface instead of terminal'
    )
    
    parser.add_argument(
        '--mode',
        choices=['solo', 'team', 'duo', 'spectate', 'ai-only'],
        help='Use preset configuration'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.mode:
        config = get_preset_config(args.mode)
        args.humans = config['humans']
        args.ais = config['ais']
        
    # Validate API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Missing OPENROUTER_API_KEY in .env file")
        print("   Please add your OpenRouter API key to the .env file")
        return 1
    
    # Launch the game
    launcher = SimpleHybridLauncher()
    
    try:
        result = asyncio.run(launcher.play_game(
            humans=args.humans,
            ais=args.ais,
            browser=args.browser,
            ai_only=args.ai_only
        ))
        
        return 0 if result and 'error' not in result else 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())