#!/usr/bin/env python3
"""
Hybrid Secret Hitler Game Launcher

This script allows you to start a Secret Hitler game with a mix of AI and human players.
Human players connect through the web interface, while AI players are managed by the LLM engine.

Usage:
    python hybrid_game_launcher.py --config config/hybrid_game.json
    python hybrid_game_launcher.py --quick-start --humans 1 --ais 4
    python hybrid_game_launcher.py --interactive
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from web_bridge import start_hybrid_system, HybridGameIntegration
from hybrid_game_coordinator import HybridGameCoordinator, HybridPlayerConfig, PlayerType
from core.game_manager import GameManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class HybridGameLauncher:
    """Main launcher class for hybrid Secret Hitler games."""
    
    def __init__(self):
        self.integration: HybridGameIntegration = None
        self.coordinator: HybridGameCoordinator = None
        self.game_task: asyncio.Task = None
        
    async def start_system(self, host: str = "localhost", port: int = 8765):
        """Start the hybrid game system."""
        logger.info("🚀 Starting Hybrid Secret Hitler Game System")
        
        # Start the integration system
        self.integration = await start_hybrid_system(host, port)
        
        logger.info(f"✅ System started on {host}:{port}")
        logger.info("🌐 Web interface available for human players")
        
    async def stop_system(self):
        """Stop the hybrid game system."""
        if self.game_task and not self.game_task.done():
            logger.info("🛑 Stopping game...")
            self.game_task.cancel()
            try:
                await self.game_task
            except asyncio.CancelledError:
                pass
        
        if self.integration:
            await self.integration.stop()
        
        logger.info("✅ System stopped")
    
    async def launch_game(self, player_configs: List[HybridPlayerConfig], 
                         openrouter_api_key: str, game_id: str = None) -> Dict[str, Any]:
        """Launch a hybrid game with the given configuration."""
        
        if not self.integration:
            raise RuntimeError("System not started. Call start_system() first.")
        
        # Generate game ID if not provided
        if not game_id:
            game_id = f"hybrid_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🎮 Launching hybrid game: {game_id}")
        logger.info(f"👥 Players: {len(player_configs)} total")
        
        human_players = [p for p in player_configs if p.type == PlayerType.HUMAN]
        ai_players = [p for p in player_configs if p.type == PlayerType.AI]
        
        logger.info(f"🤖 AI Players: {len(ai_players)}")
        logger.info(f"👤 Human Players: {len(human_players)}")
        
        # Create the hybrid coordinator
        self.coordinator = HybridGameCoordinator(
            hybrid_players=player_configs,
            openrouter_api_key=openrouter_api_key,
            bridge_server=self.integration.bridge_server,
            game_id=game_id
        )
        
        # Start the game
        self.game_task = asyncio.create_task(self.coordinator.start_hybrid_game())
        
        # Print connection instructions for human players
        if human_players:
            await self._print_human_connection_instructions(human_players, game_id)
        
        # Wait for the game to complete
        try:
            result = await self.game_task
            logger.info("🏁 Game completed!")
            return result
        except asyncio.CancelledError:
            logger.info("🛑 Game cancelled")
            return {"cancelled": True}
        except Exception as e:
            logger.error(f"❌ Game error: {e}")
            return {"error": str(e)}
    
    async def _print_human_connection_instructions(self, human_players: List[HybridPlayerConfig], game_id: str):
        """Print connection instructions for human players."""
        logger.info("\n" + "="*60)
        logger.info("🌐 HUMAN PLAYER CONNECTION INSTRUCTIONS")
        logger.info("="*60)
        logger.info(f"Game ID: {game_id}")
        logger.info(f"WebSocket URL: ws://localhost:8765")
        logger.info("")
        logger.info("Human players should connect with:")
        
        for player in human_players:
            logger.info(f"  👤 {player.name} (ID: {player.id})")
        
        logger.info("")
        logger.info("Connection format (JSON WebSocket message):")
        logger.info(json.dumps({
            "type": "authenticate_player",
            "payload": {
                "player_id": "PLAYER_ID",
                "game_id": game_id
            }
        }, indent=2))
        logger.info("="*60)
        logger.info("")
        
        # Wait generously for players to connect
        logger.info("⏳ Waiting up to 5 minutes for human players to connect...")
        logger.info("💡 Take your time - no rush!")
        
        # Check connections every 30 seconds for up to 5 minutes
        for attempt in range(10):  # 10 attempts * 30 seconds = 5 minutes
            await asyncio.sleep(30)
            connected = self.integration.bridge_server.get_connected_players(game_id)
            
            if len(connected) >= len(human_players):
                logger.info(f"✅ All human players connected: {connected}")
                break
            
            missing = [p.id for p in human_players if p.id not in connected]
            minutes_left = (9 - attempt) * 0.5  # Remaining time in minutes
            logger.info(f"⏰ Connected: {connected}, Missing: {missing} ({minutes_left:.1f}min left)")
        
        # Final check
        final_connected = self.integration.bridge_server.get_connected_players(game_id)
        if len(final_connected) < len(human_players):
            missing = [p.id for p in human_players if p.id not in final_connected]
            logger.warning(f"⚠️  Some players didn't connect: {missing}")
            logger.info("🎮 Starting game with connected players...")
    
    def create_player_config(self, player_id: str, name: str, player_type: str, 
                           model: str = None) -> HybridPlayerConfig:
        """Create a player configuration."""
        return HybridPlayerConfig(
            id=player_id,
            name=name,
            type=PlayerType.HUMAN if player_type.lower() == 'human' else PlayerType.AI,
            model=model or os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
        )


async def launch_from_config(config_file: str):
    """Launch game from configuration file."""
    launcher = HybridGameLauncher()
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Start system
        await launcher.start_system(
            host=config.get('host', 'localhost'),
            port=config.get('port', 8765)
        )
        
        # Parse player configurations
        player_configs = []
        for player_config in config['players']:
            player_configs.append(HybridPlayerConfig(
                id=player_config['id'],
                name=player_config['name'],
                type=PlayerType.HUMAN if player_config['type'].lower() == 'human' else PlayerType.AI,
                model=player_config.get('model', os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp'))
            ))
        
        # Launch game
        api_key = config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key not found in config or environment")
        
        result = await launcher.launch_game(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id=config.get('game_id')
        )
        
        logger.info(f"🎉 Game Result: {result}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await launcher.stop_system()


async def quick_start(num_humans: int, num_ais: int):
    """Quick start with specified number of humans and AIs."""
    launcher = HybridGameLauncher()
    
    try:
        # Start system
        await launcher.start_system()
        
        # Create player configurations
        player_configs = []
        
        # Add human players
        for i in range(num_humans):
            player_configs.append(HybridPlayerConfig(
                id=f"human_{i+1}",
                name=f"Human Player {i+1}",
                type=PlayerType.HUMAN
            ))
        
        # Add AI players
        for i in range(num_ais):
            player_configs.append(HybridPlayerConfig(
                id=f"ai_{i+1}",
                name=f"AI Player {i+1}",
                type=PlayerType.AI,
                model=os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
            ))
        
        # Get API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Launch game
        result = await launcher.launch_game(
            player_configs=player_configs,
            openrouter_api_key=api_key
        )
        
        logger.info(f"🎉 Game Result: {result}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await launcher.stop_system()


async def interactive_setup():
    """Interactive setup for creating a hybrid game."""
    print("🎮 Interactive Hybrid Secret Hitler Game Setup")
    print("=" * 50)
    
    # Get basic configuration
    total_players = int(input("Total number of players (5-10): "))
    if total_players < 5 or total_players > 10:
        print("❌ Invalid number of players. Must be 5-10.")
        return
    
    num_humans = int(input("Number of human players: "))
    if num_humans > total_players:
        print("❌ Number of humans can't exceed total players.")
        return
    
    num_ais = total_players - num_humans
    print(f"📊 Configuration: {num_humans} humans, {num_ais} AIs")
    
    # Get player names
    player_configs = []
    
    print(f"\n👤 Enter names for {num_humans} human players:")
    for i in range(num_humans):
        name = input(f"Human player {i+1} name: ").strip() or f"Human {i+1}"
        player_configs.append(HybridPlayerConfig(
            id=f"human_{i+1}",
            name=name,
            type=PlayerType.HUMAN
        ))
    
    print(f"\n🤖 Enter names for {num_ais} AI players:")
    for i in range(num_ais):
        name = input(f"AI player {i+1} name: ").strip() or f"AI {i+1}"
        player_configs.append(HybridPlayerConfig(
            id=f"ai_{i+1}",
            name=name,
            type=PlayerType.AI,
            model=os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
        ))
    
    # Confirm configuration
    print(f"\n📋 Final Configuration:")
    for config in player_configs:
        print(f"  {config.type.value.upper()}: {config.name}")
    
    confirm = input("\nStart the game? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Game cancelled.")
        return
    
    # Launch the game
    launcher = HybridGameLauncher()
    
    try:
        await launcher.start_system()
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        result = await launcher.launch_game(
            player_configs=player_configs,
            openrouter_api_key=api_key
        )
        
        print(f"🎉 Game Result: {result}")
        
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await launcher.stop_system()


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hybrid_game.log')
        ]
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch a hybrid Secret Hitler game with AI and human players"
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--quick-start', '-q',
        action='store_true',
        help='Quick start mode'
    )
    
    parser.add_argument(
        '--humans',
        type=int,
        default=1,
        help='Number of human players (quick start mode)'
    )
    
    parser.add_argument(
        '--ais',
        type=int,
        default=4,
        help='Number of AI players (quick start mode)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive setup mode'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='WebSocket server host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='WebSocket server port (default: 8765)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    # Validate arguments
    if args.config and not Path(args.config).exists():
        logger.error(f"❌ Configuration file not found: {args.config}")
        return 1
    
    if args.quick_start and (args.humans + args.ais) not in range(5, 11):
        logger.error("❌ Total players must be between 5 and 10")
        return 1
    
    try:
        if args.config:
            asyncio.run(launch_from_config(args.config))
        elif args.quick_start:
            asyncio.run(quick_start(args.humans, args.ais))
        elif args.interactive:
            asyncio.run(interactive_setup())
        else:
            print("❌ Please specify --config, --quick-start, or --interactive mode")
            parser.print_help()
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())