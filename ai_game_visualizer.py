#!/usr/bin/env python3
"""
AI Game Visualizer - Watch AI agents play Secret Hitler in the beautiful web UI

This script creates AI-only Secret Hitler games and provides a web interface
to watch the AI players make strategic decisions, lie, vote, and play the full game.

Usage:
    python ai_game_visualizer.py                    # Start AI game with web UI
    python ai_game_visualizer.py --browser          # Auto-open browser
    python ai_game_visualizer.py --players 7        # 7 AI players instead of 5
    python ai_game_visualizer.py --fast             # Faster decision making
    python ai_game_visualizer.py --verbose          # Show detailed AI reasoning
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
import json

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_game_coordinator import HybridGameCoordinator, HybridPlayerConfig, PlayerType
from web_bridge import start_hybrid_system
from core.game_manager import GameManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class AIGameVisualizer:
    """Visualizer for AI-only Secret Hitler games with web interface."""
    
    def __init__(self):
        self.integration = None
        self.game_task = None
        self.current_game_id = None
        self.spectator_url = None
        
    async def start_ai_game(self, players: int = 5, browser: bool = False, 
                           fast: bool = False, verbose: bool = False):
        """Start an AI-only game with web visualization."""
        
        if players < 5 or players > 10:
            print("❌ Number of players must be between 5 and 10")
            return False
            
        # Check API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key or api_key == 'your_openrouter_api_key_here':
            print("❌ No valid OpenRouter API key found!")
            print("   Add OPENROUTER_API_KEY=your_key to .env file")
            return False
        
        print("🎮 " + "="*60)
        print("🎮 SECRET HITLER - AI GAME VISUALIZER")
        print("🎮 " + "="*60)
        print(f"🤖 AI Players: {players}")
        print(f"🎯 Model: {os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')}")
        print(f"⚡ Fast mode: {'Enabled' if fast else 'Disabled'}")
        print(f"💭 Verbose reasoning: {'Enabled' if verbose else 'Disabled'}")
        print("🎭 Watch AI agents lie, strategize, and play!")
        print("=" * 60)
        
        try:
            # Start the hybrid system with spectator support
            print("\n🚀 Starting game server...")
            self.integration = await start_hybrid_system(host="localhost", port=8765)
            print("✅ Server started on localhost:8765")
            
            # Generate game ID
            self.current_game_id = f"ai_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create diverse AI player personalities
            ai_personalities = [
                ("Machiavelli", "A cunning strategist who excels at deception and manipulation"),
                ("Sherlock", "An analytical detective focused on finding logical inconsistencies"),
                ("Diplomat", "A smooth talker who builds alliances and trust"),
                ("Wildcard", "An unpredictable player who makes surprising moves"),
                ("Guardian", "A protective player who tries to shield others"),
                ("Hawk", "An aggressive player who pushes for bold actions"),
                ("Sphinx", "A mysterious player who reveals little about their thinking"),
                ("Oracle", "An intuitive player who makes decisions based on patterns"),
                ("Rebel", "A contrarian who questions everything and everyone"),
                ("Phoenix", "A resilient player who recovers quickly from setbacks")
            ]
            
            # Create AI player configurations with personalities
            player_configs = []
            for i in range(players):
                personality = ai_personalities[i % len(ai_personalities)]
                player_configs.append(HybridPlayerConfig(
                    id=f"ai_{i+1}",
                    name=personality[0],
                    type=PlayerType.AI,
                    model=os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-v3.2-exp')
                ))
            
            print(f"\n🎭 AI Player Personalities:")
            player_personalities = {}
            for i, config in enumerate(player_configs):
                personality = ai_personalities[i % len(ai_personalities)]
                player_personalities[config.id] = personality[1]
                print(f"   🤖 {config.name}: {personality[1]}")
            
            # Create spectator interface URL
            self.spectator_url = f"http://localhost:8765/spectator?gameId={self.current_game_id}"
            
            print(f"\n🌐 Web Interface URLs:")
            print(f"   👀 Spectator: {self.spectator_url}")
            print(f"   📊 Game Monitor: http://localhost:8765/monitor?gameId={self.current_game_id}")
            print(f"   📈 Live Stats: http://localhost:8765/stats?gameId={self.current_game_id}")
            
            # Open browser if requested
            if browser:
                print(f"\n🌍 Opening browser...")
                def open_browser():
                    time.sleep(2)  # Give server time to start
                    webbrowser.open(self.spectator_url)
                
                browser_thread = threading.Thread(target=open_browser)
                browser_thread.daemon = True
                browser_thread.start()
            
            print(f"\n🎯 Starting AI-only Secret Hitler game...")
            print(f"   Game ID: {self.current_game_id}")
            print(f"   Watch the game unfold in real-time!")
            print("-" * 60)
            
            # Create the hybrid coordinator
            coordinator = HybridGameCoordinator(
                hybrid_players=player_configs,
                openrouter_api_key=api_key,
                bridge_server=self.integration.bridge_server,
                game_id=self.current_game_id
            )
            
            # Set player personalities for spectator display
            if hasattr(coordinator, 'game_manager') and hasattr(coordinator.game_manager, '_set_player_personalities'):
                coordinator.game_manager._set_player_personalities(player_personalities)
            
            # Start the game
            self.game_task = asyncio.create_task(coordinator.start_hybrid_game())
            
            # Monitor the game progress
            await self._monitor_game_progress()
            
            # Wait for game completion
            result = await self.game_task
            
            print("\n" + "="*60)
            print("🎉 AI GAME COMPLETED!")
            print("="*60)
            
            if "error" in result:
                print(f"❌ Game ended with error: {result['error']}")
                return False
            else:
                print(f"🏆 Winner: {result['winner']}")
                print(f"🎯 Win Condition: {result['win_condition']}")
                print(f"💰 Cost: ${result['cost_summary']['total_cost']:.4f}")
                print(f"⏱️  Duration: {result['duration']:.1f}s")
                print(f"📊 Actions: {result['cost_summary']['total_requests']}")
                
                print(f"\n📁 Game logs saved to: logs/{self.current_game_id}/")
                print(f"🎬 Replay available at: {self.spectator_url}&replay=true")
                
                return True
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Game stopped by user")
            return False
        except Exception as e:
            print(f"\n❌ Error starting AI game: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.integration:
                await self.integration.stop()
    
    async def _monitor_game_progress(self):
        """Monitor game progress and provide real-time updates."""
        print("\n📺 LIVE GAME MONITOR")
        print("="*30)
        
        # This will be enhanced to show real-time game events
        # For now, we'll just show that monitoring is active
        while not self.game_task.done():
            await asyncio.sleep(5)
            if not self.game_task.done():
                print(f"⏰ Game in progress... (Watch at: {self.spectator_url})")
    
    async def stop_system(self):
        """Stop the AI game system."""
        if self.game_task and not self.game_task.done():
            logger.info("🛑 Stopping AI game...")
            self.game_task.cancel()
            try:
                await self.game_task
            except asyncio.CancelledError:
                pass
        
        if self.integration:
            await self.integration.stop()
        
        logger.info("✅ AI game system stopped")

async def main():
    """Main entry point for the AI Game Visualizer."""
    
    parser = argparse.ArgumentParser(
        description="AI Game Visualizer - Watch AI agents play Secret Hitler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ai_game_visualizer.py                    # 5 AI players, basic mode
    python ai_game_visualizer.py --browser          # Auto-open browser
    python ai_game_visualizer.py --players 7 --fast # 7 players, faster decisions
    python ai_game_visualizer.py --verbose          # Show detailed AI reasoning
    python ai_game_visualizer.py --players 10 --browser --verbose  # Full experience
        """
    )
    
    parser.add_argument('--players', type=int, default=5, choices=range(5, 11),
                       help='Number of AI players (5-10)')
    parser.add_argument('--browser', action='store_true',
                       help='Automatically open browser to spectator interface')
    parser.add_argument('--fast', action='store_true',
                       help='Use faster decision making (shorter prompts)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed AI reasoning in the interface')
    
    args = parser.parse_args()
    
    # Create and start the AI game visualizer
    visualizer = AIGameVisualizer()
    
    try:
        success = await visualizer.start_ai_game(
            players=args.players,
            browser=args.browser,
            fast=args.fast,
            verbose=args.verbose
        )
        
        if success:
            print(f"\n🎉 AI game completed successfully!")
            print(f"   Thank you for watching AI agents master Secret Hitler!")
        else:
            print(f"\n💥 AI game did not complete successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    finally:
        await visualizer.stop_system()

if __name__ == "__main__":
    print("🤖 Starting AI Game Visualizer...")
    print("   Watch AI agents play Secret Hitler with strategic deception!")
    
    asyncio.run(main())