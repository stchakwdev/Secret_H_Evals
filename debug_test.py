"""
Debug test to isolate the issue.
"""
import asyncio
import os
from dotenv import load_dotenv

async def debug_test():
    """Debug the initialization issue."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is available
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ No valid OpenRouter API key found!")
        print("   1. Get your API key from https://openrouter.ai/keys")
        print("   2. Add it to the .env file: OPENROUTER_API_KEY=your_key_here")
        return
    
    from core.game_manager import GameManager
    
    # Simple test config
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print("🔍 Debug Test - Creating GameManager")
    print(f"Player configs: {player_configs}")
    
    try:
        # Create game manager
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=os.environ['OPENROUTER_API_KEY'],
            game_id="debug_test"
        )
        
        print("✅ GameManager created successfully")
        print(f"Game state players: {list(game_manager.game_state.players.keys())}")
        print(f"Player contexts: {list(game_manager.player_contexts.keys())}")
        
        # Check game state
        for pid, player in game_manager.game_state.players.items():
            print(f"Player {pid}: {player.name} ({player.role.value})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_test())