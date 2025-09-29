"""
Trace test to find exact location of 'player1' error.
"""
import asyncio
import os
import traceback
from dotenv import load_dotenv

async def trace_test():
    """Trace where the 'player1' error occurs."""
    
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
    
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player4", "name": "Diana", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player5", "name": "Eve", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print("🔍 Tracing the error location...")
    
    try:
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=os.environ['OPENROUTER_API_KEY'],
            game_id="trace_test"
        )
        
        print("✅ GameManager created")
        print(f"Players in game_state: {list(game_manager.game_state.players.keys())}")
        print(f"Players in contexts: {list(game_manager.player_contexts.keys())}")
        
        # Try to start the game
        result = await game_manager.start_game()
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error caught: {repr(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trace_test())