"""
Quick test - just the role assignment phase.
"""
import asyncio
import os
from dotenv import load_dotenv

async def quick_test():
    """Test just role assignment."""
    
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
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print("🎮 Quick Test - Role Assignment Only")
    
    try:
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=os.environ['OPENROUTER_API_KEY'],
            game_id="quick_test"
        )
        
        print("✅ GameManager created")
        
        # Just test role assignment
        async with game_manager.openrouter_client:
            await game_manager.logger.log_game_start(game_manager.game_state.to_dict())
            print("✅ Game start logged")
            
            await game_manager._send_role_information()
            print("✅ Role information sent successfully!")
            
            print("\n🎭 Final Roles:")
            for pid, player in game_manager.game_state.players.items():
                print(f"  {player.name}: {player.role.value}")
            
            # Check cost
            cost_summary = game_manager.openrouter_client.get_cost_summary()
            print(f"\n💰 Total Cost: ${cost_summary.get('total_cost', 0):.4f}")
            print(f"API Requests: {cost_summary.get('total_requests', 0)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())