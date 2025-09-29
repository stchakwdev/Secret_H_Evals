#!/usr/bin/env python3
"""
Test role assignment to all players - this was working in the background test.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from core.game_manager import GameManager

async def test_role_assignments():
    """Test role assignment to all players."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🎭 ROLE ASSIGNMENT TEST")
    print("=" * 30)
    
    # Create 3-player game (faster)
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id="role_test"
    )
    
    # Show roles
    print(f"\n🎭 Role assignments:")
    for pid, player in game_manager.game_state.players.items():
        role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
        print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
    
    try:
        print(f"\n🔄 Sending role information to all players...")
        start_time = time.time()
        
        async with game_manager.openrouter_client:
            
            # Add overall timeout for the entire operation
            await asyncio.wait_for(
                game_manager._send_role_information(),
                timeout=60.0  # 60 second timeout for all 3 players
            )
            
            total_time = time.time() - start_time
            
            print(f"\n✅ ALL ROLE ASSIGNMENTS COMPLETED!")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average per player: {total_time/3:.1f}s")
            
            return True
            
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Role assignment timed out after {elapsed:.1f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Role assignment failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing role assignments...")
    result = asyncio.run(test_role_assignments())
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")