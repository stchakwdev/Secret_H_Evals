#!/usr/bin/env python3
"""
Minimal test to isolate where the hang is occurring.
"""
import asyncio
import os
from dotenv import load_dotenv

async def test_minimal():
    """Test the most basic game setup."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ No valid OpenRouter API key found!")
        return False
    
    print("🔄 Step 1: Import GameManager...")
    from core.game_manager import GameManager
    print("✅ GameManager imported")
    
    print("🔄 Step 2: Create player configs...")
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    print("✅ Player configs created")
    
    print("🔄 Step 3: Create GameManager...")
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id="minimal_test"
    )
    print("✅ GameManager created")
    
    print("🔄 Step 4: Display role assignments...")
    for pid, player in game_manager.game_state.players.items():
        print(f"   {player.name}: {player.role.value}")
    
    print("🔄 Step 5: Test single prompt template...")
    from agents.prompt_templates import PromptTemplates
    templates = PromptTemplates()
    
    # Get role info properly from game manager
    role_info = game_manager._get_role_information_for_player("player1")
    prompt = templates.get_role_assignment_prompt(
        player_name="Alice",
        role="liberal", 
        role_info=role_info,
        player_count=3
    )
    print(f"✅ Prompt generated (length: {len(prompt)})")
    
    print("🔄 Step 6: Test single LLM request...")
    try:
        async with game_manager.openrouter_client:
            response = await game_manager.openrouter_client.make_request(
                prompt="You are Alice playing Secret Hitler. You are a Liberal. Just say 'I understand my role as a Liberal.'",
                decision_type="test",
                player_id="player1"
            )
            print(f"✅ LLM request successful: {response.success}")
            print(f"   Response: {response.content[:100]}...")
            print(f"   Cost: ${response.cost:.4f}")
    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🔄 Step 7: Test role assignment method...")
    try:
        async with game_manager.openrouter_client:
            print("   Calling _send_role_information...")
            await game_manager._send_role_information()
            print("✅ Role information sent successfully")
    except Exception as e:
        print(f"❌ Role assignment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ All steps completed successfully!")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_minimal())
    print(f"\n{'✅ SUCCESS' if result else '❌ FAILED'}")