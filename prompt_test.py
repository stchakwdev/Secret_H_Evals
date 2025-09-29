#!/usr/bin/env python3
"""
Test just prompt generation to isolate the hang.
"""
import os
from dotenv import load_dotenv
from core.game_manager import GameManager

def test_prompt_generation():
    """Test prompt generation without LLM calls."""
    
    # Load environment variables
    load_dotenv()
    
    print("🧪 PROMPT GENERATION TEST")
    print("=" * 30)
    
    # Create player configs
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print("🔄 Step 1: Create GameManager...")
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key="fake_key",  # We won't make requests
        game_id="prompt_test"
    )
    print("✅ GameManager created")
    
    print("🔄 Step 2: Get public game state...")
    public_state = game_manager._get_public_game_state()
    print(f"✅ Public state keys: {list(public_state.keys())}")
    
    print("🔄 Step 3: Get private info for player...")
    private_info = game_manager._get_private_info_for_player("player1")
    print(f"✅ Private info keys: {list(private_info.keys())}")
    
    print("🔄 Step 4: Get role information...")
    role_info = game_manager._get_role_information_for_player("player1")
    print(f"✅ Role info keys: {list(role_info.keys())}")
    
    print("🔄 Step 5: Test role assignment prompt...")
    try:
        prompt = game_manager.prompt_templates.get_role_assignment_prompt(
            player_name="Alice",
            role="liberal",
            role_info=role_info,
            player_count=3
        )
        print(f"✅ Role assignment prompt generated (length: {len(prompt)})")
    except Exception as e:
        print(f"❌ Role assignment prompt failed: {e}")
        return False
    
    print("🔄 Step 6: Test nomination prompt...")
    try:
        # Get eligible chancellors
        president = game_manager.game_state.get_current_president()
        eligible_chancellors = [
            p for p in game_manager.game_state.get_alive_players()
            if p.is_eligible_chancellor and p.id != president.id
        ]
        
        print(f"   President: {president.name}")
        print(f"   Eligible: {[p.name for p in eligible_chancellors]}")
        
        # This is where it might be hanging
        print("   Generating prompt...")
        prompt = game_manager.prompt_templates.get_nomination_prompt(
            president_name=president.name,
            eligible_chancellors=[p.name for p in eligible_chancellors],
            game_state=public_state,
            private_info=private_info
        )
        print(f"✅ Nomination prompt generated (length: {len(prompt)})")
        
        # Show a preview
        print("   Preview (first 200 chars):")
        print("   " + prompt[:200] + "...")
        
    except Exception as e:
        print(f"❌ Nomination prompt failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ All prompt generation tests passed!")
    return True

if __name__ == "__main__":
    result = test_prompt_generation()
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")