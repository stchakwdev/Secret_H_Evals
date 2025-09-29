#!/usr/bin/env python3
"""
Test just the nomination phase to see AI responses.
"""
import asyncio
import os
from dotenv import load_dotenv
from core.game_manager import GameManager
from core.game_state import GamePhase

async def test_nomination():
    """Test nomination phase with detailed response logging."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🗳️  NOMINATION PHASE TEST")
    print("=" * 40)
    
    # Create player configs
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    # Create game manager
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id="nomination_test"
    )
    
    # Show roles
    print("\n🎭 Roles:")
    for pid, player in game_manager.game_state.players.items():
        role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
        print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
    
    try:
        async with game_manager.openrouter_client:
            
            # Send role information first
            print("\n📋 Sending role information...")
            await game_manager._send_role_information()
            
            # Set up nomination phase
            game_manager.game_state.phase = GamePhase.NOMINATION
            
            # Get president and eligible chancellors
            president = game_manager.game_state.get_current_president()
            eligible_chancellors = [
                p for p in game_manager.game_state.get_alive_players()
                if p.is_eligible_chancellor and p.id != president.id
            ]
            
            print(f"\n🏛️ President: {president.name}")
            print(f"🏛️ Eligible Chancellors: {', '.join(p.name for p in eligible_chancellors)}")
            
            # Build prompt for nomination
            prompt = game_manager.prompt_templates.get_nomination_prompt(
                president_name=president.name,
                eligible_chancellors=[p.name for p in eligible_chancellors],
                game_state=game_manager._get_public_game_state(),
                private_info=game_manager._get_private_info_for_player(president.id)
            )
            
            print(f"\n📝 Prompt length: {len(prompt)} characters")
            print("📝 Prompt preview:")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            
            # Make the request
            print(f"\n🤖 Requesting nomination from {president.name}...")
            response = await game_manager._make_player_request(
                player_id=president.id,
                prompt=prompt,
                decision_type="nominate_chancellor"
            )
            
            print(f"\n📨 Response received:")
            print(f"   Success: {response.success if hasattr(response, 'success') else 'Unknown'}")
            print(f"   Cost: ${response.cost if hasattr(response, 'cost') else 'Unknown':.4f}")
            print(f"   Response Content:")
            
            response_content = response.content if hasattr(response, 'content') else response
            print("   " + "-" * 50)
            print("   " + response_content.replace('\n', '\n   '))
            print("   " + "-" * 50)
            
            # Try to parse the nomination
            nominated_chancellor = game_manager._parse_nomination(response_content, eligible_chancellors)
            
            if nominated_chancellor:
                print(f"\n✅ Successfully parsed nomination: {nominated_chancellor.name}")
                
                # Try to execute the nomination
                success = game_manager.game_state.nominate_chancellor(nominated_chancellor.id)
                print(f"✅ Nomination executed: {success}")
                
            else:
                print(f"\n❌ Failed to parse nomination from response")
                print(f"   Looking for names: {[p.name for p in eligible_chancellors]}")
                print(f"   In response: {response_content.lower()}")
                
                # Try each name manually
                for chancellor in eligible_chancellors:
                    if chancellor.name.lower() in response_content.lower():
                        print(f"   ✅ Found '{chancellor.name}' in response")
                    else:
                        print(f"   ❌ '{chancellor.name}' not found in response")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_nomination())
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")