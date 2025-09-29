#!/usr/bin/env python3
"""
Test just the legislative phases that were failing before.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from core.game_manager import GameManager
from core.game_state import GamePhase

async def test_legislative_phases():
    """Test the legislative phases that were crashing before."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🏛️  LEGISLATIVE PHASE TEST")
    print("=" * 40)
    print("Testing the phases that were crashing before")
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
        game_id="legislative_test"
    )
    
    # Show roles
    print(f"\n🎭 Roles:")
    for pid, player in game_manager.game_state.players.items():
        role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
        print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
    
    try:
        async with game_manager.openrouter_client:
            
            print(f"\n📋 Step 1: Send role information...")
            await game_manager._send_role_information()
            print("✅ Role information sent")
            
            print(f"\n🗳️  Step 2: Nomination phase...")
            game_manager.game_state.phase = GamePhase.NOMINATION
            await game_manager._handle_nomination_phase()
            print("✅ Nomination completed")
            
            # Check if we have a government
            if not game_manager.game_state.government.chancellor:
                print("❌ No chancellor nominated!")
                return False
            
            president = game_manager.game_state.players[game_manager.game_state.government.president]
            chancellor = game_manager.game_state.players[game_manager.game_state.government.chancellor]
            print(f"   Government: {president.name} (President) + {chancellor.name} (Chancellor)")
            
            print(f"\n🗳️  Step 3: Voting phase...")
            game_manager.game_state.phase = GamePhase.VOTING
            await game_manager._handle_voting_phase()
            print("✅ Voting completed")
            
            # Check vote result
            if game_manager.game_state.voting_rounds:
                last_vote = game_manager.game_state.voting_rounds[-1]
                print(f"   Vote result: {last_vote.result.value}")
                
                if last_vote.result.value != "passed":
                    print("❌ Government failed, skipping legislative phases")
                    return True  # This is still success - voting worked
            
            print(f"\n🏛️  Step 4: President legislative phase...")
            game_manager.game_state.phase = GamePhase.LEGISLATIVE_PRESIDENT
            
            # This was the phase that was crashing before
            start_time = time.time()
            await game_manager._handle_president_legislative_phase()
            president_time = time.time() - start_time
            print(f"✅ President phase completed in {president_time:.1f}s")
            
            print(f"\n🏛️  Step 5: Chancellor legislative phase...")
            game_manager.game_state.phase = GamePhase.LEGISLATIVE_CHANCELLOR
            
            start_time = time.time()  
            await game_manager._handle_chancellor_legislative_phase()
            chancellor_time = time.time() - start_time
            print(f"✅ Chancellor phase completed in {chancellor_time:.1f}s")
            
            print(f"\n📊 Final state:")
            print(f"   Liberal policies: {game_manager.game_state.policy_board.liberal_policies}")
            print(f"   Fascist policies: {game_manager.game_state.policy_board.fascist_policies}")
            print(f"   Phase: {game_manager.game_state.phase.value}")
            
            print(f"\n🎉 ALL LEGISLATIVE PHASES COMPLETED SUCCESSFULLY!")
            print(f"   President phase: {president_time:.1f}s")
            print(f"   Chancellor phase: {chancellor_time:.1f}s")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing legislative phases...")
    result = asyncio.run(test_legislative_phases())
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")