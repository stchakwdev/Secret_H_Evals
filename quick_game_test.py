#!/usr/bin/env python3
"""
Quick game test - run just a few phases to validate the game flow.
"""
import asyncio
import os
from dotenv import load_dotenv
from core.game_manager import GameManager
from core.game_state import GamePhase

async def test_quick_game():
    """Test just the first few phases of the game."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🎮 Quick Game Test - First Few Phases Only")
    print("=" * 50)
    
    # Create player configs
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player4", "name": "Diana", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player5", "name": "Eve", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print(f"Players: {', '.join([p['name'] for p in player_configs])}")
    
    # Create game manager
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id="quick_test"
    )
    
    # Display role assignments
    print("\n🎭 Role Assignments:")
    for pid, player in game_manager.game_state.players.items():
        role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
        print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
    
    try:
        print("\n🔄 Starting limited game test...")
        async with game_manager.openrouter_client:
            
            # Step 1: Send role information
            print("\n📋 Phase 1: Role Assignment")
            await game_manager._send_role_information()
            print("✅ Role information sent to all players")
            
            # Step 2: Try one nomination phase
            print("\n🗳️  Phase 2: Chancellor Nomination")
            game_manager.game_state.phase = GamePhase.NOMINATION
            
            # Get current president
            president = game_manager.game_state.get_current_president()
            print(f"Current President: {president.name}")
            
            # Get eligible chancellors
            eligible_chancellors = [
                p for p in game_manager.game_state.get_alive_players()
                if p.is_eligible_chancellor and p.id != president.id
            ]
            print(f"Eligible Chancellors: {', '.join(p.name for p in eligible_chancellors)}")
            
            # Run nomination phase
            await game_manager._handle_nomination_phase()
            print("✅ Nomination phase completed")
            
            # Check if chancellor was nominated
            if game_manager.game_state.government.chancellor:
                chancellor = game_manager.game_state.players[game_manager.game_state.government.chancellor]
                print(f"Chancellor nominated: {chancellor.name}")
                
                # Step 3: Try voting phase
                print("\n🗳️  Phase 3: Government Voting")
                game_manager.game_state.phase = GamePhase.VOTING
                await game_manager._handle_voting_phase()
                print("✅ Voting phase completed")
                
                # Display vote results
                if game_manager.game_state.voting_rounds:
                    last_round = game_manager.game_state.voting_rounds[-1]
                    print(f"Vote result: {last_round.result.value}")
                    ja_votes = sum(1 for v in last_round.votes if v.vote)
                    nein_votes = sum(1 for v in last_round.votes if not v.vote)
                    print(f"Votes: {ja_votes} Ja, {nein_votes} Nein")
            
            print("\n✅ Quick game test completed successfully!")
            
            # Display final state
            print(f"\nFinal Phase: {game_manager.game_state.phase.value}")
            print(f"Liberal Policies: {game_manager.game_state.policy_board.liberal_policies}")
            print(f"Fascist Policies: {game_manager.game_state.policy_board.fascist_policies}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_quick_game())
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")