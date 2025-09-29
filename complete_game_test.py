#!/usr/bin/env python3
"""
Complete game test - verify full AI gameplay works end-to-end.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from core.game_manager import GameManager

async def test_complete_game():
    """Test a complete Secret Hitler AI game."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🎮 COMPLETE SECRET HITLER AI GAME TEST")
    print("=" * 60)
    print("Testing full gameplay with AI agents making strategic decisions")
    print("Using FREE Grok models for zero-cost testing")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create player configs (5 players for faster game)
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player4", "name": "Diana", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player5", "name": "Eve", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print(f"🎭 Players: {', '.join([p['name'] for p in player_configs])}")
    
    # Create game manager
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id=f"complete_test_{int(time.time())}"
    )
    
    # Display role assignments
    print(f"\n🎭 SECRET ROLE ASSIGNMENTS:")
    fascist_team = []
    liberal_team = []
    hitler_player = None
    
    for pid, player in game_manager.game_state.players.items():
        role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
        print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
        
        if player.role.value == "hitler":
            hitler_player = player.name
            fascist_team.append(player.name)
        elif player.role.value == "fascist":
            fascist_team.append(player.name)
        else:
            liberal_team.append(player.name)
    
    print(f"\n🎯 TEAMS:")
    print(f"   🔵 Liberals ({len(liberal_team)}): {', '.join(liberal_team)}")
    print(f"   🔴 Fascists ({len(fascist_team)}): {', '.join(fascist_team)}")
    print(f"   👑 Hitler: {hitler_player}")
    
    try:
        print(f"\n🚀 STARTING GAME...")
        print("-" * 60)
        
        # Start the game with detailed monitoring
        result = await game_manager.start_game()
        
        elapsed_time = time.time() - start_time
        
        print("-" * 60)
        
        # Check result
        if "error" in result:
            print(f"❌ GAME FAILED: {result['error']}")
            if 'game_state' in result:
                print(f"\n💥 Final state when game failed:")
                final_state = result['game_state']
                print(f"   Phase: {final_state.get('phase', 'unknown')}")
                print(f"   Liberal Policies: {final_state.get('policy_board', {}).get('liberal_policies', '?')}")
                print(f"   Fascist Policies: {final_state.get('policy_board', {}).get('fascist_policies', '?')}")
            return False
        
        # Game completed successfully!
        print(f"🎉 GAME COMPLETED SUCCESSFULLY!")
        print(f"🏆 Winner: {result['winner']}")
        print(f"🎯 Win Condition: {result['win_condition']}")
        print(f"⏱️  Duration: {elapsed_time:.1f} seconds")
        print(f"💰 Total Cost: ${result['cost_summary']['total_cost']:.4f}")
        print(f"📡 API Requests: {result['cost_summary']['total_requests']}")
        print(f"🚀 Avg Latency: {result['cost_summary']['avg_latency']:.2f}s")
        
        # Display final board state
        final_state = result['final_state']
        print(f"\n📊 FINAL BOARD STATE:")
        print(f"   🔵 Liberal Policies: {final_state['policy_board']['liberal_policies']}/5")
        print(f"   🔴 Fascist Policies: {final_state['policy_board']['fascist_policies']}/6")
        
        # Display voting history if available
        if 'voting_rounds' in final_state and final_state['voting_rounds']:
            print(f"\n🗳️  VOTING HISTORY ({len(final_state['voting_rounds'])} rounds):")
            for i, vote_round in enumerate(final_state['voting_rounds'][-5:], 1):  # Last 5
                president = vote_round['government']['president']
                chancellor = vote_round['government']['chancellor']
                president_name = final_state['players'][president]['name']
                chancellor_name = final_state['players'][chancellor]['name']
                
                ja_votes = sum(1 for v in vote_round['votes'] if v['vote'])
                nein_votes = sum(1 for v in vote_round['votes'] if not v['vote'])
                result_text = vote_round['result']
                
                print(f"   Round {i}: {president_name}+{chancellor_name} -> {ja_votes} Ja, {nein_votes} Nein ({result_text})")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        if elapsed_time > 0:
            rounds_per_minute = (len(final_state.get('voting_rounds', [])) / elapsed_time) * 60
            print(f"   Rounds per minute: {rounds_per_minute:.1f}")
        print(f"   Average cost per action: ${result['cost_summary']['total_cost'] / max(result['cost_summary']['total_requests'], 1):.4f}")
        
        # Validate the win condition
        liberal_policies = final_state['policy_board']['liberal_policies']
        fascist_policies = final_state['policy_board']['fascist_policies']
        
        print(f"\n🧪 WIN CONDITION VALIDATION:")
        if result['winner'] == "Liberals":
            if liberal_policies == 5:
                print("   ✅ Liberals won by enacting 5 Liberal policies")
            else:
                print("   ❓ Liberals won by other means (likely Hitler execution)")
        elif result['winner'] == "Fascists":
            if fascist_policies == 6:
                print("   ✅ Fascists won by enacting 6 Fascist policies")  
            else:
                print("   ❓ Fascists won by other means (likely Hitler elected)")
        
        print(f"\n📁 Detailed logs saved to: logs/{result['game_id']}/")
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ TEST FAILED after {elapsed_time:.1f}s: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Starting comprehensive AI gameplay test...")
    result = asyncio.run(test_complete_game())
    print(f"\n{'🎉 COMPLETE SUCCESS' if result else '💥 FAILED'}")
    print("=" * 60)