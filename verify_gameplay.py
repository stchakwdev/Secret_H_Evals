#!/usr/bin/env python3
"""
DEFINITIVE AI GAMEPLAY TEST - Proves the system works or fails
"""
import asyncio
import os
from dotenv import load_dotenv
from core.game_manager import GameManager

async def verify_ai_gameplay():
    """The definitive test - either works completely or shows exactly where it fails."""
    
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ FAILED: No valid OpenRouter API key")
        print("   Fix: Add OPENROUTER_API_KEY=your_key to .env file")
        return False
    
    print("🧪 DEFINITIVE AI GAMEPLAY VERIFICATION")
    print("=" * 50)
    print("This will either complete a full game or show exactly where it fails.")
    print("Using FREE models - zero cost.")
    print("=" * 50)
    
    # 5 players for a complete but faster game
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player4", "name": "Diana", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player5", "name": "Eve", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print(f"Players: {', '.join([p['name'] for p in player_configs])}")
    
    try:
        print("\n🔄 Creating game...")
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id="verification_test"
        )
        
        # Show roles for transparency
        print(f"\n🎭 Role assignments:")
        for pid, player in game_manager.game_state.players.items():
            role_emoji = "👑" if player.role.value == "hitler" else "🔴" if player.role.value == "fascist" else "🔵"
            print(f"   {role_emoji} {player.name}: {player.role.value.title()}")
        
        print(f"\n🚀 Starting complete game...")
        print("   (This should take 1-3 minutes if working)")
        
        # THE DEFINITIVE TEST - run a complete game
        result = await game_manager.start_game()
        
        # Analyze result
        if "error" in result:
            print(f"\n❌ GAME FAILED: {result['error']}")
            print(f"\nWhere it failed:")
            if 'game_state' in result:
                state = result['game_state']
                print(f"   Phase: {state.get('phase', 'unknown')}")
                print(f"   Liberal policies: {state.get('policy_board', {}).get('liberal_policies', '?')}")
                print(f"   Fascist policies: {state.get('policy_board', {}).get('fascist_policies', '?')}")
                
                if 'voting_rounds' in state:
                    print(f"   Voting rounds completed: {len(state['voting_rounds'])}")
            
            print(f"\nTo debug: Check the error message above and trace through the code.")
            return False
        
        # SUCCESS!
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"🏆 Winner: {result['winner']}")  
        print(f"🎯 Win condition: {result['win_condition']}")
        print(f"💰 Cost: ${result['cost_summary']['total_cost']:.4f}")
        print(f"📊 API requests: {result['cost_summary']['total_requests']}")
        print(f"⏱️  Duration: {result['duration']:.1f}s")
        
        final_state = result['final_state']
        print(f"\nFinal board:")
        print(f"   🔵 Liberal policies: {final_state['policy_board']['liberal_policies']}/5")
        print(f"   🔴 Fascist policies: {final_state['policy_board']['fascist_policies']}/6")
        print(f"   🗳️  Voting rounds: {len(final_state.get('voting_rounds', []))}")
        
        print(f"\n✅ AI GAMEPLAY FULLY VERIFIED AND WORKING!")
        return True
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()
        
        print(f"\nTo debug:")
        print(f"1. Check the error trace above")
        print(f"2. Most likely issues: prompt parsing, API timeouts, or game logic bugs")
        return False

if __name__ == "__main__":
    print("🧪 Running definitive AI gameplay verification...")
    result = asyncio.run(verify_ai_gameplay())
    
    if result:
        print("\n🎉 VERDICT: AI GAMEPLAY IS WORKING!")
        print("The Secret Hitler LLM system can run complete games.")
    else:
        print("\n💥 VERDICT: AI GAMEPLAY IS NOT WORKING YET")
        print("Fix the issues shown above and try again.")