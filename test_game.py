"""
Test script for Secret Hitler LLM game engine.
Basic test to validate core functionality.
"""
import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

from core.game_manager import GameManager

async def test_basic_game():
    """Test a basic 5-player game."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ No valid OpenRouter API key found!")
        print("   1. Get your API key from https://openrouter.ai/keys")
        print("   2. Add it to the .env file: OPENROUTER_API_KEY=your_key_here")
        return
    
    # Setup 5-player game
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player4", "name": "Diana", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player5", "name": "Eve", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    print("🎮 Starting Secret Hitler LLM Game Test")
    print(f"Players: {', '.join([p['name'] for p in player_configs])}")
    print("Models: Using Grok-4-Fast (FREE) for all players - no cost testing! 🆓")
    print()
    
    try:
        # Create game manager
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id="test_game_001"
        )
        
        print("🔄 Initializing game...")
        
        # Start the game
        result = await game_manager.start_game()
        
        if "error" in result:
            print(f"❌ Game failed: {result['error']}")
            if 'game_state' in result:
                print(f"Game state at failure: {result['game_state']}")
            return
        
        # Print results
        print("\n🎉 Game Completed!")
        print(f"Game ID: {result['game_id']}")
        print(f"Winner: {result['winner']}")
        print(f"Win Condition: {result['win_condition']}")
        print(f"Duration: {result['duration']:.1f} seconds")
        print(f"Total Cost: ${result['cost_summary']['total_cost']:.3f}")
        print(f"Total API Requests: {result['cost_summary']['total_requests']}")
        print(f"Average Latency: {result['cost_summary']['avg_latency']:.2f}s")
        
        # Print final board state
        final_state = result['final_state']
        print(f"\nFinal Board State:")
        print(f"Liberal Policies: {final_state['policy_board']['liberal_policies']}/5")
        print(f"Fascist Policies: {final_state['policy_board']['fascist_policies']}/6")
        
        # Print player roles
        print(f"\nPlayer Roles:")
        for pid, player_data in final_state['players'].items():
            print(f"  {player_data['name']}: {player_data['role'].title()}")
        
        # Print cost breakdown by model
        print(f"\nCost Breakdown by Model:")
        for model, cost in result['cost_summary'].get('cost_by_model', {}).items():
            requests = result['cost_summary'].get('requests_by_model', {}).get(model, 0)
            print(f"  {model}: ${cost:.3f} ({requests} requests)")
        
        print(f"\n📁 Logs saved to: logs/{result['game_id']}/")
        print("   - public.log: Public game events")
        print("   - game.log: Complete game state transitions")
        print("   - Player_[X].log: Individual player reasoning")
        print("   - metrics.json: Performance and behavioral metrics")
        print("   - api_usage.json: Detailed API usage tracking")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_cost_estimation():
    """Test cost estimation without making actual API calls."""
    
    print("💰 Testing Cost Estimation")
    
    from config.openrouter_config import estimate_cost, OPENROUTER_MODELS
    
    # Estimate costs for different scenarios
    scenarios = [
        ("Simple vote (FREE)", 100, 10, "grok-4-fast-free"),
        ("Complex nomination", 500, 50, "claude-3-sonnet"),
        ("Critical investigation", 800, 100, "claude-3-opus"),
        ("Policy discussion", 300, 30, "gpt-3.5-turbo"),
        ("Testing with free model", 1000, 100, "grok-4-fast-free")
    ]
    
    total_estimated_cost = 0
    
    print(f"{'Scenario':<20} {'Model':<20} {'Tokens':<10} {'Cost':<10}")
    print("-" * 70)
    
    for scenario, prompt_tokens, completion_tokens, model_key in scenarios:
        model_config = OPENROUTER_MODELS[model_key]
        cost = estimate_cost(prompt_tokens, completion_tokens, model_config.name)
        total_estimated_cost += cost
        
        print(f"{scenario:<20} {model_key:<20} {prompt_tokens + completion_tokens:<10} ${cost:.4f}")
    
    print("-" * 70)
    print(f"{'Total estimated':<20} {'':<20} {'':<10} ${total_estimated_cost:.4f}")
    
    # Estimate full game cost
    print(f"\n📊 Full Game Cost Estimates:")
    print(f"5-player game (40 actions): ${total_estimated_cost * 10:.2f}")
    print(f"7-player game (60 actions): ${total_estimated_cost * 15:.2f}")
    print(f"10-player game (80 actions): ${total_estimated_cost * 20:.2f}")
    
def test_game_state():
    """Test game state management without LLM calls."""
    
    print("🎲 Testing Game State Management")
    
    from core.game_state import GameState
    
    # Create test players
    player_configs = [
        {"id": "p1", "name": "Alice"},
        {"id": "p2", "name": "Bob"},
        {"id": "p3", "name": "Charlie"},
        {"id": "p4", "name": "Diana"},
        {"id": "p5", "name": "Eve"}
    ]
    
    # Initialize game state
    game = GameState(player_configs, "test_state")
    
    print(f"✅ Game initialized with {game.player_count} players")
    
    # Print roles
    print("🎭 Role assignments:")
    for pid, player in game.players.items():
        print(f"  {player.name}: {player.role.value}")
    
    # Test basic game flow
    print("\n🔄 Testing game mechanics:")
    
    # Test president selection
    president = game.get_current_president()
    print(f"  Current president: {president.name}")
    
    # Test chancellor eligibility
    eligible_chancellors = [
        p for p in game.get_alive_players()
        if p.is_eligible_chancellor and p.id != president.id
    ]
    print(f"  Eligible chancellors: {[p.name for p in eligible_chancellors]}")
    
    # Test nomination
    if eligible_chancellors:
        chancellor = eligible_chancellors[0]
        success = game.nominate_chancellor(chancellor.id)
        print(f"  Nomination {chancellor.name}: {'✅ Success' if success else '❌ Failed'}")
    
    print("  Game phase:", game.phase.value)
    print("  Liberal policies:", game.policy_board.liberal_policies)
    print("  Fascist policies:", game.policy_board.fascist_policies)
    
    print("✅ Game state test completed")

async def main():
    """Main test function."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    print("Secret Hitler LLM Engine - Test Suite")
    print("=" * 50)
    
    # Test 1: Game state management
    test_game_state()
    print()
    
    # Test 2: Cost estimation
    await test_cost_estimation()
    print()
    
    # Test 3: Full game (requires API key)
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key and api_key != 'your_openrouter_api_key_here':
        print("🚀 Running full game test with API...")
        await test_basic_game()
    else:
        print("⚠️  Skipping full game test - No valid API key found")
        print("   1. Get your API key from https://openrouter.ai/keys")
        print("   2. Add it to the .env file: OPENROUTER_API_KEY=your_key_here")

if __name__ == "__main__":
    asyncio.run(main())