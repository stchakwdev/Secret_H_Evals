#!/usr/bin/env python3
"""
Test just a single LLM request to see what's happening.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from core.game_manager import GameManager

async def test_single_request():
    """Test just a single LLM request with timeout."""
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print("🧪 SINGLE LLM REQUEST TEST")
    print("=" * 30)
    
    # Create minimal game setup (minimum 3 players required)
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player2", "name": "Bob", "model": "deepseek/deepseek-v3.2-exp"},
        {"id": "player3", "name": "Charlie", "model": "deepseek/deepseek-v3.2-exp"}
    ]
    
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        game_id="single_test"
    )
    
    print(f"🔄 Alice role: {game_manager.game_state.players['player1'].role.value}")
    
    try:
        print("🔄 Testing single LLM request with 30s timeout...")
        
        # Add request timeout at the asyncio level
        async with game_manager.openrouter_client:
            
            # Simple test prompt
            prompt = "You are Alice in Secret Hitler. You are a Liberal. Just respond with 'I understand my role.'"
            
            print("🔄 Making request...")
            start_time = time.time()
            
            # Add explicit timeout
            response = await asyncio.wait_for(
                game_manager.openrouter_client.make_request(
                    prompt=prompt,
                    decision_type="test",
                    player_id="player1"
                ),
                timeout=30.0  # 30 second timeout
            )
            
            request_time = time.time() - start_time
            
            print(f"✅ Request completed in {request_time:.1f}s")
            print(f"   Success: {response.success}")
            print(f"   Cost: ${response.cost:.4f}")
            print(f"   Model: {response.model}")
            print(f"   Response: {response.content}")
            
            if response.error:
                print(f"   Error: {response.error}")
            
            return response.success
            
    except asyncio.TimeoutError:
        print(f"❌ Request timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing single LLM request...")
    result = asyncio.run(test_single_request())
    print(f"\n{'🎉 SUCCESS' if result else '💥 FAILED'}")