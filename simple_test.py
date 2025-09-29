"""
Simplified test to isolate the API call issue.
"""
import asyncio
import os
from dotenv import load_dotenv

async def simple_api_test():
    """Test just the OpenRouter API call."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if API key is available
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("❌ No valid OpenRouter API key found!")
        print("   1. Get your API key from https://openrouter.ai/keys")
        print("   2. Add it to the .env file: OPENROUTER_API_KEY=your_key_here")
        return
    
    from agents.openrouter_client import OpenRouterClient
    
    print("🔍 Testing OpenRouter API with Grok-4-Fast Free")
    
    try:
        async with OpenRouterClient(os.environ['OPENROUTER_API_KEY']) as client:
            response = await client.make_request(
                prompt="Hello, this is a test. Please respond with 'Test successful!'",
                decision_type="test",
                player_id="test_player"
            )
            
            print(f"✅ API call successful!")
            print(f"Model: {response.model}")
            print(f"Cost: ${response.cost:.4f}")
            print(f"Response: {response.content}")
            print(f"Success: {response.success}")
            
            if response.error:
                print(f"Error: {response.error}")
                
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_api_test())