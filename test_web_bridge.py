"""
Test script for the web bridge functionality.
Demonstrates real-time monitoring of LLM games.
"""
import asyncio
import os
from core.game_manager import GameManager
from web_bridge import start_websocket_server, create_adapter

async def test_web_bridge():
    """Test the web bridge with a real game."""
    
    print("🌐 Testing LLM Game Web Bridge")
    print("=" * 50)
    
    # Start WebSocket server in background
    server_task = asyncio.create_task(start_websocket_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    
    print("✅ WebSocket server started on ws://localhost:8765")
    print("📱 You can now connect web clients to monitor the game")
    print()
    
    # Create game with web monitoring
    player_configs = [
        {"id": "player1", "name": "Alice", "model": "grok-4-fast-free"},
        {"id": "player2", "name": "Bob", "model": "grok-4-fast-free"}
    ]
    
    api_key = os.getenv('OPENROUTER_API_KEY', 'dummy-key-for-testing')
    
    try:
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=api_key,
            game_id="web_bridge_test"
        )
        
        print("🎮 Created game manager")
        
        # Create web adapter
        adapter = create_adapter(game_manager)
        
        print("🔗 Created web adapter")
        
        # Start monitoring
        await adapter.start_monitoring()
        
        print("👀 Started real-time monitoring")
        print("🌐 Game events will be broadcast to: ws://localhost:8765")
        print()
        
        # Simulate game events (since API key might not work)
        print("🎭 Broadcasting game start event...")
        await adapter._broadcast_game_start()
        
        print("⏳ Monitoring for 10 seconds...")
        await asyncio.sleep(10)
        
        # Stop monitoring
        await adapter.stop_monitoring()
        print("🛑 Stopped monitoring")
        
        # Get replay data
        print("📊 Generating replay data...")
        replay_data = await adapter.get_game_replay_data()
        
        if "error" not in replay_data:
            print(f"✅ Replay data generated:")
            print(f"   - Events: {len(replay_data.get('events', []))}")
            print(f"   - Logs: {len(replay_data.get('logs', {}))}")
        else:
            print(f"⚠️  Replay data error: {replay_data['error']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
    
    print("\n✅ Web bridge test completed!")

async def test_event_conversion():
    """Test event conversion functionality."""
    
    print("\n🔄 Testing Event Conversion")
    print("-" * 30)
    
    from web_bridge.event_converter import get_converter
    
    converter = get_converter()
    
    # Test game start event
    game_start_event = {
        "event": "game_start",
        "timestamp": "2024-01-01T00:00:00",
        "game_id": "test_game",
        "initial_state": {
            "players": {
                "player1": {"name": "Alice", "role": "liberal"},
                "player2": {"name": "Bob", "role": "hitler"}
            },
            "phase": "setup"
        }
    }
    
    web_event = converter.convert_event(game_start_event)
    print(f"✅ Game start conversion: {web_event['type']}")
    
    # Test player action event
    player_action_event = {
        "event": "player_action",
        "timestamp": "2024-01-01T00:01:00",
        "player_id": "player1",
        "action_type": "nominate_chancellor",
        "data": {
            "nominee": "Bob",
            "reasoning": "I think Bob is trustworthy",
            "public_statement": "I nominate Bob as Chancellor"
        },
        "is_deception": False
    }
    
    web_event = converter.convert_event(player_action_event)
    print(f"✅ Player action conversion: {web_event['type']}")
    
    # Test deception detection
    deception_event = converter.create_deception_event(
        "player1",
        "Bob is Hitler, I need to stop him",
        "I trust Bob completely",
        "nominate_chancellor"
    )
    print(f"✅ Deception detection: {deception_event['type']}")
    
    print("✅ Event conversion tests completed!")

async def main():
    """Main test function."""
    print("Secret Hitler LLM Web Bridge - Test Suite")
    print("=" * 60)
    
    # Test event conversion first
    await test_event_conversion()
    
    # Test web bridge
    await test_web_bridge()
    
    print("\n🎉 All tests completed!")
    print("\nTo use the web bridge in production:")
    print("1. Start the WebSocket server: python -m web_bridge.websocket_server")
    print("2. Connect your game manager with create_adapter(game_manager)")
    print("3. Connect web clients to ws://localhost:8765")

if __name__ == "__main__":
    asyncio.run(main())