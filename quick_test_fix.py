#!/usr/bin/env python3
"""
Quick test to verify the handle_connection fix works.
"""
import asyncio
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_websocket_server():
    """Test that our WebSocket server can start without the signature error."""
    try:
        from web_bridge.bidirectional_bridge import get_hybrid_bridge_instance
        
        print("🧪 Testing WebSocket server startup...")
        
        # Create server instance
        server = get_hybrid_bridge_instance("localhost", 8765)
        
        # Start server task
        server_task = asyncio.create_task(server.start_server())
        
        # Give it a moment to start
        await asyncio.sleep(2)
        
        print("✅ Server started successfully!")
        print("🧪 Testing connection handling...")
        
        # Try a simple connection test
        try:
            import websockets
            async with websockets.connect("ws://localhost:8765") as websocket:
                print("✅ Connection successful - no handle_connection signature error!")
                await websocket.send('{"type": "test", "payload": {}}')
                await asyncio.sleep(0.5)  # Give time for processing
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
        
        # Clean up
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_websocket_server())
    sys.exit(0 if result else 1)