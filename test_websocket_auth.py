#!/usr/bin/env python3
"""
Simple WebSocket authentication test to validate the 1011 error fix.
"""
import asyncio
import websockets
import json

async def test_authentication():
    """Test WebSocket authentication directly."""
    uri = "ws://localhost:8765"
    
    try:
        print("🔌 Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully")
            
            # Send authentication message
            auth_message = {
                "type": "authenticate_player",
                "payload": {
                    "player_id": "human_1",
                    "game_id": "hybrid_game_20250923_230008"
                }
            }
            
            print("📤 Sending authentication...")
            await websocket.send(json.dumps(auth_message))
            
            # Wait for response
            print("📥 Waiting for response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print("✅ Received response:")
                print(json.dumps(json.loads(response), indent=2))
                return True
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for response")
                return False
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - server not running?")
        return False
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 WebSocket Authentication Test")
    print("=" * 40)
    
    result = asyncio.run(test_authentication())
    
    if result:
        print("🎉 Authentication test PASSED!")
    else:
        print("💥 Authentication test FAILED!")