#!/usr/bin/env python3
"""
Auto-connecting WebSocket client for testing human player connection.
"""
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    """Auto-connect to the current hybrid game."""
    # Use the current game ID
    game_id = "hybrid_game_20250923_222635"
    player_id = "human_1"
    server_url = "ws://localhost:8765"
    
    print(f"🎮 Connecting as {player_id} to game {game_id}")
    print(f"🔌 Server: {server_url}")
    
    try:
        # Connect to WebSocket server
        async with websockets.connect(server_url) as websocket:
            print("✅ Connected to WebSocket server")
            
            # Send authentication message
            auth_message = {
                "type": "authenticate_player",
                "payload": {
                    "player_id": player_id,
                    "game_id": game_id
                }
            }
            
            await websocket.send(json.dumps(auth_message))
            print("📤 Sent authentication message")
            
            # Listen for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    payload = data.get("payload", {})
                    
                    print(f"\n📨 Received: {msg_type}")
                    print(f"📋 Payload: {json.dumps(payload, indent=2)}")
                    
                    if msg_type == "authenticated":
                        print("🎉 Successfully authenticated! Game should now proceed.")
                        
                    elif msg_type == "action_request":
                        print("\n🎯 ACTION REQUIRED!")
                        print("=" * 50)
                        print(f"Decision Type: {payload.get('decision_type')}")
                        print(f"Prompt: {payload.get('prompt')}")
                        print(f"Options: {payload.get('options', [])}")
                        
                        # Send a simple response for testing
                        if payload.get('decision_type') == 'vote':
                            response = "ja"  # Vote yes
                        elif payload.get('decision_type') == 'chancellor_nomination':
                            # Nominate the first available player
                            options = payload.get('options', [])
                            response = options[0] if options else "AI Player 1"
                        else:
                            response = "yes"  # Default response
                        
                        response_message = {
                            "type": "submit_action",
                            "payload": {
                                "player_id": player_id,
                                "action_type": payload.get('decision_type'),
                                "action_data": {"response": response},
                                "request_id": payload.get('request_id')
                            }
                        }
                        
                        await websocket.send(json.dumps(response_message))
                        print(f"✅ Sent response: {response}")
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message}")
                except Exception as e:
                    print(f"❌ Error handling message: {e}")
                    
    except ConnectionRefusedError:
        print("❌ Connection refused. Is the hybrid game server running?")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("🚀 Auto-connecting to hybrid Secret Hitler game...")
    asyncio.run(test_connection())