#!/usr/bin/env python3
"""
Persistent WebSocket client to test full connection flow and demonstrate 
the working hybrid Secret Hitler system.
"""
import asyncio
import websockets
import json
import signal
import sys

class PersistentTestClient:
    """A client that stays connected to test the full system."""
    
    def __init__(self, player_name="Test Player", game_id="hybrid_game_20250923_230008"):
        self.player_name = player_name
        self.game_id = game_id
        self.player_id = "human_1"
        self.websocket = None
        self.running = True
    
    async def connect_and_stay(self):
        """Connect and stay connected to demonstrate the system working."""
        uri = "ws://localhost:8765"
        
        print(f"🎮 {self.player_name} connecting to hybrid game...")
        print(f"🔌 Game ID: {self.game_id}")
        print("=" * 50)
        
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                print("✅ Connected to WebSocket server")
                
                # Authenticate
                await self._authenticate()
                
                # Listen for messages
                print("👂 Listening for game events...")
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_message(message)
                    
        except websockets.exceptions.ConnectionRefused:
            print("❌ Connection refused - is the server running?")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    async def _authenticate(self):
        """Send authentication message."""
        auth_message = {
            "type": "authenticate_player", 
            "payload": {
                "player_id": self.player_id,
                "game_id": self.game_id
            }
        }
        
        print("📤 Sending authentication...")
        await self.websocket.send(json.dumps(auth_message))
    
    async def _handle_message(self, message):
        """Handle incoming messages from server."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            payload = data.get("payload", {})
            
            if msg_type == "authenticated":
                print("🎉 Authentication successful!")
                print(f"   Player ID: {payload.get('player_id')}")
                print(f"   Game ID: {payload.get('game_id')}")
                print(f"   Status: {payload.get('status')}")
                print("💡 Now staying connected to demonstrate working system...")
                
            elif msg_type == "game_event":
                event_type = payload.get("type", "unknown")
                if event_type != "connection_status":  # Filter noise
                    print(f"🎲 Game event: {event_type}")
                    if "message" in payload:
                        print(f"   Message: {payload['message']}")
                        
            elif msg_type == "action_request":
                print(f"🎯 Action requested: {payload.get('decision_type')}")
                print(f"   Prompt: {payload.get('prompt')}")
                # For demo, just send a default response
                await self._send_default_response(payload)
                
            elif msg_type == "error":
                print(f"❌ Server error: {payload.get('message')}")
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON: {message}")
        except Exception as e:
            print(f"❌ Error handling message: {e}")
    
    async def _send_default_response(self, request_payload):
        """Send a default response to action requests."""
        decision_type = request_payload.get("decision_type")
        request_id = request_payload.get("request_id")
        
        # Send a simple default response
        response = {
            "type": "submit_action",
            "payload": {
                "player_id": self.player_id,
                "action_type": decision_type,
                "action_data": {"response": "yes"},  # Default response
                "request_id": request_id
            }
        }
        
        await self.websocket.send(json.dumps(response))
        print(f"✅ Sent default response for {decision_type}")
    
    def stop(self):
        """Stop the client."""
        self.running = False
        print("🛑 Stopping client...")

def signal_handler(signum, frame, client):
    """Handle interrupt signal."""
    print("\n🛑 Interrupt received, shutting down...")
    client.stop()
    sys.exit(0)

async def main():
    """Main function."""
    print("🎮 Persistent Test Client for Hybrid Secret Hitler")
    print("=" * 55)
    print("💡 This client will connect and stay connected to demonstrate")
    print("   the working authentication and connection tracking system.")
    print()
    
    client = PersistentTestClient()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, client))
    
    await client.connect_and_stay()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")