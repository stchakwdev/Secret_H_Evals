#!/usr/bin/env python3
"""
Auto-connecting Secret Hitler client - no game ID needed!
Automatically discovers and connects to the latest game.
"""

import asyncio
import websockets
import json
import logging
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoConnectClient:
    """Client that automatically discovers and connects to games."""
    
    def __init__(self, server_url: str = "ws://localhost:8765", player_name: str = "Human Player"):
        self.server_url = server_url
        self.player_name = player_name
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.player_id: Optional[str] = None
        self.game_id: Optional[str] = None
        self.connected = False
    
    async def connect_and_play(self):
        """Auto-discover games and connect to play."""
        print("🎮 Auto-connecting to Secret Hitler game...")
        print(f"🔌 Server: {self.server_url}")
        print(f"👤 Player: {self.player_name}")
        print()
        
        try:
            # Connect to WebSocket server
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                print("✅ Connected to server")
                
                # Try auto-connect first
                await self._auto_connect()
                
                # Listen for messages
                async for message in websocket:
                    await self._handle_message(message)
                    
        except websockets.exceptions.ConnectionRefused:
            print("❌ Connection refused. Is the game server running?")
            print("   Try: ./play --humans 1")
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    async def _auto_connect(self):
        """Attempt to auto-connect to the latest game."""
        auto_connect_message = {
            "type": "auto_connect",
            "payload": {
                "player_name": self.player_name
            }
        }
        
        await self.websocket.send(json.dumps(auto_connect_message))
        print("📡 Sent auto-connect request...")
    
    async def _handle_message(self, message: str):
        """Handle incoming message from server."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            payload = data.get("payload", {})
            
            if msg_type == "authenticated":
                self.connected = True
                self.player_id = payload.get("player_id")
                self.game_id = payload.get("game_id")
                print(f"🎉 Auto-connected successfully!")
                print(f"   Player ID: {self.player_id}")
                print(f"   Game ID: {self.game_id}")
                print()
                print("🎲 Game will start when all players are ready...")
                print("💡 You'll be prompted for actions during the game")
                print()
                
            elif msg_type == "action_request":
                await self._handle_action_request(payload)
                
            elif msg_type == "game_event":
                self._display_game_event(payload)
                
            elif msg_type == "error":
                print(f"❌ Server error: {payload.get('message')}")
                
                # If no games available, provide helpful message
                if "No active games" in payload.get('message', ''):
                    print()
                    print("💡 No games are currently waiting for players.")
                    print("   Start a new game with: ./play --humans 1")
                    print("   Or wait for someone else to start a game.")
                
            else:
                logger.debug(f"📋 Received: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON: {message}")
        except Exception as e:
            print(f"❌ Error handling message: {e}")
    
    async def _handle_action_request(self, payload):
        """Handle action request from game."""
        decision_type = payload.get("decision_type", "unknown")
        prompt = payload.get("prompt", "")
        options = payload.get("options", [])
        request_id = payload.get("request_id")
        
        print("\n" + "="*60)
        print(f"🎯 ACTION REQUIRED: {decision_type.upper()}")
        print("="*60)
        print(f"📝 {prompt}")
        
        if options:
            print("\n📋 Available options:")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
        
        print("\n💡 Enter your response:")
        
        try:
            # Get user input
            user_input = input("> ").strip()
            
            if not user_input:
                # Provide default for common actions
                if decision_type == "vote":
                    user_input = "ja"
                elif options:
                    user_input = options[0]
                else:
                    user_input = "yes"
                print(f"   Using default: {user_input}")
            
            # Send response
            response_message = {
                "type": "submit_action",
                "payload": {
                    "player_id": self.player_id,
                    "action_type": decision_type,
                    "action_data": {"response": user_input},
                    "request_id": request_id
                }
            }
            
            await self.websocket.send(json.dumps(response_message))
            print(f"✅ Sent response: {user_input}")
            
        except KeyboardInterrupt:
            print("\n🛑 Game interrupted")
            return False
        except Exception as e:
            print(f"❌ Error sending response: {e}")
            
        return True
    
    def _display_game_event(self, payload):
        """Display game events to the player."""
        event_type = payload.get("event", payload.get("type", "game_update"))
        
        # Filter out noisy events
        if event_type in ["connection_status", "heartbeat_ack"]:
            return
        
        print(f"\n🎲 GAME UPDATE: {event_type}")
        
        # Show relevant game information
        if "message" in payload:
            print(f"📢 {payload['message']}")
        
        if event_type == "game_start":
            print("🎉 The game has begun! Good luck!")
        elif event_type == "game_end":
            print("🏁 Game finished!")
            winner = payload.get("winner")
            if winner:
                print(f"🏆 Winner: {winner}")

async def main():
    """Main entry point."""
    player_name = "Human Player"
    
    # Get player name from command line if provided
    if len(sys.argv) > 1:
        player_name = " ".join(sys.argv[1:])
    
    print("🎮 Secret Hitler - Auto-Connect Client")
    print("="*40)
    
    client = AutoConnectClient(player_name=player_name)
    await client.connect_and_play()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Disconnected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")