#!/usr/bin/env python3
"""
Quick WebSocket client for testing human player connection to hybrid Secret Hitler game.
"""
import asyncio
import websockets
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanTestClient:
    """Simple WebSocket client for human player testing."""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.player_id = None
        self.game_id = None
        self.connected = False
    
    async def connect_and_authenticate(self, player_id: str, game_id: str):
        """Connect to server and authenticate as human player."""
        try:
            logger.info(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            self.player_id = player_id
            self.game_id = game_id
            
            # Send authentication message
            auth_message = {
                "type": "authenticate_player",
                "payload": {
                    "player_id": player_id,
                    "game_id": game_id
                }
            }
            
            await self.websocket.send(json.dumps(auth_message))
            logger.info(f"✅ Connected as {player_id} to game {game_id}")
            self.connected = True
            
            # Start listening for messages
            await self._listen_for_messages()
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
    
    async def _listen_for_messages(self):
        """Listen for messages from the server."""
        try:
            async for message in self.websocket:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"❌ Error listening for messages: {e}")
    
    async def _handle_message(self, message: str):
        """Handle incoming message from server."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            payload = data.get("payload", {})
            
            logger.info(f"📨 Received: {msg_type}")
            
            if msg_type == "authenticated":
                logger.info("✅ Authentication successful!")
                print(f"🎮 You are now connected as {payload.get('player_id')} in game {payload.get('game_id')}")
                
            elif msg_type == "action_request":
                await self._handle_action_request(payload)
                
            elif msg_type == "game_event":
                self._display_game_event(payload)
                
            elif msg_type == "heartbeat_ack":
                logger.debug("💓 Heartbeat acknowledged")
                
            elif msg_type == "error":
                logger.error(f"❌ Server error: {payload.get('message')}")
                
            else:
                logger.info(f"📋 Message: {json.dumps(data, indent=2)}")
                
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
    
    async def _handle_action_request(self, payload: Dict[str, Any]):
        """Handle action request from game."""
        decision_type = payload.get("decision_type", "unknown")
        prompt = payload.get("prompt", "")
        options = payload.get("options", [])
        request_id = payload.get("request_id")
        
        print("\n" + "="*60)
        print(f"🎯 ACTION REQUIRED: {decision_type}")
        print("="*60)
        print(f"📝 {prompt}")
        
        if options:
            print("\n📋 Available options:")
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
        
        print("\n💡 Enter your response (or 'help' for guidance):")
        
        # Get user input
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'help':
                self._show_help(decision_type)
                user_input = input("> ").strip()
            
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
            print("\n🛑 Interrupted by user")
            await self.disconnect()
        except Exception as e:
            logger.error(f"❌ Error sending response: {e}")
    
    def _show_help(self, decision_type: str):
        """Show help for different decision types."""
        help_text = {
            "vote": "Vote 'ja' (yes) or 'nein' (no)",
            "chancellor_nomination": "Enter the player name you want to nominate as Chancellor",
            "policy_selection": "Choose which policies to keep (e.g. '1,3' for policies 1 and 3)",
            "chancellor_policy_selection": "Choose which policy to enact",
            "veto_request": "Type 'yes' to request veto, 'no' to continue",
            "veto_response": "Type 'yes' to approve veto, 'no' to reject",
            "investigation": "Enter the name of the player you want to investigate",
            "execution": "Enter the name of the player you want to execute",
            "special_election": "Enter the name of the player for special presidential election"
        }
        
        help_msg = help_text.get(decision_type, "Enter your response based on the prompt above")
        print(f"💡 Help: {help_msg}")
    
    def _display_game_event(self, payload: Dict[str, Any]):
        """Display game events to the player."""
        event_type = payload.get("event", "game_update")
        
        print(f"\n🎲 GAME UPDATE: {event_type}")
        
        # Show relevant game information
        if "game_state" in payload:
            state = payload["game_state"]
            print(f"📊 Turn: {state.get('turn', 'N/A')} | Phase: {state.get('current_phase', 'N/A')}")
            
        if "message" in payload:
            print(f"📢 {payload['message']}")
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to maintain connection."""
        while self.connected:
            try:
                heartbeat_message = {
                    "type": "heartbeat",
                    "payload": {"player_id": self.player_id}
                }
                await self.websocket.send(json.dumps(heartbeat_message))
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"❌ Heartbeat error: {e}")
                break
    
    async def disconnect(self):
        """Disconnect from the server."""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 Disconnected from server")

async def main():
    """Main entry point for testing."""
    print("🎮 Secret Hitler - Human Player Test Client")
    print("="*50)
    
    # Get connection details
    game_id = input("Enter Game ID (or press Enter for latest): ").strip()
    if not game_id:
        game_id = "hybrid_game_20250923_211936"  # Default to current game
    
    player_id = input("Enter Player ID (default: human_1): ").strip()
    if not player_id:
        player_id = "human_1"
    
    client = HumanTestClient()
    
    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(client.send_heartbeat())
        
        # Connect and start game
        await client.connect_and_authenticate(player_id, game_id)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())