#!/usr/bin/env python3
"""
Simple AI game with real-time WebSocket broadcasting.
Directly broadcasts game events without relying on log file polling.

Author: Samuel Chakwera (stchakdev)
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from core.game_manager import GameManager
from web_bridge.websocket_server import get_server_instance

# Load environment variables
load_dotenv()


class SimpleSpectatorBridge:
    """Simple bridge that broadcasts game state directly to WebSocket."""

    def __init__(self, websocket_server, game_id):
        self.ws_server = websocket_server
        self.game_id = game_id

    async def broadcast_event(self, event_type, data):
        """Broadcast an event to all connected clients."""
        message = {
            "type": "game_event",
            "payload": {
                "game_id": self.game_id,
                "event": event_type,
                "timestamp": datetime.now().isoformat(),
                **data
            }
        }
        await self.ws_server.broadcast_game_event(self.game_id, message["payload"])

    async def broadcast_game_state(self, game_state):
        """Broadcast current game state."""
        try:
            # Get players list - could be dict or list
            players_data = []
            if hasattr(game_state, 'players') and game_state.players:
                players_iter = game_state.players.values() if isinstance(game_state.players, dict) else game_state.players
                for p in players_iter:
                    if isinstance(p, str):
                        players_data.append({"id": p, "name": p, "alive": True, "role": "unknown"})
                    else:
                        players_data.append({
                            "id": getattr(p, 'id', 'unknown'),
                            "name": getattr(p, 'name', 'unknown'),
                            "alive": getattr(p, 'is_alive', True),
                            "role": p.role.value if hasattr(p, 'role') and hasattr(p.role, 'value') else str(getattr(p, 'role', 'unknown'))
                        })

            state_dict = {
                "round": getattr(game_state, 'current_round', 1),
                "phase": game_state.phase.value if hasattr(game_state.phase, 'value') else str(game_state.phase),
                "president": game_state.government.president if hasattr(game_state, 'government') else None,
                "chancellor": game_state.government.chancellor if hasattr(game_state, 'government') else None,
                "liberal_policies": game_state.policy_board.liberal_policies if hasattr(game_state, 'policy_board') else 0,
                "fascist_policies": game_state.policy_board.fascist_policies if hasattr(game_state, 'policy_board') else 0,
                "players": players_data
            }

            await self.broadcast_event("state_update", {"state": state_dict})
        except Exception as e:
            print(f"Error broadcasting state: {e}")


async def run_game_with_live_spectator(num_players: int = 5, model: str = 'deepseek/deepseek-v3.2-exp'):
    """Run AI game with live WebSocket broadcasting."""

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("Error: OPENROUTER_API_KEY not set in .env file")
        sys.exit(1)

    # Get WebSocket server instance
    ws_server = get_server_instance(port=8765)

    # Create player configurations
    player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    player_configs = [
        {
            "id": f"player{i+1}",
            "name": player_names[i],
            "model": model,
            "type": "ai"
        }
        for i in range(num_players)
    ]

    print(f"\n{'='*70}")
    print(f"Secret Hitler - AI Game with Live Spectator")
    print(f"{'='*70}")
    print(f"Players: {num_players}")
    print(f"Model: {model}")
    print(f"WebSocket: ws://localhost:8765")
    print(f"{'='*70}\n")

    # Create game manager
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key
    )

    # Create spectator bridge
    bridge = SimpleSpectatorBridge(ws_server, game_manager.game_id)

    print(f"Game ID: {game_manager.game_id}")
    print(f"Spectator URL: file:///Users/samueltchakwera/Playground/Projects/secret-hitler/llm-game-engine/spectator.html?game={game_manager.game_id}\n")

    # Broadcast game start
    await bridge.broadcast_event("game_start", {
        "players": [p["name"] for p in player_configs],
        "num_players": num_players
    })

    # Broadcast initial state
    await bridge.broadcast_game_state(game_manager.game_state)

    # Create background task to periodically broadcast state
    async def state_broadcaster():
        while True:
            await asyncio.sleep(1)  # Broadcast every 1 second
            try:
                await bridge.broadcast_game_state(game_manager.game_state)
            except Exception as e:
                pass  # Silently ignore broadcast errors

    broadcaster_task = asyncio.create_task(state_broadcaster())

    # Run the game
    try:
        print("Starting game...\n")
        result = await game_manager.start_game()

        # Cancel broadcaster
        broadcaster_task.cancel()

        # Broadcast game end
        await bridge.broadcast_event("game_end", {
            "winner": result.get('winner', 'Unknown'),
            "rounds": result.get('rounds', 0),
            "cost": result.get('cost_summary', {})
        })

        # Print results
        print(f"\n{'='*70}")
        print(f"Game Complete!")
        print(f"{'='*70}")
        print(f"Winner: {result.get('winner', 'Unknown')}")
        print(f"Rounds: {result.get('rounds', 'Unknown')}")
        if 'cost_summary' in result:
            cost = result['cost_summary']
            print(f"Total cost: ${cost.get('total_cost', 0):.4f}")
            print(f"API calls: {cost.get('total_requests', 0)}")
        print(f"{'='*70}\n")

        return result

    except Exception as e:
        broadcaster_task.cancel()
        print(f"\nGame error: {e}")
        import traceback
        traceback.print_exc()
        raise


async def run_websocket_server():
    """Run WebSocket server."""
    server = get_server_instance(port=8765)
    await server.start_server()


async def main():
    """Main entry point - run server and game concurrently."""

    # Start WebSocket server in background
    server_task = asyncio.create_task(run_websocket_server())

    # Give server time to start
    await asyncio.sleep(1)
    print("✓ WebSocket server started on ws://localhost:8765\n")

    # Run the game
    try:
        await run_game_with_live_spectator(num_players=5)
    except KeyboardInterrupt:
        print("\n\nGame interrupted")
    finally:
        # Cancel server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
