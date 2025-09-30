#!/usr/bin/env python3
"""
Launch AI-only game with WebSocket spectator mode.
Uses event-driven architecture via GameManager's spectator_callback.

Author: Samuel Chakwera (stchakdev)
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from core.game_manager import GameManager
from web_bridge.websocket_server import get_server_instance
from web_bridge.spectator_websocket_bridge import SpectatorWebSocketBridge

# Load environment variables
load_dotenv()


async def run_websocket_server(port: int = 8765):
    """Run WebSocket server in background."""
    server = get_server_instance(port=port)
    await server.start_server()


async def run_game_with_spectator(num_players: int = 5, model: str = 'deepseek/deepseek-v3.2-exp'):
    """Run AI-only game with spectator mode enabled."""

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("Error: OPENROUTER_API_KEY not set in .env file")
        sys.exit(1)

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
    print(f"Secret Hitler - AI Game with Live Spectator Mode")
    print(f"{'='*70}")
    print(f"Players: {num_players}")
    print(f"Model: {model}")
    print(f"WebSocket Server: ws://localhost:8765")
    print(f"{'='*70}\n")

    # Get WebSocket server instance
    ws_server = get_server_instance(port=8765)

    # Create game manager first to get game_id
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key
    )

    # Create spectator bridge with callback
    bridge = SpectatorWebSocketBridge(ws_server, game_manager.game_id)
    bridge.set_event_loop(asyncio.get_event_loop())

    # Inject spectator callback into game manager
    game_manager.spectator_callback = bridge.handle_game_event

    print(f"Game ID: {game_manager.game_id}")
    spectator_path = Path(__file__).parent / "spectator_v2.html"
    print(f"\n🎭 Enhanced Spectator URL (Phases 1-7):")
    print(f"   file://{spectator_path}?ws=ws://localhost:8765&game_id={game_manager.game_id}")
    print(f"\n📂 Game Logs: logs/{game_manager.game_id}/\n")

    # Run the game
    try:
        print("Starting game...\n")
        result = await game_manager.start_game()

        # Print results
        print(f"\n{'='*70}")
        print(f"Game Complete!")
        print(f"{'='*70}")
        print(f"Winner: {result.get('winner', 'Unknown')}")
        print(f"Total rounds: {result.get('rounds', 'Unknown')}")

        if 'cost_summary' in result:
            cost = result['cost_summary']
            print(f"Total cost: ${cost.get('total_cost', 0):.4f}")
            print(f"API requests: {cost.get('total_requests', 0)}")

        print(f"\nGame logs: logs/{game_manager.game_id}/")
        print(f"{'='*70}\n")

        return result

    except Exception as e:
        print(f"\nGame error: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Main entry point - run server and game concurrently."""

    # Start WebSocket server in background
    server_task = asyncio.create_task(run_websocket_server())

    # Give server time to start
    await asyncio.sleep(1)

    print("✓ WebSocket server started on ws://localhost:8765\n")

    # Run the game
    try:
        await run_game_with_spectator(num_players=5)
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
    finally:
        # Cancel server task
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