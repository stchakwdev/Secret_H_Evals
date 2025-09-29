#!/usr/bin/env python3
"""
Quick test of the Rich terminal interface without needing a full game server.
"""
import asyncio
from ui.terminal_interface import SecretHitlerTerminalUI

async def test_terminal_ui():
    """Test the terminal UI with mock data."""
    ui = SecretHitlerTerminalUI(player_name="Test Player")
    
    # Add some mock game state
    ui.player_id = "human_1"
    ui.game_id = "test_game_123"
    ui.game_state = {
        "players": {
            "human_1": {"name": "Test Player", "connected": True, "role": "Liberal"},
            "ai_1": {"name": "AI Player 1", "connected": True, "role": "???"},
            "ai_2": {"name": "AI Player 2", "connected": True, "role": "???"},
            "ai_3": {"name": "AI Player 3", "connected": True, "role": "???"},
            "ai_4": {"name": "AI Player 4", "connected": True, "role": "???"}
        },
        "liberal_policies": 2,
        "fascist_policies": 1,
        "president": "AI Player 1",
        "chancellor": "Test Player"
    }
    
    # Add some mock events
    ui.add_event_message("🎮 Game started", "success")
    ui.add_event_message("👔 You were elected Chancellor!", "info")
    ui.add_event_message("🗳️ Vote passed 4-1", "game")
    ui.add_event_message("📜 Liberal policy enacted", "success")
    
    # Add a mock pending action
    ui.pending_action = {
        "decision_type": "vote",
        "prompt": "Do you support the government of AI Player 1 (President) and Test Player (Chancellor)?",
        "options": ["ja", "nein"],
        "request_id": "vote_123"
    }
    
    print("🧪 Testing Rich Terminal UI (press Ctrl+C to exit)")
    
    # Display the interface for a few seconds
    try:
        ui.update_display()
        from rich.console import Console
        console = Console()
        with console.screen():
            console.print(ui.layout)
            await asyncio.sleep(10)  # Show for 10 seconds
    except KeyboardInterrupt:
        pass
    
    print("✅ Terminal UI test completed")

if __name__ == "__main__":
    asyncio.run(test_terminal_ui())