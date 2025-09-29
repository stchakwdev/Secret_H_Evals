#!/usr/bin/env python3
"""
Rich Terminal UI launcher for Secret Hitler hybrid games.
Provides a beautiful terminal interface for human players.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.terminal_interface import SecretHitlerTerminalUI

async def main():
    """Launch the Rich terminal interface."""
    print("🎩 Secret Hitler - Rich Terminal Interface")
    print("=" * 50)
    print("💡 Connect to a running hybrid game server")
    print("   Make sure you have a server running on localhost:8765")
    print()
    
    try:
        # Create and run the terminal UI
        ui = SecretHitlerTerminalUI(
            player_name="Terminal Player",
            websocket_uri="ws://localhost:8765"
        )
        
        await ui.run()
        
    except KeyboardInterrupt:
        print("\n👋 Terminal interface closed")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())