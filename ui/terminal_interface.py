"""
Rich-based terminal interface for human players in Secret Hitler hybrid games.
Provides an interactive, real-time UI for game participation.
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich import box
import logging

logger = logging.getLogger(__name__)

class SecretHitlerTerminalUI:
    """Rich-based terminal interface for Secret Hitler."""
    
    def __init__(self, player_name: str = "Human Player", websocket_uri: str = "ws://localhost:8765"):
        self.console = Console()
        self.player_name = player_name
        self.player_id = None
        self.game_id = None
        self.websocket_uri = websocket_uri
        self.websocket = None
        self.game_state = {}
        self.messages = []
        self.pending_action = None
        self.running = True
        
        # UI State
        self.layout = Layout()
        self.setup_layout()
    
    def setup_layout(self):
        """Setup the Rich layout structure."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=5)
        )
        
        self.layout["main"].split_row(
            Layout(name="game_state", ratio=2),
            Layout(name="events", ratio=1)
        )
        
        self.layout["game_state"].split_column(
            Layout(name="players", size=8),
            Layout(name="board", ratio=1),
            Layout(name="actions", size=6)
        )
    
    def create_header(self) -> Panel:
        """Create the header panel."""
        title = Text("🎩 SECRET HITLER - TERMINAL INTERFACE", style="bold red")
        subtitle = Text(f"Player: {self.player_name} | Game: {self.game_id or 'Connecting...'}", style="dim")
        
        content = Align.center(
            Text.assemble(title, "\n", subtitle)
        )
        
        return Panel(
            content,
            title="🎮 Hybrid Game",
            border_style="red",
            box=box.DOUBLE
        )
    
    def create_players_panel(self) -> Panel:
        """Create the players status panel."""
        if not self.game_state:
            return Panel("Waiting for game state...", title="👥 Players")
        
        table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        table.add_column("Player", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Role", style="yellow")
        
        players = self.game_state.get("players", {})
        for player_id, player_info in players.items():
            name = player_info.get("name", player_id)
            status = "🟢 Connected" if player_info.get("connected") else "🔴 Disconnected"
            role = player_info.get("role", "Unknown") if player_id == self.player_id else "???"
            
            # Highlight current player
            if player_id == self.player_id:
                name = f"[bold]{name} (You)[/bold]"
            
            table.add_row(name, status, role)
        
        return Panel(table, title="👥 Players", border_style="blue")
    
    def create_board_panel(self) -> Panel:
        """Create the game board panel."""
        if not self.game_state:
            return Panel("Waiting for game state...", title="🏛️ Government Board")
        
        # Liberal and Fascist policy tracks
        liberal_policies = self.game_state.get("liberal_policies", 0)
        fascist_policies = self.game_state.get("fascist_policies", 0)
        
        content = []
        
        # Liberal track
        lib_track = "Liberal Policies: "
        for i in range(5):
            lib_track += "🟦 " if i < liberal_policies else "⬜ "
        content.append(Text(lib_track, style="blue"))
        
        # Fascist track
        fas_track = "Fascist Policies: "
        for i in range(6):
            fas_track += "🟥 " if i < fascist_policies else "⬜ "
        content.append(Text(fas_track, style="red"))
        
        # Current government
        president = self.game_state.get("president", "None")
        chancellor = self.game_state.get("chancellor", "None")
        content.append(Text(f"\n🎩 President: {president}", style="bold"))
        content.append(Text(f"👔 Chancellor: {chancellor}", style="bold"))
        
        return Panel(
            Text.assemble(*[c + "\n" for c in content]),
            title="🏛️ Government Board",
            border_style="yellow"
        )
    
    def create_actions_panel(self) -> Panel:
        """Create the current actions panel."""
        if self.pending_action:
            action_type = self.pending_action.get("decision_type", "Unknown")
            prompt = self.pending_action.get("prompt", "No prompt available")
            
            content = Text.assemble(
                Text(f"Action Required: {action_type}\n", style="bold red"),
                Text(prompt, style="white"),
                Text("\n\n[Use the input prompt below to respond]", style="dim italic")
            )
        else:
            content = Text("No action required. Waiting for game events...", style="dim green")
        
        return Panel(content, title="🎯 Current Action", border_style="magenta")
    
    def create_events_panel(self) -> Panel:
        """Create the events/messages panel."""
        if not self.messages:
            content = Text("No events yet...", style="dim")
        else:
            # Show last 10 messages
            recent_messages = self.messages[-10:]
            lines = []
            for msg in recent_messages:
                timestamp = msg.get("timestamp", "")[:19]  # Remove microseconds
                text = msg.get("text", "")
                lines.append(f"[dim]{timestamp}[/dim] {text}")
            content = Text("\n".join(lines))
        
        return Panel(content, title="📝 Game Events", border_style="green")
    
    def create_footer(self) -> Panel:
        """Create the footer panel."""
        if self.pending_action:
            action_type = self.pending_action.get("decision_type", "")
            content = Text(
                f"⚠️ Action required: {action_type}\nUse Ctrl+C to quit",
                style="bold yellow"
            )
        else:
            content = Text(
                "✅ Waiting for next action...\nUse Ctrl+C to quit",
                style="dim green"
            )
        
        return Panel(content, title="Status", border_style="white")
    
    def update_display(self):
        """Update all display panels."""
        self.layout["header"].update(self.create_header())
        self.layout["players"].update(self.create_players_panel())
        self.layout["board"].update(self.create_board_panel())
        self.layout["actions"].update(self.create_actions_panel())
        self.layout["events"].update(self.create_events_panel())
        self.layout["footer"].update(self.create_footer())
    
    def add_event_message(self, message: str, msg_type: str = "info"):
        """Add a message to the events list."""
        self.messages.append({
            "timestamp": datetime.now().isoformat(),
            "text": message,
            "type": msg_type
        })
    
    async def connect_to_game(self):
        """Connect to the WebSocket server and auto-discover game."""
        try:
            self.websocket = await websockets.connect(self.websocket_uri)
            self.add_event_message("🔌 Connected to game server", "success")
            
            # Send auto-connect message
            auto_connect_message = {
                "type": "auto_connect",
                "payload": {"player_name": self.player_name}
            }
            await self.websocket.send(json.dumps(auto_connect_message))
            self.add_event_message("🔍 Auto-discovering available games...", "info")
            
        except Exception as e:
            self.add_event_message(f"❌ Connection failed: {e}", "error")
            return False
        
        return True
    
    async def handle_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            payload = data.get("payload", {})
            
            if msg_type == "auto_connected":
                self.player_id = payload.get("player_id")
                self.game_id = payload.get("game_id")
                self.add_event_message(f"🎉 Connected as {self.player_id} in game {self.game_id}", "success")
            
            elif msg_type == "authenticated":
                self.add_event_message("✅ Authentication successful", "success")
            
            elif msg_type == "game_state_update":
                self.game_state = payload.get("state", {})
                self.add_event_message("🔄 Game state updated", "info")
            
            elif msg_type == "action_request":
                self.pending_action = payload
                decision_type = payload.get("decision_type", "Unknown")
                self.add_event_message(f"🎯 Action requested: {decision_type}", "action")
            
            elif msg_type == "game_event":
                event_msg = payload.get("message", "Game event occurred")
                self.add_event_message(f"🎲 {event_msg}", "game")
            
            elif msg_type == "error":
                error_msg = payload.get("message", "Unknown error")
                self.add_event_message(f"❌ Error: {error_msg}", "error")
            
        except json.JSONDecodeError:
            self.add_event_message(f"⚠️ Invalid message received", "warning")
        except Exception as e:
            self.add_event_message(f"❌ Error handling message: {e}", "error")
    
    async def send_action_response(self, response_data: Dict[str, Any]):
        """Send action response to the server."""
        if not self.websocket or not self.pending_action:
            return
        
        message = {
            "type": "submit_action",
            "payload": {
                "player_id": self.player_id,
                "action_type": self.pending_action.get("decision_type"),
                "action_data": response_data,
                "request_id": self.pending_action.get("request_id")
            }
        }
        
        await self.websocket.send(json.dumps(message))
        self.add_event_message(f"✅ Response sent: {response_data}", "success")
        self.pending_action = None
    
    async def handle_action_input(self):
        """Handle user input for actions."""
        if not self.pending_action:
            await asyncio.sleep(0.1)
            return
        
        decision_type = self.pending_action.get("decision_type", "")
        prompt = self.pending_action.get("prompt", "Enter your choice:")
        options = self.pending_action.get("options", [])
        
        try:
            if decision_type in ["vote", "ja_nein_vote"]:
                # Yes/No vote
                self.console.print(f"\n[bold yellow]🗳️ Vote Required:[/bold yellow] {prompt}")
                choice = Confirm.ask("Vote YES?", default=True)
                await self.send_action_response({"vote": "ja" if choice else "nein"})
            
            elif decision_type in ["nominate_chancellor", "select_chancellor"]:
                # Chancellor nomination
                self.console.print(f"\n[bold yellow]👔 Select Chancellor:[/bold yellow] {prompt}")
                if options:
                    self.console.print("Available players:")
                    for i, option in enumerate(options, 1):
                        self.console.print(f"  {i}. {option}")
                    
                    choice = IntPrompt.ask("Select number", choices=[str(i) for i in range(1, len(options) + 1)])
                    await self.send_action_response({"chancellor": options[choice - 1]})
                else:
                    player_name = Prompt.ask("Enter chancellor name")
                    await self.send_action_response({"chancellor": player_name})
            
            elif decision_type in ["select_policies", "discard_policy"]:
                # Policy selection
                self.console.print(f"\n[bold yellow]📜 Policy Decision:[/bold yellow] {prompt}")
                if options:
                    self.console.print("Available choices:")
                    for i, option in enumerate(options, 1):
                        self.console.print(f"  {i}. {option}")
                    
                    choice = IntPrompt.ask("Select number", choices=[str(i) for i in range(1, len(options) + 1)])
                    await self.send_action_response({"policy": options[choice - 1]})
                else:
                    policy = Prompt.ask("Enter policy choice (liberal/fascist)")
                    await self.send_action_response({"policy": policy})
            
            else:
                # Generic response
                self.console.print(f"\n[bold yellow]❓ Action Required:[/bold yellow] {prompt}")
                if options:
                    self.console.print("Available options:")
                    for i, option in enumerate(options, 1):
                        self.console.print(f"  {i}. {option}")
                    
                    choice = IntPrompt.ask("Select number", choices=[str(i) for i in range(1, len(options) + 1)])
                    await self.send_action_response({"response": options[choice - 1]})
                else:
                    response = Prompt.ask("Your response")
                    await self.send_action_response({"response": response})
        
        except KeyboardInterrupt:
            self.running = False
        except Exception as e:
            self.add_event_message(f"❌ Input error: {e}", "error")
    
    async def run(self):
        """Main run loop with Rich Live display."""
        self.console.print("[bold green]🎩 Starting Secret Hitler Terminal Interface...[/bold green]")
        
        # Connect to game
        if not await self.connect_to_game():
            return
        
        try:
            with Live(self.layout, console=self.console, screen=True, refresh_per_second=4) as live:
                # Create tasks for message handling and input
                message_task = asyncio.create_task(self.message_loop())
                input_task = asyncio.create_task(self.input_loop())
                
                # Main display update loop
                while self.running:
                    self.update_display()
                    await asyncio.sleep(0.25)
                
                # Clean up tasks
                message_task.cancel()
                input_task.cancel()
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Game session ended by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
        finally:
            if self.websocket:
                await self.websocket.close()
    
    async def message_loop(self):
        """Background loop for handling WebSocket messages."""
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.add_event_message(f"❌ Message loop error: {e}", "error")
    
    async def input_loop(self):
        """Background loop for handling user input."""
        try:
            while self.running:
                await self.handle_action_input()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

async def main():
    """Main function to run the terminal interface."""
    ui = SecretHitlerTerminalUI()
    await ui.run()

if __name__ == "__main__":
    asyncio.run(main())