"""
Game adapter that integrates the LLM game engine with the WebSocket server.
Enables real-time monitoring and replay of Secret Hitler LLM games.
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

from .websocket_server import get_server_instance
from .event_converter import get_converter, EventType

logger = logging.getLogger(__name__)

class LLMGameAdapter:
    """Adapter that connects the game engine to the web interface."""
    
    def __init__(self, game_manager, websocket_server=None):
        self.game_manager = game_manager
        self.websocket_server = websocket_server or get_server_instance()
        self.event_converter = get_converter()
        self.is_monitoring = False
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start monitoring game events and broadcasting to web clients."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info(f"Starting game monitoring for game {self.game_manager.game_id}")
        
        # Start log monitoring in background
        self._monitoring_task = asyncio.create_task(self._monitor_logs())
        
        # Broadcast initial game state
        await self._broadcast_game_start()
    
    async def stop_monitoring(self):
        """Stop monitoring game events."""
        self.is_monitoring = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped monitoring game {self.game_manager.game_id}")
    
    async def _monitor_logs(self):
        """Monitor game logs for new events."""
        game_id = self.game_manager.game_id
        log_dir = Path(f"logs/{game_id}")
        
        # Track last read positions for each log file
        log_positions = {
            "game.log": 0,
            "public.log": 0
        }
        
        while self.is_monitoring:
            try:
                # Check each log file for new entries
                for log_file, last_pos in log_positions.items():
                    log_path = log_dir / log_file
                    if log_path.exists():
                        new_pos = await self._process_log_updates(log_path, last_pos, log_file)
                        log_positions[log_file] = new_pos
                
                # Check for new player log files
                await self._check_player_logs(log_dir)
                
                # Sleep before next check
                await asyncio.sleep(0.5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring logs: {e}")
                await asyncio.sleep(1)
    
    async def _process_log_updates(self, log_path: Path, last_pos: int, log_type: str) -> int:
        """Process new log entries and broadcast them."""
        try:
            with open(log_path, 'r') as f:
                f.seek(last_pos)
                new_content = f.read()
                new_pos = f.tell()
            
            if new_content.strip():
                lines = new_content.strip().split('\n')
                for line in lines:
                    if line.strip():
                        await self._process_log_line(line, log_type)
            
            return new_pos
            
        except Exception as e:
            logger.error(f"Error processing log updates for {log_path}: {e}")
            return last_pos
    
    async def _process_log_line(self, line: str, log_type: str):
        """Process a single log line and broadcast if it's an event."""
        try:
            # Try to extract JSON event from log line
            if '{' in line and '}' in line:
                json_start = line.find('{')
                json_content = line[json_start:]
                
                try:
                    event_data = json.loads(json_content)
                    
                    # Convert to web format
                    web_event = self.event_converter.convert_event(event_data)
                    
                    # Broadcast to connected clients
                    await self.websocket_server.broadcast_game_event(
                        self.game_manager.game_id, 
                        web_event
                    )
                    
                    # Special handling for LLM-specific events
                    if event_data.get("event") == "player_action":
                        await self._handle_player_action_event(event_data)
                    
                except json.JSONDecodeError:
                    # Not a JSON event, treat as regular log entry
                    await self.websocket_server.broadcast_log_entry(
                        self.game_manager.game_id,
                        log_type,
                        {"message": line.strip()}
                    )
            else:
                # Regular log entry
                await self.websocket_server.broadcast_log_entry(
                    self.game_manager.game_id,
                    log_type,
                    {"message": line.strip()}
                )
                
        except Exception as e:
            logger.error(f"Error processing log line: {e}")
    
    async def _handle_player_action_event(self, event_data: Dict[str, Any]):
        """Handle player action events with LLM-specific processing."""
        player_id = event_data.get("player_id")
        action_data = event_data.get("data", {})
        
        # Create LLM overlay data
        llm_data = self.event_converter.create_llm_overlay_data(player_id, action_data)
        
        await self.websocket_server.broadcast_game_event(
            self.game_manager.game_id,
            llm_data
        )
        
        # Check for deception
        if event_data.get("is_deception", False):
            deception_event = self.event_converter.create_deception_event(
                player_id,
                action_data.get("reasoning", ""),
                action_data.get("public_statement", ""),
                event_data.get("action_type", "")
            )
            
            await self.websocket_server.broadcast_game_event(
                self.game_manager.game_id,
                deception_event
            )
    
    async def _check_player_logs(self, log_dir: Path):
        """Check for new player log files and monitor them."""
        try:
            player_logs = list(log_dir.glob("Player_*.log"))
            
            for player_log in player_logs:
                # For now, we'll just track that they exist
                # In a full implementation, we'd monitor these separately
                pass
                
        except Exception as e:
            logger.error(f"Error checking player logs: {e}")
    
    async def _broadcast_game_start(self):
        """Broadcast initial game state."""
        try:
            game_state = self.game_manager.game_state.to_dict()
            
            start_event = {
                "event": "game_start",
                "timestamp": datetime.now().isoformat(),
                "game_id": self.game_manager.game_id,
                "initial_state": game_state
            }
            
            web_event = self.event_converter.convert_event(start_event)
            
            await self.websocket_server.broadcast_game_event(
                self.game_manager.game_id,
                web_event
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting game start: {e}")
    
    async def broadcast_game_end(self, final_result: Dict[str, Any]):
        """Broadcast game end event."""
        try:
            end_event = {
                "event": "game_end",
                "timestamp": datetime.now().isoformat(),
                "game_id": self.game_manager.game_id,
                "final_result": final_result
            }
            
            web_event = self.event_converter.convert_event(end_event)
            
            await self.websocket_server.broadcast_game_event(
                self.game_manager.game_id,
                web_event
            )
            
            # Stop monitoring after game ends
            await self.stop_monitoring()
            
        except Exception as e:
            logger.error(f"Error broadcasting game end: {e}")
    
    async def broadcast_cost_update(self, cost_data: Dict[str, Any]):
        """Broadcast API cost update."""
        try:
            cost_event = {
                "type": EventType.API_USAGE_UPDATE.value,
                "timestamp": datetime.now().isoformat(),
                "payload": cost_data
            }
            
            await self.websocket_server.broadcast_game_event(
                self.game_manager.game_id,
                cost_event
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting cost update: {e}")
    
    async def get_game_replay_data(self) -> Dict[str, Any]:
        """Generate complete replay data for the game."""
        try:
            game_id = self.game_manager.game_id
            log_dir = Path(f"logs/{game_id}")
            
            if not log_dir.exists():
                return {"error": "Game logs not found"}
            
            replay_data = {
                "game_id": game_id,
                "events": [],
                "metrics": {},
                "logs": {}
            }
            
            # Load game events from game.log
            game_log = log_dir / "game.log"
            if game_log.exists():
                with open(game_log, 'r') as f:
                    for line in f:
                        if '{' in line:
                            try:
                                json_start = line.find('{')
                                event_data = json.loads(line[json_start:])
                                web_event = self.event_converter.convert_event(event_data)
                                replay_data["events"].append(web_event)
                            except json.JSONDecodeError:
                                continue
            
            # Load metrics
            metrics_file = log_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    replay_data["metrics"] = json.load(f)
            
            # Load public log
            public_log = log_dir / "public.log"
            if public_log.exists():
                with open(public_log, 'r') as f:
                    replay_data["logs"]["public"] = f.readlines()
            
            # Load player logs
            player_logs = {}
            for player_log in log_dir.glob("Player_*.log"):
                player_id = player_log.stem.replace("Player_", "")
                with open(player_log, 'r') as f:
                    player_logs[player_id] = f.readlines()
            
            replay_data["logs"]["players"] = player_logs
            
            return replay_data
            
        except Exception as e:
            logger.error(f"Error generating replay data: {e}")
            return {"error": str(e)}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current game metrics."""
        try:
            if hasattr(self.game_manager, 'logger') and hasattr(self.game_manager.logger, 'metrics'):
                metrics = self.game_manager.logger.metrics.copy()
                return self.event_converter.create_metrics_summary(metrics)
            return {}
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}

def create_adapter(game_manager) -> LLMGameAdapter:
    """Create a new game adapter for the given game manager."""
    return LLMGameAdapter(game_manager)