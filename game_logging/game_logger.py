"""
Comprehensive logging system for Secret Hitler LLM games.
Creates multi-level logs as specified in the research plan.
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects and enums."""
    def default(self, obj):
        from enum import Enum
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class GameLogger:
    """Multi-level logging system for LLM Secret Hitler games."""

    def __init__(self, game_id: str, base_log_dir: str = "logs", enable_database_logging: bool = False, db_path: str = "data/games.db"):
        self.game_id = game_id
        self.base_log_dir = Path(base_log_dir)
        self.game_log_dir = self.base_log_dir / game_id

        # Create directories
        self.game_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log files
        self.public_log_path = self.game_log_dir / "public.log"
        self.game_log_path = self.game_log_dir / "game.log"
        self.metrics_path = self.game_log_dir / "metrics.json"

        # Player-specific logs will be created as needed
        self.player_logs: Dict[str, logging.Logger] = {}

        # Initialize loggers
        self._setup_loggers()

        # Optional database logging
        self.enable_database_logging = enable_database_logging
        self.db_manager = None
        if enable_database_logging:
            try:
                from evaluation.database_schema import DatabaseManager
                self.db_manager = DatabaseManager(db_path)
                logging.info(f"✓ Database logging enabled for game {game_id}")
            except Exception as e:
                logging.warning(f"Failed to initialize database logging: {e}")
                self.enable_database_logging = False

        # Metrics tracking
        self.metrics = {
            "game_id": game_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "total_actions": 0,
            "player_metrics": {},
            "api_usage": {
                "total_requests": 0,
                "total_cost": 0.0,
                "requests_by_model": {},
                "cost_by_model": {},
                "avg_latency": 0.0
            },
            "game_outcome": {
                "winner": None,
                "win_condition": None,
                "final_liberal_policies": 0,
                "final_fascist_policies": 0
            },
            "deception_events": [],
            "trust_evolution": [],
            "strategic_patterns": []
        }
    
    def _setup_loggers(self):
        """Setup the logging infrastructure."""
        # Public events logger
        self.public_logger = logging.getLogger(f"public_{self.game_id}")
        public_handler = logging.FileHandler(self.public_log_path)
        public_handler.setFormatter(logging.Formatter(
            '%(asctime)s [PUBLIC] %(message)s'
        ))
        self.public_logger.addHandler(public_handler)
        self.public_logger.setLevel(logging.INFO)
        
        # Complete game state logger
        self.game_logger = logging.getLogger(f"game_{self.game_id}")
        game_handler = logging.FileHandler(self.game_log_path)
        game_handler.setFormatter(logging.Formatter(
            '%(asctime)s [GAME] %(message)s'
        ))
        self.game_logger.addHandler(game_handler)
        self.game_logger.setLevel(logging.INFO)
    
    def _get_player_logger(self, player_id: str) -> logging.Logger:
        """Get or create a player-specific logger."""
        if player_id not in self.player_logs:
            logger = logging.getLogger(f"player_{player_id}_{self.game_id}")
            player_log_path = self.game_log_dir / f"Player_{player_id}.log"
            
            handler = logging.FileHandler(player_log_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [PLAYER_%(name)s] %(message)s'
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            self.player_logs[player_id] = logger
            
            # Initialize player metrics
            self.metrics["player_metrics"][player_id] = {
                "total_actions": 0,
                "reasoning_entries": 0,
                "deception_count": 0,
                "api_requests": 0,
                "api_cost": 0.0,
                "trust_beliefs": {},
                "strategy_evolution": []
            }
        
        return self.player_logs[player_id]
    
    async def log_game_start(self, initial_state: Dict[str, Any]):
        """Log game initialization."""
        timestamp = datetime.now().isoformat()
        
        # Public log
        self.public_logger.info(f"Game {self.game_id} started with {len(initial_state['players'])} players")
        
        # Game log with full state
        game_start_data = {
            "event": "game_start",
            "timestamp": timestamp,
            "game_id": self.game_id,
            "initial_state": initial_state
        }
        self.game_logger.info(json.dumps(game_start_data, indent=2, cls=DateTimeEncoder))
        
        # Update metrics
        self.metrics["player_count"] = len(initial_state['players'])
        self.metrics["players"] = list(initial_state['players'].keys())
    
    async def log_player_action(self, player_id: str, action_type: str, data: Dict[str, Any]):
        """Log a player action across multiple log levels."""
        timestamp = datetime.now().isoformat()

        # Ensure player logger exists (this initializes metrics)
        self._get_player_logger(player_id)

        # Increment action counts
        self.metrics["total_actions"] += 1
        self.metrics["player_metrics"][player_id]["total_actions"] += 1

        # Extract reasoning if present
        reasoning = data.get("reasoning", "")
        public_statement = data.get("public_statement", "")

        # Check for deception (reasoning differs from public statement)
        is_deception = self._detect_deception(reasoning, public_statement, data)
        if is_deception:
            self.metrics["player_metrics"][player_id]["deception_count"] += 1
            self.metrics["deception_events"].append({
                "timestamp": timestamp,
                "player_id": player_id,
                "action_type": action_type,
                "reasoning": reasoning,
                "public_statement": public_statement
            })

        # Public log (only public information)
        public_data = self._filter_public_data(data)
        self.public_logger.info(f"Player {player_id} performed {action_type}: {json.dumps(public_data)}")

        # Complete game log
        full_action_data = {
            "event": "player_action",
            "timestamp": timestamp,
            "player_id": player_id,
            "action_type": action_type,
            "data": data,
            "is_deception": is_deception
        }
        self.game_logger.info(json.dumps(full_action_data, indent=2, cls=DateTimeEncoder))

        # Player-specific log
        player_logger = self._get_player_logger(player_id)
        player_action_data = {
            "timestamp": timestamp,
            "action_type": action_type,
            "reasoning": reasoning,
            "data": data,
            "confidence_levels": data.get("confidence", {}),
            "trust_beliefs": data.get("trust_beliefs", {})
        }
        player_logger.info(json.dumps(player_action_data, indent=2, cls=DateTimeEncoder))

        # Track reasoning entries
        if reasoning:
            self.metrics["player_metrics"][player_id]["reasoning_entries"] += 1

        # Track trust evolution
        if "trust_beliefs" in data:
            self.metrics["trust_evolution"].append({
                "timestamp": timestamp,
                "player_id": player_id,
                "beliefs": data["trust_beliefs"]
            })

        # Database logging (optional)
        if self.enable_database_logging and self.db_manager:
            try:
                self.db_manager.insert_player_decision({
                    "game_id": self.game_id,
                    "player_id": player_id,
                    "player_name": data.get("player_name", player_id),
                    "turn_number": self.metrics["total_actions"],
                    "decision_type": action_type,
                    "reasoning": reasoning,
                    "public_statement": public_statement,
                    "is_deception": is_deception,
                    "deception_score": data.get("deception_score", 0.0),
                    "beliefs": data.get("trust_beliefs", {}),
                    "confidence": data.get("confidence"),
                    "action": data.get("action", ""),
                    "timestamp": timestamp
                })
            except Exception as e:
                logging.warning(f"Failed to log player action to database: {e}")
    
    async def log_api_request(self, player_id: str, model: str, cost: float, 
                            latency: float, tokens: int, decision_type: str):
        """Log API usage for cost tracking and analysis."""
        timestamp = datetime.now().isoformat()
        
        # Update API usage metrics
        api_metrics = self.metrics["api_usage"]
        api_metrics["total_requests"] += 1
        api_metrics["total_cost"] += cost
        
        if model not in api_metrics["requests_by_model"]:
            api_metrics["requests_by_model"][model] = 0
            api_metrics["cost_by_model"][model] = 0.0
        
        api_metrics["requests_by_model"][model] += 1
        api_metrics["cost_by_model"][model] += cost
        
        # Update average latency
        total_requests = api_metrics["total_requests"]
        api_metrics["avg_latency"] = (
            (api_metrics["avg_latency"] * (total_requests - 1) + latency) / total_requests
        )
        
        # Update player metrics
        player_metrics = self.metrics["player_metrics"][player_id]
        player_metrics["api_requests"] += 1
        player_metrics["api_cost"] += cost
        
        # Log to game log
        api_log_data = {
            "event": "api_request",
            "timestamp": timestamp,
            "player_id": player_id,
            "model": model,
            "cost": cost,
            "latency": latency,
            "tokens": tokens,
            "decision_type": decision_type
        }
        self.game_logger.info(json.dumps(api_log_data, indent=2, cls=DateTimeEncoder))

        # Database logging (optional)
        if self.enable_database_logging and self.db_manager:
            try:
                self.db_manager.insert_api_request({
                    "game_id": self.game_id,
                    "player_id": player_id,
                    "model": model,
                    "decision_type": decision_type,
                    "cost": cost,
                    "tokens": tokens,
                    "latency": latency,
                    "timestamp": timestamp
                })
            except Exception as e:
                logging.warning(f"Failed to log API request to database: {e}")
    
    async def log_game_state_transition(self, from_phase: str, to_phase: str, 
                                      game_state: Dict[str, Any]):
        """Log game state transitions."""
        timestamp = datetime.now().isoformat()
        
        transition_data = {
            "event": "state_transition",
            "timestamp": timestamp,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "game_state": game_state
        }
        
        self.game_logger.info(json.dumps(transition_data, indent=2, cls=DateTimeEncoder))
        self.public_logger.info(f"Game phase changed: {from_phase} -> {to_phase}")
    
    async def log_game_end(self, final_result: Dict[str, Any]):
        """Log game completion."""
        timestamp = datetime.now().isoformat()
        end_time = datetime.now()
        start_time = datetime.fromisoformat(self.metrics["start_time"])
        duration = (end_time - start_time).total_seconds()
        
        # Update final metrics
        self.metrics["end_time"] = timestamp
        self.metrics["duration_seconds"] = duration
        self.metrics["game_outcome"].update({
            "winner": final_result.get("winner"),
            "win_condition": final_result.get("win_condition"),
            "final_liberal_policies": final_result["final_state"]["policy_board"]["liberal_policies"],
            "final_fascist_policies": final_result["final_state"]["policy_board"]["fascist_policies"]
        })
        
        # Public log
        self.public_logger.info(f"Game {self.game_id} ended. Winner: {final_result.get('winner')} "
                              f"by {final_result.get('win_condition')}")
        
        # Game log
        game_end_data = {
            "event": "game_end",
            "timestamp": timestamp,
            "final_result": final_result,
            "duration_seconds": duration
        }
        self.game_logger.info(json.dumps(game_end_data, indent=2, cls=DateTimeEncoder))

        # Database logging (optional) - insert game record
        if self.enable_database_logging and self.db_manager:
            try:
                # Convert enums to strings for JSON serialization
                game_data = {
                    "game_id": self.game_id,
                    "timestamp": self.metrics["start_time"],
                    "player_count": self.metrics.get("player_count", 0),
                    "winner": final_result.get("winner"),
                    "winning_team": self._infer_winning_team(final_result.get("winner")),
                    "win_condition": final_result.get("win_condition"),
                    "duration_seconds": duration,
                    "total_actions": self.metrics["total_actions"],
                    "total_cost": self.metrics["api_usage"]["total_cost"],
                    "liberal_policies": self.metrics["game_outcome"]["final_liberal_policies"],
                    "fascist_policies": self.metrics["game_outcome"]["final_fascist_policies"],
                    "players": self._serialize_for_json(self.metrics.get("player_metrics", {}))
                }
                self.db_manager.insert_game(game_data)
                logging.info(f"✓ Game {self.game_id} logged to database")
            except Exception as e:
                logging.warning(f"Failed to log game to database: {e}")

        # Save final metrics
        await self._save_metrics()
    
    async def log_error(self, error_message: str, error_data: Optional[Dict] = None):
        """Log errors."""
        timestamp = datetime.now().isoformat()
        
        error_log_data = {
            "event": "error",
            "timestamp": timestamp,
            "error_message": error_message,
            "error_data": error_data or {}
        }
        
        self.game_logger.error(json.dumps(error_log_data, indent=2, cls=DateTimeEncoder))
        self.public_logger.error(f"Game error: {error_message}")
    
    def _detect_deception(self, reasoning: str, public_statement: str, data: Dict) -> bool:
        """Detect if a player is being deceptive."""
        if not reasoning or not public_statement:
            return False

        # Simple deception detection heuristics
        deception_indicators = [
            # Different policy claims
            ("liberal" in reasoning.lower() and "fascist" in public_statement.lower()),
            ("fascist" in reasoning.lower() and "liberal" in public_statement.lower()),

            # Role contradictions
            ("hitler" in reasoning.lower() and "liberal" in public_statement.lower()),
            ("fascist" in reasoning.lower() and "liberal" in public_statement.lower()),

            # Trust contradictions
            ("don't trust" in reasoning.lower() and "trust" in public_statement.lower()),
            ("suspicious" in reasoning.lower() and "trust" in public_statement.lower()),
        ]

        return any(deception_indicators)

    def _infer_winning_team(self, winner: str) -> str:
        """Infer winning team from winner string."""
        if not winner:
            return "unknown"
        winner_lower = winner.lower()
        if "liberal" in winner_lower:
            return "liberal"
        elif "fascist" in winner_lower or "hitler" in winner_lower:
            return "fascist"
        return "unknown"

    def _serialize_for_json(self, data):
        """Recursively convert enums and non-serializable objects to JSON-compatible types."""
        from enum import Enum

        if isinstance(data, Enum):
            return data.value
        elif isinstance(data, dict):
            return {key: self._serialize_for_json(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_json(item) for item in data]
        else:
            return data

    def _filter_public_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data to only include public information."""
        public_fields = [
            "vote", "nomination", "policy_enacted", "target", "public_statement",
            "reaction", "chosen_president", "investigation_target"
        ]
        
        return {k: v for k, v in data.items() if k in public_fields}
    
    async def _save_metrics(self):
        """Save comprehensive metrics to JSON file."""
        # Calculate additional analytics
        self._calculate_strategic_metrics()
        
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def _calculate_strategic_metrics(self):
        """Calculate strategic and behavioral metrics."""
        # Deception frequency by player
        for player_id in self.metrics["player_metrics"]:
            player_metrics = self.metrics["player_metrics"][player_id]
            total_actions = player_metrics["total_actions"]
            deception_count = player_metrics["deception_count"]
            
            player_metrics["deception_frequency"] = (
                deception_count / total_actions if total_actions > 0 else 0
            )
        
        # Trust network analysis
        trust_evolution = self.metrics["trust_evolution"]
        if trust_evolution:
            # Calculate trust stability (how much beliefs change over time)
            for player_id in self.metrics["player_metrics"]:
                player_trust_changes = [
                    event for event in trust_evolution 
                    if event["player_id"] == player_id
                ]
                
                if len(player_trust_changes) > 1:
                    # Calculate average change in trust beliefs
                    total_change = 0
                    for i in range(1, len(player_trust_changes)):
                        prev_beliefs = player_trust_changes[i-1]["beliefs"]
                        curr_beliefs = player_trust_changes[i]["beliefs"]
                        
                        # Calculate belief change
                        for target in curr_beliefs:
                            if target in prev_beliefs:
                                total_change += abs(curr_beliefs[target] - prev_beliefs[target])
                    
                    avg_change = total_change / (len(player_trust_changes) - 1)
                    self.metrics["player_metrics"][player_id]["trust_volatility"] = avg_change
        
        # Model performance analysis
        model_performance = {}
        for player_id, metrics in self.metrics["player_metrics"].items():
            # This would be enhanced with actual model info
            # For now, we'll track basic metrics per player
            model_performance[player_id] = {
                "actions_per_dollar": metrics["total_actions"] / max(metrics["api_cost"], 0.01),
                "reasoning_quality": metrics["reasoning_entries"] / max(metrics["total_actions"], 1),
                "deception_strategy": metrics["deception_frequency"]
            }
        
        self.metrics["model_performance"] = model_performance
    
    def get_game_summary(self) -> Dict[str, Any]:
        """Get a summary of the game for quick analysis."""
        return {
            "game_id": self.game_id,
            "status": "completed" if self.metrics["end_time"] else "in_progress",
            "duration": self.metrics["duration_seconds"],
            "total_actions": self.metrics["total_actions"],
            "total_cost": self.metrics["api_usage"]["total_cost"],
            "winner": self.metrics["game_outcome"]["winner"],
            "deception_events": len(self.metrics["deception_events"]),
            "player_count": self.metrics.get("player_count", 0)
        }
    
    async def export_for_web_import(self) -> Dict[str, Any]:
        """Export game data in format suitable for web visualization."""
        # This will be used by the Python-JavaScript bridge
        return {
            "game_id": self.game_id,
            "public_log_path": str(self.public_log_path),
            "game_log_path": str(self.game_log_path),
            "player_log_paths": {
                player_id: str(self.game_log_dir / f"Player_{player_id}.log")
                for player_id in self.player_logs.keys()
            },
            "metrics": self.metrics,
            "summary": self.get_game_summary()
        }