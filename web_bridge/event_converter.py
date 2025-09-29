"""
Event converter for transforming Python game events into web-compatible format.
Bridges between the LLM game engine events and the Node.js Socket.IO format.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from enum import Enum

class EventType(Enum):
    """Web event types compatible with Secret Hitler frontend."""
    GAME_START = "game_start"
    GAME_END = "game_end"
    PHASE_CHANGE = "phase_change"
    PLAYER_ACTION = "player_action"
    VOTE_RESULT = "vote_result"
    POLICY_ENACTED = "policy_enacted"
    EXECUTIVE_ACTION = "executive_action"
    DECEPTION_DETECTED = "deception_detected"
    API_USAGE_UPDATE = "api_usage_update"
    ERROR = "error"

class GameEventConverter:
    """Converts Python game events to web-compatible format."""
    
    def __init__(self):
        self.conversion_map = {
            "game_start": self._convert_game_start,
            "game_end": self._convert_game_end,
            "state_transition": self._convert_phase_change,
            "player_action": self._convert_player_action,
            "vote_resolved": self._convert_vote_result,
            "api_request": self._convert_api_usage,
            "error": self._convert_error
        }
    
    def convert_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Python game event to web format."""
        event_type = event_data.get("event", "unknown")
        converter = self.conversion_map.get(event_type, self._convert_generic)
        
        try:
            return converter(event_data)
        except Exception as e:
            return self._convert_error({
                "error_message": f"Event conversion failed: {str(e)}",
                "original_event": event_data
            })
    
    def _convert_game_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert game start event."""
        initial_state = data.get("initial_state", {})
        players = initial_state.get("players", {})
        
        # Convert to web format
        web_players = {}
        for player_id, player_data in players.items():
            web_players[player_id] = {
                "id": player_id,
                "name": player_data.get("name", "Unknown"),
                "connected": True,
                "isDead": False,
                "isRemakeVoting": False,
                "role": self._convert_role_for_public(player_data.get("role")),
                "cardFlinched": None
            }
        
        return {
            "type": EventType.GAME_START.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "game_id": data.get("game_id"),
                "players": web_players,
                "gameState": {
                    "phase": initial_state.get("phase", "setup"),
                    "turnCount": 0,
                    "trackState": {
                        "liberalPolicyCount": 0,
                        "fascistPolicyCount": 0,
                        "electionTrackerCount": 0
                    },
                    "previousGovernment": {
                        "president": None,
                        "chancellor": None
                    },
                    "undrawnPolicyCount": 17,
                    "discardedPolicyCount": 0
                }
            }
        }
    
    def _convert_game_end(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert game end event."""
        final_result = data.get("final_result", {})
        
        return {
            "type": EventType.GAME_END.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "game_id": data.get("game_id"),
                "winner": final_result.get("winner"),
                "winCondition": final_result.get("win_condition"),
                "duration": final_result.get("duration", 0),
                "finalState": final_result.get("final_state", {}),
                "costSummary": final_result.get("cost_summary", {})
            }
        }
    
    def _convert_phase_change(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert phase transition event."""
        return {
            "type": EventType.PHASE_CHANGE.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "from_phase": data.get("from_phase"),
                "to_phase": data.get("to_phase"),
                "game_state": data.get("game_state", {})
            }
        }
    
    def _convert_player_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert player action event."""
        action_type = data.get("action_type", "unknown")
        action_data = data.get("data", {})
        
        # Convert different action types
        web_action = {
            "type": EventType.PLAYER_ACTION.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "player_id": data.get("player_id"),
                "action_type": action_type,
                "reasoning": action_data.get("reasoning", ""),
                "public_statement": action_data.get("public_statement", ""),
                "is_deception": data.get("is_deception", False)
            }
        }
        
        # Add action-specific data
        if action_type == "nominate_chancellor":
            web_action["payload"]["nominee"] = action_data.get("nominee")
        elif action_type == "vote":
            web_action["payload"]["vote"] = action_data.get("vote")
        elif action_type == "president_policy_selection":
            web_action["payload"]["kept_policies"] = action_data.get("kept_policies", [])
            web_action["payload"]["discarded_policy"] = action_data.get("discarded_policy")
        elif action_type == "chancellor_policy_selection":
            web_action["payload"]["chosen_policy"] = action_data.get("chosen_policy")
            web_action["payload"]["veto_requested"] = action_data.get("veto_requested", False)
        elif action_type == "investigation":
            web_action["payload"]["target"] = action_data.get("target")
            web_action["payload"]["result"] = action_data.get("result")
        elif action_type == "execution":
            web_action["payload"]["target"] = action_data.get("target")
        
        return web_action
    
    def _convert_vote_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert vote result event."""
        vote_data = data.get("data", {})
        
        return {
            "type": EventType.VOTE_RESULT.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "result": vote_data.get("result"),
                "ja_votes": vote_data.get("ja_votes", 0),
                "nein_votes": vote_data.get("nein_votes", 0),
                "new_phase": vote_data.get("new_phase"),
                "individual_votes": vote_data.get("individual_votes", [])
            }
        }
    
    def _convert_api_usage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert API usage event."""
        return {
            "type": EventType.API_USAGE_UPDATE.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "player_id": data.get("player_id"),
                "model": data.get("model"),
                "cost": data.get("cost", 0),
                "latency": data.get("latency", 0),
                "tokens": data.get("tokens", 0),
                "decision_type": data.get("decision_type")
            }
        }
    
    def _convert_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert error event."""
        return {
            "type": EventType.ERROR.value,
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": {
                "error_message": data.get("error_message", "Unknown error"),
                "error_data": data.get("error_data", {})
            }
        }
    
    def _convert_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unknown event types."""
        return {
            "type": "unknown_event",
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "payload": data
        }
    
    def _convert_role_for_public(self, role: str) -> Optional[str]:
        """Convert role for public display (hide actual roles)."""
        # In the web interface, roles are hidden until game ends
        return None
    
    def convert_log_entry(self, log_entry: str, log_type: str) -> Dict[str, Any]:
        """Convert a log entry to web format."""
        try:
            # Try to parse as JSON first
            if log_entry.strip().startswith('{'):
                json_start = log_entry.find('{')
                log_data = json.loads(log_entry[json_start:])
                return self.convert_event(log_data)
            else:
                # Handle plain text log entries
                return {
                    "type": "log_message",
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "log_type": log_type,
                        "message": log_entry.strip()
                    }
                }
        except json.JSONDecodeError:
            return {
                "type": "log_message",
                "timestamp": datetime.now().isoformat(),
                "payload": {
                    "log_type": log_type,
                    "message": log_entry.strip()
                }
            }
    
    def create_llm_overlay_data(self, player_id: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create LLM-specific overlay data for the web interface."""
        return {
            "type": "llm_reasoning",
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "player_id": player_id,
                "reasoning": action_data.get("reasoning", ""),
                "confidence": action_data.get("confidence", {}),
                "trust_beliefs": action_data.get("trust_beliefs", {}),
                "model_used": action_data.get("model", "unknown"),
                "decision_type": action_data.get("decision_type", "unknown"),
                "cost": action_data.get("cost", 0),
                "latency": action_data.get("latency", 0)
            }
        }
    
    def create_deception_event(self, player_id: str, reasoning: str, 
                              public_statement: str, action_type: str) -> Dict[str, Any]:
        """Create deception detection event."""
        return {
            "type": EventType.DECEPTION_DETECTED.value,
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "player_id": player_id,
                "action_type": action_type,
                "private_reasoning": reasoning,
                "public_statement": public_statement,
                "deception_indicators": self._analyze_deception(reasoning, public_statement)
            }
        }
    
    def _analyze_deception(self, reasoning: str, public_statement: str) -> List[str]:
        """Analyze potential deception indicators."""
        indicators = []
        
        if not reasoning or not public_statement:
            return indicators
        
        reasoning_lower = reasoning.lower()
        public_lower = public_statement.lower()
        
        # Check for contradictory policy claims
        if ("liberal" in reasoning_lower and "fascist" in public_lower) or \
           ("fascist" in reasoning_lower and "liberal" in public_lower):
            indicators.append("contradictory_policy_claims")
        
        # Check for role contradictions
        if ("hitler" in reasoning_lower and "liberal" in public_lower):
            indicators.append("role_contradiction")
        
        # Check for trust contradictions
        if ("don't trust" in reasoning_lower and "trust" in public_lower):
            indicators.append("trust_contradiction")
        
        # Check for suspicion hiding
        if ("suspicious" in reasoning_lower and "trust" in public_lower):
            indicators.append("suspicion_hiding")
        
        return indicators
    
    def create_metrics_summary(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create web-compatible metrics summary."""
        return {
            "type": "metrics_update",
            "timestamp": datetime.now().isoformat(),
            "payload": {
                "game_id": metrics_data.get("game_id"),
                "total_actions": metrics_data.get("total_actions", 0),
                "api_usage": metrics_data.get("api_usage", {}),
                "deception_events": len(metrics_data.get("deception_events", [])),
                "player_metrics": metrics_data.get("player_metrics", {}),
                "trust_evolution": metrics_data.get("trust_evolution", []),
                "model_performance": metrics_data.get("model_performance", {})
            }
        }

# Singleton converter instance
_converter_instance = None

def get_converter() -> GameEventConverter:
    """Get singleton event converter instance."""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = GameEventConverter()
    return _converter_instance