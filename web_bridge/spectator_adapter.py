#!/usr/bin/env python3
"""
Spectator Adapter - Converts game events and AI responses for web interface display.

This module bridges the gap between the game manager and the spectator web interface,
parsing AI responses and converting game state changes into spectator-friendly events.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SpectatorAdapter:
    """Converts game events and AI responses for web interface visualization."""

    def __init__(self):
        self.game_events = []
        self.player_personalities = {}
        self.current_game_state = {}
        self.recent_reasoning = {}  # Track recent reasoning per player for deception detection
    
    def set_player_personalities(self, personalities: Dict[str, str]):
        """Set AI player personalities for display."""
        self.player_personalities = personalities

    def detect_deception(self, reasoning: str, statement: str) -> Tuple[bool, float, str]:
        """
        Detect potential deception by comparing private reasoning with public statement.

        Returns:
            Tuple of (is_deceptive, deception_score, contradiction_summary)
        """
        if not reasoning or not statement:
            return False, 0.0, ""

        reasoning_lower = reasoning.lower()
        statement_lower = statement.lower()

        deception_score = 0.0
        contradictions = []

        # Check for direct contradictions in policy claims
        policy_contradictions = [
            ('liberal', 'fascist'),
            ('fascist', 'liberal'),
            ('two liberal', 'two fascist'),
            ('two fascist', 'two liberal'),
        ]

        for reasoning_term, statement_term in policy_contradictions:
            if reasoning_term in reasoning_lower and statement_term in statement_lower:
                deception_score += 0.4
                contradictions.append(f"Claims {statement_term} but thought {reasoning_term}")

        # Check for trust/distrust contradictions
        trust_patterns = [
            (r'trust\s+(\w+)', r'don\'t trust\s+\1|distrust\s+\1'),
            (r'suspicious of\s+(\w+)', r'trust\s+\1'),
        ]

        for reasoning_pattern, statement_pattern in trust_patterns:
            reasoning_match = re.search(reasoning_pattern, reasoning_lower)
            if reasoning_match:
                statement_match = re.search(statement_pattern, statement_lower)
                if statement_match:
                    deception_score += 0.3
                    contradictions.append("Contradictory trust statements")

        # Check for intentional lying keywords in reasoning
        lying_keywords = ['lie', 'deceive', 'frame', 'blame', 'misdirect', 'hide the truth']
        for keyword in lying_keywords:
            if keyword in reasoning_lower:
                deception_score += 0.25
                contradictions.append(f"Planning to {keyword}")

        # Check for opposite voting intentions
        if 'vote ja' in reasoning_lower and ('vote nein' in statement_lower or 'voting no' in statement_lower):
            deception_score += 0.5
            contradictions.append("Voting opposite to stated intention")
        elif 'vote nein' in reasoning_lower and ('vote ja' in statement_lower or 'voting yes' in statement_lower):
            deception_score += 0.5
            contradictions.append("Voting opposite to stated intention")

        # Check for semantic negation patterns
        if 'enact fascist' in reasoning_lower and 'enact liberal' in statement_lower:
            deception_score += 0.6
            contradictions.append("Planning to enact opposite policy")

        # Cap deception score at 1.0
        deception_score = min(deception_score, 1.0)

        is_deceptive = deception_score > 0.3
        contradiction_summary = "; ".join(contradictions) if contradictions else ""

        return is_deceptive, deception_score, contradiction_summary

    def parse_ai_response(self, player_id: str, player_name: str, response: str, context: str) -> Dict[str, Any]:
        """Parse AI response into structured format for spectator display."""

        # Extract REASONING, ACTION, and STATEMENT sections using regex
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=\n(?:ACTION|STATEMENT|$))', response, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r'ACTION:\s*(.*?)(?=\n(?:REASONING|STATEMENT|$))', response, re.DOTALL | re.IGNORECASE)
        statement_match = re.search(r'STATEMENT:\s*(.*?)(?=\n(?:REASONING|ACTION|$))', response, re.DOTALL | re.IGNORECASE)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        statement = statement_match.group(1).strip() if statement_match else ""

        # Store reasoning for this player
        self.recent_reasoning[player_id] = reasoning

        # Detect deception if there's both reasoning and statement
        is_deceptive, deception_score, contradiction_summary = False, 0.0, ""
        if reasoning and statement:
            is_deceptive, deception_score, contradiction_summary = self.detect_deception(reasoning, statement)

        parsed_response = {
            'player_id': player_id,
            'player_name': player_name,
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'personality': self.player_personalities.get(player_id, "Unknown AI"),
            'reasoning': reasoning,
            'action': action_match.group(1).strip() if action_match else "No action specified",
            'statement': statement,
            'raw_response': response,
            # Deception detection results
            'is_deceptive': is_deceptive,
            'deception_score': deception_score,
            'contradiction_summary': contradiction_summary
        }

        return parsed_response
    
    def create_game_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized game event for spectator display."""
        
        event = {
            'id': f"event_{len(self.game_events) + 1}",
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        self.game_events.append(event)
        return event
    
    def convert_game_start_event(self, game_id: str, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert game start to spectator event."""
        
        return self.create_game_event('game_start', {
            'game_id': game_id,
            'players': players,
            'total_players': len(players),
            'message': f"🎮 Secret Hitler game started with {len(players)} AI players!"
        })
    
    def convert_role_assignment_event(self, player_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """Convert role assignments to spectator event."""
        
        role_counts = {}
        for player_id, info in player_assignments.items():
            role = info['role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return self.create_game_event('role_assignment', {
            'assignments': player_assignments,
            'role_summary': role_counts,
            'message': f"🎭 Roles assigned: {role_counts.get('liberal', 0)} Liberals, {role_counts.get('fascist', 0)} Fascists, {1 if 'hitler' in str(player_assignments.values()) else 0} Hitler"
        })
    
    def convert_ai_decision_event(self, player_id: str, player_name: str, 
                                 response: str, context: str, game_phase: str) -> Dict[str, Any]:
        """Convert AI decision to spectator event."""
        
        parsed = self.parse_ai_response(player_id, player_name, response, context)
        
        return self.create_game_event('ai_decision', {
            'player_id': player_id,
            'player_name': player_name,
            'game_phase': game_phase,
            'personality': parsed['personality'],
            'reasoning': parsed['reasoning'],
            'action': parsed['action'],
            'statement': parsed['statement'],
            'context': context,
            'message': f"🤖 {player_name} is thinking..."
        })
    
    def convert_nomination_event(self, president: str, chancellor: str, reasoning: str = "") -> Dict[str, Any]:
        """Convert president nomination to spectator event."""
        
        return self.create_game_event('nomination', {
            'president': president,
            'chancellor': chancellor,
            'reasoning': reasoning,
            'message': f"📋 {president} nominated {chancellor} as Chancellor"
        })
    
    def convert_voting_event(self, votes: List[Dict[str, Any]], result: str) -> Dict[str, Any]:
        """Convert voting results to spectator event."""
        
        ja_votes = sum(1 for vote in votes if vote['vote'])
        nein_votes = len(votes) - ja_votes
        
        return self.create_game_event('voting_result', {
            'votes': votes,
            'ja_votes': ja_votes,
            'nein_votes': nein_votes,
            'result': result,
            'message': f"🗳️  Vote Result: {ja_votes} Ja, {nein_votes} Nein - Government {result}"
        })
    
    def convert_policy_event(self, policy_type: str, president: str, 
                           chancellor: str, phase: str) -> Dict[str, Any]:
        """Convert policy enactment to spectator event."""
        
        emoji = "🔴" if policy_type == "fascist" else "🔵"
        
        return self.create_game_event('policy_enacted', {
            'policy_type': policy_type,
            'president': president,
            'chancellor': chancellor,
            'phase': phase,
            'message': f"{emoji} {policy_type.title()} policy enacted by {president} and {chancellor}"
        })
    
    def convert_power_event(self, power_type: str, president: str, 
                          target: str = None, result: str = None) -> Dict[str, Any]:
        """Convert presidential power to spectator event."""
        
        power_icons = {
            'investigate': '🔍',
            'special_election': '🗳️',
            'execute': '💀',
            'peek': '👀'
        }
        
        icon = power_icons.get(power_type, '⚡')
        message = f"{icon} {president} used {power_type.replace('_', ' ').title()}"
        
        if target:
            message += f" on {target}"
        if result:
            message += f" - Result: {result}"
        
        return self.create_game_event('presidential_power', {
            'power_type': power_type,
            'president': president,
            'target': target,
            'result': result,
            'message': message
        })
    
    def convert_game_end_event(self, winner: str, win_condition: str, 
                             final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert game end to spectator event."""
        
        return self.create_game_event('game_end', {
            'winner': winner,
            'win_condition': win_condition,
            'final_state': final_state,
            'message': f"🏆 Game Over! {winner} wins by {win_condition}!"
        })
    
    def convert_phase_change_event(self, new_phase: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert phase change to spectator event."""
        
        phase_messages = {
            'setup': '🎭 Setting up roles...',
            'nomination': '📋 President is nominating Chancellor...',
            'voting': '🗳️  Players are voting on government...',
            'legislative_president': '📜 President is choosing policies...',
            'legislative_chancellor': '📜 Chancellor is enacting policy...',
            'presidential_power': '⚡ Presidential power activated...',
            'game_over': '🏁 Game concluded!'
        }
        
        return self.create_game_event('phase_change', {
            'new_phase': new_phase,
            'context': context,
            'message': phase_messages.get(new_phase, f"📍 Phase: {new_phase}")
        })
    
    def get_game_state_for_spectators(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal game state to spectator-friendly format."""
        
        self.current_game_state = game_state
        
        # Extract key information for spectator display
        spectator_state = {
            'game_id': game_state.get('game_id', 'unknown'),
            'phase': game_state.get('phase', 'unknown'),
            'round_info': {
                'current_president': game_state.get('government', {}).get('president', 'None'),
                'current_chancellor': game_state.get('government', {}).get('chancellor', 'None'),
                'government_active': game_state.get('government', {}).get('is_active', False)
            },
            'policy_board': {
                'liberal_policies': game_state.get('policy_board', {}).get('liberal_policies', 0),
                'fascist_policies': game_state.get('policy_board', {}).get('fascist_policies', 0),
                'max_liberal': 5,
                'max_fascist': 6
            },
            'election_tracker': {
                'failed_elections': game_state.get('election_tracker', {}).get('failed_elections', 0),
                'max_failures': 3
            },
            'players': {},
            'game_over': game_state.get('is_game_over', False),
            'winner': game_state.get('winner'),
            'win_condition': game_state.get('win_condition')
        }
        
        # Convert player information for spectator view
        for player_id, player_info in game_state.get('players', {}).items():
            spectator_state['players'][player_id] = {
                'id': player_id,
                'name': player_info.get('name', player_id),
                'is_alive': player_info.get('is_alive', True),
                'role': player_info.get('role', 'unknown'),  # Visible in spectator mode
                'personality': self.player_personalities.get(player_id, "AI Player"),
                'is_president_eligible': player_info.get('is_eligible_president', True),
                'is_chancellor_eligible': player_info.get('is_eligible_chancellor', True)
            }
        
        return spectator_state
    
    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent game events for spectator display."""
        return self.game_events[-limit:] if self.game_events else []
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all game events for full replay."""
        return self.game_events.copy()
    
    def clear_events(self):
        """Clear all stored events (for new game)."""
        self.game_events = []
        self.current_game_state = {}
    
    def export_game_log(self) -> str:
        """Export complete game log as JSON for analysis."""
        
        game_log = {
            'game_id': self.current_game_state.get('game_id', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'player_personalities': self.player_personalities,
            'final_state': self.current_game_state,
            'events': self.game_events,
            'event_count': len(self.game_events)
        }
        
        return json.dumps(game_log, indent=2, ensure_ascii=False)

# Example usage and testing
if __name__ == "__main__":
    # Test the spectator adapter
    adapter = SpectatorAdapter()
    
    # Test AI response parsing
    test_response = """
    REASONING: As a Liberal, I need to be cautious about who I trust. The president's nomination seems reasonable, but I should analyze their past behavior patterns.
    ACTION: Vote Ja
    STATEMENT: I'll support this government for now, but I'm watching both of you carefully.
    """
    
    parsed = adapter.parse_ai_response("player1", "Alice", test_response, "voting_phase")
    print("Parsed AI Response:")
    print(json.dumps(parsed, indent=2))
    
    # Test event creation
    event = adapter.convert_ai_decision_event("player1", "Alice", test_response, "voting", "voting")
    print("\nGenerated Event:")
    print(json.dumps(event, indent=2))