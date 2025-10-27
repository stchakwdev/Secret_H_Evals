"""
Secret Hitler game manager - orchestrates the entire game flow.
"""
import asyncio
import os
import json
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime
import uuid
from enum import Enum
import logging

from .game_state import GameState, GamePhase, Role, Policy
from .game_events import (
    create_reasoning_event,
    create_speech_event,
    ReasoningEvent,
    SpeechEvent
)
from agents.openrouter_client import OpenRouterClient, APIResponse, AIDecisionResponse
from game_logging.game_logger import GameLogger
from agents.prompt_templates import PromptTemplates
from web_bridge.spectator_adapter import SpectatorAdapter
from analytics.deception_detector import get_detector

logger = logging.getLogger(__name__)

class PlayerType(Enum):
    """Player types for hybrid games."""
    AI = "ai"
    HUMAN = "human"

class GameManager:
    """Manages a complete Secret Hitler game with LLM agents and human players."""
    
    def __init__(self,
                 player_configs: List[Dict[str, str]],
                 openrouter_api_key: str,
                 game_id: Optional[str] = None,
                 human_action_callback: Optional[callable] = None,
                 human_timeout: float = 60.0,
                 spectator_callback: Optional[callable] = None,
                 enable_database_logging: bool = False,
                 db_path: str = "data/games.db"):
        """
        Initialize game manager.

        Args:
            player_configs: List of player configs with 'id', 'name', 'model', 'type'
            openrouter_api_key: OpenRouter API key
            game_id: Optional game ID, will generate if not provided
            human_action_callback: Callback function for human player actions
            human_timeout: Timeout for human actions in seconds
            spectator_callback: Callback function for spectator events
            enable_database_logging: Enable SQLite database logging for Inspect AI integration
            db_path: Path to SQLite database file
        """
        self.game_id = game_id or str(uuid.uuid4())
        self.player_configs = player_configs
        self.game_state = GameState(player_configs, self.game_id)

        # LLM client
        self.openrouter_client = OpenRouterClient(openrouter_api_key)

        # Logging
        self.logger = GameLogger(
            self.game_id,
            enable_database_logging=enable_database_logging,
            db_path=db_path
        )
        
        # Prompt templates
        self.prompt_templates = PromptTemplates()
        
        # Spectator support
        self.spectator_callback = spectator_callback
        self.spectator_adapter = SpectatorAdapter()

        # Deception detection
        self.deception_detector = get_detector()

        # Game flow control
        self.is_running = False
        self.current_action_timeout = 60  # seconds
        
        # Human player support
        self.human_action_callback = human_action_callback
        self.human_timeout = human_timeout
        self.pending_human_actions: Dict[str, asyncio.Future] = {}
        
        # Player type tracking
        self.player_types: Dict[str, PlayerType] = {}
        self._initialize_player_types()
        
        # Player-specific data for LLM context
        self.player_contexts: Dict[str, Dict] = {}
        self._initialize_player_contexts()
    
    def _initialize_player_types(self):
        """Initialize player types based on configuration."""
        for player_config in self.player_configs:
            player_id = player_config['id']
            player_type_str = player_config.get('type', 'ai').lower()
            self.player_types[player_id] = PlayerType.AI if player_type_str == 'ai' else PlayerType.HUMAN
    
    def _initialize_player_contexts(self):
        """Initialize context tracking for each player."""
        for player_config in self.player_configs:
            player_id = player_config['id']
            self.player_contexts[player_id] = {
                'model': player_config.get('model', 'x-ai/grok-4-fast:free'),
                'conversation_history': [],
                'reasoning_history': [],
                'trust_beliefs': {},  # Track beliefs about other players
                'strategy_notes': [],
                'last_action_timestamp': None,
                'player_type': self.player_types[player_id]
            }
    
    def _send_spectator_event(self, event: Dict[str, Any]):
        """Send event to spectator interface if callback is available."""
        if self.spectator_callback:
            try:
                self.spectator_callback(event)
            except Exception as e:
                print(f"⚠️  Spectator callback error: {e}")
    
    def _set_player_personalities(self, personalities: Dict[str, str]):
        """Set player personalities in spectator adapter."""
        if personalities:
            self.spectator_adapter.set_player_personalities(personalities)

    def _process_ai_decision(
        self,
        response: APIResponse,
        player_id: str,
        player_name: str,
        decision_type: str,
        available_options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process AI decision with deception detection and event emission.

        Args:
            response: API response from OpenRouter
            player_id: Player identifier
            player_name: Player name for display
            decision_type: Type of decision (nomination, vote, policy_choice, etc.)
            available_options: List of available options for this decision

        Returns:
            Dict containing:
                - action: Extracted action string
                - reasoning: Reasoning summary
                - beliefs: Belief distributions
                - public_statement: Public statement (or None)
                - is_deceptive: Whether deception was detected
                - deception_score: Confidence in deception
        """
        result = {
            'action': None,
            'reasoning': None,
            'beliefs': {},
            'public_statement': None,
            'is_deceptive': False,
            'deception_score': 0.0
        }

        # Strategy 1: Try to use structured data (preferred)
        if response.structured_data:
            try:
                structured = response.structured_data

                # Extract reasoning
                reasoning_summary = structured.reasoning.summary
                full_reasoning = structured.reasoning.full_analysis
                confidence = structured.reasoning.confidence
                strategy = structured.reasoning.strategy

                # Extract beliefs (convert RoleBeliefs to dicts)
                beliefs_dict = {}
                for target_player, role_beliefs in structured.beliefs.items():
                    beliefs_dict[target_player] = {
                        'liberal': role_beliefs.liberal,
                        'fascist': role_beliefs.fascist,
                        'hitler': role_beliefs.hitler
                    }

                # Extract action and statement
                action = structured.action
                public_statement = structured.public_statement

                # Create reasoning event (private - spectators only)
                reasoning_event = create_reasoning_event(
                    player_id=player_id,
                    player_name=player_name,
                    reasoning_summary=reasoning_summary,
                    confidence=confidence,
                    beliefs=beliefs_dict,
                    decision_type=decision_type,
                    full_reasoning=full_reasoning,
                    strategy=strategy,
                    available_options=available_options,
                    chosen_option=action
                )

                # Emit reasoning event to spectators
                reasoning_dict = reasoning_event.model_dump()
                reasoning_dict['type'] = 'reasoning'
                reasoning_dict['player_id'] = player_id
                self._send_spectator_event(reasoning_dict)

                # If there's a public statement, create speech event with deception detection
                is_deceptive = False
                deception_score = 0.0
                contradiction_summary = None

                if public_statement and public_statement.strip():
                    # Run deception detection
                    full_reasoning_text = full_reasoning or reasoning_summary
                    is_deceptive, deception_score, contradiction_summary = \
                        self.deception_detector.detect_deception(
                            reasoning=full_reasoning_text,
                            statement=public_statement,
                            context={'players': [p['name'] for p in self.player_configs]}
                        )

                    # Create speech event (public)
                    speech_event = create_speech_event(
                        player_id=player_id,
                        player_name=player_name,
                        content=public_statement,
                        statement_type=self._infer_statement_type(decision_type),
                        is_deceptive=is_deceptive,
                        deception_score=deception_score,
                        contradiction_summary=contradiction_summary,
                        game_context=f"{decision_type} decision"
                    )

                    # Emit speech event
                    speech_dict = speech_event.model_dump()
                    speech_dict['type'] = 'speech'
                    speech_dict['player_id'] = player_id
                    self._send_spectator_event(speech_dict)

                    # Log deception if detected
                    if is_deceptive and deception_score > 0.5:
                        logger.info(
                            f"🎭 DECEPTION DETECTED: {player_name} - {contradiction_summary} "
                            f"(confidence: {deception_score:.2f})"
                        )

                # Update result
                result.update({
                    'action': action,
                    'reasoning': reasoning_summary,
                    'beliefs': beliefs_dict,
                    'public_statement': public_statement,
                    'is_deceptive': is_deceptive,
                    'deception_score': deception_score
                })

                return result

            except Exception as e:
                logger.warning(f"Failed to process structured data: {e}, falling back to content parsing")

        # Strategy 2: Fallback to raw content parsing (old behavior)
        if response.content:
            action = self._extract_action_from_content(response.content)
            reasoning = self._extract_reasoning(response.content)
            statement = self._extract_statement(response.content)

            result.update({
                'action': action,
                'reasoning': reasoning,
                'public_statement': statement
            })

            # Emit basic reasoning event
            basic_reasoning_event = {
                'type': 'reasoning',
                'player_id': player_id,
                'player_name': player_name,
                'summary': reasoning or "No reasoning available",
                'confidence': 0.5,
                'decision_type': decision_type,
                'timestamp': datetime.now().isoformat()
            }
            self._send_spectator_event(basic_reasoning_event)

            # If statement exists, emit speech event with deception detection
            if statement and statement.strip():
                is_deceptive, deception_score, contradiction = \
                    self.deception_detector.detect_deception(
                        reasoning=reasoning or "",
                        statement=statement,
                        context={'players': [p['name'] for p in self.player_configs]}
                    )

                speech_event = {
                    'type': 'speech',
                    'player_id': player_id,
                    'player_name': player_name,
                    'content': statement,
                    'statement_type': self._infer_statement_type(decision_type),
                    'is_deceptive': is_deceptive,
                    'deception_score': deception_score,
                    'contradiction_summary': contradiction,
                    'timestamp': datetime.now().isoformat()
                }
                self._send_spectator_event(speech_event)

                result.update({
                    'is_deceptive': is_deceptive,
                    'deception_score': deception_score
                })

                if is_deceptive and deception_score > 0.5:
                    logger.info(
                        f"🎭 DECEPTION DETECTED: {player_name} - {contradiction} "
                        f"(confidence: {deception_score:.2f})"
                    )

            return result

        # Strategy 3: Complete fallback
        logger.error(f"No valid response data for {player_name}'s {decision_type} decision")
        return result

    def _infer_statement_type(self, decision_type: str) -> str:
        """Map decision type to statement type."""
        mapping = {
            'nomination': 'nomination_reason',
            'vote': 'vote_explanation',
            'policy_choice': 'statement',
            'investigation': 'statement',
            'execution': 'statement',
        }
        return mapping.get(decision_type, 'statement')

    def _extract_action_from_content(self, content: str) -> Optional[str]:
        """Extract action from raw content (fallback for non-structured responses)."""
        # Try to find action patterns in content
        import re

        # Look for common action patterns
        patterns = [
            r'"action"\s*:\s*"([^"]+)"',
            r'ACTION:\s*([^\n]+)',
            r'I (?:will |am going to |choose to |nominate |vote |select )([^\n.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no pattern matches, return first meaningful line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            return lines[0][:100]  # First line, max 100 chars

        return None

    async def start_game(self) -> Dict[str, Any]:
        """Start and run the complete game."""
        self.is_running = True
        
        print("🔄 GameManager.start_game() called")
        
        try:
            print("🔄 Opening OpenRouter client...")
            async with self.openrouter_client:
                print("🔄 Logging game start...")
                await self.logger.log_game_start(self.game_state.to_dict())
                
                # Send game start event to spectators
                players_info = [{'id': config['id'], 'name': config['name'], 'type': config.get('type', 'ai')} 
                               for config in self.player_configs]
                game_start_event = self.spectator_adapter.convert_game_start_event(self.game_id, players_info)
                self._send_spectator_event(game_start_event)
                
                print("🔄 Sending role information to players...")
                # Send role information to players
                await self._send_role_information()
                
                print("🔄 Starting main game loop...")
                # Main game loop
                loop_count = 0
                while not self.game_state.is_game_over and self.is_running:
                    loop_count += 1
                    print(f"🔄 Game loop iteration {loop_count}, phase: {self.game_state.phase.value}")
                    await self._execute_game_phase()
                    
                    # Safety check to prevent infinite loops
                    if loop_count > 50:
                        print("⚠️  Breaking loop after 50 iterations to prevent hang")
                        self.game_state.winner = "TIMEOUT"
                        self.game_state.win_condition = "Game stopped due to excessive loops"
                        break
                
                print("🔄 Finalizing game...")
                # Game completed
                final_result = await self._finalize_game()
                print("✅ Game completed successfully")
                return final_result
                
        except Exception as e:
            print(f"❌ Game error in start_game: {str(e)}")
            await self.logger.log_error(f"Game error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "game_state": self.game_state.to_dict()}
    
    async def _send_role_information(self):
        """Send role information to all players."""
        print(f"🔄 Sending role info to {len(self.game_state.players)} players...")
        
        for player_id, player in self.game_state.players.items():
            print(f"🔄 Processing role info for {player.name} ({player.role.value})")
            
            # Build role information based on player count and role
            role_info = self._get_role_information_for_player(player_id)
            
            prompt = self.prompt_templates.get_role_assignment_prompt(
                player_name=player.name,
                role=player.role.value,
                role_info=role_info,
                player_count=self.game_state.player_count
            )
            
            print(f"🔄 Making role assignment request for {player.name}...")
            response = await self._make_player_request(
                player_id=player_id,
                prompt=prompt,
                decision_type="acknowledge_role"
            )
            
            print(f"✅ Got response from {player.name}: {response.success if hasattr(response, 'success') else 'Unknown'}")
            
            response_content = response.content if hasattr(response, 'content') else response
            await self.logger.log_player_action(
                player_id=player_id,
                action_type="role_received",
                data={
                    "role": player.role.value,
                    "response": response_content,
                    "reasoning": self._extract_reasoning(response_content)
                }
            )
        
        print("✅ Role information sent to all players")
        
        # Send role assignment event to spectators
        player_assignments = {}
        for player_id, player in self.game_state.players.items():
            player_assignments[player_id] = {
                'name': player.name,
                'role': player.role.value
            }
        
        role_assignment_event = self.spectator_adapter.convert_role_assignment_event(player_assignments)
        self._send_spectator_event(role_assignment_event)
    
    def _get_role_information_for_player(self, player_id: str) -> Dict[str, Any]:
        """Get role-specific information for a player."""
        player = self.game_state.players[player_id]
        info = {
            "your_role": player.role.value,
            "player_count": self.game_state.player_count,
            "all_players": [p.name for p in self.game_state.players.values()]
        }
        
        # Fascist information sharing
        if player.role in [Role.FASCIST, Role.HITLER]:
            fascists = self.game_state.get_fascists()
            
            # In 5-6 player games, Hitler knows fascists
            # In 7+ player games, Hitler doesn't know fascists (except in some variants)
            if player.role == Role.HITLER and self.game_state.player_count >= 7:
                info["fascist_info"] = "You are Hitler but don't know who the fascists are."
            else:
                info["fascist_players"] = [f.name for f in fascists if f.id != player_id]
                if player.role == Role.FASCIST:
                    hitler = self.game_state.get_hitler()
                    info["hitler_player"] = hitler.name if hitler else None
        
        return info
    
    async def _execute_game_phase(self):
        """Execute the current game phase."""
        phase = self.game_state.phase
        
        if phase == GamePhase.SETUP:
            self.game_state.phase = GamePhase.NOMINATION
        
        elif phase == GamePhase.NOMINATION:
            await self._handle_nomination_phase()
        
        elif phase == GamePhase.VOTING:
            await self._handle_voting_phase()
        
        elif phase == GamePhase.LEGISLATIVE_PRESIDENT:
            await self._handle_president_legislative_phase()
        
        elif phase == GamePhase.LEGISLATIVE_CHANCELLOR:
            await self._handle_chancellor_legislative_phase()
        
        elif phase == GamePhase.EXECUTIVE_ACTION:
            await self._handle_executive_action_phase()
        
        else:
            # Shouldn't reach here
            await self.logger.log_error(f"Unknown game phase: {phase}")
            self.is_running = False
    
    async def _handle_nomination_phase(self):
        """Handle chancellor nomination phase."""
        president = self.game_state.get_current_president()
        if not president:
            await self.logger.log_error("No president available")
            self.is_running = False
            return
        
        self.game_state.government.president = president.id
        
        # Get eligible chancellors
        eligible_chancellors = [
            p for p in self.game_state.get_alive_players()
            if p.is_eligible_chancellor and p.id != president.id
        ]
        
        if not eligible_chancellors:
            await self.logger.log_error("No eligible chancellors")
            self.is_running = False
            return
        
        # Ask president to nominate chancellor
        prompt = self.prompt_templates.get_nomination_prompt(
            president_name=president.name,
            eligible_chancellors=[p.name for p in eligible_chancellors],
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(president.id)
        )
        
        response = await self._make_player_request(
            player_id=president.id,
            prompt=prompt,
            decision_type="nominate_chancellor"
        )
        
        # Parse nomination
        response_content = response.content if hasattr(response, 'content') else response
        nominated_chancellor = self._parse_nomination(response_content, eligible_chancellors)
        
        if nominated_chancellor:
            success = self.game_state.nominate_chancellor(nominated_chancellor.id)
            if success:
                await self.logger.log_player_action(
                    player_id=president.id,
                    action_type="nominate_chancellor",
                    data={
                        "nominee": nominated_chancellor.name,
                        "reasoning": self._extract_reasoning(response_content),
                        "eligible_options": [p.name for p in eligible_chancellors]
                    }
                )
                
                # Announce nomination to all players
                await self._announce_nomination(president, nominated_chancellor)
            else:
                await self.logger.log_error(f"Failed to nominate {nominated_chancellor.name}")
        else:
            await self.logger.log_error(f"Could not parse nomination from: {response_content}")
    
    async def _announce_nomination(self, president, chancellor):
        """Announce nomination to all players."""
        for player_id in self.game_state.players:
            if player_id in [president.id, chancellor.id]:
                continue  # Skip president and chancellor
            
            prompt = self.prompt_templates.get_nomination_announcement_prompt(
                president_name=president.name,
                chancellor_name=chancellor.name,
                player_name=self.game_state.players[player_id].name,
                game_state=self._get_public_game_state()
            )
            
            response = await self._make_player_request(
                player_id=player_id,
                prompt=prompt,
                decision_type="acknowledge_nomination"
            )
            
            response_content = response.content if hasattr(response, 'content') else response
            await self.logger.log_player_action(
                player_id=player_id,
                action_type="nomination_announced",
                data={
                    "president": president.name,
                    "chancellor": chancellor.name,
                    "reaction": response_content,
                    "reasoning": self._extract_reasoning(response_content)
                }
            )
    
    async def _handle_voting_phase(self):
        """Handle government voting phase."""
        alive_players = self.game_state.get_alive_players()
        voting_tasks = []
        
        for player in alive_players:
            task = self._get_player_vote(player)
            voting_tasks.append(task)
        
        # Wait for all votes
        votes = await asyncio.gather(*voting_tasks, return_exceptions=True)
        
        # Process votes
        for i, vote_result in enumerate(votes):
            player = alive_players[i]
            if isinstance(vote_result, Exception):
                await self.logger.log_error(f"Voting error for {player.name}: {vote_result}")
                # Default to Nein vote
                self.game_state.cast_vote(player.id, False)
            else:
                vote, reasoning = vote_result
                self.game_state.cast_vote(player.id, vote)
                
                await self.logger.log_player_action(
                    player_id=player.id,
                    action_type="vote",
                    data={
                        "vote": "Ja" if vote else "Nein",
                        "reasoning": reasoning
                    }
                )
        
        # Resolve the vote and update game state
        await self._resolve_voting_round()
        
        # Announce results
        await self._announce_vote_results()
    
    async def _get_player_vote(self, player) -> tuple:
        """Get a single player's vote."""
        prompt = self.prompt_templates.get_voting_prompt(
            player_name=player.name,
            president_name=self.game_state.players[self.game_state.government.president].name,
            chancellor_name=self.game_state.players[self.game_state.government.chancellor].name,
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(player.id)
        )
        
        response = await self._make_player_request(
            player_id=player.id,
            prompt=prompt,
            decision_type="vote_on_government"
        )
        
        response_content = response.content if hasattr(response, 'content') else response
        vote = self._parse_vote(response_content)
        reasoning = self._extract_reasoning(response_content)
        
        return vote, reasoning
    
    async def _resolve_voting_round(self):
        """Resolve the current voting round and update game state."""
        if not self.game_state.voting_rounds:
            await self.logger.log_error("No voting round to resolve")
            return
        
        current_round = self.game_state.voting_rounds[-1]
        alive_players = len(self.game_state.get_alive_players())
        
        # Calculate the result
        result = current_round.calculate_result(alive_players)
        current_round.result = result
        
        # Process the result in game state
        self.game_state._process_vote_result(result)
        
        await self.logger.log_player_action(
            player_id="SYSTEM",
            action_type="vote_resolved",
            data={
                "result": result.value,
                "ja_votes": sum(1 for v in current_round.votes if v.vote),
                "nein_votes": sum(1 for v in current_round.votes if not v.vote),
                "new_phase": self.game_state.phase.value
            }
        )
    
    async def _announce_vote_results(self):
        """Announce voting results to all players."""
        if not self.game_state.voting_rounds:
            return
        
        current_round = self.game_state.voting_rounds[-1]
        result = current_round.result.value
        
        vote_summary = {
            "ja_votes": sum(1 for v in current_round.votes if v.vote),
            "nein_votes": sum(1 for v in current_round.votes if not v.vote),
            "result": result
        }
        
        for player_id in self.game_state.players:
            prompt = self.prompt_templates.get_vote_results_prompt(
                player_name=self.game_state.players[player_id].name,
                vote_summary=vote_summary,
                individual_votes=[(self.game_state.players[v.player_id].name, v.vote) for v in current_round.votes],
                game_state=self._get_public_game_state()
            )
            
            response = await self._make_player_request(
                player_id=player_id,
                prompt=prompt,
                decision_type="acknowledge_vote_results"
            )
            
            response_content = response.content if hasattr(response, 'content') else response
            await self.logger.log_player_action(
                player_id=player_id,
                action_type="vote_results_received",
                data={
                    "result": result,
                    "reaction": response_content,
                    "reasoning": self._extract_reasoning(response_content)
                }
            )
    
    async def _handle_president_legislative_phase(self):
        """Handle president policy selection phase."""
        president_id = self.game_state.government.president
        president = self.game_state.players[president_id]
        
        # Draw 3 policies
        policies = self.game_state.draw_policies(3)
        
        prompt = self.prompt_templates.get_president_policy_prompt(
            president_name=president.name,
            policies=[p.value for p in policies],
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(president_id)
        )
        
        response = await self._make_player_request(
            player_id=president_id,
            prompt=prompt,
            decision_type="choose_policies_as_president"
        )
        
        # Parse policy selection
        response_content = response.content if hasattr(response, 'content') else response
        print(f"🔄 President {president.name} response: {response_content[:200]}...")
        
        kept_policies = self._parse_policy_selection(response_content, policies, 2)
        print(f"🔄 Parsed kept policies: {[p.value for p in kept_policies] if kept_policies else None}")
        
        if kept_policies and len(kept_policies) == 2:
            # Calculate discarded policy safely
            discarded_policies = [p for p in policies if p not in kept_policies]
            discarded_policy_value = discarded_policies[0].value if discarded_policies else "unknown"
            
            success = self.game_state.choose_policies_president(policies, kept_policies)
            if success:
                await self.logger.log_player_action(
                    player_id=president_id,
                    action_type="president_policy_selection",
                    data={
                        "drawn_policies": [p.value for p in policies],
                        "kept_policies": [p.value for p in kept_policies],
                        "discarded_policy": discarded_policy_value,
                        "reasoning": self._extract_reasoning(response_content)
                    }
                )
                print(f"✅ President {president.name} kept {[p.value for p in kept_policies]}, discarded {discarded_policy_value}")
            else:
                await self.logger.log_error("Failed to process president policy selection")
        else:
            print(f"❌ Invalid policy selection from {president.name}: kept_policies={kept_policies}")
            print(f"   Original response: {response_content}")
            
            # Fallback: use first 2 policies if parsing fails
            print("🔄 Using fallback policy selection (first 2 policies)")
            print(f"🔄 Available policies: {len(policies)} - {[p.value for p in policies]}")
            if len(policies) < 2:
                await self.logger.log_error(f"Insufficient policies for president: only {len(policies)} available")
                # Cannot continue - end game
                self.game_state.is_game_over = True
                self.game_state.winner = "error"
                self.game_state.win_condition = f"Game error: insufficient policies ({len(policies)} < 2)"
                return
            kept_policies = policies[:2]
            discarded_policy_value = policies[2].value if len(policies) >= 3 else "unknown"
            
            success = self.game_state.choose_policies_president(policies, kept_policies)
            if success:
                await self.logger.log_player_action(
                    player_id=president_id,
                    action_type="president_policy_selection",
                    data={
                        "drawn_policies": [p.value for p in policies],
                        "kept_policies": [p.value for p in kept_policies],
                        "discarded_policy": discarded_policy_value,
                        "reasoning": "Fallback selection due to parsing failure",
                        "original_response": response_content
                    }
                )
                print(f"✅ Fallback: President {president.name} kept {[p.value for p in kept_policies]}, discarded {discarded_policy_value}")
            else:
                await self.logger.log_error("Failed to process fallback president policy selection")
    
    async def _handle_chancellor_legislative_phase(self):
        """Handle chancellor policy selection phase."""
        chancellor_id = self.game_state.government.chancellor
        chancellor = self.game_state.players[chancellor_id]
        
        # Get the 2 policies that were passed from the president
        if not hasattr(self.game_state, 'current_legislative_policies') or not self.game_state.current_legislative_policies:
            await self.logger.log_error("No policies available from president")
            return
        
        policies = self.game_state.current_legislative_policies
        
        prompt = self.prompt_templates.get_chancellor_policy_prompt(
            chancellor_name=chancellor.name,
            policies=[p.value for p in policies],
            veto_available=self.game_state.policy_board.veto_unlocked(),
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(chancellor_id)
        )
        
        response = await self._make_player_request(
            player_id=chancellor_id,
            prompt=prompt,
            decision_type="choose_policies_as_chancellor"
        )
        
        # Parse policy choice and veto decision
        response_content = response.content if hasattr(response, 'content') else response
        print(f"🔄 Chancellor {chancellor.name} response: {response_content[:200]}...")
        
        chosen_policy, veto_requested = self._parse_chancellor_decision(response_content, policies)
        print(f"🔄 Parsed chancellor decision: policy={chosen_policy.value if chosen_policy else None}, veto={veto_requested}")
        
        if chosen_policy:
            success = self.game_state.choose_policy_chancellor(chosen_policy, veto_requested)
            if success:
                await self.logger.log_player_action(
                    player_id=chancellor_id,
                    action_type="chancellor_policy_selection",
                    data={
                        "available_policies": [p.value for p in policies],
                        "chosen_policy": chosen_policy.value,
                        "veto_requested": veto_requested,
                        "reasoning": self._extract_reasoning(response_content)
                    }
                )
                print(f"✅ Chancellor {chancellor.name} chose {chosen_policy.value}, veto={veto_requested}")
            else:
                await self.logger.log_error("Failed to process chancellor policy selection")
        else:
            print(f"❌ Invalid chancellor decision from {chancellor.name}: {response_content}")
            
            # Fallback: choose first available policy
            print("🔄 Using fallback policy selection (first policy)")
            chosen_policy = policies[0]
            veto_requested = False
            
            success = self.game_state.choose_policy_chancellor(chosen_policy, veto_requested)
            if success:
                await self.logger.log_player_action(
                    player_id=chancellor_id,
                    action_type="chancellor_policy_selection",
                    data={
                        "available_policies": [p.value for p in policies],
                        "chosen_policy": chosen_policy.value,
                        "veto_requested": veto_requested,
                        "reasoning": "Fallback selection due to parsing failure",
                        "original_response": response_content
                    }
                )
                print(f"✅ Fallback: Chancellor {chancellor.name} chose {chosen_policy.value}")
            else:
                await self.logger.log_error("Failed to process fallback chancellor policy selection")
    
    async def _handle_executive_action_phase(self):
        """Handle presidential power execution."""
        power = self.game_state.policy_board.get_presidential_power(self.game_state.player_count)
        president_id = self.game_state.government.president
        president = self.game_state.players[president_id]
        
        if power == "investigate":
            await self._handle_investigation(president)
        elif power == "special_election":
            await self._handle_special_election(president)
        elif power == "execute":
            await self._handle_execution(president)
        
        # Power is completed, return to nomination phase
        self.game_state._end_legislative_session()
    
    async def _handle_investigation(self, president):
        """Handle investigation power."""
        alive_others = [p for p in self.game_state.get_alive_players() if p.id != president.id]
        
        prompt = self.prompt_templates.get_investigation_prompt(
            president_name=president.name,
            available_targets=[p.name for p in alive_others],
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(president.id)
        )
        
        response = await self._make_player_request(
            player_id=president.id,
            prompt=prompt,
            decision_type="investigate_player"
        )
        
        response_content = response.content if hasattr(response, 'content') else response
        target = self._parse_target_selection(response_content, alive_others)
        
        if target:
            role = self.game_state.execute_investigate(target.id)
            if role:
                await self.logger.log_player_action(
                    player_id=president.id,
                    action_type="investigation",
                    data={
                        "target": target.name,
                        "result": role.value,
                        "reasoning": self._extract_reasoning(response_content)
                    }
                )
                
                # Private result to president
                await self._send_investigation_result(president, target, role)
    
    async def _send_investigation_result(self, president, target, role):
        """Send investigation result privately to president."""
        prompt = self.prompt_templates.get_investigation_result_prompt(
            president_name=president.name,
            target_name=target.name,
            target_role=role.value,
            game_state=self._get_public_game_state()
        )
        
        response = await self._make_player_request(
            player_id=president.id,
            prompt=prompt,
            decision_type="acknowledge_investigation"
        )
        
        response_content = response.content if hasattr(response, 'content') else response
        await self.logger.log_player_action(
            player_id=president.id,
            action_type="investigation_result_received",
            data={
                "target": target.name,
                "result": role.value,
                "reaction": response_content,
                "reasoning": self._extract_reasoning(response_content)
            }
        )
    
    async def _handle_special_election(self, president):
        """Handle special election power."""
        alive_others = [p for p in self.game_state.get_alive_players() if p.id != president.id]
        
        prompt = self.prompt_templates.get_special_election_prompt(
            president_name=president.name,
            available_targets=[p.name for p in alive_others],
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(president.id)
        )
        
        response = await self._make_player_request(
            player_id=president.id,
            prompt=prompt,
            decision_type="special_election"
        )
        
        response_content = response.content if hasattr(response, 'content') else response
        target = self._parse_target_selection(response_content, alive_others)
        
        if target:
            success = self.game_state.execute_special_election(target.id)
            if success:
                await self.logger.log_player_action(
                    player_id=president.id,
                    action_type="special_election",
                    data={
                        "chosen_president": target.name,
                        "reasoning": self._extract_reasoning(response_content)
                    }
                )
    
    async def _handle_execution(self, president):
        """Handle execution power."""
        alive_others = [p for p in self.game_state.get_alive_players() if p.id != president.id]
        
        prompt = self.prompt_templates.get_execution_prompt(
            president_name=president.name,
            available_targets=[p.name for p in alive_others],
            game_state=self._get_public_game_state(),
            private_info=self._get_private_info_for_player(president.id)
        )
        
        response = await self._make_player_request(
            player_id=president.id,
            prompt=prompt,
            decision_type="execute_player"
        )
        
        response_content = response.content if hasattr(response, 'content') else response
        target = self._parse_target_selection(response_content, alive_others)
        
        if target:
            success = self.game_state.execute_execution(target.id)
            if success:
                await self.logger.log_player_action(
                    player_id=president.id,
                    action_type="execution",
                    data={
                        "target": target.name,
                        "target_role": target.role.value,
                        "reasoning": self._extract_reasoning(response_content)
                    }
                )
    
    async def _make_player_request(self, player_id: str, prompt: str, decision_type: str) -> Union[APIResponse, str]:
        """Make request for a specific player (AI or human)."""
        player_type = self.player_types[player_id]
        
        if player_type == PlayerType.AI:
            return await self._make_llm_request(player_id, prompt, decision_type)
        else:
            return await self._make_human_request(player_id, prompt, decision_type)
    
    async def _make_llm_request(self, player_id: str, prompt: str, decision_type: str) -> APIResponse:
        """Make LLM request for an AI player."""
        player_context = self.player_contexts[player_id]
        model_override = player_context.get('model')
        
        response = await self.openrouter_client.make_request(
            prompt=prompt,
            decision_type=decision_type,
            player_id=player_id,
            model_override=model_override
        )
        
        # Update player context
        player_context['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response.content,
            'decision_type': decision_type,
            'cost': response.cost,
            'model': response.model
        })
        
        player_context['last_action_timestamp'] = datetime.now()
        
        # Broadcast AI action to spectators if configured
        if hasattr(self, 'spectator_callback') and self.spectator_callback:
            await self._broadcast_ai_action(player_id, response, decision_type)
        
        return response
    
    async def _broadcast_ai_action(self, player_id: str, response: APIResponse, decision_type: str):
        """Broadcast AI action to spectators using spectator adapter."""
        player = self.game_state.players.get(player_id)
        if not player:
            return
        
        # Create AI decision event using spectator adapter
        ai_decision_event = self.spectator_adapter.convert_ai_decision_event(
            player_id=player_id,
            player_name=player.name,
            response=response.content,
            context=decision_type,
            game_phase=self.game_state.phase.value
        )
        
        # Send to spectators
        self._send_spectator_event(ai_decision_event)
    
    async def _make_human_request(self, player_id: str, prompt: str, decision_type: str) -> str:
        """Make request for a human player through web interface."""
        if not self.human_action_callback:
            raise ValueError("Human action callback not configured")
        
        # Create a future for this human action
        action_future = asyncio.Future()
        self.pending_human_actions[player_id] = action_future
        
        try:
            # Send request to human via web interface
            await self.human_action_callback({
                'type': 'action_request',
                'player_id': player_id,
                'prompt': prompt,
                'decision_type': decision_type,
                'game_state': self._get_public_game_state(),
                'private_info': self._get_private_info_for_player(player_id)
            })
            
            # Wait for human response with timeout
            response = await asyncio.wait_for(action_future, timeout=self.human_timeout)
            
            # Log human action
            player_context = self.player_contexts[player_id]
            player_context['conversation_history'].append({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'response': response,
                'decision_type': decision_type,
                'cost': 0.0,  # No cost for human actions
                'model': 'human'
            })
            
            player_context['last_action_timestamp'] = datetime.now()
            
            return response
            
        except asyncio.TimeoutError:
            await self.logger.log_error(f"Human player {player_id} timed out on {decision_type}")
            # Return default response based on decision type
            return self._get_default_human_response(decision_type)
        
        finally:
            # Clean up the pending action
            if player_id in self.pending_human_actions:
                del self.pending_human_actions[player_id]
    
    def _get_default_human_response(self, decision_type: str) -> str:
        """Get default response for timed-out human actions."""
        defaults = {
            "acknowledge_role": "I understand my role.",
            "nominate_chancellor": "I nominate the first available player.",
            "vote_on_government": "Nein",  # Default to Nein vote
            "choose_policies_as_president": "I choose the first two policies.",
            "choose_policies_as_chancellor": "I choose the first policy.",
            "investigate_player": "I investigate the first available player.",
            "special_election": "I choose the first available player.",
            "execute_player": "I execute the first available player."
        }
        return defaults.get(decision_type, "I take the default action.")
    
    def submit_human_action(self, player_id: str, action: str) -> bool:
        """Submit an action from a human player."""
        if player_id in self.pending_human_actions:
            future = self.pending_human_actions[player_id]
            if not future.done():
                future.set_result(action)
                return True
        return False
    
    def _get_public_game_state(self) -> Dict[str, Any]:
        """Get public game state information."""
        return {
            "phase": self.game_state.phase.value,
            "player_count": self.game_state.player_count,
            "alive_players": [p.name for p in self.game_state.get_alive_players()],
            "liberal_policies": self.game_state.policy_board.liberal_policies,
            "fascist_policies": self.game_state.policy_board.fascist_policies,
            "election_failures": self.game_state.election_tracker.failed_elections,
            "current_government": {
                "president": self.game_state.players[self.game_state.government.president].name if self.game_state.government.president else None,
                "chancellor": self.game_state.players[self.game_state.government.chancellor].name if self.game_state.government.chancellor else None
            }
        }
    
    def _get_private_info_for_player(self, player_id: str) -> Dict[str, Any]:
        """Get private information for a specific player."""
        player = self.game_state.players[player_id]
        info = {
            "your_role": player.role.value,
            "your_name": player.name
        }
        
        # Add fascist team information
        if player.role in [Role.FASCIST, Role.HITLER]:
            fascists = self.game_state.get_fascists()
            if player.role == Role.HITLER and self.game_state.player_count >= 7:
                info["team_info"] = "You are Hitler but don't know the fascists"
            else:
                info["fascist_teammates"] = [f.name for f in fascists if f.id != player_id]
                if player.role == Role.FASCIST:
                    hitler = self.game_state.get_hitler()
                    info["hitler"] = hitler.name if hitler else None
        
        # Add investigation results
        if self.game_state.last_investigation and self.game_state.last_investigation['investigator'] == player_id:
            info["last_investigation"] = self.game_state.last_investigation
        
        return info
    
    # Parsing helper methods
    def _parse_nomination(self, response: str, eligible_chancellors: List) -> Optional:
        """Parse chancellor nomination from LLM response."""
        response_lower = response.lower()
        for chancellor in eligible_chancellors:
            if chancellor.name.lower() in response_lower:
                return chancellor
        return None
    
    def _parse_vote(self, response: str) -> bool:
        """Parse vote from LLM response."""
        response_lower = response.lower()
        if 'ja' in response_lower or 'yes' in response_lower or 'approve' in response_lower:
            return True
        return False
    
    def _parse_policy_selection(self, response: str, available_policies: List[Policy], count: int) -> Optional[List[Policy]]:
        """Parse policy selection from LLM response."""
        # Simple implementation - count liberal/fascist mentions
        response_lower = response.lower()
        liberal_count = response_lower.count('liberal')
        fascist_count = response_lower.count('fascist')
        
        selected = []
        available_copy = available_policies.copy()
        
        # Try to match the intended selection
        for _ in range(count):
            if liberal_count > fascist_count and Policy.LIBERAL in available_copy:
                selected.append(Policy.LIBERAL)
                available_copy.remove(Policy.LIBERAL)
                liberal_count -= 1
            elif Policy.FASCIST in available_copy:
                selected.append(Policy.FASCIST)
                available_copy.remove(Policy.FASCIST)
                fascist_count -= 1
            elif available_copy:
                selected.append(available_copy.pop(0))
        
        return selected if len(selected) == count else None
    
    def _parse_chancellor_decision(self, response: str, policies: List[Policy]) -> tuple:
        """Parse chancellor policy decision and veto request."""
        response_lower = response.lower()
        
        # Check for veto request
        veto_requested = 'veto' in response_lower
        
        # Choose policy
        if 'liberal' in response_lower and Policy.LIBERAL in policies:
            return Policy.LIBERAL, veto_requested
        elif 'fascist' in response_lower and Policy.FASCIST in policies:
            return Policy.FASCIST, veto_requested
        else:
            # Default to first available policy
            return policies[0] if policies else None, veto_requested
    
    def _parse_target_selection(self, response: str, available_targets: List) -> Optional:
        """Parse target selection from LLM response."""
        response_lower = response.lower()
        for target in available_targets:
            if target.name.lower() in response_lower:
                return target
        return None
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from LLM response."""
        # Look for reasoning markers
        reasoning_markers = ['reasoning:', 'because', 'i think', 'my strategy']
        response_lower = response.lower()
        
        for marker in reasoning_markers:
            if marker in response_lower:
                start_idx = response_lower.find(marker)
                # Return the part after the marker
                return response[start_idx:].strip()
        
        # Return full response if no reasoning marker found
        return response.strip()
    
    async def _finalize_game(self) -> Dict[str, Any]:
        """Finalize game and return results."""
        final_state = self.game_state.to_dict()
        cost_summary = self.openrouter_client.get_cost_summary()
        
        result = {
            "game_id": self.game_id,
            "final_state": final_state,
            "winner": self.game_state.winner,
            "win_condition": self.game_state.win_condition,
            "cost_summary": cost_summary,
            "duration": (datetime.now() - self.game_state.created_at).total_seconds(),
            "player_contexts": self.player_contexts
        }
        
        await self.logger.log_game_end(result)
        
        # Export usage log
        self.openrouter_client.export_usage_log(f"logs/{self.game_id}/api_usage.json")

        return result

    def _extract_statement(self, response: str) -> Optional[str]:
        """Extract public statement from LLM response."""
        import re

        # Try to find STATEMENT section
        statement_match = re.search(r'STATEMENT:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL | re.IGNORECASE)
        if statement_match:
            statement = statement_match.group(1).strip()
            # Clean up common artifacts
            statement = re.sub(r'^["\']+|["\']+$', '', statement)  # Remove quotes
            return statement if statement else None

        # Fallback: look for quoted text after ACTION
        action_idx = response.upper().find('ACTION:')
        if action_idx >= 0:
            after_action = response[action_idx + 100:]  # Look after ACTION section
            quote_match = re.search(r'["\']([^"\']{10,})["\']', after_action)
            if quote_match:
                return quote_match.group(1).strip()

        return None