"""
Hybrid Game Coordinator - manages Secret Hitler games with both human and AI players.
Bridges between the web UI (humans) and LLM engine (AI) for seamless gameplay.
"""
import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.game_manager import GameManager
from core.game_state import GamePhase
from web_bridge.websocket_server import GameWebSocketServer


class PlayerType(Enum):
    HUMAN = "human"
    AI = "ai"


@dataclass
class HybridPlayerConfig:
    """Configuration for a hybrid game player."""
    id: str
    name: str
    type: PlayerType
    model: Optional[str] = None  # Only for AI players
    session_id: Optional[str] = None  # Only for human players


class HumanActionQueue:
    """Queue for managing human player actions."""
    
    def __init__(self):
        self._pending_actions: Dict[str, asyncio.Future] = {}
        self._timeouts: Dict[str, asyncio.Task] = {}
    
    async def wait_for_action(self, player_id: str, action_type: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for human player action with timeout."""
        action_key = f"{player_id}_{action_type}"
        
        # Create future for this action
        future = asyncio.Future()
        self._pending_actions[action_key] = future
        
        # Set up timeout
        timeout_task = asyncio.create_task(self._timeout_action(action_key, timeout))
        self._timeouts[action_key] = timeout_task
        
        try:
            # Wait for action or timeout
            result = await future
            # Cancel timeout
            timeout_task.cancel()
            return result
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Timeout waiting for {action_type} from {player_id}")
        finally:
            # Cleanup
            self._pending_actions.pop(action_key, None)
            self._timeouts.pop(action_key, None)
    
    def submit_action(self, player_id: str, action_type: str, action_data: Dict[str, Any]) -> bool:
        """Submit action from human player."""
        action_key = f"{player_id}_{action_type}"
        
        if action_key in self._pending_actions:
            future = self._pending_actions[action_key]
            if not future.done():
                future.set_result(action_data)
                return True
        return False
    
    async def _timeout_action(self, action_key: str, timeout: int):
        """Handle action timeout."""
        await asyncio.sleep(timeout)
        if action_key in self._pending_actions:
            future = self._pending_actions[action_key]
            if not future.done():
                future.set_exception(asyncio.TimeoutError(f"Action {action_key} timed out"))


class HybridGameCoordinator:
    """Coordinates hybrid games with human and AI players."""
    
    def __init__(self, 
                 hybrid_players: List[HybridPlayerConfig],
                 openrouter_api_key: str,
                 bridge_server: GameWebSocketServer,
                 game_id: Optional[str] = None):
        """
        Initialize hybrid game coordinator.
        
        Args:
            hybrid_players: List of hybrid player configurations
            openrouter_api_key: OpenRouter API key for AI players
            bridge_server: WebSocket server for human communication
            game_id: Optional game ID
        """
        self.game_id = game_id or str(uuid.uuid4())
        self.hybrid_players = hybrid_players
        self.bridge_server = bridge_server
        
        # Separate human and AI players
        self.human_players = [p for p in hybrid_players if p.type == PlayerType.HUMAN]
        self.ai_players = [p for p in hybrid_players if p.type == PlayerType.AI]
        
        # Create AI player configs for GameManager
        ai_player_configs = []
        for player in self.ai_players:
            ai_player_configs.append({
                "id": player.id,
                "name": player.name,
                "model": player.model or "deepseek/deepseek-v3.2-exp"
            })
        
        # Add human players as "dummy" AI players for game state management
        for player in self.human_players:
            ai_player_configs.append({
                "id": player.id,
                "name": player.name,
                "model": "human"  # Special marker for human players
            })
        
        # Create modified game manager
        self.game_manager = GameManager(
            player_configs=ai_player_configs,
            openrouter_api_key=openrouter_api_key,
            game_id=self.game_id
        )
        
        # Human action management
        self.human_action_queue = HumanActionQueue()
        
        # Connected human sessions
        self.human_sessions: Dict[str, str] = {}  # player_id -> session_id
        
        # Game state
        self.is_running = False
        self.current_turn_player = None
    
    async def start_hybrid_game(self) -> Dict[str, Any]:
        """Start the hybrid game."""
        self.is_running = True
        
        try:
            # Start WebSocket server monitoring
            await self.bridge_server.start_game_room(self.game_id)
            
            # Wait for human players to connect
            print(f"🎮 Starting hybrid game {self.game_id}")
            print(f"👥 Players: {len(self.human_players)} human, {len(self.ai_players)} AI")
            
            await self._wait_for_human_connections()
            
            # Start the game with modified behavior for humans
            await self._start_modified_game()
            
            return {"status": "success", "game_id": self.game_id}
            
        except Exception as e:
            await self._handle_error(f"Game error: {str(e)}")
            return {"error": str(e), "game_id": self.game_id}
    
    async def _wait_for_human_connections(self):
        """Wait for all human players to connect with generous timeout."""
        if not self.human_players:
            return  # No human players to wait for
            
        print("⏳ Waiting for human players to connect...")
        print("💡 Take your time - the game will wait for you!")
        
        connected_humans = set()
        timeout = 1800  # 30 minutes - very generous
        start_time = asyncio.get_event_loop().time()
        last_update = start_time
        
        while len(connected_humans) < len(self.human_players):
            current_time = asyncio.get_event_loop().time()
            
            # Show friendly countdown every 30 seconds
            if current_time - last_update > 30:
                remaining = timeout - (current_time - start_time)
                minutes_left = int(remaining // 60)
                if minutes_left > 0:
                    print(f"⏰ Still waiting... {minutes_left} minutes remaining")
                last_update = current_time
            
            if current_time - start_time > timeout:
                raise TimeoutError("Timeout waiting for human players to connect")
            
            # Broadcast connection status
            await self._broadcast_connection_status(connected_humans)
            
            # Check for new connections from bridge server
            for player in self.human_players:
                # Check if player is connected to bridge server
                if player.id in self.bridge_server.human_players and player.id not in connected_humans:
                    connected_humans.add(player.id)
                    print(f"✅ Human player {player.name} connected")
            
            await asyncio.sleep(1)
        
        print(f"🎉 All {len(self.human_players)} human players connected!")
    
    async def _start_modified_game(self):
        """Start game with hybrid player handling."""
        # Initialize game state
        await self.game_manager.logger.log_game_start(self.game_manager.game_state.to_dict())
        
        # Send role information
        await self._send_hybrid_role_information()
        
        # Main game loop with hybrid handling
        while not self.game_manager.game_state.is_game_over and self.is_running:
            await self._execute_hybrid_game_phase()
        
        # Game completed
        final_result = await self._finalize_hybrid_game()
        return final_result
    
    async def _send_hybrid_role_information(self):
        """Send role information to both human and AI players."""
        for player_id, player in self.game_manager.game_state.players.items():
            if self._is_human_player(player_id):
                # Send role info to human via WebSocket
                await self._send_human_role_info(player_id, player)
            else:
                # Send to AI via normal LLM process
                await self._send_ai_role_info(player_id, player)
    
    async def _send_human_role_info(self, player_id: str, player):
        """Send role information to human player."""
        role_info = self.game_manager._get_role_information_for_player(player_id)
        
        role_data = {
            "type": "role_assignment",
            "player_id": player_id,
            "role": player.role.value,
            "role_info": role_info,
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.send_to_player(player_id, role_data)
    
    async def _send_ai_role_info(self, player_id: str, player):
        """Send role information to AI player."""
        # Use existing GameManager method
        role_info = self.game_manager._get_role_information_for_player(player_id)
        
        prompt = self.game_manager.prompt_templates.get_role_assignment_prompt(
            player_name=player.name,
            role=player.role.value,
            role_info=role_info,
            player_count=self.game_manager.game_state.player_count
        )
        
        response = await self.game_manager.openrouter_client.make_request(
            prompt=prompt,
            decision_type="acknowledge_role",
            player_id=player_id
        )
        
        await self.game_manager.logger.log_player_action(
            player_id, "role_assignment", {"response": response.content}
        )
    
    async def _execute_hybrid_game_phase(self):
        """Execute game phase with hybrid player handling."""
        phase = self.game_manager.game_state.phase
        
        if phase == GamePhase.NOMINATION:
            await self._handle_hybrid_nomination()
        elif phase == GamePhase.VOTING:
            await self._handle_hybrid_voting()
        elif phase == GamePhase.LEGISLATIVE_PRESIDENT:
            await self._handle_hybrid_president_legislative()
        elif phase == GamePhase.LEGISLATIVE_CHANCELLOR:
            await self._handle_hybrid_chancellor_legislative()
        elif phase == GamePhase.EXECUTIVE_ACTION:
            await self._handle_hybrid_executive_action()
        else:
            # Default to AI handling
            await self.game_manager._execute_game_phase()
    
    async def _handle_hybrid_nomination(self):
        """Handle nomination phase with human/AI players."""
        president_id = self.game_manager.game_state.get_current_president().id
        
        if self._is_human_player(president_id):
            # Human president nomination
            await self._request_human_nomination(president_id)
        else:
            # AI president nomination
            await self.game_manager._handle_nomination_phase()
    
    async def _request_human_nomination(self, president_id: str):
        """Request nomination from human president."""
        eligible_chancellors = self.game_manager.game_state.get_eligible_chancellors()
        
        nomination_request = {
            "type": "nomination_request",
            "president_id": president_id,
            "eligible_chancellors": [p.id for p in eligible_chancellors],
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.send_to_player(president_id, nomination_request)
        
        # Wait for human response
        try:
            action_data = await self.human_action_queue.wait_for_action(
                president_id, "nominate_chancellor", timeout=120
            )
            
            chancellor_id = action_data["chancellor_id"]
            
            # Execute nomination
            success = self.game_manager.game_state.nominate_chancellor(chancellor_id)
            
            if success:
                await self.game_manager.logger.log_player_action(
                    president_id, "nominate_chancellor", 
                    {"chancellor_id": chancellor_id, "success": True}
                )
                # Broadcast to all players
                await self._broadcast_nomination(president_id, chancellor_id)
            else:
                await self._send_error_to_player(president_id, "Invalid chancellor nomination")
                
        except asyncio.TimeoutError:
            # Handle timeout - maybe auto-select or end game
            await self._handle_action_timeout(president_id, "nomination")
    
    async def _handle_hybrid_voting(self):
        """Handle voting phase with human/AI players."""
        # Collect votes from all players
        voting_tasks = []
        
        for player in self.game_manager.game_state.get_alive_players():
            if self._is_human_player(player.id):
                task = self._request_human_vote(player.id)
            else:
                task = self._request_ai_vote(player.id)
            voting_tasks.append(task)
        
        # Wait for all votes
        await asyncio.gather(*voting_tasks)
        
        # Process voting results
        await self.game_manager._resolve_voting_round()
    
    async def _request_human_vote(self, player_id: str):
        """Request vote from human player."""
        president = self.game_manager.game_state.get_current_president()
        chancellor = self.game_manager.game_state.get_current_chancellor()
        
        vote_request = {
            "type": "vote_request",
            "player_id": player_id,
            "president": president.name,
            "chancellor": chancellor.name,
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.send_to_player(player_id, vote_request)
        
        try:
            action_data = await self.human_action_queue.wait_for_action(
                player_id, "vote", timeout=60
            )
            
            vote = action_data["vote"]  # True for Ja, False for Nein
            self.game_manager.game_state.cast_vote(player_id, vote)
            
            await self.game_manager.logger.log_player_action(
                player_id, "vote", {"vote": "ja" if vote else "nein"}
            )
            
        except asyncio.TimeoutError:
            # Auto-vote Nein on timeout
            self.game_manager.game_state.cast_vote(player_id, False)
            await self._handle_action_timeout(player_id, "vote")
    
    async def _request_ai_vote(self, player_id: str):
        """Request vote from AI player using existing logic."""
        # Use existing GameManager voting logic for AI
        president = self.game_manager.game_state.get_current_president()
        chancellor = self.game_manager.game_state.get_current_chancellor()
        
        private_info = self.game_manager._get_private_info_for_player(player_id)
        game_state_dict = self.game_manager.game_state.to_dict()
        
        prompt = self.game_manager.prompt_templates.get_voting_prompt(
            player_name=self.game_manager.game_state.players[player_id].name,
            president_name=president.name,
            chancellor_name=chancellor.name,
            game_state=game_state_dict,
            private_info=private_info
        )
        
        response = await self.game_manager.openrouter_client.make_request(
            prompt=prompt,
            decision_type="vote_on_government",
            player_id=player_id
        )
        
        # Parse AI vote
        vote = self._parse_ai_vote(response.content)
        self.game_manager.game_state.cast_vote(player_id, vote)
        
        await self.game_manager.logger.log_player_action(
            player_id, "vote", 
            {"vote": "ja" if vote else "nein", "reasoning": response.content}
        )
    
    def _parse_ai_vote(self, response_content: str) -> bool:
        """Parse AI vote from response content."""
        content_lower = response_content.lower()
        if "vote ja" in content_lower or "ja" in content_lower:
            return True
        return False
    
    async def _handle_hybrid_president_legislative(self):
        """Handle president legislative phase."""
        president_id = self.game_manager.game_state.get_current_president().id
        
        # Draw policies
        policies = self.game_manager.game_state.draw_policies(3)
        
        if self._is_human_player(president_id):
            await self._request_human_president_policies(president_id, policies)
        else:
            await self._request_ai_president_policies(president_id, policies)
    
    async def _request_human_president_policies(self, president_id: str, policies: List):
        """Request policy selection from human president."""
        policy_request = {
            "type": "president_policy_request",
            "president_id": president_id,
            "policies": [p.value for p in policies],
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.send_to_player(president_id, policy_request)
        
        try:
            action_data = await self.human_action_queue.wait_for_action(
                president_id, "choose_policies_president", timeout=120
            )
            
            kept_policies_indices = action_data["kept_policies"]
            discarded_index = action_data["discarded_policy"]
            
            # Create kept policies list
            kept_policies = [policies[i] for i in kept_policies_indices]
            
            # Execute policy selection
            success = self.game_manager.game_state.choose_policies_president(policies, kept_policies)
            
            if success:
                await self.game_manager.logger.log_player_action(
                    president_id, "choose_policies_president",
                    {"policies_drawn": [p.value for p in policies], 
                     "policies_kept": [p.value for p in kept_policies]}
                )
            
        except asyncio.TimeoutError:
            await self._handle_action_timeout(president_id, "president_policies")
    
    async def _request_ai_president_policies(self, president_id: str, policies: List):
        """Request policy selection from AI president."""
        # Use existing GameManager logic
        private_info = self.game_manager._get_private_info_for_player(president_id)
        game_state_dict = self.game_manager.game_state.to_dict()
        
        prompt = self.game_manager.prompt_templates.get_president_policy_prompt(
            president_name=self.game_manager.game_state.players[president_id].name,
            policies=[p.value for p in policies],
            game_state=game_state_dict,
            private_info=private_info
        )
        
        response = await self.game_manager.openrouter_client.make_request(
            prompt=prompt,
            decision_type="choose_policies_as_president",
            player_id=president_id
        )
        
        # Parse AI choice (simplified)
        kept_policies = policies[:2]  # Default to first 2
        
        success = self.game_manager.game_state.choose_policies_president(policies, kept_policies)
        
        await self.game_manager.logger.log_player_action(
            president_id, "choose_policies_president",
            {"policies_drawn": [p.value for p in policies], 
             "policies_kept": [p.value for p in kept_policies],
             "reasoning": response.content}
        )
    
    async def _handle_hybrid_chancellor_legislative(self):
        """Handle chancellor legislative phase."""
        chancellor_id = self.game_manager.game_state.get_current_chancellor().id
        policies = self.game_manager.game_state.current_legislative_policies
        
        if self._is_human_player(chancellor_id):
            await self._request_human_chancellor_policy(chancellor_id, policies)
        else:
            await self._request_ai_chancellor_policy(chancellor_id, policies)
    
    async def _request_human_chancellor_policy(self, chancellor_id: str, policies: List):
        """Request policy enactment from human chancellor."""
        veto_available = self.game_manager.game_state.is_veto_unlocked()
        
        policy_request = {
            "type": "chancellor_policy_request",
            "chancellor_id": chancellor_id,
            "policies": [p.value for p in policies],
            "veto_available": veto_available,
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.send_to_player(chancellor_id, policy_request)
        
        try:
            action_data = await self.human_action_queue.wait_for_action(
                chancellor_id, "choose_policies_chancellor", timeout=120
            )
            
            if action_data.get("veto", False) and veto_available:
                # Handle veto request
                await self._handle_veto_request(chancellor_id)
            else:
                policy_index = action_data["policy_index"]
                chosen_policy = policies[policy_index]
                
                success = self.game_manager.game_state.enact_policy(chosen_policy)
                
                if success:
                    await self.game_manager.logger.log_player_action(
                        chancellor_id, "choose_policies_chancellor",
                        {"policies_received": [p.value for p in policies],
                         "policy_enacted": chosen_policy.value}
                    )
            
        except asyncio.TimeoutError:
            # Auto-enact first policy on timeout
            await self._handle_action_timeout(chancellor_id, "chancellor_policies")
    
    async def _request_ai_chancellor_policy(self, chancellor_id: str, policies: List):
        """Request policy enactment from AI chancellor."""
        # Use existing GameManager logic
        private_info = self.game_manager._get_private_info_for_player(chancellor_id)
        game_state_dict = self.game_manager.game_state.to_dict()
        veto_available = self.game_manager.game_state.is_veto_unlocked()
        
        prompt = self.game_manager.prompt_templates.get_chancellor_policy_prompt(
            chancellor_name=self.game_manager.game_state.players[chancellor_id].name,
            policies=[p.value for p in policies],
            veto_available=veto_available,
            game_state=game_state_dict,
            private_info=private_info
        )
        
        response = await self.game_manager.openrouter_client.make_request(
            prompt=prompt,
            decision_type="choose_policies_as_chancellor",
            player_id=chancellor_id
        )
        
        # Parse AI choice (simplified)
        chosen_policy = policies[0]  # Default to first policy
        
        success = self.game_manager.game_state.enact_policy(chosen_policy)
        
        await self.game_manager.logger.log_player_action(
            chancellor_id, "choose_policies_chancellor",
            {"policies_received": [p.value for p in policies],
             "policy_enacted": chosen_policy.value,
             "reasoning": response.content}
        )
    
    async def _handle_hybrid_executive_action(self):
        """Handle executive action phase."""
        # Use existing GameManager logic with hybrid modifications
        await self.game_manager._handle_executive_action_phase()
    
    # Helper methods
    def _is_human_player(self, player_id: str) -> bool:
        """Check if player is human."""
        return any(p.id == player_id for p in self.human_players)
    
    async def _broadcast_nomination(self, president_id: str, chancellor_id: str):
        """Broadcast nomination to all players."""
        nomination_data = {
            "type": "nomination_broadcast",
            "president_id": president_id,
            "chancellor_id": chancellor_id,
            "game_state": self.game_manager.game_state.to_dict()
        }
        
        await self.bridge_server.broadcast_to_game(self.game_id, nomination_data)
    
    async def _broadcast_connection_status(self, connected_humans: set):
        """Broadcast human connection status."""
        status_data = {
            "type": "connection_status",
            "connected_humans": list(connected_humans),
            "total_humans": len(self.human_players),
            "human_players": [{"id": p.id, "name": p.name} for p in self.human_players]
        }
        
        await self.bridge_server.broadcast_to_game(self.game_id, status_data)
    
    async def _send_error_to_player(self, player_id: str, error_message: str):
        """Send error message to specific player."""
        error_data = {
            "type": "error",
            "message": error_message
        }
        
        await self.bridge_server.send_to_player(player_id, error_data)
    
    async def _handle_action_timeout(self, player_id: str, action_type: str):
        """Handle player action timeout."""
        print(f"⏰ Timeout for {player_id} on {action_type}")
        
        timeout_data = {
            "type": "action_timeout",
            "player_id": player_id,
            "action_type": action_type
        }
        
        await self.bridge_server.broadcast_to_game(self.game_id, timeout_data)
    
    async def _handle_veto_request(self, chancellor_id: str):
        """Handle veto request from chancellor."""
        # Implementation for veto handling
        pass
    
    async def _finalize_hybrid_game(self):
        """Finalize hybrid game and return results."""
        final_result = {
            "game_id": self.game_id,
            "winner": self.game_manager.game_state.winner,
            "final_state": self.game_manager.game_state.to_dict(),
            "cost_summary": self.game_manager.openrouter_client.get_cost_summary()
        }
        
        # Broadcast game end
        await self.bridge_server.broadcast_to_game(self.game_id, {
            "type": "game_end",
            "result": final_result
        })
        
        return final_result
    
    async def _handle_error(self, error_message: str):
        """Handle game error."""
        print(f"❌ Game error: {error_message}")
        
        error_data = {
            "type": "game_error",
            "message": error_message
        }
        
        await self.bridge_server.broadcast_to_game(self.game_id, error_data)
    
    # Public interface for bridge server
    def handle_human_action(self, player_id: str, action_type: str, action_data: Dict[str, Any]) -> bool:
        """Handle action from human player via bridge server."""
        return self.human_action_queue.submit_action(player_id, action_type, action_data)
    
    def add_human_connection(self, player_id: str, session_id: str):
        """Add human player connection."""
        for player in self.hybrid_players:
            if player.id == player_id:
                player.session_id = session_id
                self.human_sessions[player_id] = session_id
                break
    
    def remove_human_connection(self, player_id: str):
        """Remove human player connection."""
        for player in self.hybrid_players:
            if player.id == player_id:
                player.session_id = None
                break
        self.human_sessions.pop(player_id, None)