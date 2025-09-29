"""
Secret Hitler game state management.
"""
from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import random
from datetime import datetime
import json

class Role(Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"
    HITLER = "hitler"

class Policy(Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"

class GamePhase(Enum):
    SETUP = "setup"
    ELECTION = "election"
    NOMINATION = "nomination"
    VOTING = "voting"
    LEGISLATIVE_PRESIDENT = "legislative_president"
    LEGISLATIVE_CHANCELLOR = "legislative_chancellor"
    EXECUTIVE_ACTION = "executive_action"
    GAME_OVER = "game_over"

class VoteResult(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"

@dataclass
class Player:
    """Represents a player in the game."""
    id: str
    name: str
    role: Role
    is_alive: bool = True
    is_eligible_chancellor: bool = True
    is_eligible_president: bool = True
    investigated_by: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'role': self.role.value,
            'is_alive': self.is_alive,
            'is_eligible_chancellor': self.is_eligible_chancellor,
            'is_eligible_president': self.is_eligible_president,
            'investigated_by': list(self.investigated_by)
        }

@dataclass 
class Government:
    """Current government configuration."""
    president: Optional[str] = None
    chancellor: Optional[str] = None
    is_active: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'president': self.president,
            'chancellor': self.chancellor,
            'is_active': self.is_active
        }

@dataclass
class ElectionTracker:
    """Track election failures."""
    failed_elections: int = 0
    MAX_FAILURES: int = 3
    
    def add_failure(self) -> bool:
        """Add election failure. Returns True if chaos should occur."""
        self.failed_elections += 1
        return self.failed_elections >= self.MAX_FAILURES
    
    def reset(self):
        """Reset after successful election."""
        self.failed_elections = 0
    
    def to_dict(self) -> Dict:
        return {
            'failed_elections': self.failed_elections,
            'max_failures': self.MAX_FAILURES
        }

@dataclass
class Vote:
    """Individual player vote."""
    player_id: str
    vote: bool  # True for Ja, False for Nein
    
    def to_dict(self) -> Dict:
        return {
            'player_id': self.player_id,
            'vote': self.vote
        }

@dataclass
class VotingRound:
    """Complete voting round."""
    government: Government
    votes: List[Vote] = field(default_factory=list)
    result: VoteResult = VoteResult.PENDING
    
    def add_vote(self, player_id: str, vote: bool):
        """Add a player's vote."""
        self.votes.append(Vote(player_id, vote))
    
    def calculate_result(self, total_players: int) -> VoteResult:
        """Calculate voting result."""
        if len(self.votes) < total_players:
            return VoteResult.PENDING
        
        ja_votes = sum(1 for vote in self.votes if vote.vote)
        nein_votes = len(self.votes) - ja_votes
        
        self.result = VoteResult.PASSED if ja_votes > nein_votes else VoteResult.FAILED
        return self.result
    
    def to_dict(self) -> Dict:
        return {
            'government': self.government.to_dict(),
            'votes': [vote.to_dict() for vote in self.votes],
            'result': self.result.value
        }

@dataclass
class PolicyBoard:
    """Track enacted policies."""
    liberal_policies: int = 0
    fascist_policies: int = 0
    
    MAX_LIBERAL: int = 5
    MAX_FASCIST: int = 6
    
    def enact_policy(self, policy: Policy) -> bool:
        """Enact a policy. Returns True if game should end."""
        if policy == Policy.LIBERAL:
            self.liberal_policies += 1
            return self.liberal_policies >= self.MAX_LIBERAL
        else:
            self.fascist_policies += 1
            return self.fascist_policies >= self.MAX_FASCIST
    
    def get_presidential_power(self, player_count: int) -> Optional[str]:
        """Get presidential power based on fascist policies and player count."""
        powers = {
            5: {3: "investigate", 4: "execute", 5: "execute"},
            6: {3: "investigate", 4: "execute", 5: "execute"},
            7: {2: "investigate", 3: "special_election", 4: "execute", 5: "execute"},
            8: {2: "investigate", 3: "special_election", 4: "execute", 5: "execute"},
            9: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execute", 5: "execute"},
            10: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execute", 5: "execute"}
        }
        
        return powers.get(player_count, {}).get(self.fascist_policies)
    
    def veto_unlocked(self) -> bool:
        """Check if veto power is unlocked."""
        return self.fascist_policies >= 5
    
    def to_dict(self) -> Dict:
        return {
            'liberal_policies': self.liberal_policies,
            'fascist_policies': self.fascist_policies,
            'max_liberal': self.MAX_LIBERAL,
            'max_fascist': self.MAX_FASCIST
        }

class GameState:
    """Complete Secret Hitler game state."""
    
    def __init__(self, player_configs: List[Dict[str, str]], game_id: str):
        self.game_id = game_id
        self.created_at = datetime.now()
        self.players = self._setup_players(player_configs)
        self.player_count = len(self.players)
        
        # Game state
        self.phase = GamePhase.SETUP
        self.current_president_index = 0
        self.government = Government()
        self.policy_board = PolicyBoard()
        self.election_tracker = ElectionTracker()
        
        # Game history
        self.voting_rounds: List[VotingRound] = []
        self.policy_deck = self._create_deck()
        self.discard_pile: List[Policy] = []
        
        # Executive actions
        self.last_investigation: Optional[Dict] = None
        self.special_election_president: Optional[str] = None
        
        # Game end conditions
        self.is_game_over = False
        self.winner: Optional[str] = None
        self.win_condition: Optional[str] = None
        
        # Logging
        self.action_log: List[Dict] = []
        
        # Legislative session tracking
        self.current_legislative_policies = None  # Policies being processed this round
        
        self._log_action("game_start", {
            "game_id": game_id,
            "player_count": self.player_count,
            "players": [p.to_dict() for p in self.players.values()]
        })
    
    def _setup_players(self, player_configs: List[Dict[str, str]]) -> Dict[str, Player]:
        """Setup players with roles."""
        players = {}
        
        # Create player objects
        for config in player_configs:
            player = Player(
                id=config['id'],
                name=config['name'],
                role=Role.LIBERAL  # Will be assigned later
            )
            players[player.id] = player
        
        # Assign roles based on player count
        self._assign_roles(players)
        
        return players
    
    def _assign_roles(self, players: Dict[str, Player]):
        """Assign roles based on player count."""
        player_list = list(players.values())
        random.shuffle(player_list)
        
        role_distributions = {
            # Test configurations (not balanced for real play)
            2: {"liberal": 1, "hitler": 1},  # Minimal test
            3: {"liberal": 2, "hitler": 1},  # Basic test
            4: {"liberal": 2, "fascist": 1, "hitler": 1},  # Small test
            
            # Official game configurations
            5: {"liberal": 3, "fascist": 1, "hitler": 1},
            6: {"liberal": 4, "fascist": 1, "hitler": 1},
            7: {"liberal": 4, "fascist": 2, "hitler": 1},
            8: {"liberal": 5, "fascist": 2, "hitler": 1},
            9: {"liberal": 5, "fascist": 3, "hitler": 1},
            10: {"liberal": 6, "fascist": 3, "hitler": 1}
        }
        
        distribution = role_distributions[len(player_list)]
        
        # Assign roles
        idx = 0
        for role, count in distribution.items():
            for _ in range(count):
                player_list[idx].role = Role(role)
                idx += 1
    
    def _create_deck(self) -> List[Policy]:
        """Create the policy deck."""
        deck = [Policy.LIBERAL] * 6 + [Policy.FASCIST] * 11
        random.shuffle(deck)
        return deck
    
    def get_fascists(self) -> List[Player]:
        """Get all fascist players (including Hitler)."""
        return [p for p in self.players.values() 
                if p.role in [Role.FASCIST, Role.HITLER] and p.is_alive]
    
    def get_liberals(self) -> List[Player]:
        """Get all liberal players."""
        return [p for p in self.players.values() 
                if p.role == Role.LIBERAL and p.is_alive]
    
    def get_hitler(self) -> Optional[Player]:
        """Get Hitler player."""
        for player in self.players.values():
            if player.role == Role.HITLER and player.is_alive:
                return player
        return None
    
    def get_alive_players(self) -> List[Player]:
        """Get all alive players."""
        return [p for p in self.players.values() if p.is_alive]
    
    def get_current_president(self) -> Optional[Player]:
        """Get current president."""
        alive_players = self.get_alive_players()
        if not alive_players:
            return None
        
        # Handle special election
        if self.special_election_president:
            return self.players.get(self.special_election_president)
        
        # Regular president rotation
        eligible_players = [p for p in alive_players if p.is_eligible_president]
        if not eligible_players:
            # Reset eligibility if all ineligible
            for p in alive_players:
                p.is_eligible_president = True
            eligible_players = alive_players
        
        return eligible_players[self.current_president_index % len(eligible_players)]
    
    def advance_president(self):
        """Advance to next president."""
        self.special_election_president = None
        alive_players = [p for p in self.get_alive_players() if p.is_eligible_president]
        self.current_president_index = (self.current_president_index + 1) % len(alive_players)
    
    def draw_policies(self, count: int) -> List[Policy]:
        """Draw policies from deck."""
        if len(self.policy_deck) < count:
            # Shuffle discard pile back into deck
            self.policy_deck.extend(self.discard_pile)
            self.discard_pile = []
            random.shuffle(self.policy_deck)
        
        drawn = self.policy_deck[:count]
        self.policy_deck = self.policy_deck[count:]
        return drawn
    
    def discard_policy(self, policy: Policy):
        """Add policy to discard pile."""
        self.discard_pile.append(policy)
    
    def nominate_chancellor(self, chancellor_id: str) -> bool:
        """Nominate chancellor."""
        if chancellor_id not in self.players or not self.players[chancellor_id].is_alive:
            return False
        
        if not self.players[chancellor_id].is_eligible_chancellor:
            return False
        
        self.government.chancellor = chancellor_id
        self.phase = GamePhase.VOTING
        
        self._log_action("nominate_chancellor", {
            "president": self.government.president,
            "chancellor": chancellor_id
        })
        
        return True
    
    def cast_vote(self, player_id: str, vote: bool) -> bool:
        """Cast vote for current government."""
        if self.phase != GamePhase.VOTING:
            return False
        
        if not self.voting_rounds:
            self.voting_rounds.append(VotingRound(government=self.government))
        
        current_round = self.voting_rounds[-1]
        
        # Check if player already voted
        if any(v.player_id == player_id for v in current_round.votes):
            return False
        
        current_round.add_vote(player_id, vote)
        
        # Check if all players voted
        alive_count = len(self.get_alive_players())
        if len(current_round.votes) == alive_count:
            result = current_round.calculate_result(alive_count)
            self._process_vote_result(result)
        
        self._log_action("cast_vote", {
            "player": player_id,
            "vote": vote
        })
        
        return True
    
    def _process_vote_result(self, result: VoteResult):
        """Process the result of a vote."""
        if result == VoteResult.PASSED:
            self.election_tracker.reset()
            self.government.is_active = True
            
            # Check Hitler chancellor win condition
            if (self.policy_board.fascist_policies >= 3 and 
                self.players[self.government.chancellor].role == Role.HITLER):
                self._end_game("fascist", "hitler_chancellor")
                return
            
            self.phase = GamePhase.LEGISLATIVE_PRESIDENT
            
            # Update eligibility
            self._update_chancellor_eligibility()
            
        else:  # FAILED
            self.government = Government()
            if self.election_tracker.add_failure():
                self._chaos_round()
            else:
                self.advance_president()
                self.phase = GamePhase.NOMINATION
    
    def _update_chancellor_eligibility(self):
        """Update chancellor eligibility after successful election."""
        # Previous president and chancellor become ineligible
        if hasattr(self, '_previous_government'):
            if self._previous_government.president:
                self.players[self._previous_government.president].is_eligible_chancellor = True
            if self._previous_government.chancellor:
                self.players[self._previous_government.chancellor].is_eligible_chancellor = True
        
        # Current government becomes ineligible
        if self.government.president:
            self.players[self.government.president].is_eligible_chancellor = False
        if self.government.chancellor:
            self.players[self.government.chancellor].is_eligible_chancellor = False
        
        self._previous_government = Government(
            president=self.government.president,
            chancellor=self.government.chancellor
        )
    
    def _chaos_round(self):
        """Handle chaos when 3 elections fail."""
        policy = self.draw_policies(1)[0]
        game_over = self.policy_board.enact_policy(policy)
        
        self._log_action("chaos_policy", {
            "policy": policy.value
        })
        
        if game_over:
            winner = "liberal" if policy == Policy.LIBERAL else "fascist"
            self._end_game(winner, "policy_majority")
        else:
            self.election_tracker.reset()
            self.advance_president()
            self.phase = GamePhase.NOMINATION
    
    def choose_policies_president(self, drawn_policies: List[Policy], kept_policies: List[Policy]) -> bool:
        """President chooses policies to keep."""
        if self.phase != GamePhase.LEGISLATIVE_PRESIDENT:
            return False
        
        if len(kept_policies) != 2:
            return False
            
        if len(drawn_policies) != 3:
            return False
        
        # Discard the third policy
        for policy in drawn_policies:
            if policy not in kept_policies:
                self.discard_policy(policy)
        
        # Store the policies for the chancellor
        self.current_legislative_policies = kept_policies
        self.phase = GamePhase.LEGISLATIVE_CHANCELLOR
        
        self._log_action("president_choose_policies", {
            "president": self.government.president,
            "kept_policies": [p.value for p in kept_policies]
        })
        
        return True
    
    def choose_policy_chancellor(self, chosen_policy: Policy, veto_requested: bool = False) -> bool:
        """Chancellor chooses final policy to enact."""
        if self.phase != GamePhase.LEGISLATIVE_CHANCELLOR:
            return False
        
        # Handle veto
        if veto_requested and self.policy_board.veto_unlocked():
            return self._request_veto()
        
        # Enact policy
        game_over = self.policy_board.enact_policy(chosen_policy)
        
        self._log_action("enact_policy", {
            "policy": chosen_policy.value,
            "chancellor": self.government.chancellor
        })
        
        if game_over:
            winner = "liberal" if chosen_policy == Policy.LIBERAL else "fascist"
            self._end_game(winner, "policy_majority")
            return True
        
        # Check for presidential power
        power = self.policy_board.get_presidential_power(self.player_count)
        if power and chosen_policy == Policy.FASCIST:
            self.phase = GamePhase.EXECUTIVE_ACTION
            self._log_action("presidential_power", {"power": power})
        else:
            self._end_legislative_session()
        
        return True
    
    def _request_veto(self) -> bool:
        """Handle veto request."""
        # In this implementation, we'll assume president always agrees to veto
        # In a real game, this would require president confirmation
        
        self.election_tracker.add_failure()
        self.government = Government()
        self.advance_president()
        self.phase = GamePhase.NOMINATION
        
        self._log_action("veto_used", {
            "president": self.government.president,
            "chancellor": self.government.chancellor
        })
        
        return True
    
    def _end_legislative_session(self):
        """End current legislative session."""
        self.government = Government()
        self.advance_president()
        self.phase = GamePhase.NOMINATION
    
    def execute_investigate(self, target_id: str) -> Optional[Role]:
        """Investigate a player's role."""
        if target_id not in self.players or not self.players[target_id].is_alive:
            return None
        
        target = self.players[target_id]
        target.investigated_by.add(self.government.president)
        
        self.last_investigation = {
            "investigator": self.government.president,
            "target": target_id,
            "result": target.role.value
        }
        
        self._log_action("investigate", {
            "investigator": self.government.president,
            "target": target_id
        })
        
        self._end_legislative_session()
        return target.role
    
    def execute_special_election(self, next_president_id: str) -> bool:
        """Choose next president (special election)."""
        if next_president_id not in self.players or not self.players[next_president_id].is_alive:
            return False
        
        self.special_election_president = next_president_id
        
        self._log_action("special_election", {
            "chooser": self.government.president,
            "next_president": next_president_id
        })
        
        self._end_legislative_session()
        return True
    
    def execute_execution(self, target_id: str) -> bool:
        """Execute a player."""
        if target_id not in self.players or not self.players[target_id].is_alive:
            return False
        
        target = self.players[target_id]
        target.is_alive = False
        
        self._log_action("execute", {
            "executor": self.government.president,
            "target": target_id,
            "target_role": target.role.value
        })
        
        # Check win conditions
        if target.role == Role.HITLER:
            self._end_game("liberal", "hitler_killed")
        else:
            self._end_legislative_session()
        
        return True
    
    def _end_game(self, winner: str, condition: str):
        """End the game with winner and condition."""
        self.is_game_over = True
        self.winner = winner
        self.win_condition = condition
        self.phase = GamePhase.GAME_OVER
        
        self._log_action("game_end", {
            "winner": winner,
            "condition": condition,
            "final_state": self.to_dict()
        })
    
    def _log_action(self, action: str, data: Dict):
        """Log a game action."""
        self.action_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
            "game_state": {
                "phase": self.phase.value,
                "current_president": self.government.president,
                "liberal_policies": self.policy_board.liberal_policies,
                "fascist_policies": self.policy_board.fascist_policies
            }
        })
    
    def to_dict(self) -> Dict:
        """Convert game state to dictionary."""
        return {
            "game_id": self.game_id,
            "created_at": self.created_at.isoformat(),
            "phase": self.phase.value,
            "players": {pid: player.to_dict() for pid, player in self.players.items()},
            "current_president_index": self.current_president_index,
            "government": self.government.to_dict(),
            "policy_board": self.policy_board.to_dict(),
            "election_tracker": self.election_tracker.to_dict(),
            "voting_rounds": [vr.to_dict() for vr in self.voting_rounds],
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "win_condition": self.win_condition,
            "last_investigation": self.last_investigation,
            "special_election_president": self.special_election_president
        }