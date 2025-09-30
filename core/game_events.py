"""
Structured game event schema using Pydantic for spectator system.

Author: Samuel Chakwera (stchakdev)
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class EventType(str, Enum):
    """Types of game events for spectator display."""
    # Game flow
    GAME_START = "game_start"
    ROUND_START = "round_start"
    PHASE_CHANGE = "phase_change"

    # Player actions
    NOMINATION = "nomination"
    VOTE_CAST = "vote_cast"
    VOTE_RESULT = "vote_result"
    POLICY_DRAW = "policy_draw"
    POLICY_DISCARD = "policy_discard"
    POLICY_ENACTED = "policy_enacted"

    # Executive actions
    INVESTIGATION = "investigation"
    SPECIAL_ELECTION = "special_election"
    POLICY_PEEK = "policy_peek"
    EXECUTION = "execution"

    # AI-specific events (KEY FOR SPECTATOR)
    REASONING = "reasoning"      # Private AI thought
    SPEECH = "speech"            # Public AI statement
    DECEPTION_ALERT = "deception_alert"  # When AI is being deceptive

    # Game end
    GAME_OVER = "game_over"


class BeliefDistribution(BaseModel):
    """Role probability beliefs for a single player."""
    liberal: float = Field(ge=0.0, le=1.0, description="Probability player is Liberal")
    fascist: float = Field(ge=0.0, le=1.0, description="Probability player is Fascist")
    hitler: float = Field(ge=0.0, le=1.0, description="Probability player is Hitler")

    class Config:
        frozen = True


class GameEvent(BaseModel):
    """Base game event with all possible fields for different event types."""

    # Core identification
    event_id: str = Field(description="Unique event identifier")
    event_type: EventType = Field(description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    game_id: str = Field(description="Game identifier")

    # Game context
    round_number: int = Field(description="Current round number")
    phase: str = Field(description="Current game phase")

    # Player context (if event is player-specific)
    player_id: Optional[str] = Field(None, description="Player who triggered event")
    player_name: Optional[str] = Field(None, description="Player name for display")

    # AI reasoning fields (for REASONING events)
    reasoning: Optional[str] = Field(None, description="Private strategic analysis")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence in decision")
    beliefs: Optional[Dict[str, BeliefDistribution]] = Field(
        None,
        description="AI's belief distribution about each player's role"
    )
    strategy: Optional[str] = Field(None, description="Current strategic plan")

    # Public statement fields (for SPEECH events)
    public_statement: Optional[str] = Field(None, description="What AI says publicly")
    statement_type: Optional[str] = Field(
        None,
        description="Type of statement: statement, vote_explanation, nomination_reason, accusation, defense"
    )
    statement_target: Optional[str] = Field(None, description="Target player if statement is directed")

    # Deception detection
    is_deceptive: Optional[bool] = Field(None, description="Whether statement contradicts reasoning")
    deception_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in deception detection")

    # Action data (varies by event type)
    action: Optional[str] = Field(None, description="Specific action taken")
    action_data: Optional[Dict[str, Any]] = Field(None, description="Additional action context")

    # Event-specific data
    data: Dict[str, Any] = Field(default_factory=dict, description="Event-specific payload")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ReasoningEvent(BaseModel):
    """Structured AI reasoning event (private - spectators only)."""
    player_id: str
    player_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Core reasoning
    summary: str = Field(description="2-3 sentence summary of reasoning")
    full_reasoning: Optional[str] = Field(None, description="Complete chain of thought")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decision")

    # Strategic context
    beliefs: Dict[str, BeliefDistribution] = Field(
        default_factory=dict,
        description="Role probability estimates for each player"
    )
    strategy: Optional[str] = Field(None, description="Current strategic plan")

    # Decision context
    decision_type: str = Field(description="Type of decision: nomination, vote, policy_choice, etc.")
    available_options: List[str] = Field(default_factory=list, description="Options available")
    chosen_option: Optional[str] = Field(None, description="Option chosen")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SpeechEvent(BaseModel):
    """Structured AI speech event (public - what other players hear)."""
    player_id: str
    player_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Statement content
    content: str = Field(description="Public statement content")
    statement_type: str = Field(
        default="statement",
        description="Type: statement, vote_explanation, nomination_reason, accusation, defense"
    )
    target_player: Optional[str] = Field(None, description="Target if directed statement")

    # Deception detection
    is_deceptive: bool = Field(default=False, description="Contradicts private reasoning")
    deception_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in deception")
    contradiction_summary: Optional[str] = Field(None, description="How it contradicts reasoning")

    # Context
    game_context: Optional[str] = Field(None, description="What's happening in game")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrustUpdate(BaseModel):
    """Trust relationship change event."""
    from_player: str
    to_player: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Trust metrics
    trust_level: float = Field(ge=-1.0, le=1.0, description="-1 (distrust) to 1 (trust)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in trust assessment")

    # Reason for change
    trigger_event: str = Field(description="What caused trust change")
    reasoning: Optional[str] = Field(None, description="Why trust changed")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Event builder helper functions

def create_reasoning_event(
    player_id: str,
    player_name: str,
    reasoning_summary: str,
    confidence: float,
    beliefs: Dict[str, Dict[str, float]],
    decision_type: str,
    full_reasoning: Optional[str] = None,
    strategy: Optional[str] = None,
    available_options: Optional[List[str]] = None,
    chosen_option: Optional[str] = None
) -> ReasoningEvent:
    """Create a reasoning event from AI decision."""

    # Convert belief dicts to BeliefDistribution objects
    belief_distributions = {}
    for target_player, probs in beliefs.items():
        belief_distributions[target_player] = BeliefDistribution(
            liberal=probs.get('liberal', 0.33),
            fascist=probs.get('fascist', 0.33),
            hitler=probs.get('hitler', 0.34)
        )

    return ReasoningEvent(
        player_id=player_id,
        player_name=player_name,
        summary=reasoning_summary,
        full_reasoning=full_reasoning,
        confidence=confidence,
        beliefs=belief_distributions,
        strategy=strategy,
        decision_type=decision_type,
        available_options=available_options or [],
        chosen_option=chosen_option
    )


def create_speech_event(
    player_id: str,
    player_name: str,
    content: str,
    statement_type: str = "statement",
    target_player: Optional[str] = None,
    is_deceptive: bool = False,
    deception_score: float = 0.0,
    contradiction_summary: Optional[str] = None,
    game_context: Optional[str] = None
) -> SpeechEvent:
    """Create a speech event from AI public statement."""

    return SpeechEvent(
        player_id=player_id,
        player_name=player_name,
        content=content,
        statement_type=statement_type,
        target_player=target_player,
        is_deceptive=is_deceptive,
        deception_score=deception_score,
        contradiction_summary=contradiction_summary,
        game_context=game_context
    )


def create_game_event(
    event_type: EventType,
    game_id: str,
    round_number: int,
    phase: str,
    data: Dict[str, Any],
    player_id: Optional[str] = None,
    player_name: Optional[str] = None,
    event_id: Optional[str] = None
) -> GameEvent:
    """Create a generic game event."""

    import uuid

    return GameEvent(
        event_id=event_id or str(uuid.uuid4()),
        event_type=event_type,
        game_id=game_id,
        round_number=round_number,
        phase=phase,
        player_id=player_id,
        player_name=player_name,
        data=data
    )