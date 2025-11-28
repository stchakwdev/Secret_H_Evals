"""
Temporal Analysis Module for Secret Hitler LLM Research.

Analyzes how game dynamics evolve over time, including:
- Trust evolution across rounds
- Deception rate changes through game phases
- Policy enactment patterns
- Turning points and critical moments

Author: Samuel Chakwera (stchakdev)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GamePhase:
    """Represents a distinct phase in game progression."""
    phase_name: str
    start_turn: int
    end_turn: int
    liberal_policies: int
    fascist_policies: int
    avg_deception_rate: float
    key_events: List[str]


@dataclass
class TurningPoint:
    """Represents a critical turning point in the game."""
    turn_number: int
    event_type: str
    description: str
    trust_impact: float
    deception_change: float
    predicted_outcome_shift: float


@dataclass
class TemporalMetrics:
    """Container for temporal analysis metrics."""
    game_id: str
    total_turns: int
    phases: List[GamePhase]
    turning_points: List[TurningPoint]
    trust_trajectory: List[float]
    deception_trajectory: List[float]
    momentum_shifts: List[int]
    early_game_deception: float
    mid_game_deception: float
    late_game_deception: float
    deception_trend: str  # "increasing", "decreasing", "stable", "volatile"


def segment_game_into_phases(
    events: List[Dict[str, Any]],
    liberal_target: int = 5,
    fascist_target: int = 6
) -> List[GamePhase]:
    """
    Segment a game into distinct phases based on policy progression.

    Phases:
    - Early game: 0-2 total policies
    - Mid game: 3-5 total policies
    - Late game: 6+ total policies or close to win condition

    Args:
        events: List of game events with timestamps and details
        liberal_target: Liberal win threshold
        fascist_target: Fascist win threshold

    Returns:
        List of GamePhase objects
    """
    if not events:
        return []

    phases = []
    current_turn = 0
    liberal_count = 0
    fascist_count = 0

    phase_events = {
        'early': {'events': [], 'start': 0, 'deceptions': []},
        'mid': {'events': [], 'start': None, 'deceptions': []},
        'late': {'events': [], 'start': None, 'deceptions': []},
    }

    current_phase = 'early'

    for event in events:
        turn = event.get('turn', current_turn)
        current_turn = max(current_turn, turn)

        # Track policy enactments
        if event.get('type') == 'policy_enacted':
            policy = event.get('policy')
            if policy == 'liberal':
                liberal_count += 1
            elif policy == 'fascist':
                fascist_count += 1

        # Track deception events
        if event.get('is_deception', False):
            deception_score = event.get('deception_score', 1.0)
            phase_events[current_phase]['deceptions'].append(deception_score)

        # Record event
        phase_events[current_phase]['events'].append(event)

        # Determine phase transitions
        total_policies = liberal_count + fascist_count

        if current_phase == 'early' and total_policies >= 3:
            current_phase = 'mid'
            phase_events['mid']['start'] = turn

        elif current_phase == 'mid':
            # Transition to late game when close to win
            if (total_policies >= 6 or
                liberal_count >= liberal_target - 1 or
                fascist_count >= fascist_target - 1):
                current_phase = 'late'
                phase_events['late']['start'] = turn

    # Build phase objects
    for phase_name, data in phase_events.items():
        if not data['events']:
            continue

        start = data['start'] if data['start'] is not None else 0
        end = max(e.get('turn', start) for e in data['events']) if data['events'] else start

        avg_deception = (
            np.mean(data['deceptions']) if data['deceptions'] else 0.0
        )

        # Extract key events
        key_events = []
        for e in data['events']:
            if e.get('type') in ['policy_enacted', 'execution', 'veto', 'special_election']:
                key_events.append(f"Turn {e.get('turn', '?')}: {e.get('type')}")

        phases.append(GamePhase(
            phase_name=phase_name,
            start_turn=start,
            end_turn=end,
            liberal_policies=liberal_count if phase_name == 'late' else 0,
            fascist_policies=fascist_count if phase_name == 'late' else 0,
            avg_deception_rate=avg_deception,
            key_events=key_events[:5]  # Limit to 5 key events
        ))

    return phases


def detect_turning_points(
    trust_scores: List[float],
    deception_scores: List[float],
    events: List[Dict[str, Any]],
    sensitivity: float = 1.5
) -> List[TurningPoint]:
    """
    Detect critical turning points in game dynamics.

    Uses change point detection on trust and deception trajectories.

    Args:
        trust_scores: Trust scores over time
        deception_scores: Deception scores over time
        events: Game events with timestamps
        sensitivity: Sensitivity for peak detection (higher = fewer peaks)

    Returns:
        List of TurningPoint objects
    """
    turning_points = []

    if len(trust_scores) < 3 or len(deception_scores) < 3:
        return turning_points

    # Smooth the trajectories
    trust_smooth = gaussian_filter1d(trust_scores, sigma=1)
    deception_smooth = gaussian_filter1d(deception_scores, sigma=1)

    # Calculate derivatives (rate of change)
    trust_derivative = np.gradient(trust_smooth)
    deception_derivative = np.gradient(deception_smooth)

    # Find peaks in absolute derivative (points of rapid change)
    trust_change_peaks, trust_props = find_peaks(
        np.abs(trust_derivative),
        prominence=np.std(trust_derivative) * sensitivity
    )

    deception_change_peaks, deception_props = find_peaks(
        np.abs(deception_derivative),
        prominence=np.std(deception_derivative) * sensitivity
    )

    # Process trust change points
    for peak_idx in trust_change_peaks:
        if peak_idx >= len(events):
            continue

        event = events[peak_idx] if peak_idx < len(events) else {}
        trust_impact = trust_derivative[peak_idx]

        # Find corresponding deception change
        deception_change = (
            deception_derivative[peak_idx]
            if peak_idx < len(deception_derivative) else 0
        )

        # Estimate outcome shift based on trust direction
        outcome_shift = trust_impact * 0.1  # Simplified model

        turning_points.append(TurningPoint(
            turn_number=peak_idx + 1,
            event_type=event.get('type', 'trust_shift'),
            description=f"Significant trust {'increase' if trust_impact > 0 else 'decrease'}",
            trust_impact=float(trust_impact),
            deception_change=float(deception_change),
            predicted_outcome_shift=float(outcome_shift)
        ))

    # Process deception change points not already captured
    for peak_idx in deception_change_peaks:
        if peak_idx in trust_change_peaks:
            continue
        if peak_idx >= len(events):
            continue

        event = events[peak_idx] if peak_idx < len(events) else {}
        deception_change = deception_derivative[peak_idx]

        turning_points.append(TurningPoint(
            turn_number=peak_idx + 1,
            event_type=event.get('type', 'deception_shift'),
            description=f"Deception rate {'spike' if deception_change > 0 else 'drop'}",
            trust_impact=0.0,
            deception_change=float(deception_change),
            predicted_outcome_shift=float(-deception_change * 0.05)  # Negative correlation
        ))

    # Sort by turn number
    turning_points.sort(key=lambda tp: tp.turn_number)

    return turning_points


def calculate_trust_trajectory(
    decisions: List[Dict[str, Any]],
    players: List[str],
    window_size: int = 3
) -> Dict[str, List[float]]:
    """
    Calculate trust trajectory for each player over time.

    Uses a sliding window of voting alignment to estimate trust.

    Args:
        decisions: List of decision events
        players: List of player IDs
        window_size: Size of sliding window for trust calculation

    Returns:
        Dict mapping player IDs to trust score trajectories
    """
    trajectories = {p: [] for p in players}

    # Group decisions by turn
    turns = {}
    for d in decisions:
        turn = d.get('turn_number', 0)
        if turn not in turns:
            turns[turn] = []
        turns[turn].append(d)

    # Calculate trust at each turn
    vote_history = {p: [] for p in players}

    for turn in sorted(turns.keys()):
        turn_decisions = turns[turn]

        # Record votes
        for d in turn_decisions:
            if d.get('decision_type') == 'vote':
                player = d.get('player_id')
                vote = d.get('action')
                if player in vote_history:
                    vote_history[player].append((turn, vote))

        # Calculate trust scores based on voting alignment with majority
        for player in players:
            recent_votes = vote_history[player][-window_size:]

            if not recent_votes:
                trajectories[player].append(0.5)  # Neutral default
                continue

            # Get majority vote for each turn in window
            alignment_score = 0
            for vote_turn, vote in recent_votes:
                # Find majority for this turn
                turn_votes = [
                    d.get('action') for d in turns.get(vote_turn, [])
                    if d.get('decision_type') == 'vote'
                ]
                if turn_votes:
                    majority = max(set(turn_votes), key=turn_votes.count)
                    if vote == majority:
                        alignment_score += 1

            trust_score = alignment_score / len(recent_votes)
            trajectories[player].append(trust_score)

    return trajectories


def calculate_deception_trajectory(
    decisions: List[Dict[str, Any]],
    window_size: int = 5
) -> List[float]:
    """
    Calculate rolling deception rate over time.

    Args:
        decisions: List of decision events
        window_size: Size of sliding window

    Returns:
        List of deception rates over time
    """
    if not decisions:
        return []

    # Sort by timestamp or turn
    sorted_decisions = sorted(
        decisions,
        key=lambda d: d.get('timestamp', d.get('turn_number', 0))
    )

    trajectory = []
    deception_window = []

    for d in sorted_decisions:
        is_deception = d.get('is_deception', False)
        deception_window.append(1 if is_deception else 0)

        # Keep window size
        if len(deception_window) > window_size:
            deception_window.pop(0)

        # Calculate rate
        rate = sum(deception_window) / len(deception_window)
        trajectory.append(rate)

    return trajectory


def detect_momentum_shifts(
    liberal_policies: List[int],
    fascist_policies: List[int]
) -> List[int]:
    """
    Detect momentum shifts in policy progression.

    A momentum shift occurs when the team that was behind
    catches up or takes the lead.

    Args:
        liberal_policies: Cumulative liberal policy count at each turn
        fascist_policies: Cumulative fascist policy count at each turn

    Returns:
        List of turn indices where momentum shifted
    """
    shifts = []

    if len(liberal_policies) < 2:
        return shifts

    # Calculate lead at each turn (positive = liberal ahead)
    leads = [lib - fas for lib, fas in zip(liberal_policies, fascist_policies)]

    # Detect sign changes (momentum shifts)
    for i in range(1, len(leads)):
        if leads[i] * leads[i-1] < 0:  # Sign changed
            shifts.append(i)
        elif leads[i-1] == 0 and leads[i] != 0:  # Broke tie
            shifts.append(i)

    return shifts


def classify_deception_trend(trajectory: List[float]) -> str:
    """
    Classify the overall deception trend.

    Args:
        trajectory: Deception rates over time

    Returns:
        Trend classification: "increasing", "decreasing", "stable", "volatile"
    """
    if len(trajectory) < 3:
        return "insufficient_data"

    # Calculate linear regression slope
    x = np.arange(len(trajectory))
    slope, _, r_value, _, _ = stats.linregress(x, trajectory)

    # Calculate volatility (standard deviation of changes)
    changes = np.diff(trajectory)
    volatility = np.std(changes)

    # Classification thresholds
    slope_threshold = 0.02  # 2% per turn
    volatility_threshold = 0.15

    if volatility > volatility_threshold:
        return "volatile"
    elif slope > slope_threshold:
        return "increasing"
    elif slope < -slope_threshold:
        return "decreasing"
    else:
        return "stable"


def analyze_game_temporal_dynamics(
    game_id: str,
    events: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    players: List[str]
) -> TemporalMetrics:
    """
    Perform comprehensive temporal analysis on a single game.

    Args:
        game_id: Game identifier
        events: List of game events
        decisions: List of player decisions
        players: List of player IDs

    Returns:
        TemporalMetrics with complete analysis
    """
    # Segment into phases
    phases = segment_game_into_phases(events)

    # Calculate trajectories
    trust_trajectories = calculate_trust_trajectory(decisions, players)
    avg_trust = [
        np.mean([t[i] for t in trust_trajectories.values() if i < len(t)])
        for i in range(max(len(t) for t in trust_trajectories.values()) or 1)
    ]

    deception_trajectory = calculate_deception_trajectory(decisions)

    # Detect turning points
    turning_points = detect_turning_points(
        avg_trust,
        deception_trajectory,
        events
    )

    # Extract policy progression
    liberal_policies = []
    fascist_policies = []
    lib_count, fas_count = 0, 0

    for event in events:
        if event.get('type') == 'policy_enacted':
            if event.get('policy') == 'liberal':
                lib_count += 1
            else:
                fas_count += 1
        liberal_policies.append(lib_count)
        fascist_policies.append(fas_count)

    momentum_shifts = detect_momentum_shifts(liberal_policies, fascist_policies)

    # Calculate phase deception rates
    n_decisions = len(decisions)
    early_cutoff = n_decisions // 3
    late_cutoff = 2 * n_decisions // 3

    early_decisions = decisions[:early_cutoff]
    mid_decisions = decisions[early_cutoff:late_cutoff]
    late_decisions = decisions[late_cutoff:]

    def phase_deception_rate(phase_decisions):
        if not phase_decisions:
            return 0.0
        deceptions = sum(1 for d in phase_decisions if d.get('is_deception', False))
        return deceptions / len(phase_decisions)

    # Classify trend
    trend = classify_deception_trend(deception_trajectory)

    return TemporalMetrics(
        game_id=game_id,
        total_turns=len(set(e.get('turn', 0) for e in events)),
        phases=phases,
        turning_points=turning_points,
        trust_trajectory=avg_trust,
        deception_trajectory=deception_trajectory,
        momentum_shifts=momentum_shifts,
        early_game_deception=phase_deception_rate(early_decisions),
        mid_game_deception=phase_deception_rate(mid_decisions),
        late_game_deception=phase_deception_rate(late_decisions),
        deception_trend=trend
    )


def compare_winning_trajectories(
    liberal_wins: List[TemporalMetrics],
    fascist_wins: List[TemporalMetrics]
) -> Dict[str, Any]:
    """
    Compare temporal patterns between liberal and fascist victories.

    Args:
        liberal_wins: Temporal metrics for games won by liberals
        fascist_wins: Temporal metrics for games won by fascists

    Returns:
        Dict with comparison statistics
    """
    comparison = {
        'liberal_avg_turns': np.mean([m.total_turns for m in liberal_wins]) if liberal_wins else 0,
        'fascist_avg_turns': np.mean([m.total_turns for m in fascist_wins]) if fascist_wins else 0,
        'liberal_avg_turning_points': np.mean([len(m.turning_points) for m in liberal_wins]) if liberal_wins else 0,
        'fascist_avg_turning_points': np.mean([len(m.turning_points) for m in fascist_wins]) if fascist_wins else 0,
        'liberal_early_deception': np.mean([m.early_game_deception for m in liberal_wins]) if liberal_wins else 0,
        'fascist_early_deception': np.mean([m.early_game_deception for m in fascist_wins]) if fascist_wins else 0,
        'liberal_momentum_shifts': np.mean([len(m.momentum_shifts) for m in liberal_wins]) if liberal_wins else 0,
        'fascist_momentum_shifts': np.mean([len(m.momentum_shifts) for m in fascist_wins]) if fascist_wins else 0,
    }

    # Statistical tests
    if liberal_wins and fascist_wins:
        # Compare game lengths
        lib_turns = [m.total_turns for m in liberal_wins]
        fas_turns = [m.total_turns for m in fascist_wins]

        if len(lib_turns) >= 2 and len(fas_turns) >= 2:
            stat, p_value = stats.mannwhitneyu(lib_turns, fas_turns, alternative='two-sided')
            comparison['game_length_test'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        # Compare early deception
        lib_early = [m.early_game_deception for m in liberal_wins]
        fas_early = [m.early_game_deception for m in fascist_wins]

        if len(lib_early) >= 2 and len(fas_early) >= 2:
            stat, p_value = stats.mannwhitneyu(lib_early, fas_early, alternative='two-sided')
            comparison['early_deception_test'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

    return comparison


def generate_temporal_report(metrics: TemporalMetrics) -> str:
    """
    Generate a narrative report of temporal analysis.

    Args:
        metrics: TemporalMetrics from analyze_game_temporal_dynamics

    Returns:
        Formatted markdown report
    """
    report = [f"# Temporal Analysis Report: Game {metrics.game_id}\n"]

    report.append(f"## Overview")
    report.append(f"- Total turns: {metrics.total_turns}")
    report.append(f"- Turning points detected: {len(metrics.turning_points)}")
    report.append(f"- Momentum shifts: {len(metrics.momentum_shifts)}")
    report.append(f"- Deception trend: {metrics.deception_trend}\n")

    report.append("## Deception by Game Phase")
    report.append(f"- Early game: {metrics.early_game_deception:.1%}")
    report.append(f"- Mid game: {metrics.mid_game_deception:.1%}")
    report.append(f"- Late game: {metrics.late_game_deception:.1%}\n")

    if metrics.phases:
        report.append("## Game Phases")
        for phase in metrics.phases:
            report.append(f"\n### {phase.phase_name.title()} Game (Turns {phase.start_turn}-{phase.end_turn})")
            report.append(f"- Average deception rate: {phase.avg_deception_rate:.1%}")
            if phase.key_events:
                report.append("- Key events:")
                for event in phase.key_events[:3]:
                    report.append(f"  - {event}")

    if metrics.turning_points:
        report.append("\n## Critical Turning Points")
        for tp in metrics.turning_points[:5]:  # Top 5
            report.append(f"\n**Turn {tp.turn_number}: {tp.event_type}**")
            report.append(f"- {tp.description}")
            report.append(f"- Trust impact: {tp.trust_impact:+.2f}")
            report.append(f"- Deception change: {tp.deception_change:+.2f}")

    return "\n".join(report)


def aggregate_temporal_patterns(
    all_metrics: List[TemporalMetrics]
) -> Dict[str, Any]:
    """
    Aggregate temporal patterns across multiple games.

    Args:
        all_metrics: List of TemporalMetrics from multiple games

    Returns:
        Aggregated statistics and patterns
    """
    if not all_metrics:
        return {}

    return {
        'total_games': len(all_metrics),
        'avg_turns': np.mean([m.total_turns for m in all_metrics]),
        'std_turns': np.std([m.total_turns for m in all_metrics]),
        'avg_turning_points': np.mean([len(m.turning_points) for m in all_metrics]),
        'avg_momentum_shifts': np.mean([len(m.momentum_shifts) for m in all_metrics]),
        'trend_distribution': {
            trend: sum(1 for m in all_metrics if m.deception_trend == trend)
            for trend in ['increasing', 'decreasing', 'stable', 'volatile']
        },
        'early_deception_mean': np.mean([m.early_game_deception for m in all_metrics]),
        'mid_deception_mean': np.mean([m.mid_game_deception for m in all_metrics]),
        'late_deception_mean': np.mean([m.late_game_deception for m in all_metrics]),
        'deception_escalation': np.mean([
            m.late_game_deception - m.early_game_deception
            for m in all_metrics
        ]),
    }
