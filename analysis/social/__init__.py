"""
Social dynamics analysis: coalitions and temporal patterns.

This module provides tools for detecting coalition formation,
analyzing voting patterns, and tracking game dynamics over time.
"""

from .coalitions import (
    CoalitionDetector,
    CoalitionResult,
    get_alignment_network_for_visualization,
)

from .temporal import (
    GamePhase,
    TurningPoint,
    TemporalMetrics,
    segment_game_into_phases,
    detect_turning_points,
    calculate_trust_trajectory,
    calculate_deception_trajectory,
    detect_momentum_shifts,
    classify_deception_trend,
    analyze_game_temporal_dynamics,
    compare_winning_trajectories,
    generate_temporal_report,
    aggregate_temporal_patterns,
)

__all__ = [
    # Coalitions
    'CoalitionDetector',
    'CoalitionResult',
    'get_alignment_network_for_visualization',
    # Temporal
    'GamePhase',
    'TurningPoint',
    'TemporalMetrics',
    'segment_game_into_phases',
    'detect_turning_points',
    'calculate_trust_trajectory',
    'calculate_deception_trajectory',
    'detect_momentum_shifts',
    'classify_deception_trend',
    'analyze_game_temporal_dynamics',
    'compare_winning_trajectories',
    'generate_temporal_report',
    'aggregate_temporal_patterns',
]
