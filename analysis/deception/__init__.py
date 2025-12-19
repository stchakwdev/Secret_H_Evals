"""
Deception detection and belief calibration analysis.

This module provides tools for detecting when AI statements
contradict private reasoning, and for evaluating belief calibration.
"""

from .detector import (
    DeceptionDetector,
    get_detector,
)

from .calibration import (
    BeliefSnapshot,
    CalibrationMetrics,
    TrustAccuracy,
    extract_beliefs_from_response,
    calculate_brier_score,
    calculate_log_loss,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
    calculate_overconfidence_rate,
    analyze_player_calibration,
    analyze_trust_accuracy,
    compare_model_calibration,
    calculate_kl_divergence_from_uniform,
    generate_calibration_report,
    aggregate_calibration_statistics,
)

__all__ = [
    # Detection
    'DeceptionDetector',
    'get_detector',
    # Calibration
    'BeliefSnapshot',
    'CalibrationMetrics',
    'TrustAccuracy',
    'extract_beliefs_from_response',
    'calculate_brier_score',
    'calculate_log_loss',
    'calculate_expected_calibration_error',
    'calculate_maximum_calibration_error',
    'calculate_overconfidence_rate',
    'analyze_player_calibration',
    'analyze_trust_accuracy',
    'compare_model_calibration',
    'calculate_kl_divergence_from_uniform',
    'generate_calibration_report',
    'aggregate_calibration_statistics',
]
