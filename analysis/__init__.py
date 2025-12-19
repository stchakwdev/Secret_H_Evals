"""
Analysis Module for Secret Hitler LLM Evaluation Framework.

This module provides comprehensive analysis tools for evaluating
LLM behavior in strategic deception scenarios.

Submodules:
    core        - Statistical utilities and streaming algorithms
    deception   - Deception detection and belief calibration
    social      - Coalition detection and temporal analysis
    models      - Model comparison and hypothesis testing
    visualization - Publication-quality figure generation

Example:
    >>> from analysis import DeceptionDetector, calculate_proportion_ci
    >>> detector = DeceptionDetector()
    >>> is_deceptive, score, summary = detector.detect_deception(reasoning, statement)
    >>> prop, lower, upper = calculate_proportion_ci(successes=58, total=100)

Author: Samuel T. Chakwera (stchakdev)
"""

# Core statistical utilities
from .core.statistical import (
    StatisticalResult,
    calculate_confidence_interval,
    calculate_proportion_ci,
    chi_square_test_proportions,
    mann_whitney_test,
    t_test_independent,
    cohens_d,
    significance_stars,
    multiple_comparison_correction,
    bootstrap_ci,
    summarize_game_statistics,
    format_ci_for_display,
)

# Streaming statistics
from .core.streaming import (
    WelfordAccumulator,
    CountAccumulator,
    HistogramAccumulator,
    StreamingGameStats,
    StreamingAnalyzer,
    stream_analyze,
)

# Deception detection
from .deception.detector import (
    DeceptionDetector,
    get_detector,
)

# Belief calibration
from .deception.calibration import (
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

# Coalition detection
from .social.coalitions import (
    CoalitionDetector,
    CoalitionResult,
    get_alignment_network_for_visualization,
)

# Temporal analysis
from .social.temporal import (
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

# Model comparison
from .models.comparison import (
    WinRateCI,
    ComparisonResult,
    ModelStats,
    chi_square_win_rates,
    cohens_h,
    cramers_v,
    odds_ratio,
    compare_win_rates,
    bonferroni_correction,
    holm_bonferroni_correction,
    calculate_elo_ratings,
    generate_comparison_matrix,
    generate_latex_comparison_table,
    generate_pairwise_significance_table,
    generate_markdown_report,
    analyze_comparison_results,
)

# Hypothesis testing
from .models.hypothesis import (
    TestType,
    HypothesisTestResult,
    interpret_effect_size,
    test_model_win_rates,
    test_deception_by_role,
    test_game_length_deception_correlation,
    test_deception_by_decision_type,
    test_early_deception_predicts_outcome,
    run_hypothesis_battery,
    generate_hypothesis_report,
)

# Visualization utilities
from .visualization.utils import (
    PUBLICATION_STYLE,
    COLORBLIND_PALETTE,
    FIGURE_SIZES,
    apply_publication_style,
    reset_style,
    get_figure,
    save_figure,
    add_significance_annotation,
    add_ci_annotation,
    create_legend_outside,
    format_percentage_axis,
    add_watermark,
    get_color,
)


__all__ = [
    # Core - Statistical
    'StatisticalResult',
    'calculate_confidence_interval',
    'calculate_proportion_ci',
    'chi_square_test_proportions',
    'mann_whitney_test',
    't_test_independent',
    'cohens_d',
    'significance_stars',
    'multiple_comparison_correction',
    'bootstrap_ci',
    'summarize_game_statistics',
    'format_ci_for_display',
    # Core - Streaming
    'WelfordAccumulator',
    'CountAccumulator',
    'HistogramAccumulator',
    'StreamingGameStats',
    'StreamingAnalyzer',
    'stream_analyze',
    # Deception
    'DeceptionDetector',
    'get_detector',
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
    # Social
    'CoalitionDetector',
    'CoalitionResult',
    'get_alignment_network_for_visualization',
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
    # Models - Comparison
    'WinRateCI',
    'ComparisonResult',
    'ModelStats',
    'chi_square_win_rates',
    'cohens_h',
    'cramers_v',
    'odds_ratio',
    'compare_win_rates',
    'bonferroni_correction',
    'holm_bonferroni_correction',
    'calculate_elo_ratings',
    'generate_comparison_matrix',
    'generate_latex_comparison_table',
    'generate_pairwise_significance_table',
    'generate_markdown_report',
    'analyze_comparison_results',
    # Models - Hypothesis
    'TestType',
    'HypothesisTestResult',
    'interpret_effect_size',
    'test_model_win_rates',
    'test_deception_by_role',
    'test_game_length_deception_correlation',
    'test_deception_by_decision_type',
    'test_early_deception_predicts_outcome',
    'run_hypothesis_battery',
    'generate_hypothesis_report',
    # Visualization
    'PUBLICATION_STYLE',
    'COLORBLIND_PALETTE',
    'FIGURE_SIZES',
    'apply_publication_style',
    'reset_style',
    'get_figure',
    'save_figure',
    'add_significance_annotation',
    'add_ci_annotation',
    'create_legend_outside',
    'format_percentage_axis',
    'add_watermark',
    'get_color',
]

__version__ = '2.0.0'
__author__ = 'Samuel T. Chakwera'
