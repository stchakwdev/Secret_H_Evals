"""Analytics module for game analysis, deception detection, and statistical analysis."""

from .statistical_analysis import (
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
    StatisticalResult,
)

from .coalition_detector import (
    CoalitionDetector,
    CoalitionResult,
    get_alignment_network_for_visualization,
)