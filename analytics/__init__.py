"""
DEPRECATED: This module has been moved to analysis/

This shim provides backward compatibility. Update your imports to:
    from analysis import ...

All functionality has been preserved in the new analysis/ module structure.
"""

import warnings

warnings.warn(
    "The 'analytics' module is deprecated and will be removed in a future version. "
    "Please update your imports to use 'from analysis import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new analysis module for backward compatibility
from analysis import (
    # Core - Statistical
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
    # Deception
    DeceptionDetector,
    get_detector,
    # Social
    CoalitionDetector,
    CoalitionResult,
    get_alignment_network_for_visualization,
)

__all__ = [
    # Statistical
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
    # Deception
    'DeceptionDetector',
    'get_detector',
    # Coalition
    'CoalitionDetector',
    'CoalitionResult',
    'get_alignment_network_for_visualization',
]
