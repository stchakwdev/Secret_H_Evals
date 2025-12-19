"""
Core statistical utilities and streaming algorithms.

This module provides foundational statistical tools for analyzing
Secret Hitler LLM evaluation experiments.
"""

from .statistical import (
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

from .streaming import (
    WelfordAccumulator,
    CountAccumulator,
    HistogramAccumulator,
    StreamingGameStats,
    StreamingAnalyzer,
    stream_analyze,
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
    # Streaming
    'WelfordAccumulator',
    'CountAccumulator',
    'HistogramAccumulator',
    'StreamingGameStats',
    'StreamingAnalyzer',
    'stream_analyze',
]
