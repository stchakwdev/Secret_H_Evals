"""
Model comparison and hypothesis testing.

This module provides tools for comparing LLM model performance
and conducting rigorous statistical hypothesis testing.
"""

from .comparison import (
    WinRateCI,
    ComparisonResult,
    ModelStats,
    significance_stars,
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

from .hypothesis import (
    TestType,
    HypothesisTestResult,
    cohens_d,
    interpret_effect_size,
    test_model_win_rates,
    test_deception_by_role,
    test_game_length_deception_correlation,
    test_deception_by_decision_type,
    test_early_deception_predicts_outcome,
    run_hypothesis_battery,
    generate_hypothesis_report,
)

__all__ = [
    # Comparison
    'WinRateCI',
    'ComparisonResult',
    'ModelStats',
    'significance_stars',
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
    # Hypothesis
    'TestType',
    'HypothesisTestResult',
    'cohens_d',
    'interpret_effect_size',
    'test_model_win_rates',
    'test_deception_by_role',
    'test_game_length_deception_correlation',
    'test_deception_by_decision_type',
    'test_early_deception_predicts_outcome',
    'run_hypothesis_battery',
    'generate_hypothesis_report',
]
