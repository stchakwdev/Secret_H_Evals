"""
Hypothesis Testing Module for Secret Hitler LLM Research.

Provides statistical tests for research questions about LLM behavior in
strategic deception games. Designed for publication-quality analysis.

Key research questions addressed:
1. Do different models have significantly different win rates?
2. Is deception rate correlated with role assignment?
3. Do fascists lie more than liberals?
4. Does game length correlate with deception frequency?

Author: Samuel Chakwera (stchakdev)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings

from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, pearsonr
from scipy.stats import fisher_exact, kruskal, ttest_ind

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.statistical_analysis import (
    calculate_proportion_ci,
    multiple_comparison_correction,
    significance_stars,
)


class TestType(Enum):
    """Types of statistical tests available."""
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MANN_WHITNEY = "mann_whitney"
    KRUSKAL_WALLIS = "kruskal_wallis"
    T_TEST = "t_test"
    SPEARMAN = "spearman"
    PEARSON = "pearson"


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test results."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: Optional[float]
    effect_size_name: Optional[str]
    confidence_interval: Optional[Tuple[float, float]]
    sample_sizes: Dict[str, int]
    conclusion: str
    significance: str  # *, **, ***, or n.s.
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'confidence_interval': self.confidence_interval,
            'sample_sizes': self.sample_sizes,
            'conclusion': self.conclusion,
            'significance': self.significance,
            'interpretation': self.interpretation,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        ci_str = ""
        if self.confidence_interval:
            ci_str = f", 95% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]"

        effect_str = ""
        if self.effect_size is not None:
            effect_str = f", {self.effect_size_name}={self.effect_size:.3f}"

        return (
            f"{self.test_name}\n"
            f"  Test: {self.test_type.value}\n"
            f"  Statistic: {self.statistic:.4f}, p={self.p_value:.4f} {self.significance}\n"
            f"  {effect_str}{ci_str}\n"
            f"  Samples: {self.sample_sizes}\n"
            f"  Conclusion: {self.conclusion}\n"
            f"  Interpretation: {self.interpretation}"
        )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Cohen's d = (M1 - M2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: First group of observations
        group2: Second group of observations

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cramers_v(contingency_table: np.ndarray) -> float:
    """
    Calculate Cramér's V effect size for chi-square test.

    Interpretation:
    - V < 0.1: negligible
    - 0.1 <= V < 0.3: small
    - 0.3 <= V < 0.5: medium
    - V >= 0.5: large

    Args:
        contingency_table: 2D array of observed frequencies

    Returns:
        Cramér's V effect size
    """
    chi2 = chi2_contingency(contingency_table)[0]
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape) - 1

    if min_dim == 0 or n == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def odds_ratio(contingency_table: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate odds ratio and 95% CI for 2x2 contingency table.

    Args:
        contingency_table: 2x2 array [[a, b], [c, d]]

    Returns:
        Tuple of (odds_ratio, (ci_lower, ci_upper))
    """
    a, b = contingency_table[0]
    c, d = contingency_table[1]

    # Handle zeros with continuity correction
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)

    # Log odds ratio standard error
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)

    # 95% CI
    log_or = np.log(or_val)
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    return or_val, (ci_lower, ci_upper)


def interpret_effect_size(effect_size: float, effect_type: str) -> str:
    """
    Interpret effect size magnitude.

    Args:
        effect_size: The effect size value
        effect_type: Type of effect size (cohens_d, cramers_v, r, etc.)

    Returns:
        Interpretation string
    """
    abs_effect = abs(effect_size)

    if effect_type in ['cohens_d', 'd']:
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    elif effect_type in ['cramers_v', 'v']:
        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "small"
        elif abs_effect < 0.5:
            return "medium"
        else:
            return "large"

    elif effect_type in ['r', 'rho', 'correlation']:
        if abs_effect < 0.1:
            return "negligible"
        elif abs_effect < 0.3:
            return "weak"
        elif abs_effect < 0.5:
            return "moderate"
        elif abs_effect < 0.7:
            return "strong"
        else:
            return "very strong"

    elif effect_type == 'odds_ratio':
        if abs_effect < 1.5:
            return "negligible"
        elif abs_effect < 2.5:
            return "small"
        elif abs_effect < 4.0:
            return "medium"
        else:
            return "large"

    return "unknown"


# =============================================================================
# Research Question 1: Model Win Rate Differences
# =============================================================================

def test_model_win_rates(
    model_data: Dict[str, Dict[str, int]],
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test if different models have significantly different win rates.

    Uses chi-square test for independence or Fisher's exact test
    for small samples.

    Args:
        model_data: Dict mapping model names to {'games': n, 'wins': w}
        alpha: Significance level

    Returns:
        HypothesisTestResult with test details
    """
    models = list(model_data.keys())

    if len(models) < 2:
        return HypothesisTestResult(
            test_name="Model Win Rate Comparison",
            test_type=TestType.CHI_SQUARE,
            statistic=0.0,
            p_value=1.0,
            effect_size=None,
            effect_size_name=None,
            confidence_interval=None,
            sample_sizes={'total_models': len(models)},
            conclusion="Insufficient models for comparison",
            significance="n.s.",
            interpretation="Need at least 2 models to compare"
        )

    # Build contingency table: rows = models, cols = [wins, losses]
    contingency = []
    sample_sizes = {}

    for model in models:
        wins = model_data[model]['wins']
        games = model_data[model]['games']
        losses = games - wins
        contingency.append([wins, losses])
        sample_sizes[model] = games

    contingency = np.array(contingency)

    # Check for small samples
    expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / contingency.sum()
    use_fisher = np.any(expected < 5)

    if len(models) == 2 and use_fisher:
        # Use Fisher's exact test for 2x2 with small samples
        stat, p_value = fisher_exact(contingency)
        test_type = TestType.FISHER_EXACT
        effect_size, ci = odds_ratio(contingency)
        effect_name = "odds_ratio"
    else:
        # Use chi-square test
        stat, p_value, dof, expected = chi2_contingency(contingency)
        test_type = TestType.CHI_SQUARE
        effect_size = cramers_v(contingency)
        effect_name = "Cramér's V"
        ci = None

    sig = significance_stars(p_value)
    is_significant = p_value < alpha

    effect_interp = interpret_effect_size(
        effect_size,
        'odds_ratio' if test_type == TestType.FISHER_EXACT else 'cramers_v'
    )

    if is_significant:
        conclusion = f"Models show significantly different win rates (p={p_value:.4f})"
        interpretation = (
            f"There is a statistically significant difference in win rates "
            f"across models with a {effect_interp} effect size ({effect_name}={effect_size:.3f}). "
            f"This suggests model choice impacts game outcomes."
        )
    else:
        conclusion = f"No significant difference in win rates (p={p_value:.4f})"
        interpretation = (
            f"The observed differences in win rates are not statistically significant. "
            f"Model choice may not substantially impact win probability in this sample."
        )

    return HypothesisTestResult(
        test_name="Model Win Rate Comparison",
        test_type=test_type,
        statistic=stat,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_name=effect_name,
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        conclusion=conclusion,
        significance=sig,
        interpretation=interpretation
    )


# =============================================================================
# Research Question 2: Deception Rate by Role
# =============================================================================

def test_deception_by_role(
    fascist_deceptions: int,
    fascist_total: int,
    liberal_deceptions: int,
    liberal_total: int,
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test if fascists deceive more than liberals.

    H0: Fascist deception rate = Liberal deception rate
    H1: Fascist deception rate > Liberal deception rate (one-tailed)

    Args:
        fascist_deceptions: Number of deceptive actions by fascists
        fascist_total: Total actions by fascists
        liberal_deceptions: Number of deceptive actions by liberals
        liberal_total: Total actions by liberals
        alpha: Significance level

    Returns:
        HypothesisTestResult with test details
    """
    # Build 2x2 contingency table
    contingency = np.array([
        [fascist_deceptions, fascist_total - fascist_deceptions],
        [liberal_deceptions, liberal_total - liberal_deceptions]
    ])

    # Calculate rates
    fasc_rate = fascist_deceptions / fascist_total if fascist_total > 0 else 0
    lib_rate = liberal_deceptions / liberal_total if liberal_total > 0 else 0

    # Check for small samples
    expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / contingency.sum()
    use_fisher = np.any(expected < 5)

    if use_fisher:
        # One-tailed Fisher's exact test
        stat, p_two = fisher_exact(contingency)
        # Convert to one-tailed (fascists > liberals)
        if fasc_rate > lib_rate:
            p_value = p_two / 2
        else:
            p_value = 1 - p_two / 2
        test_type = TestType.FISHER_EXACT
    else:
        # Chi-square test (two-tailed)
        stat, p_two, dof, exp = chi2_contingency(contingency)
        # Convert to one-tailed
        if fasc_rate > lib_rate:
            p_value = p_two / 2
        else:
            p_value = 1 - p_two / 2
        test_type = TestType.CHI_SQUARE

    # Calculate odds ratio and CI
    or_val, ci = odds_ratio(contingency)

    sig = significance_stars(p_value)
    is_significant = p_value < alpha

    effect_interp = interpret_effect_size(or_val, 'odds_ratio')

    sample_sizes = {
        'fascist_actions': fascist_total,
        'liberal_actions': liberal_total,
        'fascist_deceptions': fascist_deceptions,
        'liberal_deceptions': liberal_deceptions
    }

    rate_diff = (fasc_rate - lib_rate) * 100

    if is_significant and fasc_rate > lib_rate:
        conclusion = (
            f"Fascists deceive significantly more than liberals "
            f"({fasc_rate*100:.1f}% vs {lib_rate*100:.1f}%, p={p_value:.4f})"
        )
        interpretation = (
            f"Fascist players are {or_val:.2f}x more likely to make deceptive statements "
            f"than liberal players. This {effect_interp} effect supports the hypothesis "
            f"that LLMs can adopt role-appropriate deceptive behavior."
        )
    elif is_significant and lib_rate > fasc_rate:
        conclusion = (
            f"Liberals deceive more than fascists (unexpected) "
            f"({lib_rate*100:.1f}% vs {fasc_rate*100:.1f}%, p={p_value:.4f})"
        )
        interpretation = (
            f"Surprisingly, liberal players show higher deception rates. "
            f"This may indicate defensive deception or model confusion about roles."
        )
    else:
        conclusion = (
            f"No significant difference in deception rates "
            f"({fasc_rate*100:.1f}% vs {lib_rate*100:.1f}%, p={p_value:.4f})"
        )
        interpretation = (
            f"Fascists and liberals show similar deception rates. "
            f"LLMs may not be effectively adapting deception to their role, "
            f"or both teams employ similar strategic deception."
        )

    return HypothesisTestResult(
        test_name="Deception Rate by Role (Fascist vs Liberal)",
        test_type=test_type,
        statistic=stat,
        p_value=p_value,
        effect_size=or_val,
        effect_size_name="odds_ratio",
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        conclusion=conclusion,
        significance=sig,
        interpretation=interpretation
    )


# =============================================================================
# Research Question 3: Game Length and Deception Correlation
# =============================================================================

def test_game_length_deception_correlation(
    game_lengths: List[int],
    deception_counts: List[int],
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test if longer games have more deception.

    Uses Spearman correlation (robust to non-normality).

    Args:
        game_lengths: List of game durations (turns or seconds)
        deception_counts: List of deception counts per game
        alpha: Significance level

    Returns:
        HypothesisTestResult with correlation details
    """
    game_lengths = np.array(game_lengths)
    deception_counts = np.array(deception_counts)

    if len(game_lengths) < 3:
        return HypothesisTestResult(
            test_name="Game Length vs Deception Correlation",
            test_type=TestType.SPEARMAN,
            statistic=0.0,
            p_value=1.0,
            effect_size=None,
            effect_size_name=None,
            confidence_interval=None,
            sample_sizes={'games': len(game_lengths)},
            conclusion="Insufficient data for correlation",
            significance="n.s.",
            interpretation="Need at least 3 games for correlation analysis"
        )

    # Spearman correlation
    rho, p_value = spearmanr(game_lengths, deception_counts)

    # Fisher z-transformation for CI
    n = len(game_lengths)
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    ci_lower = np.tanh(z - 1.96 * se)
    ci_upper = np.tanh(z + 1.96 * se)

    sig = significance_stars(p_value)
    is_significant = p_value < alpha

    effect_interp = interpret_effect_size(rho, 'correlation')

    direction = "positive" if rho > 0 else "negative"

    if is_significant:
        conclusion = (
            f"Significant {direction} correlation between game length and deception "
            f"(ρ={rho:.3f}, p={p_value:.4f})"
        )
        if rho > 0:
            interpretation = (
                f"Longer games show more deception ({effect_interp} correlation). "
                f"This suggests deception may accumulate as games progress, "
                f"or that games with more deception take longer to resolve."
            )
        else:
            interpretation = (
                f"Shorter games show more deception ({effect_interp} correlation). "
                f"High deception early may lead to faster game resolution."
            )
    else:
        conclusion = (
            f"No significant correlation between game length and deception "
            f"(ρ={rho:.3f}, p={p_value:.4f})"
        )
        interpretation = (
            f"Game duration does not predict deception frequency in this sample."
        )

    return HypothesisTestResult(
        test_name="Game Length vs Deception Correlation",
        test_type=TestType.SPEARMAN,
        statistic=rho,
        p_value=p_value,
        effect_size=rho,
        effect_size_name="Spearman's ρ",
        confidence_interval=(ci_lower, ci_upper),
        sample_sizes={'games': len(game_lengths)},
        conclusion=conclusion,
        significance=sig,
        interpretation=interpretation
    )


# =============================================================================
# Research Question 4: Deception Score Distribution by Decision Type
# =============================================================================

def test_deception_by_decision_type(
    decision_type_scores: Dict[str, List[float]],
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test if deception scores differ across decision types.

    Uses Kruskal-Wallis test (non-parametric ANOVA).

    Args:
        decision_type_scores: Dict mapping decision types to deception scores
        alpha: Significance level

    Returns:
        HypothesisTestResult with test details
    """
    decision_types = list(decision_type_scores.keys())

    if len(decision_types) < 2:
        return HypothesisTestResult(
            test_name="Deception Score by Decision Type",
            test_type=TestType.KRUSKAL_WALLIS,
            statistic=0.0,
            p_value=1.0,
            effect_size=None,
            effect_size_name=None,
            confidence_interval=None,
            sample_sizes={'decision_types': len(decision_types)},
            conclusion="Insufficient decision types for comparison",
            significance="n.s.",
            interpretation="Need at least 2 decision types to compare"
        )

    # Filter out empty groups
    groups = [np.array(decision_type_scores[dt]) for dt in decision_types
              if len(decision_type_scores[dt]) > 0]
    valid_types = [dt for dt in decision_types if len(decision_type_scores[dt]) > 0]

    if len(groups) < 2:
        return HypothesisTestResult(
            test_name="Deception Score by Decision Type",
            test_type=TestType.KRUSKAL_WALLIS,
            statistic=0.0,
            p_value=1.0,
            effect_size=None,
            effect_size_name=None,
            confidence_interval=None,
            sample_sizes={'decision_types': len(valid_types)},
            conclusion="Insufficient non-empty decision types",
            significance="n.s.",
            interpretation="Need at least 2 decision types with data"
        )

    # Kruskal-Wallis test
    stat, p_value = kruskal(*groups)

    # Calculate eta-squared (effect size for Kruskal-Wallis)
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (stat - k + 1) / (n_total - k)
    eta_sq = max(0, eta_sq)  # Ensure non-negative

    sig = significance_stars(p_value)
    is_significant = p_value < alpha

    sample_sizes = {dt: len(decision_type_scores[dt]) for dt in valid_types}
    sample_sizes['total'] = n_total

    # Find highest/lowest mean deception
    means = {dt: np.mean(decision_type_scores[dt]) for dt in valid_types
             if len(decision_type_scores[dt]) > 0}
    highest_type = max(means, key=means.get)
    lowest_type = min(means, key=means.get)

    if is_significant:
        conclusion = (
            f"Deception scores differ significantly across decision types "
            f"(H={stat:.2f}, p={p_value:.4f})"
        )
        interpretation = (
            f"Different decision contexts elicit different levels of deception. "
            f"'{highest_type}' shows highest deception (M={means[highest_type]:.2f}), "
            f"while '{lowest_type}' shows lowest (M={means[lowest_type]:.2f}). "
            f"η²={eta_sq:.3f} indicates {'medium to large' if eta_sq > 0.06 else 'small'} effect."
        )
    else:
        conclusion = (
            f"No significant difference in deception across decision types "
            f"(H={stat:.2f}, p={p_value:.4f})"
        )
        interpretation = (
            f"Deception levels are relatively consistent across different decision types. "
            f"LLMs may apply similar deception strategies regardless of context."
        )

    return HypothesisTestResult(
        test_name="Deception Score by Decision Type",
        test_type=TestType.KRUSKAL_WALLIS,
        statistic=stat,
        p_value=p_value,
        effect_size=eta_sq,
        effect_size_name="η² (eta-squared)",
        confidence_interval=None,
        sample_sizes=sample_sizes,
        conclusion=conclusion,
        significance=sig,
        interpretation=interpretation
    )


# =============================================================================
# Research Question 5: Win Prediction from Early Game Behavior
# =============================================================================

def test_early_deception_predicts_outcome(
    games_with_early_deception: int,
    games_won_with_early_deception: int,
    games_without_early_deception: int,
    games_won_without_early_deception: int,
    team: str = "fascist",
    alpha: float = 0.05
) -> HypothesisTestResult:
    """
    Test if early game deception predicts win outcome.

    H0: P(win | early deception) = P(win | no early deception)
    H1: P(win | early deception) ≠ P(win | no early deception)

    Args:
        games_with_early_deception: Number of games with early deception
        games_won_with_early_deception: Wins among games with early deception
        games_without_early_deception: Games without early deception
        games_won_without_early_deception: Wins among games without early deception
        team: Team being analyzed ("fascist" or "liberal")
        alpha: Significance level

    Returns:
        HypothesisTestResult with prediction analysis
    """
    # Build 2x2 contingency table
    # Rows: early deception (yes/no), Cols: win (yes/no)
    contingency = np.array([
        [games_won_with_early_deception,
         games_with_early_deception - games_won_with_early_deception],
        [games_won_without_early_deception,
         games_without_early_deception - games_won_without_early_deception]
    ])

    total = contingency.sum()

    if total == 0:
        return HypothesisTestResult(
            test_name=f"Early Deception Predicts {team.title()} Win",
            test_type=TestType.FISHER_EXACT,
            statistic=0.0,
            p_value=1.0,
            effect_size=None,
            effect_size_name=None,
            confidence_interval=None,
            sample_sizes={'total': 0},
            conclusion="No data available",
            significance="n.s.",
            interpretation="Insufficient data for analysis"
        )

    # Calculate win rates
    rate_with = (games_won_with_early_deception / games_with_early_deception
                 if games_with_early_deception > 0 else 0)
    rate_without = (games_won_without_early_deception / games_without_early_deception
                    if games_without_early_deception > 0 else 0)

    # Fisher's exact test (handles small samples)
    stat, p_value = fisher_exact(contingency)

    # Odds ratio and CI
    or_val, ci = odds_ratio(contingency)

    sig = significance_stars(p_value)
    is_significant = p_value < alpha

    sample_sizes = {
        'with_early_deception': games_with_early_deception,
        'without_early_deception': games_without_early_deception,
        'wins_with': games_won_with_early_deception,
        'wins_without': games_won_without_early_deception
    }

    effect_interp = interpret_effect_size(or_val, 'odds_ratio')

    if is_significant:
        if or_val > 1:
            conclusion = (
                f"Early deception predicts higher {team} win rate "
                f"({rate_with*100:.1f}% vs {rate_without*100:.1f}%, p={p_value:.4f})"
            )
            interpretation = (
                f"Games where {team}s employ early deception show "
                f"{or_val:.2f}x higher odds of {team} victory ({effect_interp} effect). "
                f"Early strategic deception appears to be an effective {team} strategy."
            )
        else:
            conclusion = (
                f"Early deception predicts lower {team} win rate "
                f"({rate_with*100:.1f}% vs {rate_without*100:.1f}%, p={p_value:.4f})"
            )
            interpretation = (
                f"Games where {team}s employ early deception show "
                f"lower odds of {team} victory. "
                f"Early deception may backfire or signal {team} identity to opponents."
            )
    else:
        conclusion = (
            f"Early deception does not predict {team} win rate "
            f"({rate_with*100:.1f}% vs {rate_without*100:.1f}%, p={p_value:.4f})"
        )
        interpretation = (
            f"Whether {team}s employ early deception does not significantly "
            f"predict game outcome in this sample."
        )

    return HypothesisTestResult(
        test_name=f"Early Deception Predicts {team.title()} Win",
        test_type=TestType.FISHER_EXACT,
        statistic=stat,
        p_value=p_value,
        effect_size=or_val,
        effect_size_name="odds_ratio",
        confidence_interval=ci,
        sample_sizes=sample_sizes,
        conclusion=conclusion,
        significance=sig,
        interpretation=interpretation
    )


# =============================================================================
# Batch Testing with Multiple Comparison Correction
# =============================================================================

def run_hypothesis_battery(
    games_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    correction_method: str = 'bonferroni',
    alpha: float = 0.05
) -> Dict[str, HypothesisTestResult]:
    """
    Run a battery of hypothesis tests with multiple comparison correction.

    Args:
        games_df: DataFrame with game-level data
        decisions_df: DataFrame with decision-level data
        correction_method: 'bonferroni', 'holm', or 'fdr_bh'
        alpha: Family-wise error rate

    Returns:
        Dict mapping test names to corrected results
    """
    results = {}

    # Test 1: Model win rates (if multiple models)
    if 'model' in games_df.columns:
        model_data = {}
        for model, group in games_df.groupby('model'):
            model_data[model] = {
                'games': len(group),
                'wins': group['winner'].sum() if 'winner' in group else 0
            }
        if len(model_data) >= 2:
            results['model_win_rates'] = test_model_win_rates(model_data, alpha)

    # Test 2: Deception by role
    if 'is_fascist' in decisions_df.columns and 'is_deception' in decisions_df.columns:
        fascist = decisions_df[decisions_df['is_fascist'] == True]
        liberal = decisions_df[decisions_df['is_fascist'] == False]

        results['deception_by_role'] = test_deception_by_role(
            fascist_deceptions=int(fascist['is_deception'].sum()),
            fascist_total=len(fascist),
            liberal_deceptions=int(liberal['is_deception'].sum()),
            liberal_total=len(liberal),
            alpha=alpha
        )

    # Test 3: Game length vs deception
    if 'duration_seconds' in games_df.columns:
        game_lengths = games_df['duration_seconds'].tolist()
        deception_counts = []
        for gid in games_df['game_id']:
            count = decisions_df[
                (decisions_df['game_id'] == gid) &
                (decisions_df['is_deception'] == True)
            ].shape[0]
            deception_counts.append(count)

        if len(game_lengths) >= 3:
            results['game_length_deception'] = test_game_length_deception_correlation(
                game_lengths, deception_counts, alpha
            )

    # Test 4: Deception by decision type
    if 'decision_type' in decisions_df.columns and 'deception_score' in decisions_df.columns:
        type_scores = {}
        for dtype, group in decisions_df.groupby('decision_type'):
            scores = group['deception_score'].dropna().tolist()
            if len(scores) > 0:
                type_scores[dtype] = scores

        if len(type_scores) >= 2:
            results['deception_by_type'] = test_deception_by_decision_type(
                type_scores, alpha
            )

    # Apply multiple comparison correction
    if len(results) > 1:
        p_values = [r.p_value for r in results.values()]
        corrected = multiple_comparison_correction(p_values, alpha, correction_method)

        # Update significance based on corrected values
        for i, (name, result) in enumerate(results.items()):
            corrected_p = corrected['corrected_p_values'][i]
            is_sig = corrected['significant'][i]

            # Create new result with corrected values
            results[name] = HypothesisTestResult(
                test_name=result.test_name + f" (corrected: {correction_method})",
                test_type=result.test_type,
                statistic=result.statistic,
                p_value=corrected_p,
                effect_size=result.effect_size,
                effect_size_name=result.effect_size_name,
                confidence_interval=result.confidence_interval,
                sample_sizes=result.sample_sizes,
                conclusion=result.conclusion,
                significance='*' if is_sig else 'n.s.',
                interpretation=result.interpretation
            )

    return results


def generate_hypothesis_report(results: Dict[str, HypothesisTestResult]) -> str:
    """
    Generate a formatted report from hypothesis test results.

    Args:
        results: Dict of test results from run_hypothesis_battery

    Returns:
        Formatted markdown report
    """
    report = ["# Hypothesis Testing Report\n"]
    report.append("## Summary\n")

    significant = sum(1 for r in results.values() if r.significance != 'n.s.')
    total = len(results)

    report.append(f"- Tests conducted: {total}")
    report.append(f"- Significant results: {significant}")
    report.append(f"- Non-significant results: {total - significant}\n")

    report.append("## Individual Test Results\n")

    for name, result in results.items():
        report.append(f"### {result.test_name}\n")
        report.append(f"**{result.conclusion}**\n")
        report.append(f"- Test type: {result.test_type.value}")
        report.append(f"- Test statistic: {result.statistic:.4f}")
        report.append(f"- p-value: {result.p_value:.4f} {result.significance}")

        if result.effect_size is not None:
            report.append(f"- Effect size ({result.effect_size_name}): {result.effect_size:.4f}")

        if result.confidence_interval:
            report.append(
                f"- 95% CI: [{result.confidence_interval[0]:.4f}, "
                f"{result.confidence_interval[1]:.4f}]"
            )

        report.append(f"- Sample sizes: {result.sample_sizes}")
        report.append(f"\n*Interpretation:* {result.interpretation}\n")

    return "\n".join(report)
