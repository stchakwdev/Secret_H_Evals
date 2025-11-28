"""
Statistical analysis utilities.

Provides confidence intervals, hypothesis testing, and effect size calculations
for statistical analysis of Secret Hitler LLM experiments.
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    significance: str
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None


def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for mean.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n < 2:
        mean = np.mean(data) if n > 0 else 0.0
        return (mean, mean, mean)

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean, mean - h, mean + h)


def calculate_proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
    method: str = 'wilson'
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for proportions using Wilson score.

    Wilson score interval is better than normal approximation for small samples
    and proportions near 0 or 1.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level
        method: 'wilson' (recommended), 'normal', or 'clopper-pearson'

    Returns:
        Tuple of (proportion, lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0, 0.0)

    p = successes / total

    if method == 'wilson':
        # Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return (p, max(0, center - spread), min(1, center + spread))

    elif method == 'normal':
        # Normal approximation (Wald interval)
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(p * (1 - p) / total)
        return (p, max(0, p - z * se), min(1, p + z * se))

    elif method == 'clopper-pearson':
        # Exact binomial (Clopper-Pearson)
        alpha = 1 - confidence
        if successes == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, successes, total - successes + 1)
        if successes == total:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
        return (p, lower, upper)

    else:
        raise ValueError(f"Unknown method: {method}")


def chi_square_test_proportions(
    group1_successes: int,
    group1_total: int,
    group2_successes: int,
    group2_total: int
) -> StatisticalResult:
    """
    Chi-square test for comparing two proportions (e.g., win rates).

    Args:
        group1_successes: Successes in group 1
        group1_total: Total trials in group 1
        group2_successes: Successes in group 2
        group2_total: Total trials in group 2

    Returns:
        StatisticalResult with chi2 statistic, p-value, and interpretation
    """
    contingency = [
        [group1_successes, group1_total - group1_successes],
        [group2_successes, group2_total - group2_successes]
    ]

    # Check for valid contingency table
    if any(cell < 0 for row in contingency for cell in row):
        raise ValueError("Negative values in contingency table")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Calculate effect size (Phi coefficient for 2x2 table)
    n = group1_total + group2_total
    phi = np.sqrt(chi2 / n) if n > 0 else 0.0

    return StatisticalResult(
        statistic=chi2,
        p_value=p_value,
        significance=significance_stars(p_value),
        effect_size=phi,
        interpretation=_interpret_significance(p_value)
    )


def mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> StatisticalResult:
    """
    Mann-Whitney U test for comparing two independent samples.

    Non-parametric test suitable for non-normal distributions.

    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        StatisticalResult with U statistic, p-value, and effect size
    """
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * stat) / (n1 * n2)

    return StatisticalResult(
        statistic=stat,
        p_value=p_value,
        significance=significance_stars(p_value),
        effect_size=r,
        interpretation=_interpret_significance(p_value)
    )


def t_test_independent(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = False
) -> StatisticalResult:
    """
    Independent samples t-test (Welch's by default).

    Args:
        group1: First group of values
        group2: Second group of values
        equal_var: Whether to assume equal variances (False uses Welch's t-test)

    Returns:
        StatisticalResult with t statistic, p-value, and Cohen's d
    """
    stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    effect = cohens_d(group1, group2)

    return StatisticalResult(
        statistic=stat,
        p_value=p_value,
        significance=significance_stars(p_value),
        effect_size=effect,
        interpretation=_interpret_significance(p_value)
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def significance_stars(p_value: float) -> str:
    """
    Return significance stars for p-value (publication convention).

    Args:
        p_value: p-value from statistical test

    Returns:
        '***' for p < 0.001
        '**' for p < 0.01
        '*' for p < 0.05
        '' for p >= 0.05
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


def _interpret_significance(p_value: float) -> str:
    """Generate human-readable interpretation of p-value."""
    if p_value < 0.001:
        return "Highly significant difference (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant difference (p < 0.01)"
    elif p_value < 0.05:
        return "Significant difference (p < 0.05)"
    else:
        return "No significant difference (p >= 0.05)"


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'bonferroni'
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of uncorrected p-values
        method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)

    Returns:
        List of corrected p-values
    """
    n = len(p_values)
    if n == 0:
        return []

    if method == 'bonferroni':
        return [min(p * n, 1.0) for p in p_values]

    elif method == 'holm':
        # Holm-Bonferroni step-down method
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        corrected = np.zeros(n)

        cummax = 0.0
        for i, (idx, p) in enumerate(zip(sorted_idx, sorted_p)):
            corrected_p = p * (n - i)
            cummax = max(cummax, corrected_p)
            corrected[idx] = min(cummax, 1.0)

        return corrected.tolist()

    elif method == 'fdr_bh':
        # Benjamini-Hochberg False Discovery Rate
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]
        corrected = np.zeros(n)

        cummin = 1.0
        for i in range(n - 1, -1, -1):
            idx = sorted_idx[i]
            corrected_p = sorted_p[i] * n / (i + 1)
            cummin = min(cummin, corrected_p)
            corrected[idx] = min(cummin, 1.0)

        return corrected.tolist()

    else:
        raise ValueError(f"Unknown correction method: {method}")


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any statistic.

    Args:
        data: Input data array
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point_estimate = statistic_func(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (point_estimate, lower, upper)


def power_analysis_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8
) -> int:
    """
    Calculate required sample size for comparing two proportions.

    Uses normal approximation for power calculation.

    Args:
        p1: Expected proportion in group 1
        p2: Expected proportion in group 2
        alpha: Significance level
        power: Desired statistical power

    Returns:
        Required sample size per group
    """
    # Effect size (Cohen's h)
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    h = abs(h)

    if h == 0:
        return float('inf')

    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / h) ** 2

    return int(np.ceil(n))


def summarize_game_statistics(
    games_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive statistical summary of game results.

    Args:
        games_data: List of game result dictionaries with keys:
            - winner: 'liberal' or 'fascist'
            - duration: game duration in seconds
            - total_cost: API cost
            - deception_rate: proportion of deceptive decisions

    Returns:
        Dictionary with summary statistics and confidence intervals
    """
    n = len(games_data)
    if n == 0:
        return {"error": "No games to analyze"}

    # Extract data
    liberal_wins = sum(1 for g in games_data if g.get('winner') == 'liberal')
    fascist_wins = n - liberal_wins

    durations = np.array([g.get('duration', 0) for g in games_data])
    costs = np.array([g.get('total_cost', 0) for g in games_data])
    deception_rates = np.array([g.get('deception_rate', 0) for g in games_data])

    # Calculate statistics with CIs
    win_rate = calculate_proportion_ci(liberal_wins, n)
    duration_ci = calculate_confidence_interval(durations)
    cost_ci = calculate_confidence_interval(costs)
    deception_ci = calculate_confidence_interval(deception_rates)

    return {
        "n_games": n,
        "liberal_wins": {
            "count": liberal_wins,
            "rate": win_rate[0],
            "ci_lower": win_rate[1],
            "ci_upper": win_rate[2]
        },
        "fascist_wins": {
            "count": fascist_wins,
            "rate": 1 - win_rate[0],
            "ci_lower": 1 - win_rate[2],
            "ci_upper": 1 - win_rate[1]
        },
        "duration_seconds": {
            "mean": duration_ci[0],
            "ci_lower": duration_ci[1],
            "ci_upper": duration_ci[2],
            "median": np.median(durations),
            "std": np.std(durations)
        },
        "cost_usd": {
            "mean": cost_ci[0],
            "ci_lower": cost_ci[1],
            "ci_upper": cost_ci[2],
            "total": np.sum(costs)
        },
        "deception_rate": {
            "mean": deception_ci[0],
            "ci_lower": deception_ci[1],
            "ci_upper": deception_ci[2]
        }
    }


def format_ci_for_display(
    value: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 2,
    as_percentage: bool = False
) -> str:
    """
    Format confidence interval for display.

    Args:
        value: Point estimate
        ci_lower: Lower bound
        ci_upper: Upper bound
        decimals: Number of decimal places
        as_percentage: Whether to display as percentage

    Returns:
        Formatted string like "50.0% [45.2%, 54.8%]"
    """
    multiplier = 100 if as_percentage else 1
    suffix = '%' if as_percentage else ''

    return f"{value * multiplier:.{decimals}f}{suffix} [{ci_lower * multiplier:.{decimals}f}{suffix}, {ci_upper * multiplier:.{decimals}f}{suffix}]"
