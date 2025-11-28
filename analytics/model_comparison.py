"""
Model Comparison Analytics

Statistical comparison functions for multi-model evaluation results.
Includes hypothesis testing, effect sizes, confidence intervals,
and publication-ready output generation.

Author: Samuel Chakwera (stchakdev)
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class WinRateCI:
    """Win rate with confidence interval."""
    wins: int
    total: int
    rate: float
    ci_lower: float
    ci_upper: float
    confidence: float = 0.95

    @classmethod
    def from_counts(cls, wins: int, total: int, confidence: float = 0.95) -> 'WinRateCI':
        """Calculate Wilson score confidence interval."""
        if total == 0:
            return cls(wins=0, total=0, rate=0.0, ci_lower=0.0, ci_upper=0.0, confidence=confidence)

        p = wins / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denominator = 1 + z**2 / total

        center = (p + z**2 / (2 * total)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

        return cls(
            wins=wins,
            total=total,
            rate=p,
            ci_lower=max(0, center - spread),
            ci_upper=min(1, center + spread),
            confidence=confidence,
        )


@dataclass
class ComparisonResult:
    """Result of comparing two models."""
    model_a: str
    model_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    p_value: float
    effect_size: float
    effect_size_name: str
    significant: bool
    significance_level: str  # *, **, ***

    def to_dict(self) -> Dict:
        return {
            'model_a': self.model_a,
            'model_b': self.model_b,
            'metric': self.metric,
            'value_a': self.value_a,
            'value_b': self.value_b,
            'difference': self.difference,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'effect_size_name': self.effect_size_name,
            'significant': self.significant,
            'significance_level': self.significance_level,
        }


@dataclass
class ModelStats:
    """Statistics for a single model."""
    model_id: str
    model_name: str
    games: int
    liberal_wins: int
    fascist_wins: int
    total_cost: float
    deception_count: int = 0

    @property
    def liberal_win_rate(self) -> float:
        return self.liberal_wins / self.games if self.games > 0 else 0.0

    @property
    def fascist_win_rate(self) -> float:
        return self.fascist_wins / self.games if self.games > 0 else 0.0

    @property
    def cost_per_game(self) -> float:
        return self.total_cost / self.games if self.games > 0 else 0.0


def significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def chi_square_win_rates(
    wins_a: int, total_a: int,
    wins_b: int, total_b: int
) -> Tuple[float, float]:
    """
    Chi-square test for difference in win rates.

    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    # Construct contingency table
    # [[wins_a, losses_a], [wins_b, losses_b]]
    table = np.array([
        [wins_a, total_a - wins_a],
        [wins_b, total_b - wins_b]
    ])

    # Use chi-square with Yates correction for small samples
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=True)

    return chi2, p_value


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportions.

    Args:
        p1: First proportion
        p2: Second proportion

    Returns:
        Cohen's h effect size
    """
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def cramers_v(chi2: float, n: int, k: int) -> float:
    """
    Calculate Cramér's V effect size.

    Args:
        chi2: Chi-square statistic
        n: Total sample size
        k: Minimum of (rows, columns) in contingency table

    Returns:
        Cramér's V
    """
    return math.sqrt(chi2 / (n * (k - 1)))


def odds_ratio(
    wins_a: int, total_a: int,
    wins_b: int, total_b: int
) -> Tuple[float, float, float]:
    """
    Calculate odds ratio with 95% CI.

    Returns:
        Tuple of (odds_ratio, ci_lower, ci_upper)
    """
    # Add 0.5 to avoid division by zero (Haldane-Anscombe correction)
    a = wins_a + 0.5
    b = (total_a - wins_a) + 0.5
    c = wins_b + 0.5
    d = (total_b - wins_b) + 0.5

    or_value = (a * d) / (b * c)
    log_or = math.log(or_value)
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)

    z = 1.96  # 95% CI
    ci_lower = math.exp(log_or - z * se)
    ci_upper = math.exp(log_or + z * se)

    return or_value, ci_lower, ci_upper


def compare_win_rates(
    model_a: ModelStats,
    model_b: ModelStats,
    win_type: str = "liberal"
) -> ComparisonResult:
    """
    Compare win rates between two models using chi-square test.

    Args:
        model_a: First model statistics
        model_b: Second model statistics
        win_type: "liberal" or "fascist"

    Returns:
        ComparisonResult with test results
    """
    if win_type == "liberal":
        wins_a, wins_b = model_a.liberal_wins, model_b.liberal_wins
        rate_a, rate_b = model_a.liberal_win_rate, model_b.liberal_win_rate
    else:
        wins_a, wins_b = model_a.fascist_wins, model_b.fascist_wins
        rate_a, rate_b = model_a.fascist_win_rate, model_b.fascist_win_rate

    chi2, p_value = chi_square_win_rates(
        wins_a, model_a.games,
        wins_b, model_b.games
    )

    effect = cohens_h(rate_a, rate_b)

    return ComparisonResult(
        model_a=model_a.model_name,
        model_b=model_b.model_name,
        metric=f"{win_type}_win_rate",
        value_a=rate_a,
        value_b=rate_b,
        difference=rate_a - rate_b,
        p_value=p_value,
        effect_size=abs(effect),
        effect_size_name="Cohen's h",
        significant=p_value < 0.05,
        significance_level=significance_stars(p_value),
    )


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        List of booleans indicating significance after correction
    """
    adjusted_alpha = alpha / len(p_values)
    return [p < adjusted_alpha for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni step-down correction.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        List of booleans indicating significance after correction
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    significant = [False] * n

    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        if p <= adjusted_alpha:
            significant[idx] = True
        else:
            break  # Stop at first non-significant

    return significant


def calculate_elo_ratings(
    pairwise_results: Dict[Tuple[str, str], Tuple[int, int]],
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Calculate Elo ratings from pairwise game results.

    Args:
        pairwise_results: Dict mapping (model_a, model_b) to (wins_a, wins_b)
        initial_rating: Starting Elo rating
        k_factor: Elo K-factor
        iterations: Number of iterations for convergence

    Returns:
        Dict mapping model names to Elo ratings
    """
    # Get all models
    models = set()
    for (a, b) in pairwise_results.keys():
        models.add(a)
        models.add(b)

    # Initialize ratings
    ratings = {m: initial_rating for m in models}

    # Iterate to convergence
    for _ in range(iterations):
        for (model_a, model_b), (wins_a, wins_b) in pairwise_results.items():
            if wins_a + wins_b == 0:
                continue

            # Expected scores
            expected_a = 1 / (1 + 10 ** ((ratings[model_b] - ratings[model_a]) / 400))
            expected_b = 1 - expected_a

            # Actual scores
            total = wins_a + wins_b
            actual_a = wins_a / total
            actual_b = wins_b / total

            # Update ratings
            ratings[model_a] += k_factor * (actual_a - expected_a)
            ratings[model_b] += k_factor * (actual_b - expected_b)

    return ratings


def generate_comparison_matrix(
    model_stats: List[ModelStats],
    metric: str = "liberal_win_rate",
) -> Dict[str, Dict[str, ComparisonResult]]:
    """
    Generate pairwise comparison matrix for all models.

    Args:
        model_stats: List of model statistics
        metric: Metric to compare

    Returns:
        Nested dict of comparison results
    """
    matrix = {}

    for i, model_a in enumerate(model_stats):
        matrix[model_a.model_name] = {}
        for j, model_b in enumerate(model_stats):
            if i == j:
                matrix[model_a.model_name][model_b.model_name] = None
            else:
                result = compare_win_rates(model_a, model_b)
                matrix[model_a.model_name][model_b.model_name] = result

    return matrix


def generate_latex_comparison_table(
    model_stats: List[ModelStats],
    include_ci: bool = True,
    include_cost: bool = True,
) -> str:
    """
    Generate LaTeX table for model comparison.

    Args:
        model_stats: List of model statistics
        include_ci: Include confidence intervals
        include_cost: Include cost column

    Returns:
        LaTeX table string
    """
    # Sort by liberal win rate
    sorted_stats = sorted(model_stats, key=lambda x: x.liberal_win_rate, reverse=True)

    # Build column specification
    cols = "l" + "r" * (4 + int(include_cost))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Multi-Model Comparison: Win Rates and Performance}",
        r"\label{tab:model_comparison}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
    ]

    # Header
    header_parts = ["Model", "Games", "Liberal Win\\%", "Fascist Win\\%"]
    if include_cost:
        header_parts.append("Cost (\\$)")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for stat in sorted_stats:
        liberal_ci = WinRateCI.from_counts(stat.liberal_wins, stat.games)
        fascist_ci = WinRateCI.from_counts(stat.fascist_wins, stat.games)

        if include_ci:
            liberal_str = f"{stat.liberal_win_rate*100:.1f} ({liberal_ci.ci_lower*100:.1f}--{liberal_ci.ci_upper*100:.1f})"
            fascist_str = f"{stat.fascist_win_rate*100:.1f} ({fascist_ci.ci_lower*100:.1f}--{fascist_ci.ci_upper*100:.1f})"
        else:
            liberal_str = f"{stat.liberal_win_rate*100:.1f}\\%"
            fascist_str = f"{stat.fascist_win_rate*100:.1f}\\%"

        row_parts = [
            stat.model_name.replace("_", "\\_"),
            str(stat.games),
            liberal_str,
            fascist_str,
        ]
        if include_cost:
            row_parts.append(f"{stat.total_cost:.2f}")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_pairwise_significance_table(
    comparisons: List[ComparisonResult],
    correction: str = "bonferroni",
) -> str:
    """
    Generate LaTeX table showing pairwise significance.

    Args:
        comparisons: List of comparison results
        correction: Multiple comparison correction method

    Returns:
        LaTeX table string
    """
    # Apply correction
    p_values = [c.p_value for c in comparisons]

    if correction == "bonferroni":
        significant = bonferroni_correction(p_values)
    elif correction == "holm":
        significant = holm_bonferroni_correction(p_values)
    else:
        significant = [c.significant for c in comparisons]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Pairwise Model Comparisons (Win Rate Differences)}",
        r"\label{tab:pairwise}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model A & Model B & $\Delta$ & $p$-value & Effect Size & Sig. \\",
        r"\midrule",
    ]

    for c, sig in zip(comparisons, significant):
        sig_str = c.significance_level if sig else ""
        lines.append(
            f"{c.model_a} & {c.model_b} & {c.difference*100:+.1f}\\% & "
            f"{c.p_value:.3f} & {c.effect_size:.2f} & {sig_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\small * $p<0.05$, ** $p<0.01$, *** $p<0.001$} \\",
        f"\\multicolumn{{6}}{{l}}{{\\small Correction: {correction.capitalize()}}} \\\\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_report(
    model_stats: List[ModelStats],
    comparisons: List[ComparisonResult],
    elo_ratings: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate Markdown report of comparison results.

    Args:
        model_stats: List of model statistics
        comparisons: List of comparison results
        elo_ratings: Optional Elo ratings

    Returns:
        Markdown report string
    """
    lines = [
        "# Multi-Model Comparison Report",
        "",
        "## Summary Statistics",
        "",
        "| Model | Games | Liberal Win% | Fascist Win% | Cost |",
        "|-------|------:|-------------:|-------------:|-----:|",
    ]

    sorted_stats = sorted(model_stats, key=lambda x: x.liberal_win_rate, reverse=True)
    for stat in sorted_stats:
        lines.append(
            f"| {stat.model_name} | {stat.games} | "
            f"{stat.liberal_win_rate*100:.1f}% | {stat.fascist_win_rate*100:.1f}% | "
            f"${stat.total_cost:.2f} |"
        )

    lines.extend(["", "## Pairwise Comparisons", ""])

    sig_comparisons = [c for c in comparisons if c.significant]
    if sig_comparisons:
        lines.append("### Significant Differences (p < 0.05)")
        lines.append("")
        for c in sig_comparisons:
            lines.append(
                f"- **{c.model_a}** vs **{c.model_b}**: "
                f"{c.difference*100:+.1f}% (p={c.p_value:.3f}{c.significance_level})"
            )
    else:
        lines.append("No significant pairwise differences found.")

    if elo_ratings:
        lines.extend(["", "## Elo Ratings", ""])
        sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        lines.append("| Rank | Model | Elo |")
        lines.append("|-----:|-------|----:|")
        for rank, (model, elo) in enumerate(sorted_elo, 1):
            lines.append(f"| {rank} | {model} | {elo:.0f} |")

    return "\n".join(lines)


def analyze_comparison_results(results_file: str) -> Dict[str, Any]:
    """
    Analyze results from a comparison run.

    Args:
        results_file: Path to comparison results JSON

    Returns:
        Analysis results dict
    """
    with open(results_file) as f:
        data = json.load(f)

    # Extract model statistics
    model_stats = []
    for model_id, result in data.get('model_results', {}).items():
        stat = ModelStats(
            model_id=model_id,
            model_name=result.get('model_name', model_id),
            games=result.get('games_completed', 0),
            liberal_wins=result.get('liberal_wins', 0),
            fascist_wins=result.get('fascist_wins', 0),
            total_cost=result.get('total_cost', 0),
        )
        model_stats.append(stat)

    # Generate pairwise comparisons
    comparisons = []
    for i, model_a in enumerate(model_stats):
        for j, model_b in enumerate(model_stats):
            if i < j:  # Only upper triangle
                result = compare_win_rates(model_a, model_b)
                comparisons.append(result)

    # Calculate Elo ratings from pairwise results
    pairwise_results = {}
    for c in comparisons:
        model_a = next(s for s in model_stats if s.model_name == c.model_a)
        model_b = next(s for s in model_stats if s.model_name == c.model_b)
        pairwise_results[(c.model_a, c.model_b)] = (model_a.liberal_wins, model_b.liberal_wins)

    elo_ratings = calculate_elo_ratings(pairwise_results)

    return {
        'model_stats': model_stats,
        'comparisons': comparisons,
        'elo_ratings': elo_ratings,
        'latex_table': generate_latex_comparison_table(model_stats),
        'markdown_report': generate_markdown_report(model_stats, comparisons, elo_ratings),
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Model Comparison Analytics Module")
    print("=" * 50)

    # Create example model statistics
    example_stats = [
        ModelStats("grok", "Grok 4.1 Fast", 100, 55, 45, 0.0),
        ModelStats("llama", "Llama 4 Maverick", 100, 48, 52, 0.0),
        ModelStats("gpt5", "GPT-5 Nano", 100, 52, 48, 17.0),
    ]

    # Run comparison
    result = compare_win_rates(example_stats[0], example_stats[1])
    print(f"\nComparison: {result.model_a} vs {result.model_b}")
    print(f"  Win rates: {result.value_a*100:.1f}% vs {result.value_b*100:.1f}%")
    print(f"  Difference: {result.difference*100:+.1f}%")
    print(f"  p-value: {result.p_value:.4f} {result.significance_level}")
    print(f"  Effect size (Cohen's h): {result.effect_size:.3f}")

    # Generate LaTeX table
    print("\n" + "=" * 50)
    print("LaTeX Table:")
    print(generate_latex_comparison_table(example_stats, include_ci=False))
