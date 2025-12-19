#!/usr/bin/env python3
"""
Export Figures

Generates figures in multiple formats (PNG, SVG, PDF).
Applies consistent styling and confidence intervals.

Usage:
    python scripts/export_publication_figures.py [--output-dir OUTPUT] [--db-path DB]
    python scripts/export_publication_figures.py --formats svg pdf png

Author: Samuel Chakwera (stchakdev)
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.core.statistical import (
    calculate_proportion_ci,
    calculate_confidence_interval,
    chi_square_test_proportions,
    significance_stars,
)
from analysis.visualization.utils import (
    apply_publication_style,
    save_figure,
    get_color,
    COLORBLIND_PALETTE,
    FIGURE_SIZES,
    add_significance_annotation,
)


def connect_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Connect to games database."""
    if db_path is None:
        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "games.db"

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def get_aggregate_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Get aggregate statistics from database."""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_games,
            SUM(CASE WHEN winning_team = 'liberal' THEN 1 ELSE 0 END) as liberal_wins,
            SUM(CASE WHEN winning_team = 'fascist' THEN 1 ELSE 0 END) as fascist_wins,
            AVG(duration_seconds) as avg_duration,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost
        FROM games
    """)

    row = cursor.fetchone()
    return dict(row)


def get_deception_by_type(conn: sqlite3.Connection) -> pd.DataFrame:
    """Get deception statistics by decision type."""
    query = """
        SELECT
            decision_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_deception = 1 THEN 1 ELSE 0 END) as deceptions,
            AVG(deception_score) as avg_score
        FROM player_decisions
        GROUP BY decision_type
        ORDER BY total DESC
    """
    df = pd.read_sql_query(query, conn)
    df['deception_rate'] = df['deceptions'] / df['total']
    return df


def create_win_rate_figure(conn: sqlite3.Connection) -> plt.Figure:
    """
    Create win rate bar chart with 95% confidence intervals.

    Returns:
        Matplotlib Figure
    """
    apply_publication_style()

    stats = get_aggregate_stats(conn)
    total = stats['total_games']
    liberal_wins = stats['liberal_wins']
    fascist_wins = stats['fascist_wins']

    # Calculate CIs
    lib_rate, lib_lower, lib_upper = calculate_proportion_ci(liberal_wins, total)
    fas_rate, fas_lower, fas_upper = calculate_proportion_ci(fascist_wins, total)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    x = [0, 1]
    rates = [lib_rate * 100, fas_rate * 100]
    errors_lower = [(lib_rate - lib_lower) * 100, (fas_rate - fas_lower) * 100]
    errors_upper = [(lib_upper - lib_rate) * 100, (fas_upper - fas_rate) * 100]

    bars = ax.bar(x, rates,
                  color=[get_color('liberal'), get_color('fascist')],
                  edgecolor='black', linewidth=1)

    # Add error bars
    ax.errorbar(x, rates,
                yerr=[errors_lower, errors_upper],
                fmt='none', capsize=5, capthick=2,
                color='black', linewidth=2)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(['Liberal', 'Fascist'])
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    ax.set_title(f'Win Rates (n={total} games)')

    # Add value labels
    for bar, rate, lower, upper in zip(bars, rates, errors_lower, errors_upper):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%\n[{rate-lower:.1f}, {rate+upper:.1f}]',
                   xy=(bar.get_x() + bar.get_width() / 2, height + upper + 2),
                   ha='center', va='bottom', fontsize=9)

    # Add significance test result
    result = chi_square_test_proportions(liberal_wins, total, fascist_wins, total)
    sig_text = f"χ²={result.statistic:.2f}, p={result.p_value:.3f} {result.significance}"
    ax.text(0.5, 0.02, sig_text, transform=ax.transAxes,
           ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    return fig


def create_deception_by_type_figure(conn: sqlite3.Connection) -> plt.Figure:
    """
    Create deception rate by decision type bar chart.

    Returns:
        Matplotlib Figure
    """
    apply_publication_style()

    df = get_deception_by_type(conn)

    if df.empty:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['double_column'])
        ax.text(0.5, 0.5, 'No deception data available',
               transform=ax.transAxes, ha='center')
        return fig

    # Filter to decision types with enough data
    df = df[df['total'] >= 10]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['double_column'])

    # Calculate CIs for each type
    rates = []
    lower_errors = []
    upper_errors = []

    for _, row in df.iterrows():
        rate, lower, upper = calculate_proportion_ci(
            int(row['deceptions']), int(row['total'])
        )
        rates.append(rate * 100)
        lower_errors.append((rate - lower) * 100)
        upper_errors.append((upper - rate) * 100)

    x = range(len(df))

    bars = ax.bar(x, rates, color=get_color('fascist'), alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    ax.errorbar(x, rates,
                yerr=[lower_errors, upper_errors],
                fmt='none', capsize=3, capthick=1.5,
                color='black', linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df['decision_type'], rotation=45, ha='right')
    ax.set_ylabel('Deception Rate (%)')
    ax.set_title('Deception Rate by Decision Type')
    ax.set_ylim(0, max(rates) * 1.3)

    # Add sample sizes
    for i, (bar, n) in enumerate(zip(bars, df['total'])):
        ax.annotate(f'n={n}', xy=(bar.get_x() + bar.get_width() / 2, 2),
                   ha='center', va='bottom', fontsize=7, color='white')

    plt.tight_layout()
    return fig


def create_cost_efficiency_figure(conn: sqlite3.Connection) -> plt.Figure:
    """
    Create cost efficiency scatter plot.

    Returns:
        Matplotlib Figure
    """
    apply_publication_style()

    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_id, total_cost, duration_seconds, winning_team,
               liberal_policies, fascist_policies
        FROM games
        WHERE total_cost > 0 AND duration_seconds > 0
    """)

    games = [dict(row) for row in cursor.fetchall()]

    if not games:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])
        ax.text(0.5, 0.5, 'No cost data available',
               transform=ax.transAxes, ha='center')
        return fig

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    costs = [g['total_cost'] for g in games]
    durations = [g['duration_seconds'] / 60 for g in games]  # Convert to minutes
    colors = [get_color('liberal') if g['winning_team'] == 'liberal'
             else get_color('fascist') for g in games]

    ax.scatter(durations, costs, c=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Cost vs Duration')

    # Add trend line
    z = np.polyfit(durations, costs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(durations), max(durations), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, label='Trend')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_color('liberal'),
               markersize=8, label='Liberal Win'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=get_color('fascist'),
               markersize=8, label='Fascist Win'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    return fig


def create_deception_distribution_figure(conn: sqlite3.Connection) -> plt.Figure:
    """
    Create deception score distribution histogram.

    Returns:
        Matplotlib Figure
    """
    apply_publication_style()

    cursor = conn.cursor()
    cursor.execute("""
        SELECT deception_score FROM player_decisions
        WHERE deception_score > 0
    """)

    scores = [row[0] for row in cursor.fetchall()]

    if not scores:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])
        ax.text(0.5, 0.5, 'No deception score data available',
               transform=ax.transAxes, ha='center')
        return fig

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_column'])

    n, bins, patches_hist = ax.hist(scores, bins=20, color=get_color('fascist'),
                                    alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Deception Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Deception Scores')

    # Add statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    ax.axvline(mean_score, color='black', linestyle='--', linewidth=2,
               label=f'Mean: {mean_score:.2f}')
    ax.axvline(mean_score + std_score, color='gray', linestyle=':',
               label=f'±1 SD: {std_score:.2f}')
    ax.axvline(mean_score - std_score, color='gray', linestyle=':')

    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def create_summary_table_figure(conn: sqlite3.Connection) -> plt.Figure:
    """
    Create summary statistics table as a figure.

    Returns:
        Matplotlib Figure
    """
    apply_publication_style()

    stats = get_aggregate_stats(conn)
    total = stats['total_games']
    liberal_wins = stats['liberal_wins']

    lib_rate, lib_lower, lib_upper = calculate_proportion_ci(liberal_wins, total)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    table_data = [
        ['Metric', 'Value', '95% CI'],
        ['Total Games', f"{total}", '-'],
        ['Liberal Win Rate', f"{lib_rate*100:.1f}%", f"[{lib_lower*100:.1f}%, {lib_upper*100:.1f}%]"],
        ['Fascist Win Rate', f"{(1-lib_rate)*100:.1f}%", f"[{(1-lib_upper)*100:.1f}%, {(1-lib_lower)*100:.1f}%]"],
        ['Avg Duration', f"{stats['avg_duration']/60:.1f} min", '-'],
        ['Total Cost', f"${stats['total_cost']:.2f}", '-'],
        ['Avg Cost/Game', f"${stats['avg_cost']:.4f}", '-'],
    ]

    table = ax.table(cellText=table_data[1:],
                     colLabels=table_data[0],
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def export_all_figures(
    output_dir: str,
    db_path: Optional[str] = None,
    formats: List[str] = ['png', 'svg', 'pdf']
):
    """
    Export all publication figures.

    Args:
        output_dir: Output directory for figures
        db_path: Path to database
        formats: List of formats to export
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting publication figures to: {output_dir}")
    print(f"Formats: {formats}")

    conn = connect_db(db_path)

    figures = [
        ('win_rates', create_win_rate_figure),
        ('deception_by_type', create_deception_by_type_figure),
        ('cost_efficiency', create_cost_efficiency_figure),
        ('deception_distribution', create_deception_distribution_figure),
        ('summary_table', create_summary_table_figure),
    ]

    for name, create_func in figures:
        print(f"\nCreating {name}...")
        try:
            fig = create_func(conn)

            # Save in all formats
            base_path = output_dir / f"{name}.png"
            save_figure(fig, str(base_path), formats=formats)

            plt.close(fig)
        except Exception as e:
            print(f"  Error creating {name}: {e}")

    conn.close()
    print(f"\nFigures exported to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export figures in multiple formats"
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='visualizations/publication',
        help='Output directory (default: visualizations/publication)'
    )
    parser.add_argument(
        '--db-path', '-d',
        help='Path to games.db'
    )
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        default=['png', 'svg', 'pdf'],
        help='Output formats (default: png svg pdf)'
    )

    args = parser.parse_args()

    export_all_figures(
        output_dir=args.output_dir,
        db_path=args.db_path,
        formats=args.formats
    )


if __name__ == "__main__":
    main()
