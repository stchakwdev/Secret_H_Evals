#!/usr/bin/env python3
"""
Deception Detection Heatmap Visualization

Creates a heatmap showing when and how often each player was deceptive
by comparing their private reasoning vs public statements. Highlights
strategic lying patterns in Secret Hitler gameplay.

Author: Samuel Chakwera (stchakdev)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import numpy as np


def connect_db(db_path: str = None) -> sqlite3.Connection:
    """Connect to the games database."""
    if db_path is None:
        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "games.db"

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    return sqlite3.connect(db_path)


def get_deception_data(conn: sqlite3.Connection, game_limit: int = 10) -> pd.DataFrame:
    """
    Extract deception data from player decisions.

    Args:
        conn: Database connection
        game_limit: Maximum number of recent games to analyze

    Returns:
        DataFrame with columns: game_id, player_name, turn_number,
                               is_deception, deception_score
    """
    query = """
        SELECT
            pd.game_id,
            pd.player_name,
            pd.turn_number,
            pd.decision_type,
            pd.reasoning,
            pd.public_statement,
            pd.is_deception,
            pd.deception_score
        FROM player_decisions pd
        WHERE pd.public_statement IS NOT NULL
          AND pd.public_statement != ''
          AND pd.game_id IN (
              SELECT game_id
              FROM games
              ORDER BY timestamp DESC
              LIMIT ?
          )
        ORDER BY pd.game_id, pd.turn_number
    """

    df = pd.read_sql_query(query, conn, params=(game_limit,))
    return df


def create_deception_heatmap(df: pd.DataFrame, output_path: str, max_turns: int = 50):
    """
    Create deception heatmap visualization.

    Args:
        df: DataFrame with deception data
        output_path: Path to save PNG
        max_turns: Maximum number of turns to display
    """
    if df.empty:
        print("No deception data found")
        return

    # Create pivot table for heatmap
    # Rows: Players, Columns: Turns, Values: Deception scores
    pivot_data = df.pivot_table(
        index='player_name',
        columns='turn_number',
        values='deception_score',
        aggfunc='mean',  # Average if multiple decisions per turn
        fill_value=0
    )

    # Limit to max_turns
    if pivot_data.shape[1] > max_turns:
        pivot_data = pivot_data.iloc[:, :max_turns]

    # Sort by total deception score (most deceptive players first)
    player_totals = pivot_data.sum(axis=1).sort_values(ascending=False)
    pivot_data = pivot_data.loc[player_totals.index]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                            gridspec_kw={'height_ratios': [4, 1]},
                            constrained_layout=True)

    # Main heatmap
    ax_heatmap = axes[0]

    # Custom colormap: white (no deception) -> yellow (mild) -> red (high)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        pivot_data,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Deception Score'},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax_heatmap,
        square=False
    )

    ax_heatmap.set_title('Player Deception Heatmap - Private Reasoning vs Public Statements',
                         fontsize=14, fontweight='bold', pad=15)
    ax_heatmap.set_xlabel('Turn Number', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Player', fontsize=12, fontweight='bold')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=0)
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)

    # Deception frequency bar chart
    ax_bar = axes[1]

    # Calculate deception frequency (% of turns with deception > 0.5)
    deception_freq = (pivot_data > 0.5).sum(axis=1) / pivot_data.shape[1] * 100
    deception_freq = deception_freq.sort_values(ascending=False)

    colors = ['#d62728' if freq > 50 else '#ff7f0e' if freq > 25 else '#2ca02c'
              for freq in deception_freq.values]

    ax_bar.barh(range(len(deception_freq)), deception_freq.values, color=colors, alpha=0.7)
    ax_bar.set_yticks(range(len(deception_freq)))
    ax_bar.set_yticklabels(deception_freq.index)
    ax_bar.set_xlabel('Deception Frequency (%)', fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('Player', fontsize=11, fontweight='bold')
    ax_bar.set_title('Overall Deception Frequency', fontsize=12, fontweight='bold')
    ax_bar.grid(axis='x', alpha=0.3)
    ax_bar.set_xlim(0, 100)

    # Add percentage labels
    for i, (player, freq) in enumerate(deception_freq.items()):
        ax_bar.text(freq + 2, i, f'{freq:.1f}%', va='center', fontsize=9)

    # Overall figure title
    fig.suptitle('Deception Analysis - Secret Hitler LLM Evaluation',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Deception heatmap saved to: {output_path}")

    plt.close()


def create_deception_summary(df: pd.DataFrame, output_path: str):
    """
    Create detailed deception summary statistics visualization.

    Args:
        df: DataFrame with deception data
        output_path: Path to save PNG
    """
    if df.empty:
        print("No deception data for summary")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # 1. Deception by Decision Type
    ax1 = axes[0, 0]
    decision_deception = df.groupby('decision_type')['deception_score'].agg(['mean', 'count'])
    decision_deception = decision_deception[decision_deception['count'] >= 5]  # At least 5 samples
    decision_deception = decision_deception.sort_values('mean', ascending=False)

    ax1.barh(range(len(decision_deception)), decision_deception['mean'].values,
            color='#ff7f0e', alpha=0.7)
    ax1.set_yticks(range(len(decision_deception)))
    ax1.set_yticklabels(decision_deception.index)
    ax1.set_xlabel('Average Deception Score', fontweight='bold')
    ax1.set_title('Deception by Decision Type', fontweight='bold', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(decision_deception.iterrows()):
        ax1.text(row['mean'] + 0.01, i, f"{row['mean']:.2f} (n={int(row['count'])})",
                va='center', fontsize=9)

    # 2. Deception Distribution
    ax2 = axes[0, 1]
    deception_scores = df[df['deception_score'] > 0]['deception_score']

    ax2.hist(deception_scores, bins=20, color='#d62728', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Deception Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Deception Score Distribution', fontweight='bold', fontsize=12)
    ax2.axvline(deception_scores.median(), color='blue', linestyle='--',
               linewidth=2, label=f'Median: {deception_scores.median():.2f}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Deception Over Time (Turns)
    ax3 = axes[1, 0]
    turn_deception = df.groupby('turn_number')['deception_score'].mean()

    ax3.plot(turn_deception.index, turn_deception.values,
            marker='o', linewidth=2, markersize=4, color='#1f77b4')
    ax3.fill_between(turn_deception.index, 0, turn_deception.values, alpha=0.3)
    ax3.set_xlabel('Turn Number', fontweight='bold')
    ax3.set_ylabel('Average Deception Score', fontweight='bold')
    ax3.set_title('Deception Progression Over Game', fontweight='bold', fontsize=12)
    ax3.grid(alpha=0.3)

    # 4. Top Deceivers
    ax4 = axes[1, 1]
    player_stats = df.groupby('player_name').agg({
        'deception_score': ['mean', 'count'],
        'is_deception': 'sum'
    })
    player_stats.columns = ['avg_score', 'total_decisions', 'deception_count']
    player_stats = player_stats[player_stats['total_decisions'] >= 10]  # At least 10 decisions
    player_stats['deception_rate'] = player_stats['deception_count'] / player_stats['total_decisions'] * 100
    player_stats = player_stats.sort_values('avg_score', ascending=False).head(10)

    ax4.scatter(player_stats['deception_rate'], player_stats['avg_score'],
               s=player_stats['total_decisions']*3, alpha=0.6, color='#9467bd')

    for player, row in player_stats.iterrows():
        ax4.annotate(player, (row['deception_rate'], row['avg_score']),
                    fontsize=8, alpha=0.7)

    ax4.set_xlabel('Deception Rate (%)', fontweight='bold')
    ax4.set_ylabel('Average Deception Score', fontweight='bold')
    ax4.set_title('Player Deception Analysis\n(Bubble size = Total Decisions)',
                 fontweight='bold', fontsize=12)
    ax4.grid(alpha=0.3)

    # Overall title - position higher to avoid subplot overlap
    fig.suptitle('Deception Analytics Dashboard - Secret Hitler LLM Evaluation',
                fontsize=16, fontweight='bold', y=1.02)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Deception summary saved to: {output_path}")

    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create deception detection heatmap visualization"
    )
    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )
    parser.add_argument(
        '--output', '-o',
        default='visualizations/deception_heatmap.png',
        help='Output path for main heatmap PNG'
    )
    parser.add_argument(
        '--summary', '-s',
        default='visualizations/deception_summary.png',
        help='Output path for summary dashboard PNG'
    )
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=10,
        help='Number of recent games to analyze (default: 10)'
    )
    parser.add_argument(
        '--max-turns', '-t',
        type=int,
        default=50,
        help='Maximum turns to display in heatmap (default: 50)'
    )

    args = parser.parse_args()

    try:
        # Connect to database
        conn = connect_db(args.db)

        # Get deception data
        print(f"Analyzing {args.games} most recent games...")
        df = get_deception_data(conn, args.games)

        if df.empty:
            print("No deception data found in database")
            return 1

        print(f"Found {len(df)} player decisions with statements")

        # Create heatmap
        create_deception_heatmap(df, args.output, args.max_turns)

        # Create summary dashboard
        create_deception_summary(df, args.summary)

        conn.close()

        print("\nDeception analysis complete!")
        print(f"  - Heatmap: {args.output}")
        print(f"  - Summary: {args.summary}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
