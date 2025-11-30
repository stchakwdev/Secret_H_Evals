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
    Create deception dashboard visualization with meaningful insights.

    Args:
        df: DataFrame with deception data
        output_path: Path to save PNG
        max_turns: Maximum number of turns to display
    """
    if df.empty:
        print("No deception data found")
        return

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # === Panel 1: Deception by Decision Type ===
    ax1 = axes[0, 0]
    decision_stats = df.groupby('decision_type').agg({
        'deception_score': ['mean', 'count'],
        'is_deception': 'sum'
    }).reset_index()
    decision_stats.columns = ['decision_type', 'avg_score', 'count', 'deceptive_count']
    decision_stats['deception_rate'] = decision_stats['deceptive_count'] / decision_stats['count'] * 100
    decision_stats = decision_stats.sort_values('avg_score', ascending=True)

    colors = ['#d62728' if s > 0.25 else '#ff7f0e' if s > 0.15 else '#2ca02c'
              for s in decision_stats['avg_score']]

    bars = ax1.barh(decision_stats['decision_type'], decision_stats['avg_score'],
                    color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Average Deception Score', fontweight='bold')
    ax1.set_title('Deception by Decision Type', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 0.5)
    ax1.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(decision_stats.iterrows()):
        ax1.text(row['avg_score'] + 0.01, i,
                f"{row['avg_score']:.2f} (n={int(row['count'])})",
                va='center', fontsize=10)

    # === Panel 2: Deception Over Game Phases ===
    ax2 = axes[0, 1]

    # Bin turns into early/mid/late game phases
    df['game_phase'] = pd.cut(df['turn_number'],
                               bins=[0, 10, 25, 100],
                               labels=['Early (1-10)', 'Mid (11-25)', 'Late (26+)'])

    phase_stats = df.groupby('game_phase', observed=True).agg({
        'deception_score': ['mean', 'std', 'count']
    }).reset_index()
    phase_stats.columns = ['phase', 'mean', 'std', 'count']

    x_pos = np.arange(len(phase_stats))
    bars = ax2.bar(x_pos, phase_stats['mean'], yerr=phase_stats['std'],
                   color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8,
                   edgecolor='black', capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(phase_stats['phase'], fontweight='bold')
    ax2.set_ylabel('Avg Deception Score', fontweight='bold')
    ax2.set_title('Deception Progression Through Game', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 0.6)
    ax2.grid(axis='y', alpha=0.3)

    for i, row in phase_stats.iterrows():
        ax2.text(i, row['mean'] + row['std'] + 0.02,
                f"n={int(row['count'])}", ha='center', fontsize=10)

    # === Panel 3: Top Deceivers by Player ===
    ax3 = axes[1, 0]

    player_stats = df.groupby('player_name').agg({
        'deception_score': ['mean', 'count'],
        'is_deception': 'sum'
    }).reset_index()
    player_stats.columns = ['player', 'avg_score', 'total_decisions', 'deceptive_count']
    player_stats['deception_rate'] = player_stats['deceptive_count'] / player_stats['total_decisions'] * 100
    player_stats = player_stats.sort_values('avg_score', ascending=True)

    colors = ['#d62728' if s > 0.25 else '#ff7f0e' if s > 0.15 else '#2ca02c'
              for s in player_stats['avg_score']]

    ax3.barh(player_stats['player'], player_stats['avg_score'],
             color=colors, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Average Deception Score', fontweight='bold')
    ax3.set_title('Deception by Player', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 0.5)
    ax3.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(player_stats.iterrows()):
        ax3.text(row['avg_score'] + 0.01, i,
                f"{row['deception_rate']:.0f}% ({int(row['deceptive_count'])}/{int(row['total_decisions'])})",
                va='center', fontsize=9)

    # === Panel 4: Deception by Game ===
    ax4 = axes[1, 1]

    game_stats = df.groupby('game_id').agg({
        'deception_score': ['mean', 'count'],
        'is_deception': 'sum'
    }).reset_index()
    game_stats.columns = ['game_id', 'avg_score', 'decisions', 'deceptive']
    game_stats['game_label'] = game_stats['game_id'].str.replace('realistic_test_game_', 'Game ')
    game_stats = game_stats.sort_values('avg_score', ascending=False)

    colors = ['#d62728' if s > 0.25 else '#ff7f0e' if s > 0.15 else '#2ca02c'
              for s in game_stats['avg_score']]

    ax4.bar(game_stats['game_label'], game_stats['avg_score'],
            color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Avg Deception Score', fontweight='bold')
    ax4.set_xlabel('Game', fontweight='bold')
    ax4.set_title('Deception Intensity by Game', fontweight='bold', fontsize=12)
    ax4.set_ylim(0, 0.5)
    ax4.grid(axis='y', alpha=0.3)

    for i, row in game_stats.iterrows():
        ax4.text(list(game_stats['game_label']).index(row['game_label']),
                row['avg_score'] + 0.01,
                f"{int(row['deceptive'])}", ha='center', fontsize=10, fontweight='bold')

    # Summary stats annotation
    total_decisions = len(df)
    total_deceptive = df['is_deception'].sum()
    overall_rate = total_deceptive / total_decisions * 100 if total_decisions > 0 else 0
    avg_score = df['deception_score'].mean()

    summary_text = (f"Total: {total_decisions} decisions | "
                   f"Deceptive: {int(total_deceptive)} ({overall_rate:.1f}%) | "
                   f"Avg Score: {avg_score:.3f}")

    fig.text(0.5, -0.02, summary_text, ha='center', fontsize=11,
            style='italic', fontweight='bold')

    # Overall figure title
    fig.suptitle('Deception Analysis Dashboard - Secret Hitler LLM Evaluation',
                fontsize=16, fontweight='bold', y=1.02)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Deception dashboard saved to: {output_path}")

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
