#!/usr/bin/env python3
"""
Batch Summary Infographic Generator

Creates a comprehensive infographic summarizing batch evaluation results,
including win rates, performance metrics, and key statistics for GitHub
showcase.

Author: Samuel Chakwera (stchakdev)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Optional, Dict
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


def get_batch_statistics(conn: sqlite3.Connection, game_limit: int = 10) -> Dict:
    """
    Get comprehensive batch statistics.

    Args:
        conn: Database connection
        game_limit: Number of recent games to include

    Returns:
        Dictionary with batch statistics
    """
    # Game outcomes
    games_query = """
        SELECT
            COUNT(*) as total_games,
            SUM(CASE WHEN winner = 'liberal' THEN 1 ELSE 0 END) as liberal_wins,
            SUM(CASE WHEN winner = 'fascist' THEN 1 ELSE 0 END) as fascist_wins,
            AVG(duration_seconds) as avg_duration,
            AVG(total_actions) as avg_actions,
            AVG(total_cost) as avg_cost,
            SUM(total_cost) as total_cost,
            AVG(liberal_policies) as avg_lib_policies,
            AVG(fascist_policies) as avg_fasc_policies
        FROM games
        WHERE game_id IN (
            SELECT game_id FROM games
            ORDER BY timestamp DESC
            LIMIT ?
        )
    """

    games_df = pd.read_sql_query(games_query, conn, params=(game_limit,))

    # Player decisions
    decisions_query = """
        SELECT
            COUNT(*) as total_decisions,
            AVG(CASE WHEN is_deception = 1 THEN 1.0 ELSE 0.0 END) as deception_rate,
            COUNT(DISTINCT decision_type) as decision_types
        FROM player_decisions
        WHERE game_id IN (
            SELECT game_id FROM games
            ORDER BY timestamp DESC
            LIMIT ?
        )
    """

    decisions_df = pd.read_sql_query(decisions_query, conn, params=(game_limit,))

    # API stats
    api_query = """
        SELECT
            COUNT(*) as total_requests,
            SUM(tokens) as total_tokens,
            AVG(latency) as avg_latency
        FROM api_requests
        WHERE game_id IN (
            SELECT game_id FROM games
            ORDER BY timestamp DESC
            LIMIT ?
        )
    """

    api_df = pd.read_sql_query(api_query, conn, params=(game_limit,))

    return {
        'games': games_df.iloc[0].to_dict() if not games_df.empty else {},
        'decisions': decisions_df.iloc[0].to_dict() if not decisions_df.empty else {},
        'api': api_df.iloc[0].to_dict() if not api_df.empty else {}
    }


def create_batch_summary(stats: Dict, output_path: str, batch_name: str = "Batch Evaluation"):
    """
    Create batch summary infographic.

    Args:
        stats: Dictionary with batch statistics
        output_path: Path to save PNG
        batch_name: Name of the batch for title
    """
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f5f5f5')

    # Title area
    title_ax = fig.add_axes([0, 0.88, 1, 0.12])
    title_ax.axis('off')
    title_ax.text(0.5, 0.6, 'Secret Hitler LLM Evaluation Framework',
                 ha='center', fontsize=22, fontweight='bold')
    title_ax.text(0.5, 0.2, batch_name,
                 ha='center', fontsize=18, color='#555')

    # Main content grid
    gs = fig.add_gridspec(3, 4, left=0.05, right=0.95, top=0.85, bottom=0.08,
                         hspace=0.4, wspace=0.3)

    # === Row 1: Key Metrics ===

    # Total Games metric
    ax_games = fig.add_subplot(gs[0, 0])
    draw_metric_card(ax_games, int(stats['games'].get('total_games', 0)),
                    'Total Games', '#1f77b4')

    # Win Rate pie chart
    ax_wins = fig.add_subplot(gs[0, 1:3])
    liberal_wins = int(stats['games'].get('liberal_wins', 0))
    fascist_wins = int(stats['games'].get('fascist_wins', 0))

    if liberal_wins + fascist_wins > 0:
        sizes = [liberal_wins, fascist_wins]
        colors = ['#1f77b4', '#d62728']
        labels = [f'Liberal\n{liberal_wins} ({liberal_wins/(liberal_wins+fascist_wins)*100:.1f}%)',
                 f'Fascist\n{fascist_wins} ({fascist_wins/(liberal_wins+fascist_wins)*100:.1f}%)']

        ax_wins.pie(sizes, labels=labels, colors=colors, autopct='',
                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax_wins.set_title('Win Distribution', fontweight='bold', fontsize=14, pad=10)

    # Total Cost metric
    ax_cost = fig.add_subplot(gs[0, 3])
    total_cost = stats['games'].get('total_cost', 0)
    draw_metric_card(ax_cost, f'${total_cost:.4f}', 'Total Cost', '#2ca02c')

    # === Row 2: Performance Metrics ===

    # Average Game Duration
    ax_duration = fig.add_subplot(gs[1, 0])
    avg_duration = stats['games'].get('avg_duration', 0)
    draw_metric_card(ax_duration, f'{avg_duration/60:.1f}m',
                    'Avg Duration', '#ff7f0e')

    # Player Decisions bar
    ax_decisions = fig.add_subplot(gs[1, 1])
    total_decisions = int(stats['decisions'].get('total_decisions', 0))
    decision_types = int(stats['decisions'].get('decision_types', 0))

    ax_decisions.barh([0, 1], [total_decisions, decision_types],
                     color=['#9467bd', '#8c564b'], alpha=0.7)
    ax_decisions.set_yticks([0, 1])
    ax_decisions.set_yticklabels(['Total\nDecisions', 'Decision\nTypes'], fontsize=10)
    ax_decisions.set_xlabel('Count', fontweight='bold')
    ax_decisions.set_title('Player Decisions', fontweight='bold', fontsize=12, pad=10)
    ax_decisions.grid(axis='x', alpha=0.3)

    for i, v in enumerate([total_decisions, decision_types]):
        ax_decisions.text(v + 5, i, str(v), va='center', fontweight='bold')

    # Deception Rate
    ax_deception = fig.add_subplot(gs[1, 2])
    deception_rate = stats['decisions'].get('deception_rate', 0) * 100

    # Donut chart for deception
    sizes = [deception_rate, 100 - deception_rate]
    colors = ['#d62728', '#e0e0e0']

    wedges, texts = ax_deception.pie(sizes, colors=colors, startangle=90,
                                     wedgeprops=dict(width=0.5))
    ax_deception.text(0, 0, f'{deception_rate:.1f}%', ha='center', va='center',
                     fontsize=20, fontweight='bold', color='#d62728')
    ax_deception.set_title('Deception Rate', fontweight='bold', fontsize=12, pad=10)

    # API Efficiency
    ax_api = fig.add_subplot(gs[1, 3])
    total_requests = int(stats['api'].get('total_requests', 0))
    avg_latency = stats['api'].get('avg_latency', 0)

    ax_api.text(0.5, 0.7, str(total_requests), ha='center', va='center',
               fontsize=24, fontweight='bold', transform=ax_api.transAxes)
    ax_api.text(0.5, 0.5, 'API Requests', ha='center', va='center',
               fontsize=11, transform=ax_api.transAxes)

    # Handle None or missing latency data
    latency_text = f'{avg_latency:.2f}s avg latency' if avg_latency else 'N/A'
    ax_api.text(0.5, 0.25, latency_text, ha='center', va='center',
               fontsize=10, color='#555', transform=ax_api.transAxes)
    ax_api.axis('off')
    ax_api.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8,
                                       fill=False, edgecolor='#17becf',
                                       linewidth=3, transform=ax_api.transAxes))

    # === Row 3: Policy & Token Metrics ===

    # Average Policies
    ax_policies = fig.add_subplot(gs[2, 0:2])
    avg_lib = stats['games'].get('avg_lib_policies', 0)
    avg_fasc = stats['games'].get('avg_fasc_policies', 0)

    x = np.arange(2)
    width = 0.6
    bars = ax_policies.bar(x, [avg_lib, avg_fasc], width,
                          color=['#1f77b4', '#d62728'], alpha=0.7,
                          edgecolor='black', linewidth=2)

    ax_policies.set_xticks(x)
    ax_policies.set_xticklabels(['Liberal\nPolicies', 'Fascist\nPolicies'],
                               fontweight='bold')
    ax_policies.set_ylabel('Average per Game', fontweight='bold')
    ax_policies.set_title('Policy Enactment', fontweight='bold', fontsize=14, pad=10)
    ax_policies.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, [avg_lib, avg_fasc]):
        height = bar.get_height()
        ax_policies.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=12)

    # Token Usage
    ax_tokens = fig.add_subplot(gs[2, 2:])
    total_tokens = int(stats['api'].get('total_tokens', 0))
    avg_cost = stats['games'].get('avg_cost', 0)
    tokens_per_dollar = total_tokens / total_cost if total_cost > 0 else 0

    metrics_text = [
        f'Total Tokens: {total_tokens:,}',
        f'Avg Cost/Game: ${avg_cost:.4f}',
        f'Tokens/$: {tokens_per_dollar:,.0f}'
    ]

    ax_tokens.axis('off')
    ax_tokens.set_xlim(0, 1)
    ax_tokens.set_ylim(0, 1)

    for i, text in enumerate(metrics_text):
        y_pos = 0.7 - (i * 0.25)
        ax_tokens.text(0.5, y_pos, text, ha='center', va='center',
                      fontsize=13, fontweight='bold',
                      transform=ax_tokens.transAxes,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                               edgecolor='#2ca02c', linewidth=2))

    ax_tokens.text(0.5, 0.95, 'Efficiency Metrics', ha='center', va='top',
                  fontsize=14, fontweight='bold', transform=ax_tokens.transAxes)

    # Footer
    footer_text = (
        'Framework: Multi-agent strategic deception evaluation using Secret Hitler | '
        'Model: DeepSeek V3.2 Exp | '
        'Database: SQLite with comprehensive logging'
    )

    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9,
            style='italic', color='#666')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
    print(f"✓ Batch summary saved to: {output_path}")

    plt.close()


def draw_metric_card(ax, value, label, color):
    """Draw a metric card visualization."""
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Background box
    ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8,
                                   fill=True, facecolor='white',
                                   edgecolor=color, linewidth=3))

    # Value
    ax.text(0.5, 0.6, str(value), ha='center', va='center',
           fontsize=28, fontweight='bold', color=color,
           transform=ax.transAxes)

    # Label
    ax.text(0.5, 0.3, label, ha='center', va='center',
           fontsize=12, fontweight='bold',
           transform=ax.transAxes)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create batch summary infographic"
    )
    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )
    parser.add_argument(
        '--output', '-o',
        default='visualizations/batch_summary.png',
        help='Output path for PNG'
    )
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=10,
        help='Number of recent games to include (default: 10)'
    )
    parser.add_argument(
        '--name', '-n',
        default='Batch Evaluation Summary',
        help='Batch name for title'
    )
    parser.add_argument(
        '--total-cost',
        type=float,
        help='Override total cost value (use when DB has incorrect cost)'
    )

    args = parser.parse_args()

    try:
        conn = connect_db(args.db)

        print(f"Generating summary for {args.games} games...")
        stats = get_batch_statistics(conn, args.games)

        # Override total cost if specified (for when DB has incorrect values)
        if args.total_cost is not None:
            stats['games']['total_cost'] = args.total_cost
            stats['games']['avg_cost'] = args.total_cost / stats['games'].get('total_games', 1)
            print(f"Using override cost: ${args.total_cost:.2f}")

        create_batch_summary(stats, args.output, args.name)

        conn.close()

        print("\nBatch summary complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
