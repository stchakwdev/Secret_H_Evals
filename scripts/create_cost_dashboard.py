#!/usr/bin/env python3
"""
Cost Efficiency Dashboard Visualization

Creates comprehensive dashboard showing API costs, latency, token usage,
and efficiency metrics across games. Demonstrates cost-effectiveness of
the evaluation framework.

Author: Samuel Chakwera (stchakdev)
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
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


def get_cost_data(conn: sqlite3.Connection, game_limit: int = 10) -> pd.DataFrame:
    """
    Extract cost and performance data from API requests.

    Args:
        conn: Database connection
        game_limit: Number of recent games to analyze

    Returns:
        DataFrame with cost, tokens, latency data
    """
    query = """
        SELECT
            ar.game_id,
            ar.model,
            ar.decision_type,
            ar.cost,
            ar.tokens,
            ar.latency,
            ar.timestamp,
            g.winner,
            g.duration_seconds
        FROM api_requests ar
        JOIN games g ON ar.game_id = g.game_id
        WHERE ar.game_id IN (
            SELECT game_id
            FROM games
            ORDER BY timestamp DESC
            LIMIT ?
        )
        ORDER BY ar.timestamp
    """

    df = pd.read_sql_query(query, conn, params=(game_limit,))
    return df


def create_cost_dashboard(df: pd.DataFrame, output_path: str):
    """
    Create cost efficiency dashboard.

    Args:
        df: DataFrame with cost data
        output_path: Path to save PNG
    """
    if df.empty:
        print("No cost data found")
        return

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])  # Cost over time
    ax2 = fig.add_subplot(gs[0, 2])   # Total cost by model
    ax3 = fig.add_subplot(gs[1, 0])   # Cost by decision type
    ax4 = fig.add_subplot(gs[1, 1])   # Latency distribution
    ax5 = fig.add_subplot(gs[1, 2])   # Tokens vs Cost
    ax6 = fig.add_subplot(gs[2, :])   # Cost efficiency metrics

    # Color palette
    colors = sns.color_palette("husl", n_colors=len(df['model'].unique()))
    model_colors = dict(zip(df['model'].unique(), colors))

    # 1. Cost over time (cumulative)
    df['cumulative_cost'] = df.groupby('game_id')['cost'].cumsum()

    for game_id, game_df in df.groupby('game_id'):
        ax1.plot(range(len(game_df)), game_df['cumulative_cost'].values,
                alpha=0.6, linewidth=2, label=f"Game {game_id[:8]}...")

    ax1.set_xlabel('API Request Number', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Cumulative Cost ($)', fontweight='bold', fontsize=11)
    ax1.set_title('Cost Accumulation Over Game', fontweight='bold', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left', fontsize=8)

    # 2. Total cost by model
    model_costs = df.groupby('model')['cost'].sum().sort_values(ascending=False)

    bars = ax2.barh(range(len(model_costs)), model_costs.values,
                    color=[model_colors.get(m, '#808080') for m in model_costs.index])
    ax2.set_yticks(range(len(model_costs)))
    ax2.set_yticklabels([m.split('/')[-1][:15] for m in model_costs.index], fontsize=9)
    ax2.set_xlabel('Total Cost ($)', fontweight='bold', fontsize=10)
    ax2.set_title('Cost by Model', fontweight='bold', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    for i, (model, cost) in enumerate(model_costs.items()):
        ax2.text(cost + 0.001, i, f'${cost:.4f}', va='center', fontsize=8)

    # 3. Cost by decision type
    decision_costs = df.groupby('decision_type')['cost'].agg(['sum', 'mean', 'count'])
    decision_costs = decision_costs[decision_costs['count'] >= 5]  # At least 5 samples
    decision_costs = decision_costs.sort_values('sum', ascending=False).head(10)

    ax3.bar(range(len(decision_costs)), decision_costs['sum'].values,
           color='#2ca02c', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(decision_costs)))
    ax3.set_xticklabels(decision_costs.index, rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Total Cost ($)', fontweight='bold', fontsize=10)
    ax3.set_title('Cost by Decision Type', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Latency distribution
    latency_data = df[df['latency'] > 0]['latency']

    ax4.hist(latency_data, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax4.axvline(latency_data.median(), color='red', linestyle='--',
               linewidth=2, label=f'Median: {latency_data.median():.2f}s')
    ax4.set_xlabel('Latency (seconds)', fontweight='bold', fontsize=10)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax4.set_title('API Latency Distribution', fontweight='bold', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Tokens vs Cost scatter
    tokens_df = df[df['tokens'] > 0]

    for model in tokens_df['model'].unique():
        model_df = tokens_df[tokens_df['model'] == model]
        ax5.scatter(model_df['tokens'], model_df['cost'],
                   alpha=0.6, s=50, label=model.split('/')[-1][:15],
                   color=model_colors.get(model, '#808080'))

    ax5.set_xlabel('Tokens', fontweight='bold', fontsize=10)
    ax5.set_ylabel('Cost ($)', fontweight='bold', fontsize=10)
    ax5.set_title('Tokens vs Cost', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=8, loc='upper left')
    ax5.grid(alpha=0.3)

    # 6. Cost efficiency metrics
    game_stats = df.groupby('game_id').agg({
        'cost': 'sum',
        'tokens': 'sum',
        'latency': 'mean',
        'duration_seconds': 'first',
        'winner': 'first'
    }).reset_index()

    game_stats['cost_per_minute'] = game_stats['cost'] / (game_stats['duration_seconds'] / 60)
    game_stats['tokens_per_dollar'] = game_stats['tokens'] / game_stats['cost']

    x = np.arange(len(game_stats))
    width = 0.35

    # Normalized metrics for dual y-axis
    ax6_twin = ax6.twinx()

    bars1 = ax6.bar(x - width/2, game_stats['cost'].values, width,
                   label='Total Cost ($)', color='#1f77b4', alpha=0.7)
    bars2 = ax6_twin.bar(x + width/2, game_stats['cost_per_minute'].values, width,
                        label='Cost/Minute ($/min)', color='#ff7f0e', alpha=0.7)

    ax6.set_xlabel('Game', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Total Cost ($)', fontweight='bold', fontsize=11, color='#1f77b4')
    ax6_twin.set_ylabel('Cost per Minute ($/min)', fontweight='bold', fontsize=11, color='#ff7f0e')
    ax6.set_title('Cost Efficiency by Game', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"G{i+1}" for i in range(len(game_stats))], fontsize=9)
    ax6.tick_params(axis='y', labelcolor='#1f77b4')
    ax6_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    ax6.grid(alpha=0.3, axis='y')

    # Add game winners as annotations
    for i, (idx, row) in enumerate(game_stats.iterrows()):
        winner_color = '#1f77b4' if row['winner'] == 'liberal' else '#d62728'
        ax6.text(i, -0.001, row['winner'][0].upper(), ha='center', va='top',
                fontweight='bold', fontsize=10, color=winner_color)

    # Combined legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # Overall title
    fig.suptitle('Cost Efficiency Dashboard - Secret Hitler LLM Evaluation',
                fontsize=16, fontweight='bold')

    # Summary statistics text
    total_cost = df['cost'].sum()
    avg_cost_per_game = game_stats['cost'].mean()
    avg_latency = df['latency'].mean()
    total_tokens = df['tokens'].sum()

    summary_text = (
        f"Summary: {len(game_stats)} games | "
        f"Total Cost: ${total_cost:.4f} | "
        f"Avg Cost/Game: ${avg_cost_per_game:.4f} | "
        f"Avg Latency: {avg_latency:.2f}s | "
        f"Total Tokens: {int(total_tokens):,}"
    )

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Cost dashboard saved to: {output_path}")

    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create cost efficiency dashboard visualization"
    )
    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )
    parser.add_argument(
        '--output', '-o',
        default='visualizations/cost_dashboard.png',
        help='Output path for PNG'
    )
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=10,
        help='Number of recent games to analyze (default: 10)'
    )

    args = parser.parse_args()

    try:
        conn = connect_db(args.db)

        print(f"Analyzing cost data from {args.games} games...")
        df = get_cost_data(conn, args.games)

        if df.empty:
            print("No cost data found in database")
            return 1

        print(f"Found {len(df)} API requests")

        create_cost_dashboard(df, args.output)

        conn.close()

        print("\nCost analysis complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
