#!/usr/bin/env python3
"""
Create portfolio-quality visualizations for Secret Hitler LLM evaluation.
Generates publication-ready figures for research presentation.
"""
import sys
import argparse
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.database_schema import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
sns.set_palette("husl")


class PortfolioVisualizer:
    """Creates portfolio-quality visualizations from game database."""

    def __init__(self, db_path: str = "data/games.db", output_dir: str = "visualizations"):
        # Use DatabaseManager to get correct absolute path
        self.db = DatabaseManager(db_path)
        self.db_path = self.db.db_path  # Get resolved absolute path from DatabaseManager

        # Resolve output_dir relative to project root (llm-game-engine/)
        project_root = Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if Path(output_dir).is_absolute() else project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        logger.info(f"Loading data from {self.db_path}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant data into DataFrames."""
        conn = sqlite3.connect(str(self.db_path))

        data = {
            'games': pd.read_sql_query("SELECT * FROM games", conn),
            'decisions': pd.read_sql_query("SELECT * FROM player_decisions", conn),
            'api_requests': pd.read_sql_query("SELECT * FROM api_requests", conn)
        }

        conn.close()

        logger.info(f"Loaded {len(data['games'])} games, {len(data['decisions'])} decisions")
        return data

    def create_all_visualizations(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """Create all portfolio visualizations."""
        created_files = {}

        logger.info("Creating visualizations...")

        # 1. Game Outcomes Overview
        if len(data['games']) > 0:
            created_files['outcomes'] = self._create_outcomes_visual(data['games'])

        # 2. Deception Analysis Heatmap
        if len(data['decisions']) > 0:
            created_files['deception_heatmap'] = self._create_deception_heatmap(data['decisions'])

        # 3. Cost Efficiency Dashboard
        if len(data['games']) > 0 and len(data['api_requests']) > 0:
            created_files['cost_dashboard'] = self._create_cost_dashboard(data)

        # 4. Strategic Patterns
        if len(data['decisions']) > 0:
            created_files['strategic_patterns'] = self._create_strategic_patterns(data['decisions'])

        # 5. Model Performance Comparison (if multiple models)
        if len(data['api_requests']) > 0:
            models = data['api_requests']['model'].unique()
            if len(models) > 1:
                created_files['model_comparison'] = self._create_model_comparison(data)

        logger.info(f"Created {len(created_files)} visualization files")
        return created_files

    def _create_outcomes_visual(self, games_df: pd.DataFrame) -> Path:
        """Create comprehensive game outcomes visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Win Distribution
        winner_counts = games_df['winning_team'].value_counts()
        colors = ['#3498db', '#e74c3c']  # Blue for liberal, red for fascist
        ax1.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax1.set_title('Win Distribution by Team', fontweight='bold')

        # 2. Win Conditions
        win_conditions = games_df['win_condition'].value_counts()
        ax2.barh(range(len(win_conditions)), win_conditions.values, color='#2ecc71')
        ax2.set_yticks(range(len(win_conditions)))
        ax2.set_yticklabels([cond[:40] + '...' if len(cond) > 40 else cond
                             for cond in win_conditions.index])
        ax2.set_xlabel('Count')
        ax2.set_title('Win Conditions Distribution', fontweight='bold')
        ax2.invert_yaxis()

        # 3. Policies Enacted Distribution (filter out unknown winners)
        games_with_winner = games_df[games_df['winning_team'].isin(['liberal', 'fascist'])]
        if len(games_with_winner) > 0:
            colors = games_with_winner['winning_team'].map({'liberal': '#3498db', 'fascist': '#e74c3c'})
            ax3.scatter(games_with_winner['liberal_policies'], games_with_winner['fascist_policies'],
                       alpha=0.6, s=100, c=colors)
        ax3.set_xlabel('Liberal Policies')
        ax3.set_ylabel('Fascist Policies')
        ax3.set_title('Policy Board States at Game End', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(['Liberal Win', 'Fascist Win'])

        # 4. Game Duration Distribution
        ax4.hist(games_df['duration_seconds'], bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.axvline(games_df['duration_seconds'].mean(), color='red', linestyle='--',
                   label=f'Mean: {games_df["duration_seconds"].mean():.0f}s')
        ax4.set_xlabel('Duration (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Game Duration Distribution', fontweight='bold')
        ax4.legend()

        plt.tight_layout()
        output_file = self.output_dir / "game_outcomes.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Created: {output_file}")
        return output_file

    def _create_deception_heatmap(self, decisions_df: pd.DataFrame) -> Path:
        """Create deception frequency heatmap."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Filter decisions with deception data
        deception_df = decisions_df[decisions_df['is_deception'].notna()].copy()

        if len(deception_df) == 0:
            logger.warning("No deception data available")
            plt.close()
            return None

        # 1. Deception by Player and Decision Type
        pivot_data = deception_df.groupby(['player_name', 'decision_type'])['is_deception'].mean().unstack(fill_value=0)

        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Deception Rate'})
        ax1.set_title('Deception Rate by Player and Decision Type', fontweight='bold')
        ax1.set_xlabel('Decision Type')
        ax1.set_ylabel('Player')

        # 2. Deception Score Distribution
        deception_scores = deception_df[deception_df['is_deception'] == 1]['deception_score']
        if len(deception_scores) > 0:
            ax2.hist(deception_scores, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Deception Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Deception Severity Distribution', fontweight='bold')
            ax2.axvline(deception_scores.mean(), color='darkred', linestyle='--',
                       label=f'Mean: {deception_scores.mean():.2f}')
            ax2.legend()

        plt.tight_layout()
        output_file = self.output_dir / "deception_analysis.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Created: {output_file}")
        return output_file

    def _create_cost_dashboard(self, data: Dict[str, pd.DataFrame]) -> Path:
        """Create comprehensive cost analysis dashboard."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        games_df = data['games']
        api_df = data['api_requests']

        # 1. Cost per Game
        ax1 = fig.add_subplot(gs[0, :2])
        games_df_sorted = games_df.sort_values('timestamp')
        ax1.plot(range(len(games_df_sorted)), games_df_sorted['total_cost'],
                marker='o', linestyle='-', alpha=0.6, color='#3498db')
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Cost ($)')
        ax1.set_title('Cost per Game Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add rolling average
        if len(games_df) >= 5:
            rolling_avg = games_df_sorted['total_cost'].rolling(window=5).mean()
            ax1.plot(range(len(games_df_sorted)), rolling_avg,
                    color='red', linestyle='--', label='5-game moving average')
            ax1.legend()

        # 2. Cost Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.boxplot(games_df['total_cost'], vert=True)
        ax2.set_ylabel('Cost ($)')
        ax2.set_title('Cost Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Cost by Model
        ax3 = fig.add_subplot(gs[1, :])
        cost_by_model = api_df.groupby('model')['cost'].agg(['sum', 'mean', 'count'])
        cost_by_model = cost_by_model.sort_values('sum', ascending=False)

        x = range(len(cost_by_model))
        width = 0.35
        ax3.bar([i - width/2 for i in x], cost_by_model['sum'], width,
               label='Total Cost', color='#3498db')
        ax3_twin = ax3.twinx()
        ax3_twin.bar([i + width/2 for i in x], cost_by_model['count'], width,
                    label='Request Count', color='#2ecc71', alpha=0.7)

        ax3.set_xlabel('Model')
        ax3.set_ylabel('Total Cost ($)', color='#3498db')
        ax3_twin.set_ylabel('Request Count', color='#2ecc71')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cost_by_model.index, rotation=45, ha='right')
        ax3.set_title('Cost and Usage by Model', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Cost per Decision Type
        ax4 = fig.add_subplot(gs[2, :2])
        cost_by_decision = api_df.groupby('decision_type')['cost'].mean().sort_values(ascending=False)
        ax4.barh(range(len(cost_by_decision)), cost_by_decision.values, color='#9b59b6')
        ax4.set_yticks(range(len(cost_by_decision)))
        ax4.set_yticklabels(cost_by_decision.index)
        ax4.set_xlabel('Average Cost per Request ($)')
        ax4.set_title('Cost Efficiency by Decision Type', fontweight='bold')
        ax4.invert_yaxis()

        # 5. Cost Efficiency Metrics
        ax5 = fig.add_subplot(gs[2, 2])
        total_cost = games_df['total_cost'].sum()
        total_games = len(games_df)
        total_requests = len(api_df)

        metrics_text = f"""
        Cost Metrics:

        Total Cost: ${total_cost:.2f}

        Per Game: ${total_cost/total_games:.3f}

        Per Request: ${total_cost/total_requests:.4f}

        Games/Dollar: {total_games/total_cost:.1f}
        """

        ax5.text(0.1, 0.5, metrics_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax5.axis('off')

        output_file = self.output_dir / "cost_analysis.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Created: {output_file}")
        return output_file

    def _create_strategic_patterns(self, decisions_df: pd.DataFrame) -> Path:
        """Analyze strategic decision patterns."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Decision Type Distribution
        decision_counts = decisions_df['decision_type'].value_counts()
        ax1.bar(range(len(decision_counts)), decision_counts.values, color='#3498db')
        ax1.set_xticks(range(len(decision_counts)))
        ax1.set_xticklabels(decision_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Count')
        ax1.set_title('Decision Type Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Deception Rate by Decision Type
        deception_by_type = decisions_df.groupby('decision_type')['is_deception'].mean().sort_values(ascending=False)
        colors = ['#e74c3c' if rate > 0.3 else '#f39c12' if rate > 0.15 else '#2ecc71'
                 for rate in deception_by_type.values]
        ax2.barh(range(len(deception_by_type)), deception_by_type.values, color=colors)
        ax2.set_yticks(range(len(deception_by_type)))
        ax2.set_yticklabels(deception_by_type.index)
        ax2.set_xlabel('Deception Rate')
        ax2.set_title('Strategic Deception by Decision Type', fontweight='bold')
        ax2.invert_yaxis()
        ax2.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='High deception threshold')
        ax2.legend()

        # 3. Player Activity Distribution
        player_activity = decisions_df['player_name'].value_counts()
        ax3.bar(range(len(player_activity)), player_activity.values, color='#9b59b6')
        ax3.set_xticks(range(len(player_activity)))
        ax3.set_xticklabels(player_activity.index, rotation=45, ha='right')
        ax3.set_ylabel('Decision Count')
        ax3.set_title('Player Decision Frequency', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Confidence Distribution
        if 'confidence' in decisions_df.columns:
            confidence_data = decisions_df[decisions_df['confidence'].notna()]['confidence']
            if len(confidence_data) > 0:
                ax4.hist(confidence_data, bins=20, color='#1abc9c', alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Confidence Score')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Decision Confidence Distribution', fontweight='bold')
                ax4.axvline(confidence_data.mean(), color='darkgreen', linestyle='--',
                           label=f'Mean: {confidence_data.mean():.2f}')
                ax4.legend()

        plt.tight_layout()
        output_file = self.output_dir / "strategic_patterns.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Created: {output_file}")
        return output_file

    def _create_model_comparison(self, data: Dict[str, pd.DataFrame]) -> Path:
        """Compare performance across different models."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        api_df = data['api_requests']
        games_df = data['games']

        # 1. Cost per Model
        cost_by_model = api_df.groupby('model').agg({
            'cost': ['sum', 'mean', 'std'],
            'tokens': 'sum',
            'latency': 'mean'
        })

        models = cost_by_model.index
        x = range(len(models))

        ax1.bar(x, cost_by_model[('cost', 'sum')], color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Total Cost ($)')
        ax1.set_title('Total Cost by Model', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Average Latency
        ax2.bar(x, cost_by_model[('latency', 'mean')], color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Average Latency (s)')
        ax2.set_title('Response Time by Model', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Token Efficiency
        cost_per_token = cost_by_model[('cost', 'sum')] / cost_by_model[('tokens', 'sum')]
        ax3.bar(x, cost_per_token * 1000000, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('Cost per Million Tokens ($)')
        ax3.set_title('Cost Efficiency by Model', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Usage Distribution
        request_counts = api_df['model'].value_counts()
        ax4.pie(request_counts.values, labels=request_counts.index, autopct='%1.1f%%', startangle=90)
        ax4.set_title('API Request Distribution by Model', fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / "model_comparison.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Created: {output_file}")
        return output_file

    def generate_summary_stats(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate summary statistics for README."""
        games_df = data['games']
        decisions_df = data['decisions']
        api_df = data['api_requests']

        stats = {
            'total_games': len(games_df),
            'total_decisions': len(decisions_df),
            'total_api_requests': len(api_df),
            'total_cost': float(games_df['total_cost'].sum()),
            'avg_cost_per_game': float(games_df['total_cost'].mean()),
            'liberal_wins': int((games_df['winning_team'] == 'liberal').sum()),
            'fascist_wins': int((games_df['winning_team'] == 'fascist').sum()),
            'avg_game_duration': float(games_df['duration_seconds'].mean()),
            'deception_rate': float(decisions_df['is_deception'].mean()) if len(decisions_df) > 0 else 0,
            'unique_models': list(api_df['model'].unique())
        }

        stats['liberal_win_rate'] = stats['liberal_wins'] / max(stats['total_games'], 1)
        stats['fascist_win_rate'] = stats['fascist_wins'] / max(stats['total_games'], 1)

        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create portfolio-quality visualizations for Secret Hitler LLM evaluation"
    )
    parser.add_argument(
        '--db-path',
        default='data/games.db',
        help='Path to SQLite database (default: data/games.db)'
    )
    parser.add_argument(
        '--output-dir',
        default='visualizations',
        help='Output directory for visualizations (default: visualizations/)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary statistics JSON'
    )

    args = parser.parse_args()

    try:
        visualizer = PortfolioVisualizer(args.db_path, args.output_dir)
        data = visualizer.load_data()

        if len(data['games']) == 0:
            logger.error("No games found in database")
            return 1

        # Create visualizations
        files = visualizer.create_all_visualizations(data)

        logger.info(f"\n✓ Created {len(files)} visualizations in {args.output_dir}/")
        for name, path in files.items():
            logger.info(f"  - {name}: {path.name}")

        # Generate summary stats if requested
        if args.summary:
            stats = visualizer.generate_summary_stats(data)
            stats_file = Path(args.output_dir) / "summary_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"\n✓ Summary statistics saved to {stats_file}")

            # Print key stats
            print(f"\n{'='*60}")
            print("Summary Statistics")
            print(f"{'='*60}")
            print(f"Total Games: {stats['total_games']}")
            print(f"Total Cost: ${stats['total_cost']:.2f}")
            print(f"Average Cost/Game: ${stats['avg_cost_per_game']:.3f}")
            print(f"Liberal Win Rate: {stats['liberal_win_rate']:.1%}")
            print(f"Fascist Win Rate: {stats['fascist_win_rate']:.1%}")
            print(f"Deception Rate: {stats['deception_rate']:.1%}")
            print(f"Avg Game Duration: {stats['avg_game_duration']:.1f}s")
            print(f"{'='*60}\n")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
