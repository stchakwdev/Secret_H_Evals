"""
Analytics module for processing Secret Hitler LLM experiment results.
Provides statistical analysis, visualization, and research insights.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class ExperimentAnalyzer:
    """Analyzes results from Secret Hitler LLM experiments."""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.report_path = self.experiment_dir / "final_report.json"
        self.config_path = self.experiment_dir / "experiment_config.json"
        
        if not self.report_path.exists():
            raise FileNotFoundError(f"No final report found at {self.report_path}")
        
        self.report = self._load_report()
        self.config = self._load_config()
        self.games_df = self._create_games_dataframe()
    
    def _load_report(self) -> Dict[str, Any]:
        """Load the experiment report."""
        with open(self.report_path, 'r') as f:
            return json.load(f)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the experiment configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _create_games_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from game results."""
        successful_games = [
            game for game in self.report['raw_results'] 
            if game['success']
        ]
        
        if not successful_games:
            return pd.DataFrame()
        
        # Extract key metrics for each game
        data = []
        for game in successful_games:
            game_data = {
                'game_id': game['game_id'],
                'game_number': game['game_number'],
                'winner': game['winner'],
                'win_condition': game['win_condition'],
                'duration': game['duration'],
                'cost': game['cost'],
                'timestamp': game['timestamp']
            }
            
            # Extract final state metrics
            final_state = game.get('final_state', {})
            game_data['liberal_policies'] = final_state.get('policy_board', {}).get('liberal_policies', 0)
            game_data['fascist_policies'] = final_state.get('policy_board', {}).get('fascist_policies', 0)
            game_data['total_policies'] = game_data['liberal_policies'] + game_data['fascist_policies']
            
            # Extract cost summary
            cost_summary = game.get('cost_summary', {})
            game_data['total_requests'] = cost_summary.get('total_requests', 0)
            game_data['avg_latency'] = cost_summary.get('avg_latency', 0)
            
            data.append(game_data)
        
        return pd.DataFrame(data)
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if self.games_df.empty:
            return {"error": "No successful games to analyze"}
        
        analysis = {
            'experiment_info': self._get_experiment_info(),
            'basic_statistics': self._calculate_basic_stats(),
            'win_analysis': self._analyze_win_patterns(),
            'cost_analysis': self._analyze_costs(),
            'performance_analysis': self._analyze_performance(),
            'temporal_analysis': self._analyze_temporal_patterns(),
            'statistical_tests': self._perform_statistical_tests()
        }
        
        return analysis
    
    def _get_experiment_info(self) -> Dict[str, Any]:
        """Get basic experiment information."""
        return {
            'experiment_id': self.report['experiment_summary']['experiment_id'],
            'name': self.report['experiment_summary']['name'],
            'description': self.report['experiment_summary']['description'],
            'total_games': len(self.games_df),
            'success_rate': self.report['game_statistics']['success_rate'],
            'total_duration': self.report['experiment_summary']['duration_seconds'],
            'total_cost': self.report['game_statistics']['total_cost']
        }
    
    def _calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic statistical measures."""
        if self.games_df.empty:
            return {}
        
        numeric_cols = ['duration', 'cost', 'total_requests', 'avg_latency', 'total_policies']
        stats_dict = {}
        
        for col in numeric_cols:
            if col in self.games_df.columns:
                stats_dict[col] = {
                    'mean': float(self.games_df[col].mean()),
                    'median': float(self.games_df[col].median()),
                    'std': float(self.games_df[col].std()),
                    'min': float(self.games_df[col].min()),
                    'max': float(self.games_df[col].max()),
                    'q25': float(self.games_df[col].quantile(0.25)),
                    'q75': float(self.games_df[col].quantile(0.75))
                }
        
        return stats_dict
    
    def _analyze_win_patterns(self) -> Dict[str, Any]:
        """Analyze win patterns and conditions."""
        winner_counts = self.games_df['winner'].value_counts()
        win_condition_counts = self.games_df['win_condition'].value_counts()
        
        # Analyze win conditions by winner
        win_cross_tab = pd.crosstab(self.games_df['winner'], self.games_df['win_condition'])
        
        # Calculate win rates
        total_games = len(self.games_df)
        
        analysis = {
            'winner_distribution': winner_counts.to_dict(),
            'win_condition_distribution': win_condition_counts.to_dict(),
            'win_rates': {
                winner: count / total_games 
                for winner, count in winner_counts.items()
            },
            'win_condition_by_winner': win_cross_tab.to_dict(),
            'average_policies_per_game': {
                'liberal': float(self.games_df['liberal_policies'].mean()),
                'fascist': float(self.games_df['fascist_policies'].mean()),
                'total': float(self.games_df['total_policies'].mean())
            }
        }
        
        return analysis
    
    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost patterns and efficiency."""
        cost_analysis = {
            'total_cost': float(self.games_df['cost'].sum()),
            'cost_per_game': {
                'mean': float(self.games_df['cost'].mean()),
                'median': float(self.games_df['cost'].median()),
                'std': float(self.games_df['cost'].std())
            },
            'cost_by_winner': self.games_df.groupby('winner')['cost'].agg(['mean', 'std']).to_dict(),
            'cost_efficiency': {
                'cost_per_policy': float(self.games_df['cost'].sum() / self.games_df['total_policies'].sum()),
                'cost_per_request': float(self.games_df['cost'].sum() / self.games_df['total_requests'].sum()),
                'games_per_dollar': len(self.games_df) / float(self.games_df['cost'].sum())
            }
        }
        
        # Cost correlation analysis
        if len(self.games_df) > 3:
            cost_corr = self.games_df[['cost', 'duration', 'total_policies', 'total_requests']].corr()['cost']
            cost_analysis['cost_correlations'] = cost_corr.drop('cost').to_dict()
        
        return cost_analysis
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze game performance metrics."""
        performance = {
            'game_duration': {
                'mean_seconds': float(self.games_df['duration'].mean()),
                'median_seconds': float(self.games_df['duration'].median()),
                'duration_by_winner': self.games_df.groupby('winner')['duration'].mean().to_dict()
            },
            'api_performance': {
                'avg_requests_per_game': float(self.games_df['total_requests'].mean()),
                'avg_latency': float(self.games_df['avg_latency'].mean()),
                'latency_std': float(self.games_df['avg_latency'].std())
            },
            'game_length': {
                'avg_policies_enacted': float(self.games_df['total_policies'].mean()),
                'policy_distribution': self.games_df['total_policies'].value_counts().to_dict(),
                'policies_by_winner': self.games_df.groupby('winner')['total_policies'].mean().to_dict()
            }
        }
        
        return performance
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns over time during the experiment."""
        if 'timestamp' not in self.games_df.columns:
            return {}
        
        self.games_df['timestamp'] = pd.to_datetime(self.games_df['timestamp'])
        self.games_df = self.games_df.sort_values('timestamp')
        
        # Rolling averages
        window_size = min(5, len(self.games_df) // 2)
        if window_size >= 2:
            rolling_cost = self.games_df['cost'].rolling(window=window_size).mean()
            rolling_duration = self.games_df['duration'].rolling(window=window_size).mean()
            
            temporal_analysis = {
                'trends': {
                    'cost_trend': 'stable',  # Simple trend detection
                    'duration_trend': 'stable',
                    'cost_slope': float(np.polyfit(range(len(self.games_df)), self.games_df['cost'], 1)[0]),
                    'duration_slope': float(np.polyfit(range(len(self.games_df)), self.games_df['duration'], 1)[0])
                },
                'experiment_progression': {
                    'first_half_avg_cost': float(self.games_df.iloc[:len(self.games_df)//2]['cost'].mean()),
                    'second_half_avg_cost': float(self.games_df.iloc[len(self.games_df)//2:]['cost'].mean()),
                    'first_half_avg_duration': float(self.games_df.iloc[:len(self.games_df)//2]['duration'].mean()),
                    'second_half_avg_duration': float(self.games_df.iloc[len(self.games_df)//2:]['duration'].mean())
                }
            }
            
            return temporal_analysis
        
        return {}
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        # Test if liberal/fascist win rates differ from 50%
        if 'winner' in self.games_df.columns:
            liberal_wins = (self.games_df['winner'] == 'liberal').sum()
            total_games = len(self.games_df)
            
            # Binomial test for win rate
            p_value = stats.binom_test(liberal_wins, total_games, 0.5)
            tests['liberal_win_rate_test'] = {
                'liberal_wins': int(liberal_wins),
                'total_games': int(total_games),
                'liberal_win_rate': float(liberal_wins / total_games),
                'null_hypothesis': 'Win rate = 50%',
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        # Test for differences in cost by winner
        if len(self.games_df) > 10:
            liberal_costs = self.games_df[self.games_df['winner'] == 'liberal']['cost']
            fascist_costs = self.games_df[self.games_df['winner'] == 'fascist']['cost']
            
            if len(liberal_costs) > 0 and len(fascist_costs) > 0:
                t_stat, p_value = stats.ttest_ind(liberal_costs, fascist_costs)
                tests['cost_by_winner_test'] = {
                    'test': 't-test',
                    'null_hypothesis': 'No difference in cost by winner',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'liberal_mean_cost': float(liberal_costs.mean()),
                    'fascist_mean_cost': float(fascist_costs.mean())
                }
        
        return tests
    
    def create_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Create visualization plots and save them."""
        if output_dir is None:
            output_dir = self.experiment_dir / "visualizations"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        created_files = {}
        
        # 1. Win rate distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Winner distribution pie chart
        winner_counts = self.games_df['winner'].value_counts()
        ax1.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Winner Distribution')
        
        # Win condition distribution
        win_condition_counts = self.games_df['win_condition'].value_counts()
        ax2.bar(win_condition_counts.index, win_condition_counts.values)
        ax2.set_title('Win Conditions')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = output_dir / "win_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['win_analysis'] = str(plot_file)
        
        # 2. Cost and performance metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cost distribution
        ax1.hist(self.games_df['cost'], bins=10, alpha=0.7, edgecolor='black')
        ax1.set_title('Cost Distribution')
        ax1.set_xlabel('Cost ($)')
        ax1.set_ylabel('Frequency')
        
        # Duration distribution
        ax2.hist(self.games_df['duration'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_title('Game Duration Distribution')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Frequency')
        
        # Cost vs Duration scatter
        ax3.scatter(self.games_df['duration'], self.games_df['cost'], alpha=0.6)
        ax3.set_title('Cost vs Duration')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Cost ($)')
        
        # Policies enacted distribution
        ax4.hist(self.games_df['total_policies'], bins=range(int(self.games_df['total_policies'].max()) + 2), 
                alpha=0.7, color='green', edgecolor='black')
        ax4.set_title('Total Policies Enacted')
        ax4.set_xlabel('Number of Policies')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plot_file = output_dir / "performance_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        created_files['performance_metrics'] = str(plot_file)
        
        # 3. Box plots by winner
        if len(self.games_df['winner'].unique()) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Cost by winner
            self.games_df.boxplot(column='cost', by='winner', ax=ax1)
            ax1.set_title('Cost by Winner')
            ax1.set_xlabel('Winner')
            ax1.set_ylabel('Cost ($)')
            
            # Duration by winner
            self.games_df.boxplot(column='duration', by='winner', ax=ax2)
            ax2.set_title('Duration by Winner')
            ax2.set_xlabel('Winner')
            ax2.set_ylabel('Duration (seconds)')
            
            plt.tight_layout()
            plot_file = output_dir / "winner_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files['winner_comparison'] = str(plot_file)
        
        logger.info(f"Visualizations saved to {output_dir}")
        return created_files
    
    def export_csv(self, output_file: Optional[str] = None) -> str:
        """Export game data to CSV for external analysis."""
        if output_file is None:
            output_file = self.experiment_dir / "games_data.csv"
        
        self.games_df.to_csv(output_file, index=False)
        logger.info(f"Game data exported to {output_file}")
        return str(output_file)

def compare_experiments(experiment_dirs: List[str]) -> Dict[str, Any]:
    """Compare multiple experiments."""
    analyzers = [ExperimentAnalyzer(exp_dir) for exp_dir in experiment_dirs]
    
    comparison = {
        'experiments': [],
        'comparative_analysis': {}
    }
    
    # Collect basic stats for each experiment
    for analyzer in analyzers:
        exp_info = analyzer._get_experiment_info()
        win_analysis = analyzer._analyze_win_patterns()
        cost_analysis = analyzer._analyze_costs()
        
        comparison['experiments'].append({
            'experiment_id': exp_info['experiment_id'],
            'name': exp_info['name'],
            'total_games': exp_info['total_games'],
            'liberal_win_rate': win_analysis['win_rates'].get('liberal', 0),
            'fascist_win_rate': win_analysis['win_rates'].get('fascist', 0),
            'avg_cost_per_game': cost_analysis['cost_per_game']['mean'],
            'total_cost': cost_analysis['total_cost']
        })
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison['experiments'])
    
    if len(comp_df) > 1:
        comparison['comparative_analysis'] = {
            'win_rate_variance': {
                'liberal_std': float(comp_df['liberal_win_rate'].std()),
                'fascist_std': float(comp_df['fascist_win_rate'].std())
            },
            'cost_variance': {
                'cost_per_game_std': float(comp_df['avg_cost_per_game'].std()),
                'cost_range': float(comp_df['avg_cost_per_game'].max() - comp_df['avg_cost_per_game'].min())
            },
            'best_experiment': {
                'lowest_cost': comp_df.loc[comp_df['avg_cost_per_game'].idxmin(), 'name'],
                'most_balanced': comp_df.loc[(abs(comp_df['liberal_win_rate'] - 0.5)).idxmin(), 'name']
            }
        }
    
    return comparison

def main():
    """CLI for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Secret Hitler LLM experiment results")
    parser.add_argument('experiment_dir', help='Path to experiment results directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization plots')
    parser.add_argument('--export-csv', action='store_true', help='Export data to CSV')
    parser.add_argument('--output', type=str, help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ExperimentAnalyzer(args.experiment_dir)
    analysis = analyzer.generate_analysis_report()
    
    # Save analysis report
    output_dir = Path(args.output) if args.output else Path(args.experiment_dir)
    analysis_file = output_dir / "analysis_report.json"
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis report saved to: {analysis_file}")
    
    # Create visualizations if requested
    if args.visualize:
        viz_files = analyzer.create_visualizations(output_dir / "visualizations")
        print(f"Visualizations created: {list(viz_files.keys())}")
    
    # Export CSV if requested
    if args.export_csv:
        csv_file = analyzer.export_csv(output_dir / "games_data.csv")
        print(f"Data exported to: {csv_file}")
    
    # Print summary
    exp_info = analysis['experiment_info']
    win_analysis = analysis['win_analysis']
    
    print(f"\nExperiment Summary:")
    print(f"Name: {exp_info['name']}")
    print(f"Games: {exp_info['total_games']}")
    print(f"Success Rate: {exp_info['success_rate']:.1%}")
    print(f"Total Cost: ${exp_info['total_cost']:.4f}")
    print(f"Liberal Win Rate: {win_analysis['win_rates'].get('liberal', 0):.1%}")
    print(f"Fascist Win Rate: {win_analysis['win_rates'].get('fascist', 0):.1%}")

if __name__ == "__main__":
    main()