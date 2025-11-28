"""
Model Comparator for Multi-Model Evaluation Batches

Orchestrates running comparison batches across multiple models with
proper role balancing, seed management, and result aggregation.

Author: Samuel Chakwera (stchakdev)
"""

import asyncio
import json
import os
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Import model configurations
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_comparison_config import (
    ModelConfig, BatchConfig, ComparisonGroup,
    CURRENT_RUN_MODELS, DEFAULT_BATCH, get_model_by_id
)


class ComparisonStatus(Enum):
    """Status of a comparison run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelResult:
    """Results for a single model in the comparison."""
    model_id: str
    model_name: str
    games_completed: int = 0
    games_failed: int = 0
    liberal_wins: int = 0
    fascist_wins: int = 0
    total_cost: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    avg_game_duration: float = 0.0
    deception_events: int = 0
    game_ids: List[str] = field(default_factory=list)

    @property
    def games_total(self) -> int:
        return self.games_completed + self.games_failed

    @property
    def win_rate_liberal(self) -> float:
        if self.games_completed == 0:
            return 0.0
        return self.liberal_wins / self.games_completed

    @property
    def win_rate_fascist(self) -> float:
        if self.games_completed == 0:
            return 0.0
        return self.fascist_wins / self.games_completed

    @property
    def cost_per_game(self) -> float:
        if self.games_completed == 0:
            return 0.0
        return self.total_cost / self.games_completed

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'win_rate_liberal': self.win_rate_liberal,
            'win_rate_fascist': self.win_rate_fascist,
            'cost_per_game': self.cost_per_game,
        }


@dataclass
class ComparisonProgress:
    """Progress tracking for a comparison batch."""
    comparison_id: str
    batch_name: str
    status: ComparisonStatus = ComparisonStatus.PENDING
    total_games: int = 0
    completed_games: int = 0
    failed_games: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_model: Optional[str] = None
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.completed_games / self.total_games) * 100

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.model_results.values())

    @property
    def elapsed_time(self) -> Optional[float]:
        if not self.start_time:
            return None
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now()
        return (end - start).total_seconds()

    def to_dict(self) -> Dict:
        return {
            'comparison_id': self.comparison_id,
            'batch_name': self.batch_name,
            'status': self.status.value,
            'total_games': self.total_games,
            'completed_games': self.completed_games,
            'failed_games': self.failed_games,
            'progress_percent': self.progress_percent,
            'total_cost': self.total_cost,
            'elapsed_time': self.elapsed_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'current_model': self.current_model,
            'model_results': {k: v.to_dict() for k, v in self.model_results.items()},
            'errors': self.errors,
        }


class ModelComparator:
    """
    Orchestrates multi-model comparison experiments.

    Features:
    - Runs games across multiple models with balanced role distribution
    - Uses fixed seeds for reproducible scenarios
    - Tracks per-model statistics
    - Generates comparison reports
    """

    def __init__(
        self,
        api_key: str,
        batch_config: BatchConfig = DEFAULT_BATCH,
        output_dir: str = "results/comparisons",
        num_players: int = 7,
        enable_db_logging: bool = True,
        seed: Optional[int] = None,
    ):
        self.api_key = api_key
        self.batch_config = batch_config
        self.output_dir = Path(output_dir)
        self.num_players = num_players
        self.enable_db_logging = enable_db_logging
        self.base_seed = seed or random.randint(0, 1_000_000)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize progress
        self.progress = ComparisonProgress(
            comparison_id=f"cmp-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
            batch_name=batch_config.name,
            total_games=batch_config.total_games,
        )

        # Initialize per-model results
        for model in batch_config.models:
            self.progress.model_results[model.openrouter_id] = ModelResult(
                model_id=model.openrouter_id,
                model_name=model.name,
            )

    async def run_comparison(
        self,
        concurrency: int = 3,
        rate_limit: int = 60,
        resume: bool = False,
    ) -> ComparisonProgress:
        """
        Run the full comparison batch.

        Args:
            concurrency: Number of concurrent games
            rate_limit: API requests per minute limit
            resume: Whether to resume from previous progress

        Returns:
            ComparisonProgress with results
        """
        from core.game_manager import GameManager

        self.progress.status = ComparisonStatus.RUNNING
        self.progress.start_time = datetime.now().isoformat()

        try:
            # Run games for each model
            for model in self.batch_config.models:
                self.progress.current_model = model.name
                print(f"\n{'='*60}")
                print(f"Running games for: {model.name}")
                print(f"OpenRouter ID: {model.openrouter_id}")
                print(f"Games: {self.batch_config.games_per_model}")
                print(f"{'='*60}\n")

                model_result = self.progress.model_results[model.openrouter_id]

                for game_idx in range(self.batch_config.games_per_model):
                    # Generate deterministic seed for this game
                    game_seed = self.base_seed + hash(model.openrouter_id) + game_idx

                    try:
                        result = await self._run_single_game(
                            model=model,
                            game_index=game_idx,
                            seed=game_seed,
                        )

                        # Update model results
                        model_result.games_completed += 1
                        if result.get('winner') == 'liberal':
                            model_result.liberal_wins += 1
                        elif result.get('winner') == 'fascist':
                            model_result.fascist_wins += 1

                        cost_summary = result.get('cost_summary', {})
                        model_result.total_cost += cost_summary.get('total_cost', 0)
                        model_result.total_tokens_in += cost_summary.get('total_input_tokens', 0)
                        model_result.total_tokens_out += cost_summary.get('total_output_tokens', 0)

                        if 'game_id' in result:
                            model_result.game_ids.append(result['game_id'])

                        self.progress.completed_games += 1

                    except Exception as e:
                        model_result.games_failed += 1
                        self.progress.failed_games += 1
                        self.progress.errors.append(f"{model.name} game {game_idx}: {str(e)}")
                        print(f"Error in game {game_idx} for {model.name}: {e}")

                    # Progress update
                    if (game_idx + 1) % 10 == 0:
                        print(f"  [{model.name}] {game_idx + 1}/{self.batch_config.games_per_model} games")
                        self._save_progress()

                # Save progress after each model
                self._save_progress()

            self.progress.status = ComparisonStatus.COMPLETED

        except Exception as e:
            self.progress.status = ComparisonStatus.FAILED
            self.progress.errors.append(f"Fatal error: {str(e)}")
            raise

        finally:
            self.progress.end_time = datetime.now().isoformat()
            self.progress.current_model = None
            self._save_progress()
            self._generate_report()

        return self.progress

    async def _run_single_game(
        self,
        model: ModelConfig,
        game_index: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single game with the specified model."""
        from core.game_manager import GameManager

        # Player names for the game
        player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]

        # Create player configurations - all using the same model
        player_configs = [
            {
                "id": f"player{i+1}",
                "name": player_names[i],
                "model": model.openrouter_id,
                "type": "ai"
            }
            for i in range(self.num_players)
        ]

        # Set random seed for reproducibility
        random.seed(seed)

        # Create and run game
        game_manager = GameManager(
            player_configs=player_configs,
            openrouter_api_key=self.api_key,
            enable_database_logging=self.enable_db_logging,
        )

        result = await game_manager.start_game()
        result['seed'] = seed
        result['model'] = model.openrouter_id

        return result

    def _save_progress(self):
        """Save current progress to file."""
        progress_file = self.output_dir / f"{self.progress.comparison_id}_progress.json"
        with open(progress_file, 'w') as f:
            json.dump(self.progress.to_dict(), f, indent=2)

    def _generate_report(self):
        """Generate comparison report."""
        report_file = self.output_dir / f"{self.progress.comparison_id}_report.json"

        report = {
            'comparison_id': self.progress.comparison_id,
            'batch_name': self.progress.batch_name,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_games': self.progress.total_games,
                'completed_games': self.progress.completed_games,
                'failed_games': self.progress.failed_games,
                'total_cost': self.progress.total_cost,
                'elapsed_seconds': self.progress.elapsed_time,
            },
            'model_rankings': self._calculate_rankings(),
            'model_results': {k: v.to_dict() for k, v in self.progress.model_results.items()},
            'comparison_groups': self._analyze_comparison_groups(),
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_file}")
        return report

    def _calculate_rankings(self) -> List[Dict]:
        """Calculate model rankings based on win rates."""
        rankings = []

        for model_id, result in self.progress.model_results.items():
            model = get_model_by_id(model_id)
            rankings.append({
                'model_id': model_id,
                'model_name': result.model_name,
                'games_completed': result.games_completed,
                'liberal_win_rate': result.win_rate_liberal,
                'fascist_win_rate': result.win_rate_fascist,
                'total_cost': result.total_cost,
                'cost_per_game': result.cost_per_game,
                'is_free': model.is_free if model else False,
                'provider': model.provider.value if model else 'unknown',
            })

        # Sort by liberal win rate (higher is better for balanced play)
        rankings.sort(key=lambda x: x['liberal_win_rate'], reverse=True)

        # Add rank
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return rankings

    def _analyze_comparison_groups(self) -> Dict[str, Dict]:
        """Analyze predefined comparison groups."""
        from config.model_comparison_config import COMPARISON_GROUPS

        analysis = {}

        for group in COMPARISON_GROUPS:
            group_models = [m.openrouter_id for m in group.models]
            group_results = {
                mid: self.progress.model_results[mid].to_dict()
                for mid in group_models
                if mid in self.progress.model_results
            }

            if not group_results:
                continue

            # Calculate group statistics
            total_liberal_wins = sum(r['liberal_wins'] for r in group_results.values())
            total_games = sum(r['games_completed'] for r in group_results.values())

            analysis[group.name] = {
                'hypothesis': group.hypothesis,
                'models': list(group_results.keys()),
                'total_games': total_games,
                'avg_liberal_win_rate': total_liberal_wins / total_games if total_games > 0 else 0,
                'results': group_results,
            }

        return analysis

    def generate_latex_table(self) -> str:
        """Generate LaTeX table of results."""
        rankings = self._calculate_rankings()

        latex = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Multi-Model Comparison Results}",
            r"\label{tab:model_comparison}",
            r"\begin{tabular}{lccccr}",
            r"\toprule",
            r"Model & Provider & Games & Liberal Win\% & Fascist Win\% & Cost \\",
            r"\midrule",
        ]

        for r in rankings:
            row = (
                f"{r['model_name']} & {r['provider']} & {r['games_completed']} & "
                f"{r['liberal_win_rate']*100:.1f}\\% & {r['fascist_win_rate']*100:.1f}\\% & "
                f"\\${r['total_cost']:.2f} \\\\"
            )
            latex.append(row)

        latex.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(latex)

    def print_summary(self):
        """Print a summary of the comparison results."""
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY: {self.progress.batch_name}")
        print(f"{'='*70}")
        print(f"Status: {self.progress.status.value}")
        print(f"Total games: {self.progress.completed_games}/{self.progress.total_games}")
        print(f"Failed games: {self.progress.failed_games}")
        print(f"Total cost: ${self.progress.total_cost:.2f}")

        if self.progress.elapsed_time:
            elapsed_min = self.progress.elapsed_time / 60
            print(f"Elapsed time: {elapsed_min:.1f} minutes")

        print(f"\n{'Model Results':^70}")
        print("-" * 70)
        print(f"{'Model':<25} {'Games':>8} {'Liberal%':>10} {'Fascist%':>10} {'Cost':>10}")
        print("-" * 70)

        for model_id, result in sorted(
            self.progress.model_results.items(),
            key=lambda x: x[1].win_rate_liberal,
            reverse=True
        ):
            print(
                f"{result.model_name:<25} "
                f"{result.games_completed:>8} "
                f"{result.win_rate_liberal*100:>9.1f}% "
                f"{result.win_rate_fascist*100:>9.1f}% "
                f"${result.total_cost:>9.2f}"
            )

        print("-" * 70)
        print(f"{'TOTAL':<25} {self.progress.completed_games:>8} {'':<10} {'':<10} ${self.progress.total_cost:>9.2f}")
        print("=" * 70)


async def run_model_comparison(
    api_key: str,
    models: Optional[List[ModelConfig]] = None,
    games_per_model: int = 500,
    num_players: int = 7,
    concurrency: int = 3,
    output_dir: str = "results/comparisons",
) -> ComparisonProgress:
    """
    Convenience function to run a model comparison.

    Args:
        api_key: OpenRouter API key
        models: List of models to compare (default: CURRENT_RUN_MODELS)
        games_per_model: Number of games per model
        num_players: Number of players per game
        concurrency: Number of concurrent games
        output_dir: Output directory for results

    Returns:
        ComparisonProgress with results
    """
    if models is None:
        models = CURRENT_RUN_MODELS

    batch = BatchConfig(
        name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        models=models,
        games_per_model=games_per_model,
    )

    comparator = ModelComparator(
        api_key=api_key,
        batch_config=batch,
        output_dir=output_dir,
        num_players=num_players,
    )

    return await comparator.run_comparison(concurrency=concurrency)


if __name__ == "__main__":
    # Print batch configuration
    from config.model_comparison_config import print_cost_summary
    print_cost_summary()
