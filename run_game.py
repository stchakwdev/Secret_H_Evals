#!/usr/bin/env python3
"""
Secret Hitler LLM Evaluation Framework
Main entry point for running evaluations

Author: Samuel Chakwera (stchakdev)
"""
import asyncio
import argparse
import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional

from core.game_manager import GameManager
from config.model_comparison_config import (
    ALL_MODELS, FREE_MODELS, PAID_MODELS, CURRENT_RUN_MODELS,
    BatchConfig, print_cost_summary, get_model_by_id
)

# Load environment variables
load_dotenv()


async def run_single_game(num_players: int, model: str, enable_db_logging: bool = False) -> Dict:
    """Run a single game evaluation."""

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("Error: OPENROUTER_API_KEY not set in .env file")
        sys.exit(1)

    # Create player configurations
    player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack"]
    player_configs = [
        {
            "id": f"player{i+1}",
            "name": player_names[i],
            "model": model,
            "type": "ai"
        }
        for i in range(num_players)
    ]

    print(f"\n{'='*60}")
    print(f"Secret Hitler LLM Evaluation")
    print(f"{'='*60}")
    print(f"Players: {num_players}")
    print(f"Model: {model}")
    print(f"Database logging: {'Enabled' if enable_db_logging else 'Disabled'}")
    print(f"{'='*60}\n")

    # Create and run game
    game_manager = GameManager(
        player_configs=player_configs,
        openrouter_api_key=api_key,
        enable_database_logging=enable_db_logging
    )

    result = await game_manager.start_game()

    # Print results
    print(f"\n{'='*60}")
    print(f"Game Complete")
    print(f"{'='*60}")
    print(f"Winner: {result.get('winner', 'Unknown')}")
    print(f"Total rounds: {result.get('rounds', 'Unknown')}")

    if 'cost_summary' in result:
        cost = result['cost_summary']
        print(f"Total cost: ${cost.get('total_cost', 0):.4f}")
        print(f"API requests: {cost.get('total_requests', 0)}")

    print(f"{'='*60}\n")

    return result


async def run_batch_evaluation(
    num_games: int,
    num_players: int,
    model: str,
    output_dir: str,
    enable_db_logging: bool = False,
    batch_id: Optional[str] = None,
    batch_tag: Optional[str] = None
):
    """Run batch evaluation with multiple games."""

    # Generate batch_id if not provided
    if batch_id is None:
        batch_id = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

    # Create batch metadata
    start_time = datetime.now()
    batch_metadata = {
        "batch_id": batch_id,
        "batch_tag": batch_tag,
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "target_games": num_games,
        "players": num_players,
        "model": model,
        "output_dir": output_dir,
        "database_logging": enable_db_logging,
        "log_dir": str(Path(__file__).parent / "logs")
    }

    # Write metadata to logs/.current_batch
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = logs_dir / ".current_batch"

    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch Evaluation")
    print(f"{'='*60}")
    print(f"Batch ID: {batch_id}")
    if batch_tag:
        print(f"Batch tag: {batch_tag}")
    print(f"Games: {num_games}")
    print(f"Players per game: {num_players}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"Database logging: {'Enabled' if enable_db_logging else 'Disabled'}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"{'='*60}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    game_durations = []
    batch_start_time = datetime.now()

    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        game_start = datetime.now()
        result = await run_single_game(num_players, model, enable_db_logging)
        game_end = datetime.now()

        game_duration = (game_end - game_start).total_seconds()
        game_durations.append(game_duration)
        results.append(result)

        print(f"Game {i+1} completed in {game_duration/60:.1f} minutes ({game_duration:.0f} seconds)")

    # Aggregate results
    batch_end_time = datetime.now()
    batch_duration = (batch_end_time - batch_start_time).total_seconds()

    liberal_wins = sum(1 for r in results if r.get('winner') == 'liberal')
    fascist_wins = sum(1 for r in results if r.get('winner') == 'fascist')
    total_cost = sum(r.get('cost_summary', {}).get('total_cost', 0) for r in results)

    # Timing statistics
    avg_duration = sum(game_durations) / len(game_durations) if game_durations else 0
    min_duration = min(game_durations) if game_durations else 0
    max_duration = max(game_durations) if game_durations else 0
    games_per_hour = (num_games / batch_duration) * 3600 if batch_duration > 0 else 0

    print(f"\n{'='*60}")
    print(f"Batch Evaluation Complete")
    print(f"{'='*60}")
    print(f"Liberal wins: {liberal_wins}/{num_games} ({liberal_wins/num_games*100:.1f}%)")
    print(f"Fascist wins: {fascist_wins}/{num_games} ({fascist_wins/num_games*100:.1f}%)")
    print(f"\nTiming Statistics:")
    print(f"Total batch runtime: {batch_duration/60:.1f} minutes ({batch_duration:.0f} seconds)")
    print(f"Average game duration: {avg_duration/60:.1f} minutes ({avg_duration:.0f} seconds)")
    print(f"Min/Max game duration: {min_duration/60:.1f} - {max_duration/60:.1f} minutes")
    print(f"Games per hour: {games_per_hour:.1f}")
    print(f"\nCost Statistics:")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per game: ${total_cost/num_games:.4f}")
    print(f"Cost per minute: ${total_cost/(batch_duration/60):.4f}")
    print(f"Time efficiency: {avg_duration/total_cost if total_cost > 0 else 0:.0f} seconds per dollar")
    print(f"{'='*60}\n")


async def run_model_comparison(
    games_per_model: int,
    num_players: int,
    models: Optional[List[str]] = None,
    output_dir: str = "results/comparisons",
    enable_db_logging: bool = True,
    concurrency: int = 3,
    free_only: bool = False,
    paid_only: bool = False,
):
    """
    Run multi-model comparison experiment.

    Args:
        games_per_model: Number of games to run per model
        num_players: Number of players per game
        models: Specific model IDs to compare (None = use defaults)
        output_dir: Output directory for results
        enable_db_logging: Enable database logging
        concurrency: Number of concurrent games
        free_only: Only use free models
        paid_only: Only use paid models
    """
    from experiments.model_comparator import ModelComparator

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("Error: OPENROUTER_API_KEY not set in .env file")
        sys.exit(1)

    # Determine which models to use
    if models:
        # Use specific models provided
        model_configs = [get_model_by_id(m) for m in models]
        model_configs = [m for m in model_configs if m is not None]
        if not model_configs:
            print(f"Error: No valid models found from: {models}")
            print("\nAvailable models:")
            for m in ALL_MODELS:
                print(f"  {m.openrouter_id} ({m.name})")
            sys.exit(1)
    elif free_only:
        model_configs = FREE_MODELS
    elif paid_only:
        model_configs = PAID_MODELS
    else:
        model_configs = CURRENT_RUN_MODELS

    # Create batch configuration
    batch = BatchConfig(
        name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        models=model_configs,
        games_per_model=games_per_model,
        description=f"Multi-model comparison: {len(model_configs)} models, {games_per_model} games each"
    )

    # Print cost summary
    print_cost_summary(batch)

    # Create and run comparator
    comparator = ModelComparator(
        api_key=api_key,
        batch_config=batch,
        output_dir=output_dir,
        num_players=num_players,
        enable_db_logging=enable_db_logging,
    )

    # Run comparison
    progress = await comparator.run_comparison(concurrency=concurrency)

    # Print summary
    comparator.print_summary()

    # Generate LaTeX table
    latex = comparator.generate_latex_table()
    latex_file = Path(output_dir) / f"{progress.comparison_id}_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_file}")

    return progress


def list_available_models():
    """Print all available models for comparison."""
    print(f"\n{'='*70}")
    print("Available Models for Phase 4 Multi-Model Comparison")
    print(f"{'='*70}\n")

    print("FREE TIER (9 models):")
    print(f"{'Model':<25} {'ID':<45} {'Context':>10}")
    print("-" * 82)
    for m in FREE_MODELS:
        print(f"{m.name:<25} {m.openrouter_id:<45} {m.context_length:>10,}")

    print(f"\nPAID TIER (2 models):")
    print(f"{'Model':<25} {'ID':<45} {'$/M In':>8} {'$/M Out':>8}")
    print("-" * 94)
    for m in PAID_MODELS:
        print(f"{m.name:<25} {m.openrouter_id:<45} ${m.input_cost_per_million:>6.2f} ${m.output_cost_per_million:>6.2f}")

    print(f"\n{'='*70}")
    print("Cost Estimate for Default Batch (500 games/model):")
    print_cost_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Secret Hitler LLM Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single game with 5 players
  python run_game.py --players 5

  # Run batch evaluation
  python run_game.py --batch --games 10 --players 5

  # Use specific model
  python run_game.py --model anthropic/claude-3-sonnet

  # Run multi-model comparison (Phase 4)
  python run_game.py --model-comparison --games-per-model 500

  # Compare only free models
  python run_game.py --model-comparison --games-per-model 100 --free-only

  # Compare specific models
  python run_game.py --model-comparison --games-per-model 50 --compare-models x-ai/grok-4.1-fast:free,deepseek/deepseek-chat

  # List available models
  python run_game.py --list-models

Author: Samuel Chakwera (stchakdev)
        """
    )

    parser.add_argument(
        '--players', '-p',
        type=int,
        default=5,
        choices=range(5, 11),
        help='Number of players (5-10)'
    )

    parser.add_argument(
        '--model', '-m',
        default='deepseek/deepseek-v3.2-exp',
        help='OpenRouter model ID (default: deepseek/deepseek-v3.2-exp)'
    )

    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Run batch evaluation'
    )

    parser.add_argument(
        '--games', '-g',
        type=int,
        default=10,
        help='Number of games for batch evaluation (default: 10)'
    )

    parser.add_argument(
        '--output', '-o',
        default='results',
        help='Output directory for batch results (default: results/)'
    )

    parser.add_argument(
        '--enable-db-logging',
        action='store_true',
        help='Enable SQLite database logging for Inspect AI integration'
    )

    parser.add_argument(
        '--batch-id',
        type=str,
        help='Custom batch identifier (auto-generated if not provided)'
    )

    parser.add_argument(
        '--batch-tag',
        type=str,
        help='Human-readable batch tag/description'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel execution for batch mode (recommended for large batches)'
    )

    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=3,
        help='Number of concurrent games in parallel mode (default: 3)'
    )

    parser.add_argument(
        '--rate-limit',
        type=int,
        default=60,
        help='API requests per minute limit (default: 60)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous batch progress (parallel mode only)'
    )

    # Model comparison arguments (Phase 4)
    parser.add_argument(
        '--model-comparison',
        action='store_true',
        help='Run multi-model comparison experiment (Phase 4)'
    )

    parser.add_argument(
        '--games-per-model',
        type=int,
        default=500,
        help='Number of games per model in comparison mode (default: 500)'
    )

    parser.add_argument(
        '--compare-models',
        type=str,
        help='Comma-separated list of specific model IDs to compare'
    )

    parser.add_argument(
        '--free-only',
        action='store_true',
        help='Only use free models in comparison mode'
    )

    parser.add_argument(
        '--paid-only',
        action='store_true',
        help='Only use paid models in comparison mode'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models for comparison and exit'
    )

    args = parser.parse_args()

    try:
        # Handle --list-models first (no API key needed)
        if args.list_models:
            list_available_models()
            sys.exit(0)

        # Handle --model-comparison mode (Phase 4)
        if args.model_comparison:
            models = None
            if args.compare_models:
                models = [m.strip() for m in args.compare_models.split(',')]

            asyncio.run(run_model_comparison(
                games_per_model=args.games_per_model,
                num_players=args.players,
                models=models,
                output_dir=args.output,
                enable_db_logging=args.enable_db_logging,
                concurrency=args.concurrency,
                free_only=args.free_only,
                paid_only=args.paid_only,
            ))
            sys.exit(0)

        if args.batch:
            if args.parallel:
                # Use parallel runner for large-scale batches
                from experiments.parallel_runner import run_parallel_batch

                print(f"\n{'='*60}")
                print("Parallel Batch Evaluation")
                print(f"{'='*60}")
                print(f"Games: {args.games}")
                print(f"Concurrency: {args.concurrency}")
                print(f"Rate limit: {args.rate_limit} req/min")
                print(f"Resume: {args.resume}")
                print(f"{'='*60}\n")

                progress = asyncio.run(run_parallel_batch(
                    num_games=args.games,
                    num_players=args.players,
                    model=args.model,
                    concurrency=args.concurrency,
                    batch_id=args.batch_id,
                    batch_tag=args.batch_tag,
                    resume=args.resume
                ))

                print(f"\n{'='*60}")
                print("Parallel Batch Complete")
                print(f"{'='*60}")
                print(f"Completed: {progress.completed}/{progress.total_games}")
                print(f"Failed: {progress.failed}")
                print(f"Liberal wins: {progress.liberal_wins}")
                print(f"Fascist wins: {progress.fascist_wins}")
                print(f"Total cost: ${progress.total_cost:.4f}")
                print(f"{'='*60}\n")
            else:
                # Use sequential batch runner
                asyncio.run(run_batch_evaluation(
                    num_games=args.games,
                    num_players=args.players,
                    model=args.model,
                    output_dir=args.output,
                    enable_db_logging=args.enable_db_logging,
                    batch_id=args.batch_id,
                    batch_tag=args.batch_tag
                ))
        else:
            asyncio.run(run_single_game(
                num_players=args.players,
                model=args.model,
                enable_db_logging=args.enable_db_logging
            ))
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()