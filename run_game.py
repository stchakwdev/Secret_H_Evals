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
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

from core.game_manager import GameManager

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


async def run_batch_evaluation(num_games: int, num_players: int, model: str, output_dir: str, enable_db_logging: bool = False):
    """Run batch evaluation with multiple games."""

    print(f"\n{'='*60}")
    print(f"Batch Evaluation")
    print(f"{'='*60}")
    print(f"Games: {num_games}")
    print(f"Players per game: {num_players}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print(f"Database logging: {'Enabled' if enable_db_logging else 'Disabled'}")
    print(f"{'='*60}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        result = await run_single_game(num_players, model, enable_db_logging)
        results.append(result)

    # Aggregate results
    liberal_wins = sum(1 for r in results if r.get('winner') == 'liberals')
    fascist_wins = sum(1 for r in results if r.get('winner') == 'fascists')
    total_cost = sum(r.get('cost_summary', {}).get('total_cost', 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Batch Evaluation Complete")
    print(f"{'='*60}")
    print(f"Liberal wins: {liberal_wins}/{num_games} ({liberal_wins/num_games*100:.1f}%)")
    print(f"Fascist wins: {fascist_wins}/{num_games} ({fascist_wins/num_games*100:.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Average cost per game: ${total_cost/num_games:.4f}")
    print(f"{'='*60}\n")


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

    args = parser.parse_args()

    try:
        if args.batch:
            asyncio.run(run_batch_evaluation(
                num_games=args.games,
                num_players=args.players,
                model=args.model,
                output_dir=args.output,
                enable_db_logging=args.enable_db_logging
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