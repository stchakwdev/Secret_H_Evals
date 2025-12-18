#!/usr/bin/env python3
"""
Analyze Secret Hitler games using Inspect AI log format.
"""
import argparse
import sys
from pathlib import Path
import json
import logging
import pandas as pd
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_inspect_log(log_path: Path) -> Dict[str, Any]:
    """Load an Inspect AI format log file."""
    with open(log_path, 'r') as f:
        return json.load(f)


def analyze_experiment(log_dir: str, output_dir: str = "./reports") -> Dict[str, Any]:
    """
    Analyze Secret Hitler games using Inspect logs.

    Args:
        log_dir: Directory containing Inspect format JSON logs
        output_dir: Directory for output reports

    Returns:
        Dictionary containing analysis results
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all log files
    log_files = list(log_dir.glob("*.json"))
    if not log_files:
        logger.error(f"No JSON log files found in {log_dir}")
        return {}

    logger.info(f"Loading {len(log_files)} game logs...")

    logs = []
    for log_file in log_files:
        try:
            log = load_inspect_log(log_file)
            logs.append(log)
        except Exception as e:
            logger.warning(f"Failed to load {log_file}: {e}")

    if not logs:
        logger.error("No valid logs loaded")
        return {}

    logger.info(f"✓ Loaded {len(logs)} games")

    # Extract all samples for analysis
    all_samples = []
    for log in logs:
        game_id = log.get("run_id", "unknown")
        for sample in log.get("samples", []):
            metadata = sample.get("metadata", {})
            all_samples.append({
                "game_id": game_id,
                "player_id": metadata.get("player_id", ""),
                "player_name": metadata.get("player_name", ""),
                "decision_type": metadata.get("decision_type", ""),
                "is_deception": metadata.get("is_deception", False),
                "deception_score": metadata.get("deception_score", 0.0),
                "confidence": metadata.get("confidence", 0.0),
                "reasoning": metadata.get("reasoning", ""),
                "public_statement": metadata.get("public_statement", ""),
                "action": sample.get("output", ""),
                "timestamp": metadata.get("timestamp", "")
            })

    df = pd.DataFrame(all_samples)

    # Perform analyses
    analyses = {}

    # 1. Deception Analysis
    logger.info("\n" + "="*60)
    logger.info("Deception Analysis")
    logger.info("="*60)

    if "is_deception" in df.columns and not df.empty:
        deception_by_player = df.groupby("player_name")["is_deception"].agg([
            ('total_actions', 'count'),
            ('deception_count', 'sum'),
            ('deception_rate', 'mean')
        ]).round(3)

        logger.info("\nDeception by Player:")
        logger.info(deception_by_player.to_string())

        analyses["deception_by_player"] = deception_by_player.to_dict()

        # Deception by decision type
        deception_by_type = df.groupby("decision_type")["is_deception"].agg([
            ('count', 'count'),
            ('deception_rate', 'mean')
        ]).round(3)

        logger.info("\nDeception by Decision Type:")
        logger.info(deception_by_type.to_string())

        analyses["deception_by_type"] = deception_by_type.to_dict()

    # 2. Game Outcome Analysis
    logger.info("\n" + "="*60)
    logger.info("Game Outcome Analysis")
    logger.info("="*60)

    outcomes = []
    costs = []
    durations = []

    for log in logs:
        metadata = log.get("metadata", {})
        results = log.get("results", {})

        # Extract scores from new nested format
        # Format: {"name": "category", "metrics": {"metric_name": {"value": X}}}
        scores = {}
        for score_category in results.get("scores", []):
            metrics = score_category.get("metrics", {})
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "value" in metric_data:
                    scores[metric_name] = metric_data["value"]

        outcomes.append({
            "game_id": log.get("eval", {}).get("task_id") or metadata.get("game_id"),
            "winner": metadata.get("winner"),
            "winning_team": metadata.get("winning_team"),
            "win_condition": metadata.get("win_condition"),
            "liberal_win": scores.get("win_rate_liberal", 0),
            "fascist_win": scores.get("win_rate_fascist", 0),
            "deception_freq": scores.get("deception_frequency", 0),
            "total_cost": scores.get("total_cost", 0),
            "duration": scores.get("game_duration", 0),
            "player_count": metadata.get("player_count", 0)
        })

        costs.append(scores.get("total_cost", 0))
        durations.append(scores.get("game_duration", 0))

    outcomes_df = pd.DataFrame(outcomes)

    # Win rate analysis
    if not outcomes_df.empty:
        liberal_wins = outcomes_df["liberal_win"].sum()
        fascist_wins = outcomes_df["fascist_win"].sum()
        total_games = len(outcomes_df)

        logger.info(f"\nTotal Games: {total_games}")
        logger.info(f"Liberal Wins: {int(liberal_wins)} ({liberal_wins/total_games*100:.1f}%)")
        logger.info(f"Fascist Wins: {int(fascist_wins)} ({fascist_wins/total_games*100:.1f}%)")

        analyses["win_rates"] = {
            "total_games": total_games,
            "liberal_wins": int(liberal_wins),
            "fascist_wins": int(fascist_wins),
            "liberal_win_rate": liberal_wins / total_games,
            "fascist_win_rate": fascist_wins / total_games
        }

        # Win conditions
        win_conditions = outcomes_df["win_condition"].value_counts()
        logger.info("\nWin Conditions:")
        logger.info(win_conditions.to_string())

        analyses["win_conditions"] = win_conditions.to_dict()

    # 3. Cost Analysis
    logger.info("\n" + "="*60)
    logger.info("Cost Analysis")
    logger.info("="*60)

    if costs:
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs)

        logger.info(f"Total Cost: ${total_cost:.4f}")
        logger.info(f"Average Cost per Game: ${avg_cost:.4f}")
        logger.info(f"Min Cost: ${min(costs):.4f}")
        logger.info(f"Max Cost: ${max(costs):.4f}")

        analyses["cost_analysis"] = {
            "total_cost": total_cost,
            "avg_cost_per_game": avg_cost,
            "min_cost": min(costs),
            "max_cost": max(costs),
            "games_per_dollar": len(costs) / max(total_cost, 0.01)
        }

    # 4. Performance Metrics
    logger.info("\n" + "="*60)
    logger.info("Performance Metrics")
    logger.info("="*60)

    if durations:
        avg_duration = sum(durations) / len(durations)
        logger.info(f"Average Game Duration: {avg_duration:.1f} seconds")

        analyses["performance"] = {
            "avg_duration_seconds": avg_duration,
            "total_duration_seconds": sum(durations)
        }

    # Export detailed results to CSV
    csv_path = output_dir / "inspect_analysis.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved detailed analysis to {csv_path}")

    # Export outcomes to CSV
    outcomes_csv = output_dir / "game_outcomes.csv"
    outcomes_df.to_csv(outcomes_csv, index=False)
    logger.info(f"✓ Saved game outcomes to {outcomes_csv}")

    # Save JSON summary
    summary_json = output_dir / "analysis_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    logger.info(f"✓ Saved analysis summary to {summary_json}")

    return analyses


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Secret Hitler games using Inspect AI logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all games in inspect_logs directory
  python scripts/analyze_with_inspect.py --input data/inspect_logs/

  # Specify custom output directory
  python scripts/analyze_with_inspect.py --input data/inspect_logs/ --output results/

  # Verbose output
  python scripts/analyze_with_inspect.py --input data/inspect_logs/ --verbose
        """
    )

    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='./data/inspect_logs',
        help='Directory containing Inspect format logs (default: ./data/inspect_logs)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./reports',
        help='Output directory for analysis reports (default: ./reports)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    try:
        analyses = analyze_experiment(args.input, args.output)

        if not analyses:
            logger.error("Analysis failed or no data found")
            sys.exit(1)

        logger.info("\n✓ Analysis complete!")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
