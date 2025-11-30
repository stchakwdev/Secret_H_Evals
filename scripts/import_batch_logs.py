#!/usr/bin/env python3
"""
Import game results from batch log files into the database.
Handles the 300-game overnight batch that wasn't saved to DB during execution.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.database_schema import DatabaseManager


def import_batch_logs(
    logs_dir: Path,
    db_path: str = "data/games.db",
    batch_pattern: str = "batch-20251128-232324-3e429120-game-*"
) -> dict:
    """
    Import all game metrics from batch log directories into database.

    Args:
        logs_dir: Directory containing game log folders
        db_path: Path to SQLite database
        batch_pattern: Glob pattern to match game directories

    Returns:
        Summary statistics dict
    """
    db = DatabaseManager(db_path)
    db.enable_wal_mode()  # Enable WAL for better performance

    stats = {
        'total_found': 0,
        'imported': 0,
        'skipped': 0,
        'errors': 0,
        'outcomes': {}
    }

    game_dirs = sorted(logs_dir.glob(batch_pattern))
    stats['total_found'] = len(game_dirs)

    print(f"Found {len(game_dirs)} game directories matching '{batch_pattern}'")

    for game_dir in game_dirs:
        metrics_file = game_dir / "metrics.json"

        if not metrics_file.exists():
            stats['skipped'] += 1
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Check if game already exists
            existing = db.get_game(metrics.get('game_id'))
            if existing:
                stats['skipped'] += 1
                continue

            # Transform metrics to match database schema
            outcome = metrics.get('game_outcome', {})
            winner = outcome.get('winner', 'unknown')
            win_condition = outcome.get('win_condition', 'unknown')

            # Track outcome distribution
            outcome_key = f"{winner}:{win_condition}"
            stats['outcomes'][outcome_key] = stats['outcomes'].get(outcome_key, 0) + 1

            game_data = {
                'game_id': metrics.get('game_id'),
                'timestamp': metrics.get('start_time', datetime.now().isoformat()),
                'player_count': metrics.get('player_count', 5),
                'players': {},  # Will default to 'unknown' model
                'winner': winner,
                'winning_team': winner,  # Same as winner for this game
                'win_condition': win_condition,
                'duration_seconds': metrics.get('duration_seconds', 0),
                'total_actions': metrics.get('total_actions', 0),
                'total_cost': metrics.get('api_usage', {}).get('total_cost', 0.0),
                'liberal_policies': outcome.get('final_liberal_policies', 0),
                'fascist_policies': outcome.get('final_fascist_policies', 0),
            }

            success = db.insert_game(game_data)
            if success:
                stats['imported'] += 1
                if stats['imported'] % 50 == 0:
                    print(f"  Imported {stats['imported']} games...")
            else:
                stats['errors'] += 1

        except Exception as e:
            print(f"  Error processing {game_dir.name}: {e}")
            stats['errors'] += 1

    # Optimize database after bulk insert
    db.analyze_tables()

    return stats


def main():
    logs_dir = project_root / "logs"
    db_path = str(project_root / "data" / "games.db")

    print("=" * 60)
    print("Secret Hitler Batch Log Importer")
    print("=" * 60)
    print(f"Logs directory: {logs_dir}")
    print(f"Database: {db_path}")
    print()

    # Import the overnight batch
    stats = import_batch_logs(logs_dir, db_path)

    print()
    print("=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"Total directories found: {stats['total_found']}")
    print(f"Successfully imported: {stats['imported']}")
    print(f"Skipped (already exists or no metrics): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print()

    print("Outcome Distribution:")
    for outcome, count in sorted(stats['outcomes'].items(), key=lambda x: -x[1]):
        pct = count / max(sum(stats['outcomes'].values()), 1) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # Verify database contents
    db = DatabaseManager(db_path)
    db_stats = db.get_game_stats()
    print()
    print("Database Statistics:")
    print(f"  Total games in DB: {db_stats.get('total_games', 0)}")
    print(f"  Total decisions: {db_stats.get('total_decisions', 0)}")
    print(f"  Total cost: ${db_stats.get('total_cost', 0):.4f}")


if __name__ == "__main__":
    main()
