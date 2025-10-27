#!/usr/bin/env python3
"""
Migrate historical Secret Hitler game logs to SQLite database and Inspect format.
"""
import argparse
import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.database_schema import DatabaseManager
from evaluation.inspect_adapter import SecretHitlerInspectAdapter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalMigrator:
    """Migrates historical JSON logs to database and Inspect format."""

    def __init__(self, logs_dir: str, db_path: str):
        self.logs_dir = Path(logs_dir)
        self.db = DatabaseManager(db_path)
        self.adapter = SecretHitlerInspectAdapter(
            game_logs_dir=str(logs_dir),
            db_path=db_path
        )

    def scan_game_directories(self) -> List[Path]:
        """Scan logs directory for game folders."""
        if not self.logs_dir.exists():
            logger.error(f"Logs directory not found: {self.logs_dir}")
            return []

        game_dirs = [d for d in self.logs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Found {len(game_dirs)} game directories")
        return game_dirs

    def load_game_metrics(self, game_dir: Path) -> Dict[str, Any]:
        """Load metrics.json from a game directory."""
        metrics_file = game_dir / "metrics.json"

        if not metrics_file.exists():
            raise FileNotFoundError(f"No metrics.json in {game_dir}")

        with open(metrics_file, 'r') as f:
            return json.load(f)

    def migrate_game(self, game_dir: Path) -> bool:
        """Migrate a single game to database."""
        game_id = game_dir.name

        try:
            # Load metrics
            metrics = self.load_game_metrics(game_dir)

            # Add game_id if not present
            if "game_id" not in metrics:
                metrics["game_id"] = game_id

            # Add timestamp if not present
            if "timestamp" not in metrics:
                # Use file modification time as fallback
                metrics["timestamp"] = datetime.fromtimestamp(
                    game_dir.stat().st_mtime
                ).isoformat()

            # Extract required fields
            game_data = {
                "game_id": game_id,
                "timestamp": metrics.get("start_time", metrics.get("timestamp")),
                "player_count": metrics.get("player_count", len(metrics.get("players", []))),
                "winner": metrics.get("game_outcome", {}).get("winner"),
                "winning_team": self._infer_winning_team(metrics),
                "win_condition": metrics.get("game_outcome", {}).get("win_condition"),
                "duration_seconds": metrics.get("duration_seconds", 0),
                "total_actions": metrics.get("total_actions", 0),
                "total_cost": metrics.get("api_usage", {}).get("total_cost", 0.0),
                "liberal_policies": metrics.get("game_outcome", {}).get("final_liberal_policies", 0),
                "fascist_policies": metrics.get("game_outcome", {}).get("final_fascist_policies", 0),
                "players": metrics.get("player_metrics", {})
            }

            # Insert into database
            success = self.db.insert_game(game_data)

            if success:
                logger.debug(f"✓ Migrated {game_id} to database")
                return True
            else:
                logger.warning(f"✗ Failed to migrate {game_id}")
                return False

        except Exception as e:
            logger.error(f"✗ Error migrating {game_id}: {e}")
            return False

    def _infer_winning_team(self, metrics: Dict[str, Any]) -> str:
        """Infer winning team from game outcome."""
        winner = metrics.get("game_outcome", {}).get("winner", "")
        if "liberal" in winner.lower():
            return "liberal"
        elif "fascist" in winner.lower() or "hitler" in winner.lower():
            return "fascist"
        return "unknown"

    def migrate_all(self, export_to_inspect: bool = False, dry_run: bool = False) -> Dict[str, int]:
        """
        Migrate all games.

        Args:
            export_to_inspect: Also export to Inspect format after migration
            dry_run: Preview migration without making changes

        Returns:
            Dictionary with success/failure counts
        """
        game_dirs = self.scan_game_directories()

        if not game_dirs:
            logger.warning("No games found to migrate")
            return {"total": 0, "success": 0, "failed": 0, "exported": 0}

        if dry_run:
            logger.info(f"DRY RUN: Would migrate {len(game_dirs)} games")
            for game_dir in game_dirs[:10]:  # Show first 10
                logger.info(f"  - {game_dir.name}")
            if len(game_dirs) > 10:
                logger.info(f"  ... and {len(game_dirs) - 10} more")
            return {"total": len(game_dirs), "success": 0, "failed": 0, "exported": 0}

        results = {"total": len(game_dirs), "success": 0, "failed": 0, "exported": 0}

        # Migrate games
        logger.info(f"Migrating {len(game_dirs)} games to database...")
        with tqdm(total=len(game_dirs), desc="Migrating games") as pbar:
            for game_dir in game_dirs:
                if self.migrate_game(game_dir):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                pbar.update(1)

        logger.info(f"\n✓ Migration complete: {results['success']}/{results['total']} successful")

        # Export to Inspect if requested
        if export_to_inspect and results["success"] > 0:
            logger.info("\nExporting to Inspect format...")
            try:
                exported_paths = self.adapter.export_all_games()
                results["exported"] = len(exported_paths)
                logger.info(f"✓ Exported {results['exported']} games to Inspect format")
            except Exception as e:
                logger.error(f"Export to Inspect failed: {e}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate historical Secret Hitler logs to database and Inspect format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration (dry run)
  python scripts/migrate_historical.py --dry-run

  # Migrate to database only
  python scripts/migrate_historical.py --all

  # Migrate and export to Inspect format
  python scripts/migrate_historical.py --all --export-inspect

  # Use custom paths
  python scripts/migrate_historical.py --all --logs-dir custom/logs --db-path custom/db/games.db
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate all games'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview migration without making changes'
    )

    parser.add_argument(
        '--export-inspect',
        action='store_true',
        help='Also export to Inspect format after migration'
    )

    parser.add_argument(
        '--logs-dir',
        type=str,
        default='./logs',
        help='Path to game logs directory (default: ./logs)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='./data/games.db',
        help='Path to SQLite database (default: ./data/games.db)'
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

    # Initialize migrator
    migrator = HistoricalMigrator(
        logs_dir=args.logs_dir,
        db_path=args.db_path
    )

    # Execute migration
    try:
        if not args.all and not args.dry_run:
            parser.print_help()
            print("\n❌ Error: Must specify --all or --dry-run")
            sys.exit(1)

        results = migrator.migrate_all(
            export_to_inspect=args.export_inspect,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print("\n✓ Dry run complete")
            sys.exit(0)

        print(f"\n{'='*60}")
        print(f"Migration Summary")
        print(f"{'='*60}")
        print(f"Total games: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        if results['exported'] > 0:
            print(f"Exported to Inspect: {results['exported']}")
        print(f"{'='*60}")

        sys.exit(0 if results['failed'] == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
