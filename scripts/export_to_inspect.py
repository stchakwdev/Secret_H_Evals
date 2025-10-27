#!/usr/bin/env python3
"""
Batch export Secret Hitler games to Inspect AI format.
"""
import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.inspect_adapter import SecretHitlerInspectAdapter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_single_game(adapter: SecretHitlerInspectAdapter, game_id: str):
    """Export a single game."""
    try:
        output_path = adapter.export_game(game_id)
        logger.info(f"✓ Exported {game_id} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to export {game_id}: {e}")
        return False


def export_all_games(adapter: SecretHitlerInspectAdapter, limit: int = None):
    """Export all games with progress bar."""
    logger.info("Starting batch export...")

    try:
        # Get game list
        games = adapter.db.get_all_games(limit=limit)

        if not games:
            logger.warning("No games found in database. Trying log directories...")
            # Fallback to scanning directories
            if adapter.game_logs_dir.exists():
                game_dirs = [d.name for d in adapter.game_logs_dir.iterdir() if d.is_dir()]
                if limit:
                    game_dirs = game_dirs[:limit]
                games = [{"game_id": gid} for gid in game_dirs]

        if not games:
            logger.error("No games found to export")
            return 0

        logger.info(f"Found {len(games)} games to export")

        # Export with progress bar
        success_count = 0
        with tqdm(total=len(games), desc="Exporting games") as pbar:
            for game in games:
                game_id = game["game_id"]
                if export_single_game(adapter, game_id):
                    success_count += 1
                pbar.update(1)

        logger.info(f"\n✓ Successfully exported {success_count}/{len(games)} games")
        return success_count

    except Exception as e:
        logger.error(f"Batch export failed: {e}")
        return 0


def export_latest_n(adapter: SecretHitlerInspectAdapter, n: int):
    """Export the N most recent games."""
    logger.info(f"Exporting {n} most recent games...")
    return export_all_games(adapter, limit=n)


def main():
    parser = argparse.ArgumentParser(
        description="Export Secret Hitler games to Inspect AI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export specific game
  python scripts/export_to_inspect.py --game-id game_001

  # Export all games
  python scripts/export_to_inspect.py --all

  # Export 10 most recent games
  python scripts/export_to_inspect.py --latest 10

  # Specify custom database path
  python scripts/export_to_inspect.py --all --db-path custom/path/games.db
        """
    )

    parser.add_argument(
        '--game-id',
        type=str,
        help='Export specific game by ID'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Export all games'
    )

    parser.add_argument(
        '--latest',
        type=int,
        metavar='N',
        help='Export N most recent games'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit total number of games to export'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='./data/games.db',
        help='Path to SQLite database (default: ./data/games.db)'
    )

    parser.add_argument(
        '--logs-dir',
        type=str,
        default='./logs',
        help='Path to game logs directory (default: ./logs)'
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

    # Initialize adapter
    adapter = SecretHitlerInspectAdapter(
        game_logs_dir=args.logs_dir,
        db_path=args.db_path
    )

    # Execute command
    try:
        if args.game_id:
            # Export single game
            success = export_single_game(adapter, args.game_id)
            sys.exit(0 if success else 1)

        elif args.all:
            # Export all games
            count = export_all_games(adapter, limit=args.limit)
            sys.exit(0 if count > 0 else 1)

        elif args.latest:
            # Export latest N games
            count = export_latest_n(adapter, args.latest)
            sys.exit(0 if count > 0 else 1)

        else:
            parser.print_help()
            print("\n❌ Error: Must specify --game-id, --all, or --latest")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
