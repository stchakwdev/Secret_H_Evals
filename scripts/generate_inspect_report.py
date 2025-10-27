#!/usr/bin/env python3
"""
Generate HTML reports using Inspect AI CLI tools.
"""
import argparse
import sys
import subprocess
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_html_report(
    input_dir: str,
    output_path: str,
    title: str = "Secret Hitler LLM Evaluation"
) -> bool:
    """
    Generate HTML report using inspect CLI command.

    Args:
        input_dir: Directory containing Inspect JSON logs
        output_path: Path for output HTML file
        title: Report title

    Returns:
        True if successful, False otherwise
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return False

    logger.info(f"Found {len(json_files)} game logs")
    logger.info(f"Generating HTML report: {output_path}")

    # Note: Inspect CLI doesn't have a 'report' command for static HTML
    # Instead, it provides an interactive viewer via 'inspect view start'
    # For sharing, you can bundle logs with 'inspect view bundle'

    logger.info("\n" + "="*70)
    logger.info("Inspect CLI Usage Information")
    logger.info("="*70)
    logger.info("\nNote: Inspect uses an interactive web viewer instead of static HTML reports.\n")

    logger.info("To view logs interactively (starts web server):")
    logger.info(f"  inspect view start {input_dir}/*.json\n")

    logger.info("To bundle logs for sharing:")
    logger.info(f"  inspect view bundle {input_dir}/*.json --output bundle.json\n")

    logger.info("For static CSV/JSON analysis reports:")
    logger.info(f"  python scripts/analyze_with_inspect.py --input {input_dir}\n")

    logger.info("Reports already generated:")
    reports_dir = Path("reports")
    if reports_dir.exists():
        for report_file in reports_dir.glob("*"):
            logger.info(f"  - {report_file}")

    logger.info("\n" + "="*70)
    return True


def view_logs(input_dir: str, filter_expr: str = None):
    """
    Open logs in Inspect viewer.

    Args:
        input_dir: Directory containing Inspect JSON logs
        filter_expr: Optional filter expression
    """
    input_dir = Path(input_dir)

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return False

    logger.info(f"Opening {len(json_files)} logs in Inspect viewer...")

    try:
        cmd = ["inspect", "view", str(input_dir / "*.json")]

        if filter_expr:
            cmd.extend(["--filter", filter_expr])

        logger.info(f"Running: {' '.join(cmd)}")

        # Run in foreground (interactive)
        subprocess.run(cmd, check=True)
        return True

    except FileNotFoundError:
        logger.error("'inspect' command not found. Please install: pip install inspect-ai")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to view logs: {e}")
        return False


def compare_games(input_dir: str, game_ids: list):
    """
    Compare specific games side-by-side.

    Args:
        input_dir: Directory containing Inspect JSON logs
        game_ids: List of game IDs to compare
    """
    input_dir = Path(input_dir)

    # Find files for specified game IDs
    files = [input_dir / f"{game_id}.json" for game_id in game_ids]

    missing = [f for f in files if not f.exists()]
    if missing:
        logger.error(f"Missing files: {missing}")
        return False

    logger.info(f"Comparing {len(files)} games...")

    try:
        cmd = ["inspect", "view"] + [str(f) for f in files]

        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True

    except Exception as e:
        logger.error(f"Failed to compare games: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports using Inspect AI tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate HTML report
  python scripts/generate_inspect_report.py --report --output reports/experiment1.html

  # View logs in browser
  python scripts/generate_inspect_report.py --view

  # Compare specific games
  python scripts/generate_inspect_report.py --compare game_001 game_002 game_003

  # Filter logs by model
  python scripts/generate_inspect_report.py --view --filter "metadata.model == 'gpt-4'"
        """
    )

    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='./data/inspect_logs',
        help='Directory containing Inspect logs (default: ./data/inspect_logs)'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML report'
    )

    parser.add_argument(
        '--view',
        action='store_true',
        help='Open logs in Inspect viewer'
    )

    parser.add_argument(
        '--compare',
        nargs='+',
        metavar='GAME_ID',
        help='Compare specific games (provide game IDs)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='./reports/inspect_report.html',
        help='Output path for HTML report (default: ./reports/inspect_report.html)'
    )

    parser.add_argument(
        '--title',
        '-t',
        type=str,
        default='Secret Hitler LLM Evaluation',
        help='Report title'
    )

    parser.add_argument(
        '--filter',
        '-f',
        type=str,
        help='Filter expression for logs'
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

    # Execute command
    try:
        if args.report:
            success = generate_html_report(
                input_dir=args.input,
                output_path=args.output,
                title=args.title
            )
            sys.exit(0 if success else 1)

        elif args.view:
            success = view_logs(
                input_dir=args.input,
                filter_expr=args.filter
            )
            sys.exit(0 if success else 1)

        elif args.compare:
            success = compare_games(
                input_dir=args.input,
                game_ids=args.compare
            )
            sys.exit(0 if success else 1)

        else:
            parser.print_help()
            print("\n❌ Error: Must specify --report, --view, or --compare")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
