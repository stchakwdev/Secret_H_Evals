#!/usr/bin/env python3
"""
Master Visualization Generator

Runs all visualization scripts to generate a complete set of visuals
for GitHub showcase. Simplifies the process of creating all visualizations
with a single command.

Author: Samuel Chakwera (stchakdev)
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def run_script(script_name: str, args: list = None) -> int:
    """
    Run a visualization script.

    Args:
        script_name: Name of the script to run
        args: Additional arguments for the script

    Returns:
        Exit code from the script
    """
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]

    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
        else:
            print(f"✗ {script_name} failed with exit code {result.returncode}")
        return result.returncode
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate all visualizations for GitHub showcase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations from latest 3 games
  python generate_all_visuals.py --games 3

  # Generate with custom database
  python generate_all_visuals.py --db /path/to/games.db

  # Generate visualizations with custom output directory
  python generate_all_visuals.py --output custom_viz/

Author: Samuel Chakwera (stchakdev)
        """
    )

    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )

    parser.add_argument(
        '--games', '-g',
        type=int,
        default=3,
        help='Number of recent games to analyze (default: 3)'
    )

    parser.add_argument(
        '--output', '-o',
        default='visualizations',
        help='Output directory for visualizations (default: visualizations/)'
    )

    parser.add_argument(
        '--batch-name', '-n',
        help='Batch name for summary infographic'
    )

    parser.add_argument(
        '--skip', '-s',
        nargs='+',
        choices=['timeline', 'deception', 'network', 'cost', 'summary'],
        help='Skip specific visualizations'
    )

    args = parser.parse_args()

    # Determine batch name
    if args.batch_name:
        batch_name = args.batch_name
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        batch_name = f"Batch Evaluation - {timestamp}"

    # Build common database argument
    db_args = ['--db', args.db] if args.db else []

    skip_list = args.skip or []

    # Track results
    results = {}
    output_path = Path(args.output)

    print("\n" + "="*60)
    print("SECRET HITLER LLM EVALUATION - VISUALIZATION GENERATOR")
    print("="*60)
    print(f"Games to analyze: {args.games}")
    print(f"Output directory: {output_path}")
    print(f"Database: {args.db or 'data/games.db (default)'}")
    if skip_list:
        print(f"Skipping: {', '.join(skip_list)}")
    print("="*60)

    # 1. Policy Progression Timeline
    if 'timeline' not in skip_list:
        script_args = db_args + ['--output', str(output_path / 'policy_progression_timeline.png'),
                                 '--limit', str(args.games)]
        results['timeline'] = run_script('create_policy_timeline.py', script_args)
    else:
        print("\nSkipping policy progression timeline")

    # 2. Deception Heatmap
    if 'deception' not in skip_list:
        script_args = db_args + ['--output', str(output_path / 'deception_heatmap.png'),
                                 '--summary', str(output_path / 'deception_summary.png'),
                                 '--games', str(args.games)]
        results['deception'] = run_script('create_deception_heatmap.py', script_args)
    else:
        print("\nSkipping deception heatmap")

    # 3. Vote Network Graph
    if 'network' not in skip_list:
        script_args = db_args + ['--output', str(output_path / 'vote_network.png'),
                                 '--combined', '--games', str(args.games)]
        results['network'] = run_script('create_vote_network.py', script_args)
    else:
        print("\nSkipping vote network graph")

    # 4. Cost Efficiency Dashboard
    if 'cost' not in skip_list:
        script_args = db_args + ['--output', str(output_path / 'cost_dashboard.png'),
                                 '--games', str(args.games)]
        results['cost'] = run_script('create_cost_dashboard.py', script_args)
    else:
        print("\nSkipping cost dashboard")

    # 5. Batch Summary Infographic
    if 'summary' not in skip_list:
        script_args = db_args + ['--output', str(output_path / 'batch_summary.png'),
                                 '--games', str(args.games),
                                 '--name', batch_name]
        results['summary'] = run_script('create_batch_summary.py', script_args)
    else:
        print("\nSkipping batch summary")

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)

    success_count = sum(1 for code in results.values() if code == 0)
    total_count = len(results)

    for name, code in results.items():
        status = "✓ SUCCESS" if code == 0 else "✗ FAILED"
        print(f"{name.upper():20s}: {status}")

    print(f"\nTotal: {success_count}/{total_count} successful")

    if success_count == total_count:
        print("\n✓ All visualizations generated successfully!")
        print(f"\nOutput files in: {output_path.absolute()}/")
        print("\nGenerated files:")
        for file in sorted(output_path.glob('*.png')):
            print(f"  - {file.name}")
        return 0
    else:
        print(f"\n✗ {total_count - success_count} visualization(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
