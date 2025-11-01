#!/usr/bin/env python3
"""
Real-time batch experiment progress tracker.
Usage: python check_batch_progress.py [--watch]
"""
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import argparse
import json
import re

def count_games_since(start_time):
    """Count completed games since start_time."""
    logs_dir = Path(__file__).parent / "logs"

    if not logs_dir.exists():
        return 0, 0

    total_games = 0
    completed_games = 0

    for game_dir in logs_dir.iterdir():
        if not game_dir.is_dir():
            continue

        game_log = game_dir / "game.log"
        if not game_log.exists():
            continue

        # Check if game started after our start time
        mtime = datetime.fromtimestamp(game_log.stat().st_mtime)
        if mtime < start_time:
            continue

        total_games += 1

        # Check if game completed (look for duration_seconds at end of log)
        try:
            with open(game_log, 'r') as f:
                content = f.read()
                if 'duration_seconds' in content:
                    completed_games += 1
        except Exception:
            pass

    return total_games, completed_games

def parse_game_state(game_log_path):
    """Extract detailed game state from game log."""
    try:
        with open(game_log_path, 'r') as f:
            content = f.read()

        # Find the last JSON object containing game state
        state_pattern = r'"phase":\s*"([^"]+)"'
        phase_matches = list(re.finditer(state_pattern, content))
        phase = phase_matches[-1].group(1) if phase_matches else "unknown"

        # Extract policy counts
        lib_pattern = r'"liberal_policies":\s*(\d+)'
        fasc_pattern = r'"fascist_policies":\s*(\d+)'
        lib_matches = list(re.finditer(lib_pattern, content))
        fasc_matches = list(re.finditer(fasc_pattern, content))

        liberal_policies = int(lib_matches[-1].group(1)) if lib_matches else 0
        fascist_policies = int(fasc_matches[-1].group(1)) if fasc_matches else 0

        # Check for game end and winner
        game_end_pattern = r'"event":\s*"game_end"'
        winner_pattern = r'"winner":\s*"([^"]+)"'
        win_condition_pattern = r'"win_condition":\s*"([^"]+)"'

        is_complete = bool(re.search(game_end_pattern, content))
        winner_matches = list(re.finditer(winner_pattern, content))
        win_cond_matches = list(re.finditer(win_condition_pattern, content))

        winner = None
        win_condition = None
        if winner_matches:
            winner_val = winner_matches[-1].group(1)
            if winner_val != "null":
                winner = winner_val
        if win_cond_matches:
            win_cond_val = win_cond_matches[-1].group(1)
            if win_cond_val != "null":
                win_condition = win_cond_val

        # Extract government info
        pres_pattern = r'"president":\s*"([^"]+)"'
        chan_pattern = r'"chancellor":\s*"([^"]+)"'
        pres_matches = list(re.finditer(pres_pattern, content))
        chan_matches = list(re.finditer(chan_pattern, content))

        current_president = None
        current_chancellor = None
        if pres_matches:
            pres_val = pres_matches[-1].group(1)
            if pres_val not in ["null", ""]:
                current_president = pres_val
        if chan_matches:
            chan_val = chan_matches[-1].group(1)
            if chan_val not in ["null", ""]:
                current_chancellor = chan_val

        return {
            'phase': phase,
            'liberal_policies': liberal_policies,
            'fascist_policies': fascist_policies,
            'is_complete': is_complete,
            'winner': winner,
            'win_condition': win_condition,
            'current_president': current_president,
            'current_chancellor': current_chancellor
        }
    except Exception as e:
        return None

def get_latest_game_info():
    """Get info about the most recent game."""
    logs_dir = Path(__file__).parent / "logs"

    if not logs_dir.exists():
        return None

    # Find most recently modified game log
    latest_game = None
    latest_time = 0

    for game_dir in logs_dir.iterdir():
        if not game_dir.is_dir():
            continue

        game_log = game_dir / "game.log"
        if not game_log.exists():
            continue

        mtime = game_log.stat().st_mtime
        if mtime > latest_time:
            latest_time = mtime
            latest_game = game_dir

    if not latest_game:
        return None

    game_log = latest_game / "game.log"

    # Parse game state
    try:
        with open(game_log, 'r') as f:
            lines = sum(1 for _ in f)

        state = parse_game_state(game_log)
        if not state:
            state = {'phase': 'unknown', 'is_complete': False}

        return {
            'game_id': latest_game.name,
            'log_lines': lines,
            'completed': state['is_complete'],
            'last_modified': datetime.fromtimestamp(latest_time),
            **state  # Merge all state fields
        }
    except Exception:
        return None

def estimate_completion(started_games, target_games, elapsed_hours):
    """Estimate completion time."""
    if started_games == 0:
        return "Unknown"

    games_per_hour = started_games / elapsed_hours if elapsed_hours > 0 else 0
    if games_per_hour == 0:
        return "Unknown"

    remaining_games = target_games - started_games
    hours_remaining = remaining_games / games_per_hour

    return f"{hours_remaining:.1f} hours (~{int(hours_remaining * 60)} minutes)"

def display_progress(start_time, target_games=100, clear_screen=True):
    """Display current progress."""
    if clear_screen:
        os.system('clear' if os.name != 'nt' else 'cls')

    started, completed = count_games_since(start_time)
    elapsed = datetime.now() - start_time
    elapsed_hours = elapsed.total_seconds() / 3600

    latest_game = get_latest_game_info()

    print("=" * 70)
    print("SECRET HITLER BATCH EXPERIMENT PROGRESS")
    print("=" * 70)
    print(f"\nBatch started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time:  {int(elapsed.total_seconds() / 60)} minutes ({elapsed_hours:.2f} hours)")
    print()
    print(f"Games started:   {started}/{target_games} ({started/target_games*100:.1f}%)")
    print(f"Games completed: {completed}/{target_games} ({completed/target_games*100:.1f}%)")
    print()

    # Progress bar
    progress = started / target_games
    bar_length = 50
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"Progress: [{bar}] {progress*100:.1f}%")
    print()

    # Rate and ETA
    if started > 0 and elapsed_hours > 0:
        rate = started / elapsed_hours
        print(f"Rate: {rate:.2f} games/hour ({rate/60:.2f} games/minute)")

        remaining = target_games - started
        eta_hours = remaining / rate
        eta_time = datetime.now() + (datetime.now() - start_time) * (remaining / started) if started > 0 else None

        if eta_time:
            print(f"ETA:  {eta_time.strftime('%Y-%m-%d %H:%M:%S')} ({eta_hours:.1f} hours remaining)")
    print()

    # Latest game info
    if latest_game:
        print("Latest Game:")
        print(f"  ID: {latest_game['game_id'][:16]}...")
        status_icon = '✓ Completed' if latest_game['completed'] else '⏳ In Progress'
        print(f"  Status: {status_icon}")

        # Show game state details
        phase = latest_game.get('phase', 'unknown').replace('_', ' ').title()
        print(f"  Phase: {phase}")

        # Policy board
        lib_pol = latest_game.get('liberal_policies', 0)
        fasc_pol = latest_game.get('fascist_policies', 0)
        lib_bar = '🔵' * lib_pol + '⚪' * (5 - lib_pol)
        fasc_bar = '🔴' * fasc_pol + '⚪' * (6 - fasc_pol)
        print(f"  Policies: Liberal {lib_bar} ({lib_pol}/5) | Fascist {fasc_bar} ({fasc_pol}/6)")

        # Government
        if latest_game.get('current_president') or latest_game.get('current_chancellor'):
            pres = latest_game.get('current_president', 'None')
            chan = latest_game.get('current_chancellor', 'None')
            print(f"  Government: President={pres}, Chancellor={chan}")

        # Winner info if game completed
        if latest_game['completed']:
            winner = latest_game.get('winner', 'Unknown')
            win_condition = latest_game.get('win_condition', 'Unknown')
            if winner:
                win_cond_display = win_condition.replace('_', ' ').title() if win_condition else 'Unknown'
                print(f"  Winner: {winner} ({win_cond_display})")

        print(f"  Log lines: {latest_game['log_lines']:,}")
        print(f"  Last update: {latest_game['last_modified'].strftime('%H:%M:%S')}")

    print("=" * 70)
    print("\nPress Ctrl+C to stop watching")

def load_current_batch():
    """Load batch metadata from logs/.current_batch file."""
    metadata_file = Path(__file__).parent / "logs" / ".current_batch"

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not read batch metadata: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Track batch experiment progress')
    parser.add_argument('--watch', action='store_true', help='Watch progress in real-time')
    parser.add_argument('--start-time', type=str, help='Batch start time (YYYY-MM-DD HH:MM:SS) - auto-detected if not provided')
    parser.add_argument('--target', type=int, help='Target number of games - auto-detected if not provided')
    parser.add_argument('--interval', type=int, help='Update interval in seconds (watch mode)', default=5)
    parser.add_argument('--batch-id', type=str, help='Track specific batch by ID (uses latest if not provided)')

    args = parser.parse_args()

    # Try to load batch metadata
    batch_metadata = load_current_batch()

    # Determine start_time and target_games
    if args.start_time:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
    elif batch_metadata:
        start_time = datetime.strptime(batch_metadata['start_time'], '%Y-%m-%d %H:%M:%S')
        print(f"Auto-detected batch: {batch_metadata.get('batch_id', 'Unknown')}")
        if batch_metadata.get('batch_tag'):
            print(f"Batch tag: {batch_metadata['batch_tag']}")
    else:
        print("Error: No --start-time provided and no batch metadata found in logs/.current_batch")
        print("Run a batch with run_game.py --batch first, or specify --start-time manually")
        sys.exit(1)

    if args.target:
        target_games = args.target
    elif batch_metadata:
        target_games = batch_metadata['target_games']
    else:
        target_games = 100  # Default fallback

    if args.watch:
        print("Starting real-time progress tracker...")
        print(f"Monitoring games since {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        try:
            while True:
                display_progress(start_time, target_games, clear_screen=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nStopped monitoring.")
            display_progress(start_time, target_games, clear_screen=False)
    else:
        display_progress(start_time, target_games, clear_screen=False)

if __name__ == '__main__':
    main()
