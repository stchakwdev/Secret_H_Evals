#!/usr/bin/env python3
"""
Policy Progression Timeline Visualization

Creates a horizontal timeline showing liberal vs fascist policy progression
across games in the verification batch. Visualizes the strategic race to
enact 5 policies and win the game.

Author: Samuel Chakwera (stchakdev)
"""

import sqlite3
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse


def connect_db(db_path: str = None) -> sqlite3.Connection:
    """Connect to the games database."""
    if db_path is None:
        # Default to project data directory
        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "games.db"

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    return sqlite3.connect(db_path)


def extract_policy_events(game_data_json: str) -> List[Dict]:
    """Extract policy enactment events from game data JSON."""
    try:
        game_data = json.loads(game_data_json)
        events = []

        # Track cumulative policy counts
        liberal_count = 0
        fascist_count = 0

        # Extract from game history if available
        if 'history' in game_data:
            for entry in game_data['history']:
                if entry.get('type') == 'policy_enacted':
                    policy_type = entry.get('policy')
                    if policy_type == 'liberal':
                        liberal_count += 1
                    elif policy_type == 'fascist':
                        fascist_count += 1

                    events.append({
                        'policy': policy_type,
                        'liberal_count': liberal_count,
                        'fascist_count': fascist_count,
                        'round': len(events) + 1
                    })

        return events
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse game data: {e}")
        return []


def get_batch_games(conn: sqlite3.Connection, batch_id: str = None, limit: int = 10) -> List[Dict]:
    """
    Get games from database.

    Args:
        conn: Database connection
        batch_id: Optional batch ID to filter (uses latest games if None)
        limit: Maximum number of games to return

    Returns:
        List of game dictionaries with metadata and policy progressions
    """
    cursor = conn.cursor()

    if batch_id:
        # Filter by batch_id if provided (requires metadata in game_data_json)
        query = """
            SELECT game_id, winner, liberal_policies, fascist_policies,
                   game_data_json, timestamp
            FROM games
            WHERE json_extract(game_data_json, '$.batch_id') = ?
              AND winner IN ('liberal', 'fascist')
              AND (liberal_policies > 0 OR fascist_policies > 0)
            ORDER BY (liberal_policies + fascist_policies) DESC
        """
        cursor.execute(query, (batch_id,))
    else:
        # Get games with actual policy data (exclude TIMEOUT/errors)
        query = """
            SELECT game_id, winner, liberal_policies, fascist_policies,
                   game_data_json, timestamp
            FROM games
            WHERE winner IN ('liberal', 'fascist')
              AND (liberal_policies > 0 OR fascist_policies > 0)
            ORDER BY (liberal_policies + fascist_policies) DESC
            LIMIT ?
        """
        cursor.execute(query, (limit,))

    games = []
    for row in cursor.fetchall():
        game_id, winner, lib_final, fasc_final, game_data_json, timestamp = row

        # Extract policy progression events
        events = extract_policy_events(game_data_json)

        # If no events from history, reconstruct from final counts
        if not events and (lib_final > 0 or fasc_final > 0):
            # Simple reconstruction: alternate policies (not accurate but better than nothing)
            total = lib_final + fasc_final
            events = []
            lib_count = 0
            fasc_count = 0

            for i in range(total):
                if lib_count < lib_final and (fasc_count >= fasc_final or i % 2 == 0):
                    lib_count += 1
                    events.append({
                        'policy': 'liberal',
                        'liberal_count': lib_count,
                        'fascist_count': fasc_count,
                        'round': i + 1
                    })
                else:
                    fasc_count += 1
                    events.append({
                        'policy': 'fascist',
                        'liberal_count': lib_count,
                        'fascist_count': fasc_count,
                        'round': i + 1
                    })

        games.append({
            'game_id': game_id,
            'winner': winner,
            'liberal_final': lib_final,
            'fascist_final': fasc_final,
            'events': events,
            'timestamp': timestamp
        })

    return games


def get_interesting_games(conn: sqlite3.Connection, limit: int = 10) -> List[Dict]:
    """
    Select interesting games with variety of outcomes.

    Selects a mix of liberal and fascist wins, prioritizing games with
    high total policy counts (close, exciting games).

    Args:
        conn: Database connection
        limit: Maximum number of games to return

    Returns:
        List of game dictionaries with metadata and policy progressions
    """
    cursor = conn.cursor()

    # Get mix: ~50% liberal wins, ~50% fascist wins, sorted by policy count
    half_limit = max(1, limit // 2)

    query = """
        SELECT * FROM (
            SELECT game_id, winner, liberal_policies, fascist_policies,
                   game_data_json, timestamp,
                   (liberal_policies + fascist_policies) AS total_policies
            FROM games
            WHERE winner = 'liberal'
              AND (liberal_policies + fascist_policies) >= 5
            ORDER BY total_policies DESC
            LIMIT ?
        )
        UNION ALL
        SELECT * FROM (
            SELECT game_id, winner, liberal_policies, fascist_policies,
                   game_data_json, timestamp,
                   (liberal_policies + fascist_policies) AS total_policies
            FROM games
            WHERE winner = 'fascist'
              AND (liberal_policies + fascist_policies) >= 5
            ORDER BY total_policies DESC
            LIMIT ?
        )
        ORDER BY total_policies DESC
    """
    cursor.execute(query, (half_limit, limit - half_limit))

    games = []
    for row in cursor.fetchall():
        # Unpack 7 columns: ..., total_policies (ignored)
        game_id, winner, lib_final, fasc_final, game_data_json, timestamp, _ = row

        # Extract policy progression events
        events = extract_policy_events(game_data_json)

        # If no events from history, reconstruct from final counts
        if not events and (lib_final > 0 or fasc_final > 0):
            total = lib_final + fasc_final
            events = []
            lib_count = 0
            fasc_count = 0

            for i in range(total):
                if lib_count < lib_final and (fasc_count >= fasc_final or i % 2 == 0):
                    lib_count += 1
                    events.append({
                        'policy': 'liberal',
                        'liberal_count': lib_count,
                        'fascist_count': fasc_count,
                        'round': i + 1
                    })
                else:
                    fasc_count += 1
                    events.append({
                        'policy': 'fascist',
                        'liberal_count': lib_count,
                        'fascist_count': fasc_count,
                        'round': i + 1
                    })

        games.append({
            'game_id': game_id,
            'winner': winner,
            'liberal_final': lib_final,
            'fascist_final': fasc_final,
            'events': events,
            'timestamp': timestamp
        })

    return games


def create_policy_timeline(games: List[Dict], output_path: str):
    """
    Create policy progression timeline visualization.

    Args:
        games: List of game dictionaries with policy events
        output_path: Path to save the PNG visualization
    """
    n_games = len(games)
    if n_games == 0:
        print("No games found to visualize")
        return

    # Figure setup
    fig, axes = plt.subplots(n_games, 1, figsize=(14, 3 * n_games),
                            constrained_layout=True)

    if n_games == 1:
        axes = [axes]

    # Color scheme
    liberal_color = '#1f77b4'  # Blue
    fascist_color = '#d62728'  # Red
    grid_color = '#e0e0e0'

    for idx, (game, ax) in enumerate(zip(games, axes)):
        events = game['events']

        # Setup axis
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1, 2)
        ax.set_aspect('equal')

        # Draw policy tracks
        liberal_y = 1
        fascist_y = 0

        # Draw track backgrounds
        ax.add_patch(patches.Rectangle(
            (0, liberal_y - 0.3), 10, 0.6,
            facecolor=liberal_color, alpha=0.1, edgecolor=liberal_color, linewidth=2
        ))
        ax.add_patch(patches.Rectangle(
            (0, fascist_y - 0.3), 10, 0.6,
            facecolor=fascist_color, alpha=0.1, edgecolor=fascist_color, linewidth=2
        ))

        # Draw tick marks
        for i in range(6):
            ax.plot([i, i], [liberal_y - 0.35, liberal_y - 0.45],
                   color=liberal_color, linewidth=2)
            ax.plot([i, i], [fascist_y - 0.35, fascist_y - 0.45],
                   color=fascist_color, linewidth=2)

        # Track labels
        ax.text(-0.3, liberal_y, 'Liberal', fontsize=12, fontweight='bold',
               color=liberal_color, va='center', ha='right')
        ax.text(-0.3, fascist_y, 'Fascist', fontsize=12, fontweight='bold',
               color=fascist_color, va='center', ha='right')

        # Draw policy enactments
        lib_count = 0
        fasc_count = 0

        for event in events:
            if event['policy'] == 'liberal':
                lib_count = event['liberal_count']
                # Draw liberal policy marker
                circle = patches.Circle(
                    (lib_count - 1, liberal_y), 0.25,
                    facecolor=liberal_color, edgecolor='white', linewidth=2, zorder=10
                )
                ax.add_patch(circle)
                ax.text(lib_count - 1, liberal_y, str(event['round']),
                       fontsize=9, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11)

            elif event['policy'] == 'fascist':
                fasc_count = event['fascist_count']
                # Draw fascist policy marker
                circle = patches.Circle(
                    (fasc_count - 1, fascist_y), 0.25,
                    facecolor=fascist_color, edgecolor='white', linewidth=2, zorder=10
                )
                ax.add_patch(circle)
                ax.text(fasc_count - 1, fascist_y, str(event['round']),
                       fontsize=9, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11)

        # Game outcome annotation
        winner_text = f"Winner: {game['winner'].upper()}"
        final_score = f"({game['liberal_final']}-{game['fascist_final']})"

        winner_color = liberal_color if game['winner'] == 'liberal' else fascist_color

        ax.text(10.3, 0.5, winner_text, fontsize=11, fontweight='bold',
               color=winner_color, va='center')
        ax.text(10.3, 0.1, final_score, fontsize=9, color='gray', va='center')

        # Game title
        game_num = idx + 1
        ax.set_title(f"Game {game_num} - {game['game_id'][:8]}...",
                    fontsize=12, fontweight='bold', pad=10)

        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Overall title - y=1.02 places it above figure, bbox_inches='tight' captures it
    fig.suptitle('Policy Progression Timeline - Secret Hitler LLM Evaluation',
                fontsize=16, fontweight='bold', y=1.02)

    # Save figure with support for multiple formats
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect format from extension and save appropriately
    ext = output_path.suffix.lower()
    if ext in ['.svg', '.pdf', '.eps']:
        plt.savefig(output_path, format=ext[1:], bbox_inches='tight', facecolor='white')
    else:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Saved: {output_path}")

    # Also save vector formats for publication if requested
    if ext == '.png':
        # Auto-generate SVG and PDF versions
        svg_path = output_path.with_suffix('.svg')
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"Saved: {svg_path}")
        print(f"Saved: {pdf_path}")

    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create policy progression timeline visualization"
    )
    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )
    parser.add_argument(
        '--batch-id', '-b',
        help='Batch ID to filter games (default: use latest games)'
    )
    parser.add_argument(
        '--output', '-o',
        default='visualizations/policy_progression_timeline.png',
        help='Output path for PNG (default: visualizations/policy_progression_timeline.png)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Maximum number of games to visualize (default: 10)'
    )
    parser.add_argument(
        '--interesting', '-i',
        action='store_true',
        help='Select interesting games (close games with variety of outcomes)'
    )

    args = parser.parse_args()

    try:
        # Connect to database
        conn = connect_db(args.db)

        # Get games based on selection mode
        if args.interesting:
            games = get_interesting_games(conn, args.limit)
            print(f"Selected {len(games)} interesting games (mix of liberal/fascist wins)")
        else:
            games = get_batch_games(conn, args.batch_id, args.limit)
            print(f"Found {len(games)} games to visualize")

        # Create visualization
        create_policy_timeline(games, args.output)

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
