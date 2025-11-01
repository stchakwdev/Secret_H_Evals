#!/usr/bin/env python3
"""
Vote Alignment Network Graph Visualization

Creates a force-directed network graph showing voting patterns and
player alignment. Nodes represent players, edges represent vote
agreement frequency, colored by final roles.

Author: Samuel Chakwera (stchakdev)
"""

import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from collections import defaultdict


def connect_db(db_path: str = None) -> sqlite3.Connection:
    """Connect to the games database."""
    if db_path is None:
        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "games.db"

    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    return sqlite3.connect(db_path)


def extract_vote_data(conn: sqlite3.Connection, game_id: str) -> Dict:
    """
    Extract voting data from a specific game.

    Args:
        conn: Database connection
        game_id: Game ID to analyze

    Returns:
        Dictionary with player roles and vote alignments
    """
    # Get game data for player roles
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_data_json FROM games WHERE game_id = ?
    """, (game_id,))

    row = cursor.fetchone()
    if not row:
        return None

    try:
        game_data = json.loads(row[0])
    except json.JSONDecodeError:
        return None

    # Extract player roles
    player_roles = {}
    if 'players' in game_data:
        for player in game_data['players']:
            player_roles[player.get('name')] = {
                'role': player.get('role', 'unknown'),
                'party': player.get('party', 'unknown')
            }

    # Get voting data from turns
    cursor.execute("""
        SELECT action_data_json
        FROM turns
        WHERE game_id = ? AND action_type = 'vote'
        ORDER BY turn_number
    """, (game_id,))

    votes = []
    for row in cursor.fetchall():
        try:
            vote_data = json.loads(row[0])
            if 'votes' in vote_data:
                votes.append(vote_data['votes'])
        except json.JSONDecodeError:
            continue

    return {
        'game_id': game_id,
        'player_roles': player_roles,
        'votes': votes
    }


def calculate_vote_alignment(votes: List[Dict]) -> Dict[Tuple[str, str], float]:
    """
    Calculate vote alignment between all player pairs.

    Args:
        votes: List of vote dictionaries {player_name: True/False}

    Returns:
        Dictionary mapping (player1, player2) -> alignment_score (0-1)
    """
    alignment = defaultdict(lambda: {'agreed': 0, 'total': 0})

    for vote_round in votes:
        players = list(vote_round.keys())

        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                if player1 in vote_round and player2 in vote_round:
                    # Both voted same way
                    if vote_round[player1] == vote_round[player2]:
                        alignment[(player1, player2)]['agreed'] += 1
                    alignment[(player1, player2)]['total'] += 1

    # Calculate alignment scores
    scores = {}
    for pair, stats in alignment.items():
        if stats['total'] > 0:
            scores[pair] = stats['agreed'] / stats['total']

    return scores


def create_vote_network(vote_data: Dict, output_path: str):
    """
    Create force-directed network graph of voting patterns.

    Args:
        vote_data: Dictionary with player_roles and votes
        output_path: Path to save PNG
    """
    if not vote_data or not vote_data['votes']:
        print("No vote data available")
        return

    # Calculate alignment scores
    alignment_scores = calculate_vote_alignment(vote_data['votes'])

    # Create network graph
    G = nx.Graph()

    # Add nodes (players)
    player_roles = vote_data['player_roles']
    for player, role_data in player_roles.items():
        G.add_node(player, **role_data)

    # Add edges (vote alignment)
    for (player1, player2), score in alignment_scores.items():
        if score > 0.3:  # Only show edges with >30% alignment
            G.add_edge(player1, player2, weight=score)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors by party
    node_colors = []
    for node in G.nodes():
        party = G.nodes[node].get('party', 'unknown')
        if party == 'liberal':
            node_colors.append('#1f77b4')  # Blue
        elif party == 'fascist':
            node_colors.append('#d62728')  # Red
        else:
            node_colors.append('#7f7f7f')  # Gray

    # Edge colors and widths by alignment strength
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        # Color gradient: green (weak) -> yellow (medium) -> orange (strong)
        if weight < 0.5:
            edge_colors.append('#90ee90')  # Light green
        elif weight < 0.75:
            edge_colors.append('#ffd700')  # Gold
        else:
            edge_colors.append('#ff8c00')  # Dark orange

        edge_widths.append(weight * 5)  # Scale width by alignment

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500,
                          alpha=0.9, ax=ax, edgecolors='black', linewidths=2)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=edge_widths, alpha=0.6, ax=ax)

    # Draw labels
    labels = {}
    for node in G.nodes():
        role = G.nodes[node].get('role', '?')
        labels[node] = f"{node}\n({role})"

    nx.draw_networkx_labels(G, pos, labels, font_size=9,
                           font_weight='bold', ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Liberal', edgecolor='black'),
        Patch(facecolor='#d62728', label='Fascist', edgecolor='black'),
        Patch(facecolor='#7f7f7f', label='Unknown', edgecolor='black'),
        plt.Line2D([0], [0], color='#90ee90', linewidth=3, label='Vote Alignment: 30-50%'),
        plt.Line2D([0], [0], color='#ffd700', linewidth=4, label='Vote Alignment: 50-75%'),
        plt.Line2D([0], [0], color='#ff8c00', linewidth=5, label='Vote Alignment: 75-100%')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    ax.set_title(f'Vote Alignment Network - Game {vote_data["game_id"][:8]}...',
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Vote network saved to: {output_path}")

    plt.close()


def create_combined_network(conn: sqlite3.Connection, game_limit: int,
                            output_path: str):
    """
    Create combined vote network from multiple games.

    Args:
        conn: Database connection
        game_limit: Number of recent games to include
        output_path: Path to save PNG
    """
    # Get recent game IDs
    cursor = conn.cursor()
    cursor.execute("""
        SELECT game_id FROM games
        ORDER BY timestamp DESC
        LIMIT ?
    """, (game_limit,))

    game_ids = [row[0] for row in cursor.fetchall()]

    if not game_ids:
        print("No games found")
        return

    # Aggregate alignment across all games
    all_alignments = defaultdict(lambda: {'agreed': 0, 'total': 0})
    player_stats = defaultdict(lambda: {'liberal': 0, 'fascist': 0, 'games': 0})

    for game_id in game_ids:
        vote_data = extract_vote_data(conn, game_id)
        if not vote_data or not vote_data['votes']:
            continue

        # Track player party affiliations
        for player, role_data in vote_data['player_roles'].items():
            party = role_data.get('party', 'unknown')
            if party in ['liberal', 'fascist']:
                player_stats[player][party] += 1
                player_stats[player]['games'] += 1

        # Aggregate alignments
        alignment = calculate_vote_alignment(vote_data['votes'])
        for pair, score in alignment.items():
            # Assume each vote round is one data point
            votes_per_pair = len([v for v in vote_data['votes']
                                 if pair[0] in v and pair[1] in v])
            agreed = int(score * votes_per_pair)
            all_alignments[pair]['agreed'] += agreed
            all_alignments[pair]['total'] += votes_per_pair

    # Calculate overall alignment scores
    alignment_scores = {}
    for pair, stats in all_alignments.items():
        if stats['total'] >= 3:  # At least 3 votes together
            alignment_scores[pair] = stats['agreed'] / stats['total']

    # Create graph
    G = nx.Graph()

    # Add nodes with predominant party
    for player, stats in player_stats.items():
        if stats['games'] > 0:
            if stats['liberal'] > stats['fascist']:
                party = 'liberal'
            elif stats['fascist'] > stats['liberal']:
                party = 'fascist'
            else:
                party = 'mixed'

            G.add_node(player, party=party, games=stats['games'])

    # Add edges
    for (player1, player2), score in alignment_scores.items():
        if score > 0.4 and player1 in G.nodes() and player2 in G.nodes():
            G.add_edge(player1, player2, weight=score)

    if len(G.nodes()) == 0:
        print("No network data to visualize")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

    # Node colors
    node_colors = []
    for node in G.nodes():
        party = G.nodes[node].get('party', 'unknown')
        if party == 'liberal':
            node_colors.append('#1f77b4')
        elif party == 'fascist':
            node_colors.append('#d62728')
        else:
            node_colors.append('#9467bd')  # Purple for mixed

    # Node sizes by games played
    node_sizes = [G.nodes[node].get('games', 1) * 300 for node in G.nodes()]

    # Edge colors and widths
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight < 0.5:
            edge_colors.append('#b0e0b0')
        elif weight < 0.7:
            edge_colors.append('#ffe680')
        else:
            edge_colors.append('#ffb366')
        edge_widths.append(weight * 4)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                          width=edge_widths, alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Mostly Liberal', edgecolor='black'),
        Patch(facecolor='#d62728', label='Mostly Fascist', edgecolor='black'),
        Patch(facecolor='#9467bd', label='Mixed Roles', edgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=8, label='Node size = Games played'),
        plt.Line2D([0], [0], color='#b0e0b0', linewidth=3, label='Alignment: 40-50%'),
        plt.Line2D([0], [0], color='#ffe680', linewidth=4, label='Alignment: 50-70%'),
        plt.Line2D([0], [0], color='#ffb366', linewidth=5, label='Alignment: 70-100%')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_title(f'Combined Vote Alignment Network - {len(game_ids)} Games',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Combined vote network saved to: {output_path}")

    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create vote alignment network visualization"
    )
    parser.add_argument(
        '--db', '-d',
        help='Path to games.db (default: data/games.db)'
    )
    parser.add_argument(
        '--output', '-o',
        default='visualizations/vote_network.png',
        help='Output path for PNG'
    )
    parser.add_argument(
        '--game-id', '-g',
        help='Specific game ID to visualize (default: latest game)'
    )
    parser.add_argument(
        '--combined', '-c',
        action='store_true',
        help='Create combined network from multiple games'
    )
    parser.add_argument(
        '--games', '-n',
        type=int,
        default=10,
        help='Number of games for combined network (default: 10)'
    )

    args = parser.parse_args()

    try:
        conn = connect_db(args.db)

        if args.combined:
            # Create combined network
            print(f"Creating combined network from {args.games} games...")
            create_combined_network(conn, args.games, args.output)
        else:
            # Single game network
            if args.game_id:
                game_id = args.game_id
            else:
                # Get latest game
                cursor = conn.cursor()
                cursor.execute("SELECT game_id FROM games ORDER BY timestamp DESC LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    print("No games found in database")
                    return 1
                game_id = row[0]

            print(f"Creating network for game {game_id[:8]}...")
            vote_data = extract_vote_data(conn, game_id)
            if vote_data:
                create_vote_network(vote_data, args.output)
            else:
                print(f"No data found for game {game_id}")
                return 1

        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
