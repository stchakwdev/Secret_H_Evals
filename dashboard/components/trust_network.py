"""
Trust Network Visualization Component.

Creates interactive force-directed graph showing trust relationships
between players in a Secret Hitler game.
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analytics.coalition_detector import CoalitionDetector, get_alignment_network_for_visualization


def create_trust_network_figure(
    data_loader,
    game_id: str,
    show_roles: bool = True,
    min_edge_weight: float = 0.3
) -> go.Figure:
    """
    Create interactive trust network visualization.

    Args:
        data_loader: DataLoader instance
        game_id: Game identifier
        show_roles: Whether to color nodes by actual roles
        min_edge_weight: Minimum edge weight to display

    Returns:
        Plotly Figure with network visualization
    """
    # Get voting history and roles
    voting_history = data_loader.get_voting_history(game_id)
    roles = data_loader.get_player_roles(game_id) if show_roles else {}

    if not voting_history:
        fig = go.Figure()
        fig.add_annotation(
            text="No voting data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Build alignment matrix and detect coalitions
    detector = CoalitionDetector(min_alignment_threshold=min_edge_weight)
    alignment_matrix, player_names = detector.build_vote_alignment_matrix(voting_history)

    if len(player_names) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No player data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    partition = detector.detect_coalitions(alignment_matrix, player_names)

    # Get network data for visualization
    network_data = get_alignment_network_for_visualization(
        alignment_matrix, player_names, partition, roles, min_edge_weight
    )

    # Create figure
    fig = go.Figure()

    # Add edges
    for edge in network_data['edges']:
        source_node = next(n for n in network_data['nodes'] if n['id'] == edge['source'])
        target_node = next(n for n in network_data['nodes'] if n['id'] == edge['target'])

        # Edge width based on weight
        width = edge['weight'] * 5

        # Edge color based on alignment
        color = 'rgba(46, 204, 113, 0.5)' if edge['same_coalition'] else 'rgba(189, 195, 199, 0.3)'

        fig.add_trace(go.Scatter(
            x=[source_node['x'], target_node['x'], None],
            y=[source_node['y'], target_node['y'], None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            hovertext=f"{edge['source']} ↔ {edge['target']}: {edge['weight']:.0%} alignment",
            showlegend=False
        ))

    # Add nodes
    node_x = [n['x'] for n in network_data['nodes']]
    node_y = [n['y'] for n in network_data['nodes']]
    node_colors = [n['color'] for n in network_data['nodes']]
    node_labels = [n['label'] for n in network_data['nodes']]
    node_roles = [n['role'] for n in network_data['nodes']]

    hover_text = [
        f"<b>{label}</b><br>Role: {role}<br>Coalition: {node['coalition']}"
        for label, role, node in zip(node_labels, node_roles, network_data['nodes'])
    ]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=node_labels,
        textposition='top center',
        hoverinfo='text',
        hovertext=hover_text,
        name='Players'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text="Vote Alignment Network",
            x=0.5
        ),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        hovermode='closest',
        annotations=[
            dict(
                text="Blue: Liberal | Red: Fascist | Black: Hitler",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    return fig


def create_trust_evolution_figure(
    data_loader,
    game_id: str,
    observer_player: Optional[str] = None
) -> go.Figure:
    """
    Create trust evolution over time visualization.

    Args:
        data_loader: DataLoader instance
        game_id: Game identifier
        observer_player: If specified, show this player's trust of others

    Returns:
        Plotly Figure with trust evolution lines
    """
    trust_evolution = data_loader.get_trust_evolution(game_id)
    roles = data_loader.get_player_roles(game_id)

    if not trust_evolution:
        fig = go.Figure()
        fig.add_annotation(
            text="No trust evolution data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    # Role colors
    role_colors = {
        'liberal': '#3498db',
        'fascist': '#e74c3c',
        'hitler': '#2c3e50',
        'unknown': '#95a5a6'
    }

    # If observer specified, show their trust of others
    if observer_player and observer_player in trust_evolution:
        snapshots = trust_evolution[observer_player]
        other_players = set()

        for snapshot in snapshots:
            beliefs = snapshot.get('beliefs', {})
            other_players.update(beliefs.keys())

        for target in other_players:
            turns = []
            trust_scores = []

            for snapshot in snapshots:
                turn = snapshot.get('turn', 0)
                beliefs = snapshot.get('beliefs', {})

                if target in beliefs:
                    # Trust score = P(liberal) - P(fascist+hitler)
                    belief = beliefs[target]
                    if isinstance(belief, dict):
                        trust = belief.get('liberal', 0.5) - belief.get('fascist', 0.25) - belief.get('hitler', 0.25)
                    else:
                        trust = float(belief) if belief else 0

                    turns.append(turn)
                    trust_scores.append(trust)

            if turns:
                role = roles.get(target, 'unknown')
                color = role_colors.get(role, '#95a5a6')

                fig.add_trace(go.Scatter(
                    x=turns,
                    y=trust_scores,
                    mode='lines+markers',
                    name=f"{target} ({role})",
                    line=dict(color=color),
                    marker=dict(size=6)
                ))

        fig.update_layout(
            title=f"{observer_player}'s Trust of Other Players Over Time",
            xaxis_title="Turn",
            yaxis_title="Trust Score",
            yaxis=dict(range=[-1, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    else:
        # Show average trust evolution for all players
        fig.add_annotation(
            text="Select a player to view their trust evolution",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    return fig


def create_coalition_purity_chart(
    data_loader,
    game_ids: List[str]
) -> go.Figure:
    """
    Create chart showing coalition purity across games.

    Args:
        data_loader: DataLoader instance
        game_ids: List of game identifiers

    Returns:
        Plotly Figure with purity scores
    """
    detector = CoalitionDetector()
    results = []

    for game_id in game_ids:
        voting_history = data_loader.get_voting_history(game_id)
        roles = data_loader.get_player_roles(game_id)

        if voting_history and roles:
            result = detector.analyze_game_coalitions(voting_history, roles)
            results.append({
                'game_id': game_id[:8],
                'purity': result.purity_score,
                'modularity': result.modularity,
                'num_coalitions': result.num_coalitions
            })

    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="No coalition data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    import pandas as pd
    df = pd.DataFrame(results)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['game_id'],
        y=df['purity'],
        name='Coalition Purity',
        marker_color='#3498db'
    ))

    fig.update_layout(
        title="Coalition Purity by Game",
        xaxis_title="Game",
        yaxis_title="Purity Score",
        yaxis=dict(range=[0, 1])
    )

    return fig
