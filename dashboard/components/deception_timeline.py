"""
Deception Timeline Visualization Component.

Creates heatmaps showing deception patterns over game turns.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def create_deception_heatmap(
    data_loader,
    game_id: str,
    show_tooltips: bool = True
) -> go.Figure:
    """
    Create deception timeline heatmap.

    Shows deception score for each player at each turn.

    Args:
        data_loader: DataLoader instance
        game_id: Game identifier
        show_tooltips: Whether to show detailed tooltips

    Returns:
        Plotly Figure with heatmap visualization
    """
    # Get deception data
    df = data_loader.get_deception_timeline_data(game_id)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No deception data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Pivot to create matrix: players x turns
    pivot_df = df.pivot_table(
        index='player_name',
        columns='turn',
        values='deception_score',
        aggfunc='mean'
    ).fillna(0)

    # Sort players by total deception
    player_totals = pivot_df.sum(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[player_totals.index]

    # Create hover text with details
    if show_tooltips:
        hover_df = df.pivot_table(
            index='player_name',
            columns='turn',
            values=['reasoning', 'public_statement', 'decision_type'],
            aggfunc='first'
        )

        hover_text = []
        for player in pivot_df.index:
            player_hover = []
            for turn in pivot_df.columns:
                score = pivot_df.loc[player, turn]
                try:
                    reasoning = hover_df.loc[player, ('reasoning', turn)][:100] if pd.notna(hover_df.loc[player, ('reasoning', turn)]) else "N/A"
                    statement = hover_df.loc[player, ('public_statement', turn)][:100] if pd.notna(hover_df.loc[player, ('public_statement', turn)]) else "N/A"
                    decision = hover_df.loc[player, ('decision_type', turn)] if pd.notna(hover_df.loc[player, ('decision_type', turn)]) else "N/A"
                except (KeyError, TypeError):
                    reasoning = "N/A"
                    statement = "N/A"
                    decision = "N/A"

                text = f"<b>{player}</b> - Turn {turn}<br>"
                text += f"Score: {score:.2f}<br>"
                text += f"Type: {decision}<br>"
                text += f"<br><b>Private:</b> {reasoning}...<br>"
                text += f"<b>Public:</b> {statement}..."
                player_hover.append(text)
            hover_text.append(player_hover)
    else:
        hover_text = None

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[f"Turn {t}" for t in pivot_df.columns],
        y=pivot_df.index,
        colorscale='Reds',
        colorbar=dict(title="Deception<br>Score"),
        hovertemplate='%{hovertext}<extra></extra>' if hover_text else '%{y}<br>Turn %{x}<br>Score: %{z:.2f}<extra></extra>',
        hovertext=hover_text
    ))

    fig.update_layout(
        title="Deception Timeline by Player",
        xaxis_title="Game Turn",
        yaxis_title="Player",
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_deception_trend_chart(
    data_loader,
    game_ids: Optional[List[str]] = None
) -> go.Figure:
    """
    Create chart showing deception trends over game progression.

    Args:
        data_loader: DataLoader instance
        game_ids: Optional list of games to analyze

    Returns:
        Plotly Figure with trend lines
    """
    all_decisions = data_loader.get_player_decisions(limit=10000)

    if all_decisions.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No deception data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Filter by game_ids if provided
    if game_ids:
        all_decisions = all_decisions[all_decisions['game_id'].isin(game_ids)]

    # Group by turn and calculate mean deception
    turn_stats = all_decisions.groupby('turn_number').agg({
        'deception_score': ['mean', 'std', 'count'],
        'is_deception': 'mean'
    }).reset_index()

    turn_stats.columns = ['turn', 'mean_score', 'std_score', 'count', 'deception_rate']

    # Filter turns with enough data
    turn_stats = turn_stats[turn_stats['count'] >= 5]

    if turn_stats.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    # Mean deception score with confidence band
    fig.add_trace(go.Scatter(
        x=turn_stats['turn'],
        y=turn_stats['mean_score'],
        mode='lines+markers',
        name='Mean Deception Score',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))

    # Add confidence band (mean ± 1 std)
    fig.add_trace(go.Scatter(
        x=list(turn_stats['turn']) + list(turn_stats['turn'][::-1]),
        y=list(turn_stats['mean_score'] + turn_stats['std_score']) +
          list((turn_stats['mean_score'] - turn_stats['std_score'])[::-1]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
        showlegend=True
    ))

    # Deception rate on secondary axis
    fig.add_trace(go.Scatter(
        x=turn_stats['turn'],
        y=turn_stats['deception_rate'],
        mode='lines+markers',
        name='Deception Rate',
        line=dict(color='#9b59b6', width=2, dash='dash'),
        marker=dict(size=6),
        yaxis='y2'
    ))

    fig.update_layout(
        title="Deception Trends Over Game Progression",
        xaxis_title="Turn Number",
        yaxis=dict(title="Deception Score", range=[0, 1]),
        yaxis2=dict(
            title="Deception Rate",
            overlaying='y',
            side='right',
            range=[0, 1],
            tickformat='.0%'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_deception_by_role_chart(
    data_loader,
    game_ids: Optional[List[str]] = None
) -> go.Figure:
    """
    Create chart comparing deception rates by player role.

    Args:
        data_loader: DataLoader instance
        game_ids: Optional list of games to filter

    Returns:
        Plotly Figure with role comparison
    """
    games_df = data_loader.get_all_games()
    decisions_df = data_loader.get_player_decisions(limit=10000)

    if decisions_df.empty or games_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for role analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Build role mapping from games
    role_mapping = {}
    for _, game in games_df.iterrows():
        game_id = game['game_id']
        game_details = data_loader.get_game_details(game_id)
        game_data = game_details.get('game_data', {})

        for player_id, player_info in game_data.get('players', {}).items():
            key = (game_id, player_id)
            role_mapping[key] = player_info.get('role', 'unknown')

    # Add role to decisions
    decisions_df['role'] = decisions_df.apply(
        lambda row: role_mapping.get((row['game_id'], row['player_id']), 'unknown'),
        axis=1
    )

    # Normalize roles
    role_map = {'liberal': 'Liberal', 'fascist': 'Fascist', 'hitler': 'Hitler'}
    decisions_df['role_clean'] = decisions_df['role'].map(role_map).fillna('Unknown')

    # Calculate stats by role
    role_stats = decisions_df.groupby('role_clean').agg({
        'deception_score': ['mean', 'std', 'count'],
        'is_deception': 'mean'
    }).reset_index()

    role_stats.columns = ['role', 'mean_score', 'std_score', 'count', 'deception_rate']
    role_stats = role_stats[role_stats['role'] != 'Unknown']

    # Colors by role
    colors = {'Liberal': '#3498db', 'Fascist': '#e74c3c', 'Hitler': '#2c3e50'}

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=role_stats['role'],
        y=role_stats['deception_rate'],
        marker_color=[colors.get(r, '#95a5a6') for r in role_stats['role']],
        error_y=dict(
            type='data',
            array=role_stats['std_score'] / np.sqrt(role_stats['count']),  # SE
            visible=True
        ),
        text=[f"{r:.1%}" for r in role_stats['deception_rate']],
        textposition='outside'
    ))

    fig.update_layout(
        title="Deception Rate by Player Role",
        xaxis_title="Role",
        yaxis_title="Deception Rate",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False
    )

    return fig


def create_reasoning_vs_statement_comparison(
    data_loader,
    game_id: str,
    player_name: Optional[str] = None
) -> go.Figure:
    """
    Create visualization comparing private reasoning vs public statements.

    Args:
        data_loader: DataLoader instance
        game_id: Game identifier
        player_name: Optional player to focus on

    Returns:
        Plotly Figure with text comparison
    """
    df = data_loader.get_deception_timeline_data(game_id)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    if player_name:
        df = df[df['player_name'] == player_name]

    # Get high deception instances
    high_deception = df[df['deception_score'] > 0.5].sort_values('deception_score', ascending=False)

    if high_deception.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No high-deception instances found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create table-like visualization
    fig = go.Figure()

    # Add as annotations for readable text
    y_pos = 0.95
    for i, (_, row) in enumerate(high_deception.head(5).iterrows()):
        reasoning = row['reasoning'][:150] + "..." if len(str(row['reasoning'])) > 150 else str(row['reasoning'])
        statement = row['public_statement'][:150] + "..." if len(str(row['public_statement'])) > 150 else str(row['public_statement'])

        fig.add_annotation(
            text=f"<b>Turn {row['turn']} - {row['player_name']} (Score: {row['deception_score']:.2f})</b><br>"
                 f"<b>Private:</b> {reasoning}<br>"
                 f"<b>Public:</b> {statement}",
            xref="paper", yref="paper",
            x=0, y=y_pos - i * 0.2,
            showarrow=False,
            align="left",
            font=dict(size=10),
            xanchor="left"
        )

    fig.update_layout(
        title="High Deception Examples: Private Reasoning vs Public Statement",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=500
    )

    return fig
