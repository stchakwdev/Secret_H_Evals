"""
Main Dash Application for Secret Hitler LLM Evaluation Dashboard.

Provides interactive visualizations for research analysis including:
- Game overview with aggregate statistics
- Trust network visualization
- Deception timeline heatmaps
- Model comparison with confidence intervals
- Coalition analysis

Usage:
    python -m dashboard.app [--port PORT] [--debug] [--db-path PATH]
"""

import argparse
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from .data_loader import get_data_loader, DataLoader
from .components.trust_network import create_trust_network_figure
from .components.deception_timeline import create_deception_heatmap
from .components.model_comparison import create_model_comparison_figure

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis import (
    calculate_proportion_ci,
    summarize_game_statistics,
    format_ci_for_display,
)


# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Secret Hitler LLM Evaluation Dashboard"
)

# Initialize data loader (will be set with actual path at runtime)
data_loader: DataLoader = None


def create_navbar():
    """Create navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Secret Hitler LLM Evaluation", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Game Analysis", href="/game", active="exact")),
                dbc.NavItem(dbc.NavLink("Model Comparison", href="/models", active="exact")),
                dbc.NavItem(dbc.NavLink("Deception Analysis", href="/deception", active="exact")),
            ], className="ms-auto", navbar=True),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4"
    )


def create_overview_cards(stats: dict):
    """Create summary statistic cards."""
    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Total Games"),
            dbc.CardBody([
                html.H2(stats.get('total_games', 0), className="card-title"),
            ])
        ], color="primary", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Liberal Win Rate"),
            dbc.CardBody([
                html.H2(f"{stats.get('liberal_win_rate', 0):.1%}", className="card-title"),
                html.Small(f"{stats.get('liberal_wins', 0)} / {stats.get('total_games', 0)}")
            ])
        ], color="info", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Total Cost"),
            dbc.CardBody([
                html.H2(f"${stats.get('total_cost', 0):.2f}", className="card-title"),
                html.Small(f"Avg: ${stats.get('avg_cost_per_game', 0):.3f}/game")
            ])
        ], color="success", inverse=True), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("Avg Duration"),
            dbc.CardBody([
                html.H2(f"{stats.get('avg_duration', 0)/60:.1f} min", className="card-title"),
            ])
        ], color="warning", inverse=True), width=3),
    ], className="mb-4")


def create_overview_layout():
    """Create main overview page layout."""
    return html.Div([
        html.H2("Dashboard Overview", className="mb-4"),

        # Stats cards placeholder (populated by callback)
        html.Div(id="overview-stats-cards"),

        dbc.Row([
            # Win rate chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Win Rates by Team"),
                    dbc.CardBody([
                        dcc.Graph(id="win-rate-chart")
                    ])
                ])
            ], width=6),

            # Cost over time
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cumulative Cost Over Games"),
                    dbc.CardBody([
                        dcc.Graph(id="cost-timeline-chart")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            # Games table
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Games"),
                    dbc.CardBody([
                        html.Div(id="games-table")
                    ])
                ])
            ], width=12),
        ])
    ])


def create_game_analysis_layout():
    """Create single game analysis page layout."""
    return html.Div([
        html.H2("Game Analysis", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Select Game"),
                dcc.Dropdown(id="game-selector", placeholder="Select a game...")
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            # Trust network
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Trust Network"),
                    dbc.CardBody([
                        dcc.Graph(id="trust-network-graph")
                    ])
                ])
            ], width=6),

            # Deception timeline
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Deception Timeline"),
                    dbc.CardBody([
                        dcc.Graph(id="deception-timeline-graph")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            # Game details
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Game Details"),
                    dbc.CardBody(id="game-details-content")
                ])
            ], width=12),
        ])
    ])


def create_model_comparison_layout():
    """Create model comparison page layout."""
    return html.Div([
        html.H2("Model Comparison", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Win Rates by Model (with 95% CI)"),
                    dbc.CardBody([
                        dcc.Graph(id="model-winrate-chart")
                    ])
                ])
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cost Efficiency by Model"),
                    dbc.CardBody([
                        dcc.Graph(id="model-cost-chart")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Performance Statistics"),
                    dbc.CardBody(id="model-stats-table")
                ])
            ], width=12),
        ])
    ])


def create_deception_analysis_layout():
    """Create deception analysis page layout."""
    return html.Div([
        html.H2("Deception Analysis", className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Deception Rate by Decision Type"),
                    dbc.CardBody([
                        dcc.Graph(id="deception-by-type-chart")
                    ])
                ])
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Deception Score Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="deception-distribution-chart")
                    ])
                ])
            ], width=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Deceivers"),
                    dbc.CardBody(id="top-deceivers-table")
                ])
            ], width=12),
        ])
    ])


# Main layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    dbc.Container([
        html.Div(id='page-content')
    ], fluid=True)
])


# Callbacks

@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route to appropriate page based on URL."""
    if pathname == '/game':
        return create_game_analysis_layout()
    elif pathname == '/models':
        return create_model_comparison_layout()
    elif pathname == '/deception':
        return create_deception_analysis_layout()
    else:
        return create_overview_layout()


@callback(
    [Output('overview-stats-cards', 'children'),
     Output('win-rate-chart', 'figure'),
     Output('cost-timeline-chart', 'figure'),
     Output('games-table', 'children')],
    Input('url', 'pathname')
)
def update_overview(pathname):
    """Update overview page data."""
    if pathname != '/' and pathname is not None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if data_loader is None:
        return html.Div("No database loaded"), go.Figure(), go.Figure(), html.Div()

    try:
        stats = data_loader.get_aggregate_statistics()
        games_df = data_loader.get_all_games()

        # Stats cards
        cards = create_overview_cards(stats)

        # Win rate chart with CI
        liberal_wins = stats.get('liberal_wins', 0)
        total = stats.get('total_games', 1)
        fascist_wins = stats.get('fascist_wins', 0)

        lib_rate, lib_lower, lib_upper = calculate_proportion_ci(liberal_wins, total)
        fas_rate, fas_lower, fas_upper = calculate_proportion_ci(fascist_wins, total)

        win_fig = go.Figure()
        win_fig.add_trace(go.Bar(
            x=['Liberal', 'Fascist'],
            y=[lib_rate * 100, fas_rate * 100],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[(lib_upper - lib_rate) * 100, (fas_upper - fas_rate) * 100],
                arrayminus=[(lib_rate - lib_lower) * 100, (fas_rate - fas_lower) * 100]
            ),
            marker_color=['#3498db', '#e74c3c']
        ))
        win_fig.update_layout(
            yaxis_title="Win Rate (%)",
            yaxis_range=[0, 100],
            showlegend=False
        )

        # Cost timeline
        if not games_df.empty:
            games_df = games_df.sort_values('timestamp')
            games_df['cumulative_cost'] = games_df['total_cost'].cumsum()

            cost_fig = px.line(
                games_df,
                x=range(len(games_df)),
                y='cumulative_cost',
                labels={'x': 'Game Number', 'cumulative_cost': 'Cumulative Cost ($)'}
            )
        else:
            cost_fig = go.Figure()

        # Games table
        if not games_df.empty:
            table_df = games_df[['game_id', 'timestamp', 'winning_team', 'total_cost', 'duration_seconds']].head(10)
            table_df['duration_seconds'] = (table_df['duration_seconds'] / 60).round(1)
            table_df.columns = ['Game ID', 'Timestamp', 'Winner', 'Cost ($)', 'Duration (min)']

            table = dbc.Table.from_dataframe(
                table_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
        else:
            table = html.Div("No games found")

        return cards, win_fig, cost_fig, table

    except Exception as e:
        return html.Div(f"Error: {e}"), go.Figure(), go.Figure(), html.Div()


@callback(
    Output('game-selector', 'options'),
    Input('url', 'pathname')
)
def populate_game_selector(pathname):
    """Populate game selector dropdown."""
    if data_loader is None:
        return []

    try:
        games_df = data_loader.get_all_games()
        options = [
            {'label': f"{row['game_id'][:8]}... ({row['winning_team']})", 'value': row['game_id']}
            for _, row in games_df.head(50).iterrows()
        ]
        return options
    except Exception:
        return []


@callback(
    [Output('trust-network-graph', 'figure'),
     Output('deception-timeline-graph', 'figure'),
     Output('game-details-content', 'children')],
    Input('game-selector', 'value')
)
def update_game_analysis(game_id):
    """Update game analysis visualizations."""
    if not game_id or data_loader is None:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, html.Div("Select a game to view details")

    try:
        # Trust network
        trust_fig = create_trust_network_figure(data_loader, game_id)

        # Deception timeline
        deception_fig = create_deception_heatmap(data_loader, game_id)

        # Game details
        game = data_loader.get_game_details(game_id)
        details = html.Div([
            html.P(f"Winner: {game.get('winning_team', 'Unknown')}"),
            html.P(f"Win Condition: {game.get('win_condition', 'Unknown')}"),
            html.P(f"Duration: {game.get('duration_seconds', 0)/60:.1f} minutes"),
            html.P(f"Total Cost: ${game.get('total_cost', 0):.4f}"),
            html.P(f"Liberal Policies: {game.get('liberal_policies', 0)}"),
            html.P(f"Fascist Policies: {game.get('fascist_policies', 0)}"),
        ])

        return trust_fig, deception_fig, details

    except Exception as e:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, html.Div(f"Error: {e}")


@callback(
    [Output('model-winrate-chart', 'figure'),
     Output('model-cost-chart', 'figure'),
     Output('model-stats-table', 'children')],
    Input('url', 'pathname')
)
def update_model_comparison(pathname):
    """Update model comparison visualizations."""
    if pathname != '/models':
        return dash.no_update, dash.no_update, dash.no_update

    if data_loader is None:
        return go.Figure(), go.Figure(), html.Div()

    try:
        model_df = data_loader.get_model_comparison()

        if model_df.empty:
            return go.Figure(), go.Figure(), html.Div("No model data available")

        # Win rate chart with CI
        winrate_fig = create_model_comparison_figure(model_df)

        # Cost chart
        cost_fig = px.bar(
            model_df,
            x='model',
            y='avg_cost_per_game',
            labels={'model': 'Model', 'avg_cost_per_game': 'Avg Cost per Game ($)'},
            color='model'
        )

        # Stats table
        table_df = model_df[['model', 'games', 'liberal_win_rate', 'avg_cost_per_game']].copy()
        table_df['liberal_win_rate'] = (table_df['liberal_win_rate'] * 100).round(1).astype(str) + '%'
        table_df['avg_cost_per_game'] = '$' + table_df['avg_cost_per_game'].round(4).astype(str)
        table_df.columns = ['Model', 'Games', 'Liberal Win Rate', 'Avg Cost/Game']

        table = dbc.Table.from_dataframe(
            table_df,
            striped=True,
            bordered=True,
            hover=True
        )

        return winrate_fig, cost_fig, table

    except Exception as e:
        return go.Figure(), go.Figure(), html.Div(f"Error: {e}")


@callback(
    [Output('deception-by-type-chart', 'figure'),
     Output('deception-distribution-chart', 'figure'),
     Output('top-deceivers-table', 'children')],
    Input('url', 'pathname')
)
def update_deception_analysis(pathname):
    """Update deception analysis visualizations."""
    if pathname != '/deception':
        return dash.no_update, dash.no_update, dash.no_update

    if data_loader is None:
        return go.Figure(), go.Figure(), html.Div()

    try:
        # Deception by type
        type_df = data_loader.get_deception_by_decision_type()

        if type_df.empty:
            return go.Figure(), go.Figure(), html.Div("No deception data available")

        type_fig = px.bar(
            type_df,
            x='decision_type',
            y='deception_rate',
            labels={'decision_type': 'Decision Type', 'deception_rate': 'Deception Rate'},
            color='deception_rate',
            color_continuous_scale='Reds'
        )
        type_fig.update_layout(yaxis_tickformat='.0%')

        # Distribution
        decisions_df = data_loader.get_player_decisions(limit=5000)
        dist_fig = px.histogram(
            decisions_df[decisions_df['deception_score'] > 0],
            x='deception_score',
            nbins=20,
            labels={'deception_score': 'Deception Score'},
            title="Distribution of Deception Scores"
        )

        # Top deceivers
        deception_summary = data_loader.get_deception_summary()
        if not deception_summary.empty:
            top_deceivers = deception_summary.nlargest(10, 'deception_rate')
            table_df = top_deceivers[['player_name', 'total_decisions', 'deception_count', 'deception_rate']].copy()
            table_df['deception_rate'] = (table_df['deception_rate'] * 100).round(1).astype(str) + '%'
            table_df.columns = ['Player', 'Decisions', 'Deceptions', 'Deception Rate']

            table = dbc.Table.from_dataframe(
                table_df,
                striped=True,
                bordered=True,
                hover=True
            )
        else:
            table = html.Div("No data available")

        return type_fig, dist_fig, table

    except Exception as e:
        return go.Figure(), go.Figure(), html.Div(f"Error: {e}")


def run_server(
    port: int = 8050,
    debug: bool = False,
    db_path: str = None
):
    """
    Run the dashboard server.

    Args:
        port: Port to run on (default 8050)
        debug: Enable debug mode
        db_path: Path to database file
    """
    global data_loader

    if db_path:
        data_loader = get_data_loader(Path(db_path))
    else:
        data_loader = get_data_loader()

    print(f"Starting dashboard on http://localhost:{port}")
    print(f"Database: {data_loader.db_path}")

    app.run(debug=debug, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secret Hitler LLM Evaluation Dashboard")
    parser.add_argument('--port', type=int, default=8050, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--db-path', type=str, help='Path to database file')

    args = parser.parse_args()
    run_server(port=args.port, debug=args.debug, db_path=args.db_path)
