"""
Model Comparison Visualization Component.

Creates publication-ready model comparison charts with confidence intervals
and statistical significance indicators.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from analytics.statistical_analysis import (
    calculate_proportion_ci,
    chi_square_test_proportions,
    significance_stars,
)


def create_model_comparison_figure(
    model_df: pd.DataFrame,
    metric: str = 'liberal_win_rate',
    confidence: float = 0.95
) -> go.Figure:
    """
    Create model comparison chart with confidence intervals.

    Args:
        model_df: DataFrame with columns: model, games, liberal_wins, liberal_win_rate
        metric: Metric to compare ('liberal_win_rate', 'avg_cost_per_game')
        confidence: Confidence level for intervals

    Returns:
        Plotly Figure with error bars
    """
    if model_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No model data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    # Calculate confidence intervals for each model
    models = []
    rates = []
    lower_errors = []
    upper_errors = []
    colors = []

    color_palette = px.colors.qualitative.Plotly

    for i, (_, row) in enumerate(model_df.iterrows()):
        model = row['model']
        games = row['games']
        wins = row['liberal_wins']

        rate, lower, upper = calculate_proportion_ci(int(wins), int(games), confidence)

        models.append(model)
        rates.append(rate * 100)  # Convert to percentage
        lower_errors.append((rate - lower) * 100)
        upper_errors.append((upper - rate) * 100)
        colors.append(color_palette[i % len(color_palette)])

    # Create bar chart with error bars
    fig.add_trace(go.Bar(
        x=models,
        y=rates,
        error_y=dict(
            type='data',
            symmetric=False,
            array=upper_errors,
            arrayminus=lower_errors,
            visible=True,
            thickness=2,
            width=6
        ),
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>" +
                      "Win Rate: %{y:.1f}%<br>" +
                      f"{int(confidence*100)}% CI: [%{{customdata[0]:.1f}}%, %{{customdata[1]:.1f}}%]<extra></extra>",
        customdata=list(zip(
            [r - e for r, e in zip(rates, lower_errors)],
            [r + e for r, e in zip(rates, upper_errors)]
        ))
    ))

    fig.update_layout(
        title=dict(
            text=f"Liberal Win Rate by Model (with {int(confidence*100)}% CI)",
            x=0.5
        ),
        xaxis_title="Model",
        yaxis_title="Liberal Win Rate (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        bargap=0.3
    )

    return fig


def create_model_comparison_with_significance(
    model_df: pd.DataFrame,
    reference_model: Optional[str] = None
) -> go.Figure:
    """
    Create model comparison with statistical significance tests.

    Compares each model against a reference (or first model) and shows
    significance stars.

    Args:
        model_df: DataFrame with model statistics
        reference_model: Model to compare others against (default: first)

    Returns:
        Plotly Figure with significance annotations
    """
    if model_df.empty or len(model_df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 models for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Set reference model
    if reference_model is None:
        reference_model = model_df.iloc[0]['model']

    ref_row = model_df[model_df['model'] == reference_model].iloc[0]
    ref_wins = int(ref_row['liberal_wins'])
    ref_total = int(ref_row['games'])

    fig = go.Figure()

    # Calculate significance for each model
    models = []
    rates = []
    lower_errors = []
    upper_errors = []
    significance = []
    colors = []

    color_palette = px.colors.qualitative.Plotly

    for i, (_, row) in enumerate(model_df.iterrows()):
        model = row['model']
        games = int(row['games'])
        wins = int(row['liberal_wins'])

        rate, lower, upper = calculate_proportion_ci(wins, games)

        models.append(model)
        rates.append(rate * 100)
        lower_errors.append((rate - lower) * 100)
        upper_errors.append((upper - rate) * 100)
        colors.append(color_palette[i % len(color_palette)])

        # Test significance against reference
        if model != reference_model:
            result = chi_square_test_proportions(
                ref_wins, ref_total,
                wins, games
            )
            significance.append(result.significance)
        else:
            significance.append("(ref)")

    # Create bars
    fig.add_trace(go.Bar(
        x=models,
        y=rates,
        error_y=dict(
            type='data',
            symmetric=False,
            array=upper_errors,
            arrayminus=lower_errors,
            visible=True
        ),
        marker_color=colors,
        text=[f"{r:.1f}% {s}" for r, s in zip(rates, significance)],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>vs Reference: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Model Comparison (Reference: {reference_model})",
            x=0.5
        ),
        xaxis_title="Model",
        yaxis_title="Liberal Win Rate (%)",
        yaxis=dict(range=[0, 120]),  # Extra space for text
        showlegend=False,
        annotations=[
            dict(
                text="* p<0.05  ** p<0.01  *** p<0.001",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    return fig


def create_cost_efficiency_scatter(
    model_df: pd.DataFrame
) -> go.Figure:
    """
    Create scatter plot of cost vs performance.

    Args:
        model_df: DataFrame with model statistics

    Returns:
        Plotly Figure with scatter plot
    """
    if model_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No model data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    # Size based on number of games
    max_games = model_df['games'].max()
    sizes = (model_df['games'] / max_games * 30 + 10).tolist()

    fig.add_trace(go.Scatter(
        x=model_df['avg_cost_per_game'],
        y=model_df['liberal_win_rate'] * 100,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=model_df['liberal_win_rate'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Win Rate")
        ),
        text=model_df['model'].apply(lambda x: x.split('/')[-1][:15]),  # Short names
        textposition='top center',
        hovertemplate="<b>%{text}</b><br>" +
                      "Cost: $%{x:.4f}/game<br>" +
                      "Win Rate: %{y:.1f}%<br>" +
                      "<extra></extra>"
    ))

    # Add efficiency frontier line (Pareto optimal)
    sorted_df = model_df.sort_values('avg_cost_per_game')
    pareto_x = []
    pareto_y = []
    max_win_rate = 0

    for _, row in sorted_df.iterrows():
        if row['liberal_win_rate'] >= max_win_rate:
            pareto_x.append(row['avg_cost_per_game'])
            pareto_y.append(row['liberal_win_rate'] * 100)
            max_win_rate = row['liberal_win_rate']

    if len(pareto_x) > 1:
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode='lines',
            name='Efficiency Frontier',
            line=dict(color='green', dash='dash', width=2),
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="Cost vs Performance Efficiency",
        xaxis_title="Average Cost per Game ($)",
        yaxis_title="Liberal Win Rate (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_model_performance_table(
    model_df: pd.DataFrame,
    confidence: float = 0.95
) -> go.Figure:
    """
    Create detailed performance table as a figure.

    Args:
        model_df: DataFrame with model statistics
        confidence: Confidence level for intervals

    Returns:
        Plotly Table figure
    """
    if model_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No model data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Prepare table data
    table_data = []

    for _, row in model_df.iterrows():
        model = row['model'].split('/')[-1]  # Short name
        games = int(row['games'])
        wins = int(row['liberal_wins'])

        rate, lower, upper = calculate_proportion_ci(wins, games, confidence)

        table_data.append({
            'Model': model,
            'Games': games,
            'Liberal Wins': wins,
            'Win Rate': f"{rate*100:.1f}%",
            f'{int(confidence*100)}% CI': f"[{lower*100:.1f}%, {upper*100:.1f}%]",
            'Avg Cost': f"${row['avg_cost_per_game']:.4f}",
            'Total Cost': f"${row['total_cost']:.2f}"
        })

    df = pd.DataFrame(table_data)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#3498db',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[['#f8f9fa', '#ffffff'] * (len(df) // 2 + 1)][:len(df)],
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="Model Performance Summary",
        height=100 + len(df) * 30
    )

    return fig


def create_deception_by_model_chart(
    data_loader,
    models: Optional[List[str]] = None
) -> go.Figure:
    """
    Compare deception rates across models.

    Args:
        data_loader: DataLoader instance
        models: Optional list of models to include

    Returns:
        Plotly Figure with deception comparison
    """
    # This requires joining decisions with games to get model info
    # For now, create a placeholder
    fig = go.Figure()

    fig.add_annotation(
        text="Deception by model analysis requires enhanced data schema",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )

    return fig
