#!/usr/bin/env python3
"""
Automated Asset Generator for Documentation.

Generates presentation-quality screenshots, diagrams, and GIFs:
- Pipeline overview diagrams
- Game progression animations
- Trust network evolution GIFs
- Deception detection visualizations
- Dashboard screenshots
- Architecture diagrams

Usage:
    python scripts/generate_assets.py --all
    python scripts/generate_assets.py --diagrams
    python scripts/generate_assets.py --gifs
    python scripts/generate_assets.py --screenshots

Author: Samuel T. Chakwera (stchakdev)
"""

import argparse
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
import numpy as np
import networkx as nx

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. GIF generation disabled.")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Dashboard screenshots disabled.")

# Import visualization utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.visualization.utils import (
    apply_publication_style,
    save_figure,
    COLORBLIND_PALETTE,
    FIGURE_SIZES,
    get_figure,
    add_watermark,
)

# Output directories
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
DIAGRAMS_DIR = ASSETS_DIR / "diagrams"
GIFS_DIR = ASSETS_DIR / "gifs"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"
FIGURES_DIR = ASSETS_DIR / "figures"


def ensure_directories():
    """Create output directories if they don't exist."""
    for d in [DIAGRAMS_DIR, GIFS_DIR, SCREENSHOTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DIAGRAM GENERATORS
# =============================================================================

def create_pipeline_diagram(output_path: Optional[Path] = None) -> str:
    """
    Create architecture/pipeline overview diagram.

    Shows the flow from game engine through LLM agents to analysis.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color scheme
    colors = {
        'core': '#3498db',      # Blue - Core Engine
        'agents': '#e74c3c',    # Red - LLM Agents
        'data': '#2ecc71',      # Green - Data/Logging
        'analysis': '#9b59b6',  # Purple - Analysis
        'output': '#f39c12',    # Orange - Output
        'arrow': '#34495e',     # Dark gray
    }

    # Box style
    box_style = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=2)

    # Helper function for boxes
    def draw_box(x, y, w, h, label, color, sublabel=None):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white')
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha='center', va='center', fontsize=8,
                    color='white', alpha=0.9)

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, label=None, curved=False):
        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color=colors['arrow'], lw=1.5)
        if curved:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle="arc3,rad=0.2", **kw)
        else:
            arrow = FancyArrowPatch((x1, y1), (x2, y2), **kw)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.3, label, ha='center', va='bottom', fontsize=8,
                    style='italic', color=colors['arrow'])

    # Title
    ax.text(6, 7.5, "Secret Hitler LLM Evaluation Framework",
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(6, 7.1, "Architecture Overview",
            ha='center', va='center', fontsize=12, color='gray')

    # Layer 1: Core Game Engine
    draw_box(2, 5.5, 2.5, 1.2, "Game Engine", colors['core'], "core/")
    draw_box(6, 5.5, 2.5, 1.2, "Game State", colors['core'], "Policies, Roles")
    draw_box(10, 5.5, 2.5, 1.2, "Turn Manager", colors['core'], "Flow Control")

    # Layer 2: LLM Agents
    draw_box(2, 3.5, 2.5, 1.2, "LLM Agents", colors['agents'], "agents/")
    draw_box(6, 3.5, 2.5, 1.2, "OpenRouter", colors['agents'], "API Client")
    draw_box(10, 3.5, 2.5, 1.2, "Prompts", colors['agents'], "Templates")

    # Layer 3: Data/Logging
    draw_box(2, 1.5, 2.5, 1.2, "Game Logger", colors['data'], "JSON Logs")
    draw_box(6, 1.5, 2.5, 1.2, "SQLite DB", colors['data'], "Structured Data")
    draw_box(10, 1.5, 2.5, 1.2, "Inspect AI", colors['data'], "Export Format")

    # Analysis box (spanning)
    rect = FancyBboxPatch((0.5, -0.3), 11, 1.0,
                          boxstyle="round,pad=0.02",
                          facecolor=colors['analysis'], edgecolor='black',
                          linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(6, 0.2, "Analysis Module", ha='center', va='center', fontsize=12,
            fontweight='bold', color='white')
    ax.text(2, 0.2, "Deception", ha='center', va='center', fontsize=9, color='white')
    ax.text(4.5, 0.2, "Coalitions", ha='center', va='center', fontsize=9, color='white')
    ax.text(7.5, 0.2, "Statistics", ha='center', va='center', fontsize=9, color='white')
    ax.text(10, 0.2, "Temporal", ha='center', va='center', fontsize=9, color='white')

    # Arrows between layers
    # Core layer horizontal
    draw_arrow(3.4, 5.5, 4.6, 5.5)
    draw_arrow(7.4, 5.5, 8.6, 5.5)

    # Core to Agents
    draw_arrow(2, 4.9, 2, 4.1)
    draw_arrow(6, 4.9, 6, 4.1)
    draw_arrow(10, 4.9, 10, 4.1)

    # Agents layer horizontal
    draw_arrow(3.4, 3.5, 4.6, 3.5, "Decisions")
    draw_arrow(7.4, 3.5, 8.6, 3.5)

    # Agents to Data
    draw_arrow(2, 2.9, 2, 2.1)
    draw_arrow(6, 2.9, 6, 2.1, "Store")
    draw_arrow(10, 2.9, 10, 2.1)

    # Data layer horizontal
    draw_arrow(3.4, 1.5, 4.6, 1.5)
    draw_arrow(7.4, 1.5, 8.6, 1.5, "Export")

    # Data to Analysis
    draw_arrow(4, 0.9, 4, 0.7)
    draw_arrow(8, 0.9, 8, 0.7)

    # Legend
    legend_y = 7.5
    legend_items = [
        ('Core Engine', colors['core']),
        ('LLM Integration', colors['agents']),
        ('Data Layer', colors['data']),
        ('Analysis', colors['analysis']),
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 0.5 + i * 2.8
        rect = FancyBboxPatch((x, legend_y - 0.15), 0.3, 0.3,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.45, legend_y, label, ha='left', va='center', fontsize=9)

    add_watermark(fig)

    output_path = output_path or DIAGRAMS_DIR / "pipeline_overview.png"
    save_figure(fig, str(output_path), formats=['svg', 'pdf'])
    plt.close(fig)

    return str(output_path)


def create_architecture_diagram(output_path: Optional[Path] = None) -> str:
    """
    Create detailed module architecture diagram.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, "Module Architecture",
            ha='center', va='center', fontsize=18, fontweight='bold')

    # Module boxes
    modules = [
        # (x, y, w, h, name, submodules, color)
        (2, 7, 3, 2, "core/", ["game_state.py", "game_manager.py", "game_events.py"], '#3498db'),
        (7, 7, 3, 2, "agents/", ["openrouter_client.py", "prompt_templates.py"], '#e74c3c'),
        (12, 7, 2, 2, "config/", ["openrouter_config.py"], '#95a5a6'),
        (2, 4, 3, 2, "analysis/", ["core/", "deception/", "social/", "models/"], '#9b59b6'),
        (7, 4, 3, 2, "evaluation/", ["database_schema.py", "inspect_adapter.py"], '#2ecc71'),
        (12, 4, 2, 2, "dashboard/", ["app.py", "components/"], '#f39c12'),
        (2, 1, 3, 2, "experiments/", ["batch_runner.py", "analytics.py"], '#1abc9c'),
        (7, 1, 3, 2, "scripts/", ["generate_assets.py", "visualizations"], '#e67e22'),
        (12, 1, 2, 2, "web_bridge/", ["websocket", "adapters"], '#34495e'),
    ]

    for x, y, w, h, name, subs, color in modules:
        # Main box
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black',
                              linewidth=2, alpha=0.85)
        ax.add_patch(rect)

        # Module name
        ax.text(x, y + h/2 - 0.3, name, ha='center', va='top',
                fontsize=11, fontweight='bold', color='white')

        # Submodules
        for i, sub in enumerate(subs[:3]):  # Max 3 shown
            ax.text(x, y + h/2 - 0.6 - i*0.35, sub, ha='center', va='top',
                    fontsize=8, color='white', alpha=0.9)

    add_watermark(fig)

    output_path = output_path or DIAGRAMS_DIR / "architecture.png"
    save_figure(fig, str(output_path), formats=['svg'])
    plt.close(fig)

    return str(output_path)


# =============================================================================
# GIF GENERATORS
# =============================================================================

def create_game_progression_gif(
    game_data: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    fps: int = 2
) -> str:
    """
    Create animated GIF showing game progression.

    Shows policy board filling up over turns.
    """
    if not HAS_IMAGEIO:
        print("Skipping GIF generation: imageio not installed")
        return ""

    # Use sample data if not provided
    if game_data is None:
        game_data = _get_sample_game_data()

    apply_publication_style()
    frames = []

    # Generate frames for each turn
    policies = game_data.get('policies', [])
    liberal_count = 0
    fascist_count = 0

    for turn_idx in range(len(policies) + 1):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw policy boards
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1, 5)
        ax.axis('off')

        # Title
        ax.text(5, 4.5, f"Game Progression - Turn {turn_idx}",
                ha='center', va='center', fontsize=14, fontweight='bold')

        # Liberal board (5 slots)
        ax.text(2.5, 3.5, "Liberal Policies", ha='center', fontsize=12)
        for i in range(5):
            color = COLORBLIND_PALETTE['liberal'] if i < liberal_count else 'lightgray'
            rect = FancyBboxPatch((i * 1.1 + 0.2, 2.5), 0.9, 0.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

        # Fascist board (6 slots)
        ax.text(7.5, 3.5, "Fascist Policies", ha='center', fontsize=12)
        for i in range(6):
            color = COLORBLIND_PALETTE['fascist'] if i < fascist_count else 'lightgray'
            rect = FancyBboxPatch((i * 1.1 + 5.2, 2.5), 0.9, 0.8,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)

        # Score display
        ax.text(2.5, 1.5, f"Liberal: {liberal_count}/5",
                ha='center', fontsize=11, color=COLORBLIND_PALETTE['liberal'])
        ax.text(7.5, 1.5, f"Fascist: {fascist_count}/6",
                ha='center', fontsize=11, color=COLORBLIND_PALETTE['fascist'])

        # Election tracker
        ax.text(5, 0.5, "Election Tracker", ha='center', fontsize=10)
        for i in range(3):
            tracker_color = '#ffcc00' if i < (turn_idx % 3) else 'lightgray'
            circle = Circle((4 + i * 1, 0), 0.2, facecolor=tracker_color,
                            edgecolor='black', linewidth=1)
            ax.add_patch(circle)

        add_watermark(fig)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

        # Update counts for next frame
        if turn_idx < len(policies):
            if policies[turn_idx] == 'liberal':
                liberal_count += 1
            else:
                fascist_count += 1

    # Add final frame multiple times for pause at end
    for _ in range(3):
        frames.append(frames[-1])

    output_path = output_path or GIFS_DIR / "game_progression.gif"
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Saved: {output_path}")

    return str(output_path)


def create_trust_network_gif(
    game_data: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    fps: int = 1
) -> str:
    """
    Create animated GIF showing trust network evolution.
    """
    if not HAS_IMAGEIO:
        print("Skipping GIF generation: imageio not installed")
        return ""

    # Use sample data if not provided
    if game_data is None:
        game_data = _get_sample_trust_data()

    apply_publication_style()
    frames = []

    players = game_data.get('players', ['P1', 'P2', 'P3', 'P4', 'P5'])
    roles = game_data.get('roles', ['L', 'L', 'L', 'F', 'H'])
    trust_evolution = game_data.get('trust_evolution', [])

    # Create graph
    G = nx.complete_graph(len(players))
    pos = nx.circular_layout(G)

    for turn_idx, trust_matrix in enumerate(trust_evolution):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')

        # Title
        ax.text(0, 1.35, f"Trust Network - Turn {turn_idx + 1}",
                ha='center', va='center', fontsize=14, fontweight='bold')

        # Draw edges with trust weights
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                trust = trust_matrix[i][j]
                width = max(0.5, trust * 5)
                alpha = max(0.2, trust)
                color = 'green' if trust > 0.5 else 'red' if trust < 0.3 else 'gray'
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                       color=color, alpha=alpha, linewidth=width)

        # Draw nodes
        for i, (player, role) in enumerate(zip(players, roles)):
            color = COLORBLIND_PALETTE['liberal'] if role == 'L' else \
                    COLORBLIND_PALETTE['fascist'] if role == 'F' else '#ffcc00'
            circle = Circle(pos[i], 0.12, facecolor=color,
                           edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[i][0], pos[i][1], player, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # Legend
        ax.text(-1.3, -1.3, "L=Liberal  F=Fascist  H=Hitler",
                fontsize=9, style='italic')

        add_watermark(fig)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # Add pause at end
    for _ in range(3):
        frames.append(frames[-1])

    output_path = output_path or GIFS_DIR / "trust_network_evolution.gif"
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Saved: {output_path}")

    return str(output_path)


# =============================================================================
# SCREENSHOT GENERATORS
# =============================================================================

def create_dashboard_screenshot(output_path: Optional[Path] = None) -> str:
    """
    Create a static representation of the dashboard.
    """
    if not HAS_PLOTLY:
        print("Skipping dashboard screenshot: plotly not installed")
        return ""

    # Create a multi-panel figure representing dashboard
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Win Rates by Team", "Deception Over Time",
                       "Model Performance", "Cost Analysis"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )

    # Win rates bar chart
    fig.add_trace(
        go.Bar(x=['Liberal', 'Fascist'], y=[0.58, 0.42],
               marker_color=[COLORBLIND_PALETTE['liberal'], COLORBLIND_PALETTE['fascist']],
               name='Win Rate'),
        row=1, col=1
    )

    # Deception heatmap
    z = np.random.rand(5, 10) * 0.5 + 0.25
    fig.add_trace(
        go.Heatmap(z=z, colorscale='RdYlGn_r', showscale=False),
        row=1, col=2
    )

    # Model performance scatter
    models = ['DeepSeek', 'Claude', 'GPT-4', 'Llama']
    costs = [0.02, 0.15, 0.10, 0.05]
    performance = [0.72, 0.78, 0.75, 0.68]
    fig.add_trace(
        go.Scatter(x=costs, y=performance, mode='markers+text',
                  text=models, textposition='top center',
                  marker=dict(size=15, color=COLORBLIND_PALETTE['info'])),
        row=2, col=1
    )

    # Cost pie chart
    fig.add_trace(
        go.Pie(labels=['API Calls', 'Processing', 'Storage'],
               values=[85, 10, 5], hole=0.4),
        row=2, col=2
    )

    fig.update_layout(
        title_text="Secret Hitler LLM Evaluation Dashboard",
        showlegend=False,
        height=800,
        width=1200,
    )

    output_path = output_path or SCREENSHOTS_DIR / "dashboard_overview.png"
    pio.write_image(fig, str(output_path), scale=2)
    print(f"Saved: {output_path}")

    return str(output_path)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_sample_game_data() -> Dict:
    """Generate sample game data for demonstrations."""
    return {
        'policies': ['fascist', 'liberal', 'fascist', 'liberal', 'fascist',
                    'liberal', 'fascist', 'liberal', 'liberal', 'liberal'],
        'turns': 10,
    }


def _get_sample_trust_data() -> Dict:
    """Generate sample trust evolution data."""
    np.random.seed(42)
    n_players = 5
    n_turns = 6

    trust_evolution = []
    base_trust = np.ones((n_players, n_players)) * 0.5
    np.fill_diagonal(base_trust, 1.0)

    for _ in range(n_turns):
        # Add some random evolution
        delta = np.random.randn(n_players, n_players) * 0.1
        delta = (delta + delta.T) / 2  # Symmetric
        base_trust = np.clip(base_trust + delta, 0, 1)
        np.fill_diagonal(base_trust, 1.0)
        trust_evolution.append(base_trust.tolist())

    return {
        'players': ['Alice', 'Bob', 'Carol', 'David', 'Eve'],
        'roles': ['L', 'L', 'L', 'F', 'H'],
        'trust_evolution': trust_evolution,
    }


def load_game_from_database(db_path: Path, game_id: Optional[str] = None) -> Optional[Dict]:
    """Load game data from SQLite database."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return None

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get latest game if not specified
    if game_id is None:
        cursor.execute("SELECT game_id FROM games ORDER BY created_at DESC LIMIT 1")
        result = cursor.fetchone()
        if result:
            game_id = result[0]
        else:
            conn.close()
            return None

    # Load game data
    cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
    game = cursor.fetchone()

    conn.close()

    if game:
        return {'game_id': game_id}  # Expand as needed
    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for asset generation."""
    parser = argparse.ArgumentParser(
        description="Generate presentation assets for documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all assets
    python scripts/generate_assets.py --all

    # Generate only diagrams
    python scripts/generate_assets.py --diagrams

    # Generate only GIFs
    python scripts/generate_assets.py --gifs

    # Generate from specific database
    python scripts/generate_assets.py --all --db data/games.db
"""
    )

    parser.add_argument('--all', action='store_true',
                        help='Generate all assets')
    parser.add_argument('--diagrams', action='store_true',
                        help='Generate architecture diagrams')
    parser.add_argument('--gifs', action='store_true',
                        help='Generate animated GIFs')
    parser.add_argument('--screenshots', action='store_true',
                        help='Generate dashboard screenshots')
    parser.add_argument('--db', type=str, default=None,
                        help='Path to SQLite database for real data')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output directory')

    args = parser.parse_args()

    # Default to --all if no specific option given
    if not (args.diagrams or args.gifs or args.screenshots):
        args.all = True

    # Setup output directory
    global ASSETS_DIR, DIAGRAMS_DIR, GIFS_DIR, SCREENSHOTS_DIR, FIGURES_DIR
    if args.output:
        ASSETS_DIR = Path(args.output)
        DIAGRAMS_DIR = ASSETS_DIR / "diagrams"
        GIFS_DIR = ASSETS_DIR / "gifs"
        SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"
        FIGURES_DIR = ASSETS_DIR / "figures"

    ensure_directories()

    print("=" * 60)
    print("Asset Generation for Secret Hitler LLM Evaluation Framework")
    print("=" * 60)

    generated = []

    # Load database data if available
    game_data = None
    if args.db:
        db_path = Path(args.db)
        game_data = load_game_from_database(db_path)

    # Generate diagrams
    if args.all or args.diagrams:
        print("\nGenerating diagrams...")
        generated.append(create_pipeline_diagram())
        generated.append(create_architecture_diagram())

    # Generate GIFs
    if args.all or args.gifs:
        print("\nGenerating GIFs...")
        generated.append(create_game_progression_gif(game_data))
        generated.append(create_trust_network_gif())

    # Generate screenshots
    if args.all or args.screenshots:
        print("\nGenerating screenshots...")
        generated.append(create_dashboard_screenshot())

    # Summary
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Generated {len([g for g in generated if g])} assets")
    print(f"Output directory: {ASSETS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
