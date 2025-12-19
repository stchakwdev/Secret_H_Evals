"""
Shared visualization utilities for figures.

Provides consistent styling and export functionality for all visualization scripts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional, Tuple


# Standard style settings
PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.dpi': 300,
}

# Colorblind-safe palette
COLORBLIND_PALETTE = {
    'liberal': '#0077BB',      # Blue
    'fascist': '#CC3311',      # Red
    'neutral': '#EE7733',      # Orange
    'success': '#009988',      # Teal
    'warning': '#EE3377',      # Magenta
    'info': '#33BBEE',         # Cyan
    'gray': '#BBBBBB',         # Gray
    'dark': '#000000',         # Black
}

# Standard figure sizes (in inches) for different publication formats
FIGURE_SIZES = {
    'single_column': (3.5, 2.625),      # Single column width
    'double_column': (7.0, 5.25),       # Double column width
    'full_page': (7.0, 9.0),            # Full page
    'square': (5.0, 5.0),               # Square
    'wide': (10.0, 4.0),                # Wide format
    'dashboard': (12.0, 8.0),           # Dashboard/presentation
}


def apply_publication_style():
    """Apply standard matplotlib style settings."""
    plt.rcParams.update(PUBLICATION_STYLE)


def reset_style():
    """Reset to matplotlib defaults."""
    mpl.rcdefaults()


def get_figure(
    size: str = 'double_column',
    custom_size: Optional[Tuple[float, float]] = None,
    publication_style: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with standard settings.

    Args:
        size: Preset size name from FIGURE_SIZES
        custom_size: Custom (width, height) in inches
        publication_style: Whether to apply publication style

    Returns:
        Tuple of (Figure, Axes)
    """
    if publication_style:
        apply_publication_style()

    figsize = custom_size or FIGURE_SIZES.get(size, FIGURE_SIZES['double_column'])
    fig, ax = plt.subplots(figsize=figsize)

    return fig, ax


def save_figure(
    fig: plt.Figure,
    output_path: str,
    formats: Optional[list] = None,
    dpi: int = 300,
    transparent: bool = False,
    tight: bool = True
) -> list:
    """
    Save figure to multiple formats including vector graphics.

    Args:
        fig: Matplotlib figure to save
        output_path: Base output path (extension determines format)
        formats: List of additional formats to save ['svg', 'pdf', 'png']
        dpi: Resolution for raster formats
        transparent: Whether to use transparent background
        tight: Whether to use tight bounding box

    Returns:
        List of saved file paths
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    base_name = output_path.stem
    output_dir = output_path.parent

    # Determine formats to save
    primary_format = output_path.suffix.lower().lstrip('.')
    all_formats = [primary_format]

    if formats:
        all_formats.extend([f.lower().lstrip('.') for f in formats if f.lower().lstrip('.') not in all_formats])

    # Common save options
    save_kwargs = {
        'bbox_inches': 'tight' if tight else None,
        'transparent': transparent,
        'facecolor': fig.get_facecolor() if not transparent else 'none',
        'edgecolor': 'none'
    }

    for fmt in all_formats:
        file_path = output_dir / f"{base_name}.{fmt}"

        if fmt in ['svg', 'pdf', 'eps']:
            # Vector formats
            fig.savefig(str(file_path), format=fmt, **save_kwargs)
        else:
            # Raster formats
            fig.savefig(str(file_path), format=fmt, dpi=dpi, **save_kwargs)

        saved_paths.append(str(file_path))
        print(f"Saved: {file_path}")

    return saved_paths


def add_significance_annotation(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    significance: str,
    height: float = 0.05
):
    """
    Add significance bracket annotation to plot.

    Args:
        ax: Matplotlib axes
        x1, x2: X positions for bracket ends
        y: Y position (top of bracket)
        significance: Significance stars (*, **, ***)
        height: Height of bracket arms
    """
    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y - height, y, y, y - height], 'k-', linewidth=1)

    # Add significance text
    ax.text((x1 + x2) / 2, y, significance, ha='center', va='bottom', fontsize=12)


def add_ci_annotation(
    ax: plt.Axes,
    x: float,
    y: float,
    ci_lower: float,
    ci_upper: float,
    label: str = '',
    orientation: str = 'vertical'
):
    """
    Add confidence interval annotation.

    Args:
        ax: Matplotlib axes
        x, y: Center position
        ci_lower, ci_upper: CI bounds
        label: Optional label
        orientation: 'vertical' or 'horizontal'
    """
    if orientation == 'vertical':
        ax.errorbar(x, y, yerr=[[y - ci_lower], [ci_upper - y]],
                   fmt='none', capsize=3, color='black', linewidth=1.5)
    else:
        ax.errorbar(x, y, xerr=[[x - ci_lower], [ci_upper - x]],
                   fmt='none', capsize=3, color='black', linewidth=1.5)

    if label:
        ax.annotate(label, (x, ci_upper), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=9)


def create_legend_outside(ax: plt.Axes, location: str = 'right'):
    """
    Create legend outside the plot area.

    Args:
        ax: Matplotlib axes
        location: 'right', 'bottom', or 'top'
    """
    if location == 'right':
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    elif location == 'bottom':
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
                 ncol=3, borderaxespad=0)
    elif location == 'top':
        ax.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center',
                 ncol=3, borderaxespad=0)


def format_percentage_axis(ax: plt.Axes, axis: str = 'y'):
    """
    Format axis as percentage.

    Args:
        ax: Matplotlib axes
        axis: 'x', 'y', or 'both'
    """
    from matplotlib.ticker import PercentFormatter

    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))


def add_watermark(fig: plt.Figure, text: str = 'Secret Hitler LLM Evaluation'):
    """
    Add subtle watermark to figure.

    Args:
        fig: Matplotlib figure
        text: Watermark text
    """
    fig.text(0.99, 0.01, text, fontsize=8, color='gray',
            ha='right', va='bottom', alpha=0.5, style='italic')


# Convenience function to get colors
def get_color(name: str) -> str:
    """
    Get color from colorblind-safe palette.

    Args:
        name: Color name (liberal, fascist, neutral, success, etc.)

    Returns:
        Hex color code
    """
    return COLORBLIND_PALETTE.get(name, COLORBLIND_PALETTE['gray'])
