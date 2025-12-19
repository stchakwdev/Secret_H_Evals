"""
Visualization utilities for publication-quality figures.

This module provides consistent styling, colorblind-safe palettes,
and export functionality for all visualization needs.
"""

from .utils import (
    # Style
    PUBLICATION_STYLE,
    COLORBLIND_PALETTE,
    FIGURE_SIZES,
    apply_publication_style,
    reset_style,
    # Figure creation
    get_figure,
    save_figure,
    # Annotations
    add_significance_annotation,
    add_ci_annotation,
    create_legend_outside,
    format_percentage_axis,
    add_watermark,
    # Utilities
    get_color,
)

__all__ = [
    # Style
    'PUBLICATION_STYLE',
    'COLORBLIND_PALETTE',
    'FIGURE_SIZES',
    'apply_publication_style',
    'reset_style',
    # Figure creation
    'get_figure',
    'save_figure',
    # Annotations
    'add_significance_annotation',
    'add_ci_annotation',
    'create_legend_outside',
    'format_percentage_axis',
    'add_watermark',
    # Utilities
    'get_color',
]
