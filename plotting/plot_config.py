#!/usr/bin/env python3
"""
Plot Configuration
========================================================

Usage:
    from plot_config import setup_plot_style, COLORS, FONTS, SIZES

    setup_plot_style()
    plt.plot(x, y, color=COLORS['train'], linewidth=SIZES['linewidth'])
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# COLORS
# ============================================================================

COLORS = {
    # Training plots
    'train': '#2E86AB',         # Blue
    'validation': '#E63946',    # Red

    # Completeness/Purity plots
    'completeness': '#2ecc71',  # Green
    'purity': '#3498db',        # Blue

    # Feature group colors (for physical results)
    'group_palette': [
        '#f48c8c',  # Coral
        '#7ddad1',  # Teal
        '#71c5e8',  # Sky blue
        '#f4d28c',  # Yellow
        '#dda0dd',  # Plum
        '#90ee90',  # Light green
        '#ffb347',  # Orange
        '#87cefa'   # Light sky blue
    ],

    # Layout colors
    'grid': '#CCCCCC',
    'edge': '#000000',
}

# Paired colormap (for sectors/histograms)
PAIRED_COLORS = plt.cm.Paired.colors

# FONTS
# ============================================================================

FONTS = {
    'title': 20,
    'subtitle': 16,
    'label': 18,
    'tick': 14,
    'legend': 14,
    'annotation': 16,
    'small': 12,
    'family': 'serif',
    'mathtext': 'cm',
}


# SIZES
# ============================================================================

SIZES = {
    # Line widths
    'linewidth': 2.0,
    'linewidth_thin': 1.2,

    # Markers
    'markersize': 8,

    # Axes
    'axes_linewidth': 1.2,
    'tick_major_width': 1.2,
    'tick_minor_width': 0.8,
    'tick_major_size': 6,
    'tick_minor_size': 4,

    # Output
    'dpi': 300,
    'figsize_single': (10, 7),
    'figsize_double': (16, 7),
}


# STYLE SETTINGS
# ============================================================================

STYLE = {
    'grid': True,
    'grid_alpha': 0.3,
    'grid_linestyle': '--',
    'grid_linewidth': 0.8,
    'legend_frameon': True,
    'legend_framealpha': 0.9,
    'legend_edgecolor': '#000000',
    'legend_fancybox': False,
    'tick_direction': 'in',
    'tick_top': True,
    'tick_right': True,
    'tick_bottom': True,
    'tick_left': True,
}


# SETUP FUNCTION
# ============================================================================

def setup_plot_style(use_latex=False, use_seaborn=False, seaborn_style='whitegrid'):
    """
    Apply consistent plotting style to matplotlib.

    Args:
        use_latex: If True, use LaTeX for text rendering
        use_seaborn: If True, apply seaborn styling
        seaborn_style: Seaborn style ('whitegrid', 'darkgrid', etc.)
    """
    mpl.rcdefaults()

    plt.rcParams.update({
        # Figure
        'figure.dpi': SIZES['dpi'],
        'figure.figsize': SIZES['figsize_single'],
        'savefig.dpi': SIZES['dpi'],
        'savefig.bbox': 'tight',

        # Fonts
        'font.family': FONTS['family'],
        'font.size': FONTS['tick'],
        'text.usetex': use_latex,
        'mathtext.fontset': FONTS['mathtext'],

        # Axes
        'axes.linewidth': SIZES['axes_linewidth'],
        'axes.labelsize': FONTS['label'],
        'axes.titlesize': FONTS['title'],
        'axes.grid': STYLE['grid'],
        'axes.edgecolor': COLORS['edge'],

        # Ticks
        'xtick.labelsize': FONTS['tick'],
        'ytick.labelsize': FONTS['tick'],
        'xtick.direction': STYLE['tick_direction'],
        'ytick.direction': STYLE['tick_direction'],
        'xtick.major.size': SIZES['tick_major_size'],
        'ytick.major.size': SIZES['tick_major_size'],
        'xtick.minor.size': SIZES['tick_minor_size'],
        'ytick.minor.size': SIZES['tick_minor_size'],
        'xtick.major.width': SIZES['tick_major_width'],
        'ytick.major.width': SIZES['tick_major_width'],
        'xtick.minor.width': SIZES['tick_minor_width'],
        'ytick.minor.width': SIZES['tick_minor_width'],
        'xtick.top': STYLE['tick_top'],
        'xtick.bottom': STYLE['tick_bottom'],
        'ytick.left': STYLE['tick_left'],
        'ytick.right': STYLE['tick_right'],

        # Lines
        'lines.linewidth': SIZES['linewidth'],
        'lines.markersize': SIZES['markersize'],

        # Legend
        'legend.fontsize': FONTS['legend'],
        'legend.frameon': STYLE['legend_frameon'],
        'legend.framealpha': STYLE['legend_framealpha'],
        'legend.edgecolor': STYLE['legend_edgecolor'],
        'legend.fancybox': STYLE['legend_fancybox'],

        # Grid
        'grid.alpha': STYLE['grid_alpha'],
        'grid.linestyle': STYLE['grid_linestyle'],
        'grid.linewidth': STYLE['grid_linewidth'],
        'grid.color': COLORS['grid'],
    })

    if use_seaborn:
        import seaborn as sns
        sns.set_style(seaborn_style)


# UTILITY FUNCTION
# ============================================================================

def get_rank_colors():
    """Get consistent colors for rank categories (used in histograms)."""
    return {
        'Rank 1': PAIRED_COLORS[0],
        'Rank-1': PAIRED_COLORS[0],
        'Rank 2': PAIRED_COLORS[2],
        'Rank-2': PAIRED_COLORS[2],
        'Rank 3': PAIRED_COLORS[4],
        'Rank-3': PAIRED_COLORS[4],
        'Rest': PAIRED_COLORS[6],
    }
