#!/usr/bin/env python3
"""
Unified Plot Configuration for BCG Candidate Classifier
========================================================

This module provides consistent plotting styles across all visualization routines
including diagnostic plots, feature importance, completeness/purity analysis, and
training curves.

Usage:
    from plot_config import setup_plot_style, COLORS, FONTS, SIZES

    # Apply style at the beginning of your plotting script
    setup_plot_style()

    # Use consistent colors
    plt.plot(x, y, color=COLORS['primary'])

    # Use consistent font sizes
    ax.set_xlabel('X Label', fontsize=FONTS['label'])
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# ============================================================================
# COLOR SCHEMES
# ============================================================================

COLORS = {
    # Primary colors for lines and markers
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange

    # Specialized colors
    'completeness': '#2ecc71',  # Green
    'purity': '#3498db',        # Blue
    'train': '#2E86AB',         # Blue
    'validation': '#E63946',    # Red

    # Rank colors (consistent with sectors plot)
    'rank1': '#A6CEE3',         # Light blue (Paired colormap)
    'rank2': '#B2DF8A',         # Light green
    'rank3': '#FB9A99',         # Light red
    'rest': '#FDBF6F',          # Light orange

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

    # Neutral colors
    'grid': '#CCCCCC',
    'edge': '#000000',
    'white': '#FFFFFF',
}

# Colormap for Paired colors (used in sectors and histograms)
PAIRED_COLORS = plt.cm.Paired.colors


# ============================================================================
# FONT CONFIGURATION
# ============================================================================

FONTS = {
    # Font sizes (standardized)
    'title': 20,           # Main plot titles
    'subtitle': 18,        # Subplot titles
    'label': 18,           # Axis labels
    'tick': 14,            # Tick labels
    'legend': 14,          # Legend text
    'annotation': 16,      # Annotations and text boxes
    'small': 12,           # Small text (e.g., notes)

    # Font family
    'family': 'serif',
    'mathtext': 'cm',      # Computer Modern for math
}


# ============================================================================
# SIZE CONFIGURATION
# ============================================================================

SIZES = {
    # Line properties
    'linewidth': 2.0,
    'linewidth_thin': 1.2,
    'linewidth_thick': 2.5,

    # Marker properties
    'markersize': 8,
    'markersize_small': 6,
    'markersize_large': 10,

    # Axes properties
    'axes_linewidth': 1.2,
    'tick_major_width': 1.2,
    'tick_minor_width': 0.8,
    'tick_major_size': 6,
    'tick_minor_size': 4,

    # Figure properties
    'dpi': 300,
    'figsize_single': (10, 7),
    'figsize_double': (16, 7),
    'figsize_square': (10, 10),
}


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

STYLE = {
    # Grid settings
    'grid': True,
    'grid_alpha': 0.3,
    'grid_linestyle': '--',
    'grid_linewidth': 0.8,

    # Legend settings
    'legend_frameon': True,
    'legend_framealpha': 0.9,
    'legend_edgecolor': '#000000',
    'legend_fancybox': False,

    # Spine settings
    'show_top_spine': True,
    'show_right_spine': True,
    'show_left_spine': True,
    'show_bottom_spine': True,

    # Tick settings
    'tick_direction': 'in',
    'tick_top': True,
    'tick_right': True,
    'tick_bottom': True,
    'tick_left': True,
}


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_plot_style(use_latex=False, use_seaborn=False, seaborn_style='whitegrid'):
    """
    Apply consistent plotting style to matplotlib.

    Args:
        use_latex: If True, use LaTeX for text rendering (default: False)
        use_seaborn: If True, apply seaborn styling on top of matplotlib (default: False)
        seaborn_style: Seaborn style to use if use_seaborn=True

    Example:
        >>> setup_plot_style()
        >>> fig, ax = plt.subplots()
        >>> # Your plotting code here
    """
    # Reset to defaults first
    mpl.rcdefaults()

    # Apply custom rcParams
    plt.rcParams.update({
        # Figure settings
        'figure.dpi': SIZES['dpi'],
        'figure.figsize': SIZES['figsize_single'],
        'savefig.dpi': SIZES['dpi'],
        'savefig.bbox': 'tight',

        # Font settings
        'font.family': FONTS['family'],
        'font.size': FONTS['tick'],
        'text.usetex': use_latex,
        'mathtext.fontset': FONTS['mathtext'],

        # Axes settings
        'axes.linewidth': SIZES['axes_linewidth'],
        'axes.labelsize': FONTS['label'],
        'axes.titlesize': FONTS['title'],
        'axes.grid': STYLE['grid'],
        'axes.edgecolor': COLORS['edge'],

        # Tick settings
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

        # Line settings
        'lines.linewidth': SIZES['linewidth'],
        'lines.markersize': SIZES['markersize'],

        # Legend settings
        'legend.fontsize': FONTS['legend'],
        'legend.frameon': STYLE['legend_frameon'],
        'legend.framealpha': STYLE['legend_framealpha'],
        'legend.edgecolor': STYLE['legend_edgecolor'],
        'legend.fancybox': STYLE['legend_fancybox'],

        # Grid settings
        'grid.alpha': STYLE['grid_alpha'],
        'grid.linestyle': STYLE['grid_linestyle'],
        'grid.linewidth': STYLE['grid_linewidth'],
        'grid.color': COLORS['grid'],
    })

    # Apply seaborn styling if requested
    if use_seaborn:
        sns.set_style(seaborn_style)
        sns.set_palette("husl")


def configure_axis(ax, xlabel=None, ylabel=None, title=None, grid=None,
                   show_legend=True, legend_loc='best'):
    """
    Configure axis with consistent styling.

    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label (optional)
        ylabel: Y-axis label (optional)
        title: Plot title (optional)
        grid: Override grid setting (True/False/None for default)
        show_legend: Whether to show legend
        legend_loc: Legend location

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> configure_axis(ax, xlabel='X', ylabel='Y', title='My Plot')
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTS['label'])

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTS['label'])

    if title:
        ax.set_title(title, fontsize=FONTS['title'], pad=15)

    # Configure grid
    if grid is not None:
        ax.grid(grid, alpha=STYLE['grid_alpha'],
                linestyle=STYLE['grid_linestyle'],
                linewidth=STYLE['grid_linewidth'])

    # Configure tick parameters
    ax.tick_params(axis='both', labelsize=FONTS['tick'],
                   width=SIZES['tick_major_width'],
                   direction=STYLE['tick_direction'])

    # Configure legend
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=FONTS['legend'], loc=legend_loc,
                 frameon=STYLE['legend_frameon'],
                 framealpha=STYLE['legend_framealpha'],
                 edgecolor=STYLE['legend_edgecolor'])


def save_figure(fig, filepath, formats=['png', 'pdf'], dpi=None, **kwargs):
    """
    Save figure in multiple formats with consistent settings.

    Args:
        fig: Matplotlib figure object
        filepath: Base filepath (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg', etc.)
        dpi: DPI for raster formats (default: use SIZES['dpi'])
        **kwargs: Additional arguments passed to savefig

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> save_figure(fig, 'my_plot', formats=['png', 'pdf'])
    """
    if dpi is None:
        dpi = SIZES['dpi']

    # Remove extension if provided
    import os
    filepath_base = os.path.splitext(filepath)[0]

    saved_files = []
    for fmt in formats:
        output_file = f"{filepath_base}.{fmt}"
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight', **kwargs)
        saved_files.append(output_file)
        print(f"Saved: {output_file}")

    return saved_files


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_color_cycle(n_colors, palette='default'):
    """
    Get a color cycle with n colors.

    Args:
        n_colors: Number of colors needed
        palette: Color palette name ('default', 'groups', 'paired', 'viridis')

    Returns:
        List of color hex codes
    """
    if palette == 'groups':
        colors = COLORS['group_palette']
        return [colors[i % len(colors)] for i in range(n_colors)]
    elif palette == 'paired':
        return [PAIRED_COLORS[i % len(PAIRED_COLORS)] for i in range(n_colors)]
    elif palette == 'viridis':
        cmap = plt.cm.viridis
        return [mpl.colors.rgb2hex(cmap(i / n_colors)) for i in range(n_colors)]
    else:  # default
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        return [colors[i % len(colors)] for i in range(n_colors)]


def get_rank_colors():
    """
    Get consistent colors for rank categories.

    Returns:
        Dictionary mapping rank names to colors
    """
    return {
        'Rank 1': PAIRED_COLORS[0],
        'Rank-1': PAIRED_COLORS[0],
        'Rank 2': PAIRED_COLORS[2],
        'Rank-2': PAIRED_COLORS[2],
        'Rank 3': PAIRED_COLORS[4],
        'Rank-3': PAIRED_COLORS[4],
        'Rest': PAIRED_COLORS[6],
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating the plot configuration usage.
    """
    import numpy as np

    # Setup plot style
    setup_plot_style()

    # Generate example data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create plot with consistent styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    # Plot 1: Line plot
    ax1.plot(x, y1, color=COLORS['primary'], linewidth=SIZES['linewidth'],
             label='sin(x)')
    ax1.plot(x, y2, color=COLORS['secondary'], linewidth=SIZES['linewidth'],
             label='cos(x)')
    configure_axis(ax1, xlabel='X', ylabel='Y', title='Trigonometric Functions',
                   grid=True)

    # Plot 2: Scatter plot
    ax2.scatter(x, y1, color=COLORS['accent'], s=SIZES['markersize']**2,
                alpha=0.6, label='Data')
    configure_axis(ax2, xlabel='X', ylabel='Y', title='Scatter Plot',
                   grid=True)

    plt.tight_layout()

    # Save figure
    save_figure(fig, 'example_plot', formats=['png', 'pdf'])

    plt.show()
