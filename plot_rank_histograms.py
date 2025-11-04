#!/usr/bin/env python3
"""
Rank-based Histogram Analysis for BCG Candidate Classifier

This module creates histogram plots showing:
1. Radial distance (r_center) of detected BCG candidates from image center
2. RedMapper probabilities (p_RM) of detected BCG candidates

Both histograms are differentiated by rank (Rank-1, Rank-2, Rank-3, Rest)
to analyze the spatial and probability distributions of successful detections.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_evaluation_results(results_file):
    """Load and validate evaluation results CSV file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)

    # Validate required columns
    required_cols = ['pred_x', 'pred_y', 'bcg_rank']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Loaded {len(df)} evaluation results")
    return df


def calculate_radial_distance(pred_x, pred_y, image_size=512, pixel_scale=None):
    """
    Calculate radial distance of predicted BCG from image center.

    Args:
        pred_x: Predicted x coordinate
        pred_y: Predicted y coordinate
        image_size: Size of the image (assumes square images)
        pixel_scale: Conversion factor from pixels to arcmin (arcmin/pixel)
                    If None, returns distance in pixels

    Returns:
        Radial distance in arcmin (if pixel_scale provided) or pixels
    """
    center = image_size / 2.0
    r_center_pixels = np.sqrt((pred_x - center)**2 + (pred_y - center)**2)

    if pixel_scale is not None:
        return r_center_pixels * pixel_scale
    else:
        return r_center_pixels


def prepare_histogram_data(df, image_size=512, pixel_scale=None):
    """
    Prepare data for histogram plotting by rank categories.

    Args:
        df: DataFrame with evaluation results
        image_size: Size of the image for r_center calculation
        pixel_scale: Conversion factor from pixels to arcmin (arcmin/pixel)

    Returns:
        Dictionary with data for each rank category
    """
    # Calculate r_center for all predictions
    df['r_center'] = calculate_radial_distance(df['pred_x'], df['pred_y'], image_size, pixel_scale)

    # Categorize by rank
    rank_categories = {
        'Rank-1': df[df['bcg_rank'] == 1],
        'Rank-2': df[df['bcg_rank'] == 2],
        'Rank-3': df[df['bcg_rank'] == 3],
        'Rest': df[(df['bcg_rank'] > 3) | (df['bcg_rank'].isna())]
    }

    # Extract data for each category
    histogram_data = {}
    for category, data in rank_categories.items():
        if len(data) > 0:
            histogram_data[category] = {
                'r_center': data['r_center'].values,
                'p_RM': data['bcg_prob'].values if 'bcg_prob' in data.columns else None,
                'count': len(data)
            }
        else:
            histogram_data[category] = {
                'r_center': np.array([]),
                'p_RM': np.array([]) if 'bcg_prob' in df.columns else None,
                'count': 0
            }

    return histogram_data


def create_rank_histograms(results_file, output_dir=None, image_size=512,
                          dataset_type='3p8arcmin', figsize=(16, 7)):
    """
    Create histogram plots differentiated by rank.

    Args:
        results_file: Path to evaluation_results.csv
        output_dir: Directory to save plots (defaults to same dir as results)
        image_size: Size of the image for r_center calculation
        dataset_type: Dataset type ('2p2arcmin' or '3p8arcmin') for pixel scale conversion
        figsize: Figure size for the plot

    Returns:
        fig: Matplotlib figure object
    """
    # Load data
    df = load_evaluation_results(results_file)

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate pixel scale based on dataset type
    if dataset_type == '2p2arcmin':
        # 2.2 arcmin field of view / 512 pixels
        pixel_scale = 2.2 / image_size  # arcmin per pixel
    elif dataset_type == '3p8arcmin':
        # 3.8 arcmin field of view / 512 pixels
        pixel_scale = 3.8 / image_size  # arcmin per pixel
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Expected '2p2arcmin' or '3p8arcmin'")

    print(f"Using dataset type: {dataset_type}")
    print(f"Pixel scale: {pixel_scale:.6f} arcmin/pixel")

    # Prepare histogram data
    histogram_data = prepare_histogram_data(df, image_size, pixel_scale)

    # Check if we have p_RM data
    has_p_RM = 'bcg_prob' in df.columns and not df['bcg_prob'].isna().all()

    # Set style consistent with plot_sectors_hardcoded.py
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2
    })

    # Define colors for each rank category (using Paired colormap like sectors plot)
    colors = plt.cm.Paired.colors
    rank_colors = {
        'Rank-1': colors[0],
        'Rank-2': colors[2],
        'Rank-3': colors[4],
        'Rest': colors[6]
    }

    # Create figure with 2 subplots
    if has_p_RM:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]))

    # Subplot 1: r_center histogram
    for category in ['Rank-1', 'Rank-2', 'Rank-3', 'Rest']:
        if histogram_data[category]['count'] > 0:
            r_center_data = histogram_data[category]['r_center']
            ax1.hist(r_center_data, bins=20, alpha=0.6,
                    label=f"{category} (n={histogram_data[category]['count']})",
                    color=rank_colors[category], edgecolor='black', linewidth=1.2)

    ax1.set_xlabel(r'$r_{\rm center}$ (arcmin)', fontsize=22)
    ax1.set_ylabel('Count', fontsize=22)
    ax1.set_title('Radial Distance from Image Center', fontsize=24, pad=15)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.legend(fontsize=18, frameon=True, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Subplot 2: p_RM histogram (if available)
    if has_p_RM:
        for category in ['Rank-1', 'Rank-2', 'Rank-3', 'Rest']:
            if histogram_data[category]['count'] > 0 and histogram_data[category]['p_RM'] is not None:
                p_RM_data = histogram_data[category]['p_RM']
                # Filter out NaN values
                p_RM_data = p_RM_data[~np.isnan(p_RM_data)]
                if len(p_RM_data) > 0:
                    ax2.hist(p_RM_data, bins=20, alpha=0.6,
                            label=f"{category} (n={len(p_RM_data)})",
                            color=rank_colors[category], edgecolor='black', linewidth=1.2)

        ax2.set_xlabel(r'$p_{\rm RM}$ (RedMapper Probability)', fontsize=22)
        ax2.set_ylabel('Count', fontsize=22)
        ax2.set_title('RedMapper BCG Probability', fontsize=24, pad=15)
        ax2.tick_params(axis='both', labelsize=20)
        ax2.legend(fontsize=18, frameon=True, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 1)  # Probabilities are 0-1

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, 'rank_histograms.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Rank histograms saved to: {output_file}")

    # Also save as PDF for publication quality
    pdf_file = os.path.join(output_dir, 'rank_histograms.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"High-quality PDF saved to: {pdf_file}")

    # Print statistics summary
    print("\nRank Histogram Statistics:")
    for category in ['Rank-1', 'Rank-2', 'Rank-3', 'Rest']:
        if histogram_data[category]['count'] > 0:
            r_center_data = histogram_data[category]['r_center']
            print(f"  {category}:")
            print(f"    Count: {histogram_data[category]['count']}")
            print(f"    r_center mean: {np.mean(r_center_data):.4f} arcmin")
            print(f"    r_center median: {np.median(r_center_data):.4f} arcmin")
            print(f"    r_center std: {np.std(r_center_data):.4f} arcmin")

            if has_p_RM and histogram_data[category]['p_RM'] is not None:
                p_RM_data = histogram_data[category]['p_RM']
                p_RM_data = p_RM_data[~np.isnan(p_RM_data)]
                if len(p_RM_data) > 0:
                    print(f"    p_RM mean: {np.mean(p_RM_data):.4f}")
                    print(f"    p_RM median: {np.median(p_RM_data):.4f}")
                    print(f"    p_RM std: {np.std(p_RM_data):.4f}")

    return fig


def main():
    """Command line interface for rank histogram plotting."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate rank-based histogram plots from BCG evaluation results')
    parser.add_argument('results_file', help='Path to evaluation_results.csv file')
    parser.add_argument('--output_dir', help='Output directory for plots (default: same as results file)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size in pixels for r_center calculation (default: 512)')
    parser.add_argument('--dataset_type', type=str, default='3p8arcmin',
                       choices=['2p2arcmin', '3p8arcmin'],
                       help='Dataset type for pixel scale conversion (default: 3p8arcmin)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 7],
                       help='Figure size (width height) in inches')

    args = parser.parse_args()

    try:
        fig = create_rank_histograms(args.results_file, args.output_dir,
                                     args.image_size, args.dataset_type,
                                     tuple(args.figsize))
        plt.show()
    except Exception as e:
        print(f"Error creating rank histograms: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
