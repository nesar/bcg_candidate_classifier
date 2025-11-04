#!/usr/bin/env python3
"""
Completeness and Purity Analysis for BCG Candidate Classifier

This module computes and plots completeness and purity as functions of:
1. Redshift (z)
2. Delta M* z (delta_mstar_z)

Definitions:
- Completeness (Recall): Fraction of true BCGs successfully detected
  = (# images where true BCG is detected within threshold) / (total # images)

- Purity (Precision): Fraction of detections that are actually true BCGs
  = (# correct detections) / (total # detections across all images)

For single-prediction models: purity = completeness
For multi-detection models (UQ): purity accounts for multiple detections per image
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from plot_config import setup_plot_style, COLORS, FONTS, SIZES

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_merge_data(evaluation_csv, bcg_csv=None):
    """
    Load evaluation results and merge with BCG catalog to get delta_mstar_z.

    Args:
        evaluation_csv: Path to evaluation_results.csv
        bcg_csv: Path to BCG CSV with delta_mstar_z. If None, tries to infer from directory structure.

    Returns:
        DataFrame with evaluation results + delta_mstar_z
    """
    if not os.path.exists(evaluation_csv):
        raise FileNotFoundError(f"Evaluation results not found: {evaluation_csv}")

    df_eval = pd.read_csv(evaluation_csv)
    print(f"Loaded {len(df_eval)} evaluation results")

    # Check if delta_mstar_z is already in the evaluation results
    if 'delta_mstar_z' in df_eval.columns:
        print("delta_mstar_z already present in evaluation results")
        return df_eval

    # Otherwise, try to load from BCG CSV
    if bcg_csv is None:
        # Try to infer BCG CSV path from directory structure
        # Evaluation results are typically in: .../trained_models/experiment_name/evaluation_results/evaluation_results.csv
        # We need to find the corresponding BCG CSV based on the experiment
        print("Warning: BCG CSV path not provided. delta_mstar_z will not be available.")
        df_eval['delta_mstar_z'] = np.nan
        return df_eval

    if not os.path.exists(bcg_csv):
        print(f"Warning: BCG CSV not found: {bcg_csv}")
        df_eval['delta_mstar_z'] = np.nan
        return df_eval

    # Load BCG catalog
    df_bcg = pd.read_csv(bcg_csv)
    print(f"Loaded {len(df_bcg)} BCG catalog entries")

    # Merge on cluster_name to get delta_mstar_z
    if 'cluster_name' in df_eval.columns and 'Cluster name' in df_bcg.columns:
        # Create a mapping from cluster name to delta_mstar_z
        cluster_to_delta = df_bcg.groupby('Cluster name')['delta_mstar_z'].first().to_dict()
        df_eval['delta_mstar_z'] = df_eval['cluster_name'].map(cluster_to_delta)
        n_matched = df_eval['delta_mstar_z'].notna().sum()
        print(f"Matched delta_mstar_z for {n_matched}/{len(df_eval)} samples")
    else:
        print("Warning: Cannot match cluster names. delta_mstar_z will not be available.")
        df_eval['delta_mstar_z'] = np.nan

    return df_eval


def compute_metrics_by_bins(df, bin_column, n_bins=10, distance_threshold=10.0,
                            use_multi_detection=False, bin_edges=None):
    """
    Compute completeness and purity as a function of a binned variable.

    Args:
        df: DataFrame with evaluation results
        bin_column: Column name to bin by (e.g., 'z' or 'delta_mstar_z')
        n_bins: Number of bins (ignored if bin_edges is provided)
        distance_threshold: Distance threshold in pixels for successful detection
        use_multi_detection: Whether to use multi-detection analysis (for UQ models)
        bin_edges: Custom bin edges. If None, bins are created automatically.

    Returns:
        bin_centers, completeness, purity, n_samples_per_bin
    """
    # Filter out NaN values in the bin column
    df_valid = df[df[bin_column].notna()].copy()

    if len(df_valid) == 0:
        print(f"Warning: No valid data for {bin_column}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Create bins
    if bin_edges is None:
        bin_edges = np.linspace(df_valid[bin_column].min(), df_valid[bin_column].max(), n_bins + 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    completeness = []
    purity = []
    n_samples_per_bin = []

    # Assign each sample to a bin
    df_valid['bin_idx'] = pd.cut(df_valid[bin_column], bins=bin_edges, labels=False, include_lowest=True)

    for bin_idx in range(len(bin_edges) - 1):
        df_bin = df_valid[df_valid['bin_idx'] == bin_idx]
        n_samples = len(df_bin)
        n_samples_per_bin.append(n_samples)

        if n_samples == 0:
            completeness.append(np.nan)
            purity.append(np.nan)
            continue

        if use_multi_detection and 'n_detections' in df_bin.columns:
            # Multi-detection analysis (for UQ models)
            # Completeness: fraction of images where at least one detection matches true BCG
            if 'matches_any_target' in df_bin.columns:
                # Use multi-target matching if available
                n_detected = df_bin['matches_any_target'].sum()
            else:
                # Fallback: use distance error with threshold
                n_detected = (df_bin['distance_error'] <= distance_threshold).sum()

            completeness_bin = n_detected / n_samples

            # Purity: fraction of all detections that are correct
            total_detections = df_bin['n_detections'].sum()
            if total_detections > 0:
                # Approximate: assume each correct image contributes 1 correct detection
                # Cap at 1.0 (100%) since approximation can exceed 100% with sparse detections
                purity_bin = min(1.0, n_detected / total_detections)
            else:
                purity_bin = np.nan

        else:
            # Single-prediction analysis (traditional or non-UQ models)
            # Completeness: fraction of images with correct prediction
            if 'matches_any_target' in df_bin.columns:
                n_correct = df_bin['matches_any_target'].sum()
            else:
                n_correct = (df_bin['distance_error'] <= distance_threshold).sum()

            completeness_bin = n_correct / n_samples

            # Purity: same as completeness for single-prediction models
            purity_bin = completeness_bin

        completeness.append(completeness_bin)
        purity.append(purity_bin)

    return bin_centers, np.array(completeness), np.array(purity), np.array(n_samples_per_bin)


def plot_completeness_purity(evaluation_csv, output_dir=None, bcg_csv=None,
                             distance_threshold=10.0, n_bins=15, figsize=(16, 8)):
    """
    Create completeness and purity plots as functions of redshift and delta_mstar_z.

    Args:
        evaluation_csv: Path to evaluation_results.csv
        output_dir: Directory to save plots (defaults to same dir as evaluation results)
        bcg_csv: Path to BCG CSV file (to get delta_mstar_z)
        distance_threshold: Distance threshold in pixels for successful detection
        n_bins: Number of bins for each variable
        figsize: Figure size for the subplot grid
    """
    # Apply consistent plot style
    setup_plot_style(use_seaborn=True, seaborn_style='whitegrid')

    # Load and merge data
    df = load_and_merge_data(evaluation_csv, bcg_csv)

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(evaluation_csv))

    # Determine if this is a UQ model (has n_detections column)
    use_multi_detection = 'n_detections' in df.columns
    if use_multi_detection:
        print(f"Using multi-detection analysis (UQ model)")
        print(f"Average detections per image: {df['n_detections'].mean():.2f}")
    else:
        print(f"Using single-prediction analysis")

    # Create figure with 2 subplots (1x2) - no suptitle
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Define colors - use consistent colors from plot_config
    color_completeness = COLORS['completeness']  # Green
    color_purity = COLORS['purity']  # Blue

    # Determine common x-axis limits for matching scales
    z_min, z_max = None, None
    delta_min, delta_max = None, None
    if 'z' in df.columns and df['z'].notna().sum() > 0:
        z_min, z_max = df['z'].min(), df['z'].max()
    if 'delta_mstar_z' in df.columns and df['delta_mstar_z'].notna().sum() > 0:
        delta_min, delta_max = df['delta_mstar_z'].min(), df['delta_mstar_z'].max()

    # ============================================================================
    # Plot 1: Completeness and Purity vs Redshift
    # ============================================================================
    ax1 = axes[0]

    if 'z' in df.columns and df['z'].notna().sum() > 0:
        bin_centers_z, completeness_z, purity_z, n_samples_z = compute_metrics_by_bins(
            df, 'z', n_bins=n_bins, distance_threshold=distance_threshold,
            use_multi_detection=use_multi_detection
        )

        if len(bin_centers_z) > 0:
            # Plot completeness
            ax1.plot(bin_centers_z, completeness_z * 100, 'o-', color=color_completeness,
                    linewidth=SIZES['linewidth'], markersize=SIZES['markersize'], label='Completeness')

            # Add error bars based on binomial statistics
            completeness_err = np.sqrt(completeness_z * (1 - completeness_z) / n_samples_z) * 100
            completeness_err = np.nan_to_num(completeness_err, nan=0.0)
            ax1.fill_between(bin_centers_z,
                           (completeness_z - completeness_err/100) * 100,
                           (completeness_z + completeness_err/100) * 100,
                           alpha=0.3, color=color_completeness)

            # Add overall completeness line
            overall_completeness = np.nanmean(completeness_z) * 100
            ax1.axhline(overall_completeness, color=color_completeness, linestyle=':',
                       alpha=0.5, linewidth=SIZES['linewidth'], label=f'Overall: {overall_completeness:.1f}%')

            # Plot purity
            ax1.plot(bin_centers_z, purity_z * 100, 's-', color=color_purity,
                    linewidth=SIZES['linewidth'], markersize=SIZES['markersize'], label='Purity')

            # Add error bars for purity
            if use_multi_detection:
                # For multi-detection, purity uncertainty is more complex
                # Use a simple approximation
                purity_err = np.sqrt(purity_z * (1 - purity_z) / n_samples_z) * 100
                purity_err = np.nan_to_num(purity_err, nan=0.0)
            else:
                # For single-prediction, purity = completeness
                purity_err = completeness_err

            ax1.fill_between(bin_centers_z,
                           (purity_z - purity_err/100) * 100,
                           (purity_z + purity_err/100) * 100,
                           alpha=0.3, color=color_purity)

            # Add overall purity line
            overall_purity = np.nanmean(purity_z) * 100
            ax1.axhline(overall_purity, color=color_purity, linestyle=':',
                       alpha=0.5, linewidth=SIZES['linewidth'], label=f'Overall: {overall_purity:.1f}%')

            ax1.set_xlabel(r'$z$', fontsize=FONTS['label'])
            ax1.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])
            ax1.tick_params(axis='both', labelsize=FONTS['tick'])
            ax1.set_ylim([0, 105])
            if z_min is not None and z_max is not None:
                ax1.set_xlim([z_min, z_max])
            ax1.legend(fontsize=FONTS['legend'], loc='lower left')

            # Add note about single-prediction vs multi-detection
            if not use_multi_detection:
                ax1.text(0.02, 0.98, 'Single-prediction:\nPurity = Completeness',
                        transform=ax1.transAxes, fontsize=FONTS['small'], va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, 'Insufficient redshift data', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=FONTS['tick'])
            ax1.set_xlabel(r'$z$', fontsize=FONTS['label'])
            ax1.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])
    else:
        ax1.text(0.5, 0.5, 'No redshift data available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTS['tick'])
        ax1.set_xlabel(r'$z$', fontsize=FONTS['label'])
        ax1.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])

    # ============================================================================
    # Plot 2: Completeness and Purity vs Delta M* z
    # ============================================================================
    ax2 = axes[1]

    if 'delta_mstar_z' in df.columns and df['delta_mstar_z'].notna().sum() > 0:
        bin_centers_dm, completeness_dm, purity_dm, n_samples_dm = compute_metrics_by_bins(
            df, 'delta_mstar_z', n_bins=n_bins, distance_threshold=distance_threshold,
            use_multi_detection=use_multi_detection
        )

        if len(bin_centers_dm) > 0:
            # Plot completeness
            ax2.plot(bin_centers_dm, completeness_dm * 100, 'o-', color=color_completeness,
                    linewidth=SIZES['linewidth'], markersize=SIZES['markersize'], label='Completeness')

            # Add error bars
            completeness_err = np.sqrt(completeness_dm * (1 - completeness_dm) / n_samples_dm) * 100
            completeness_err = np.nan_to_num(completeness_err, nan=0.0)
            ax2.fill_between(bin_centers_dm,
                           (completeness_dm - completeness_err/100) * 100,
                           (completeness_dm + completeness_err/100) * 100,
                           alpha=0.3, color=color_completeness)

            # Add overall completeness line
            overall_completeness = np.nanmean(completeness_dm) * 100
            ax2.axhline(overall_completeness, color=color_completeness, linestyle=':',
                       alpha=0.5, linewidth=SIZES['linewidth'], label=f'Overall: {overall_completeness:.1f}%')

            # Plot purity
            ax2.plot(bin_centers_dm, purity_dm * 100, 's-', color=color_purity,
                    linewidth=SIZES['linewidth'], markersize=SIZES['markersize'], label='Purity')

            # Add error bars for purity
            if use_multi_detection:
                purity_err = np.sqrt(purity_dm * (1 - purity_dm) / n_samples_dm) * 100
                purity_err = np.nan_to_num(purity_err, nan=0.0)
            else:
                purity_err = completeness_err

            ax2.fill_between(bin_centers_dm,
                           (purity_dm - purity_err/100) * 100,
                           (purity_dm + purity_err/100) * 100,
                           alpha=0.3, color=color_purity)

            # Add overall purity line
            overall_purity = np.nanmean(purity_dm) * 100
            ax2.axhline(overall_purity, color=color_purity, linestyle=':',
                       alpha=0.5, linewidth=SIZES['linewidth'], label=f'Overall: {overall_purity:.1f}%')

            ax2.set_xlabel(r'$\delta m^*_z$', fontsize=FONTS['label'])
            ax2.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])
            ax2.tick_params(axis='both', labelsize=FONTS['tick'])
            ax2.set_ylim([0, 105])
            if delta_min is not None and delta_max is not None:
                ax2.set_xlim([delta_min, delta_max])
            ax2.legend(fontsize=FONTS['legend'], loc='lower left')

            # Add note about single-prediction vs multi-detection
            if not use_multi_detection:
                ax2.text(0.02, 0.98, 'Single-prediction:\nPurity = Completeness',
                        transform=ax2.transAxes, fontsize=FONTS['small'], va='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        else:
            ax2.text(0.5, 0.5, 'Insufficient delta_mstar_z data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=FONTS['tick'])
            ax2.set_xlabel(r'$\delta m^*_z$', fontsize=FONTS['label'])
            ax2.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])
    else:
        ax2.text(0.5, 0.5, 'No delta_mstar_z data available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=FONTS['tick'])
        ax2.set_xlabel(r'$\delta m^*_z$', fontsize=FONTS['label'])
        ax2.set_ylabel('Completeness / Purity (%)', fontsize=FONTS['label'])

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, 'completeness_purity_plots.png')
    plt.savefig(output_file, dpi=SIZES['dpi'], bbox_inches='tight')
    print(f"Completeness and purity plots saved to: {output_file}")

    # Also save as PDF for publication quality
    pdf_file = os.path.join(output_dir, 'completeness_purity_plots.pdf')
    plt.savefig(pdf_file, dpi=SIZES['dpi'], bbox_inches='tight')
    print(f"High-quality PDF saved to: {pdf_file}")

    plt.close()

    return fig


def main():
    """Command line interface for completeness/purity plotting."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate completeness and purity plots from BCG evaluation results'
    )
    parser.add_argument('evaluation_csv', help='Path to evaluation_results.csv file')
    parser.add_argument('--output_dir', help='Output directory for plots (default: parent of evaluation CSV)')
    parser.add_argument('--bcg_csv', help='Path to BCG CSV file (to get delta_mstar_z)')
    parser.add_argument('--distance_threshold', type=float, default=10.0,
                       help='Distance threshold in pixels for successful detection (default: 10.0)')
    parser.add_argument('--n_bins', type=int, default=15,
                       help='Number of bins for each variable (default: 15)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[16, 8],
                       help='Figure size (width height) in inches')

    args = parser.parse_args()

    try:
        fig = plot_completeness_purity(
            args.evaluation_csv,
            output_dir=args.output_dir,
            bcg_csv=args.bcg_csv,
            distance_threshold=args.distance_threshold,
            n_bins=args.n_bins,
            figsize=tuple(args.figsize)
        )
        print("Completeness and purity analysis completed successfully!")
    except Exception as e:
        print(f"Error creating completeness/purity plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
