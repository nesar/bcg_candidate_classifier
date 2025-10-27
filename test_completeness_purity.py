#!/usr/bin/env python3
"""
Test script for completeness and purity plotting on existing experimental results.

Usage:
    python test_completeness_purity.py <experiment_dir>

Example:
    python test_completeness_purity.py ./trained_models/candidate_classifier_color_uq_run_20251027_120000/
"""

import sys
import os
from plot_completeness_purity import plot_completeness_purity


def find_evaluation_csv(experiment_dir):
    """Find evaluation_results.csv in experiment directory."""
    # Common locations for evaluation_results.csv
    possible_paths = [
        os.path.join(experiment_dir, "evaluation_results", "evaluation_results.csv"),
        os.path.join(experiment_dir, "evaluation_results.csv"),
        os.path.join(experiment_dir, "test_results", "evaluation_results.csv"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def find_bcg_csv(experiment_dir):
    """Try to infer BCG CSV path from experiment directory."""
    # Common BCG CSV locations on the cluster
    base_dir = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs'

    # Try to infer from experiment name
    if '2p2arcmin' in experiment_dir or '2.2' in experiment_dir:
        bcg_csv = f'{base_dir}/bcgs_2p2arcmin_clean_matched.csv'
    elif '3p8arcmin' in experiment_dir or '3.8' in experiment_dir:
        bcg_csv = f'{base_dir}/bcgs_3p8arcmin_clean_matched.csv'
    else:
        # Default to 3.8 arcmin
        bcg_csv = f'{base_dir}/bcgs_3p8arcmin_clean_matched.csv'

    if os.path.exists(bcg_csv):
        return bcg_csv

    # Try alternative path without '_clean_matched'
    bcg_csv_alt = bcg_csv.replace('_clean_matched', '_with_coordinates')
    if os.path.exists(bcg_csv_alt):
        return bcg_csv_alt

    return None


def main():
    """Main function to run completeness/purity analysis on existing results."""
    if len(sys.argv) < 2:
        print("Usage: python test_completeness_purity.py <experiment_dir>")
        print("\nExample:")
        print("  python test_completeness_purity.py ./trained_models/candidate_classifier_color_uq_run_20251027_120000/")
        sys.exit(1)

    experiment_dir = sys.argv[1]

    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    print("="*80)
    print("COMPLETENESS AND PURITY ANALYSIS")
    print("="*80)
    print(f"Experiment directory: {experiment_dir}")
    print()

    # Find evaluation_results.csv
    evaluation_csv = find_evaluation_csv(experiment_dir)
    if evaluation_csv is None:
        print("Error: Could not find evaluation_results.csv in experiment directory")
        print("Searched in:")
        print("  - <experiment_dir>/evaluation_results/evaluation_results.csv")
        print("  - <experiment_dir>/evaluation_results.csv")
        print("  - <experiment_dir>/test_results/evaluation_results.csv")
        sys.exit(1)

    print(f"Found evaluation results: {evaluation_csv}")

    # Find BCG CSV
    bcg_csv = find_bcg_csv(experiment_dir)
    if bcg_csv:
        print(f"Found BCG catalog: {bcg_csv}")
    else:
        print("Warning: Could not find BCG catalog CSV. Delta M* z plots may not be available.")

    # Set output directory to experiment root
    output_dir = experiment_dir

    print()
    print("Generating completeness and purity plots...")
    print()

    try:
        plot_completeness_purity(
            evaluation_csv,
            output_dir=output_dir,
            bcg_csv=bcg_csv,
            distance_threshold=10.0,
            n_bins=10
        )

        print()
        print("="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Plots saved to:")
        print(f"  - {output_dir}/completeness_purity_plots.png")
        print(f"  - {output_dir}/completeness_purity_plots.pdf")
        print()

    except Exception as e:
        print()
        print("="*80)
        print("ERROR!")
        print("="*80)
        print(f"Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
