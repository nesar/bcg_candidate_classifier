#!/usr/bin/env python3
"""
Local plotting script for SHAP physical importance results.

This script reads the CSV data files generated on HPC and recreates the 
shap_physical_importance.png and shap_physical_breakdown.png plots locally.

Usage:
    python plot_physical_results.py <path_to_csv_reports_directory>
    
Example:
    python plot_physical_results.py ./trained_models/candidate_classifier_color_uq_run_20250926_225200/feature_importance_analysis/csv_reports/
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_physical_importance(csv_path, output_path=None):
    """
    Recreate shap_physical_importance.png from CSV data.
    
    Args:
        csv_path: Path to shap_physical_importance_data.csv
        output_path: Optional output path for the plot
    """
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Extract data for plotting
    groups = df['group_title'].tolist()
    importances = df['total_importance'].tolist()
    colors = df['color'].tolist()
    descriptions = df['description'].tolist()
    feature_counts = df['feature_count'].tolist()
    
    # Create plot (data is already sorted by importance in CSV)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(groups)), importances, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel('SHAP Importance Score', fontsize=12)
    ax.set_title('Physical Feature Group Importance (SHAP)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        width = bar.get_width()
        ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
               f'{importance:.3f}', ha='left', va='center', fontweight='bold')
    
    # Add feature count annotations
    for i, feature_count in enumerate(feature_counts):
        ax.text(0.02 * max(importances), i, f'({feature_count} features)', 
               ha='left', va='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved physical importance plot to: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_physical_breakdown(csv_path, output_path=None):
    """
    Recreate shap_physical_breakdown.png from CSV data.
    
    Args:
        csv_path: Path to shap_physical_breakdown_data.csv
        output_path: Optional output path for the plot
    """
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Group by feature group
    groups = df['group_name'].unique()
    group_data = {}
    
    for group in groups:
        group_df = df[df['group_name'] == group].copy()
        # Sort by importance within each group
        group_df = group_df.sort_values('importance', ascending=False)
        group_data[group] = group_df
    
    # Create subplot for each group with adaptive height
    n_groups = len(groups)
    
    # Calculate adaptive height based on number of features in each group
    total_height = 0
    group_heights = []
    for group in groups:
        n_features = len(group_data[group])
        # Minimum 3 inches, then 0.4 inches per feature, with some padding
        group_height = max(3, n_features * 0.4 + 1.5)
        group_heights.append(group_height)
        total_height += group_height
    
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, max(total_height, 6)))
    if n_groups == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        group_df = group_data[group]
        
        # Extract data for this group
        physical_names = group_df['physical_feature_name'].tolist()
        importances = group_df['importance'].tolist()
        group_title = group_df['group_title'].iloc[0]
        group_description = group_df['group_description'].iloc[0]
        group_color = group_df['group_color'].iloc[0]
        
        if len(physical_names) > 0:
            # Create horizontal bar plot
            bars = ax.barh(range(len(physical_names)), importances, 
                          color=group_color, alpha=0.7, edgecolor='black')
            
            # Adjust font size based on number of features
            n_features = len(physical_names)
            if n_features > 15:
                label_fontsize = 8
                title_fontsize = 10
            elif n_features > 10:
                label_fontsize = 9
                title_fontsize = 11
            else:
                label_fontsize = 10
                title_fontsize = 12
            
            ax.set_yticks(range(len(physical_names)))
            ax.set_yticklabels(physical_names, fontsize=label_fontsize)
            ax.set_xlabel('Importance Score', fontsize=label_fontsize)
            ax.set_title(f"{group_title}: {group_description}", 
                       fontsize=title_fontsize, fontweight='bold')
            
            # Add value labels with adaptive font size
            value_fontsize = max(6, label_fontsize - 2)
            for bar, importance in zip(bars, importances):
                width = bar.get_width()
                ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontsize=value_fontsize)
        else:
            ax.text(0.5, 0.5, 'No features found for this group', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{group_title}: {group_description}", 
                       fontsize=12, fontweight='bold')
    
    # Use better spacing for plots with many subplots
    plt.tight_layout(pad=2.0, h_pad=3.0)
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved physical breakdown plot to: {output_path}")
    else:
        plt.show()
    
    return fig


def main():
    """Main function to process command line arguments and generate plots."""
    if len(sys.argv) < 2:
        print("Usage: python plot_physical_results.py <path_to_csv_reports_directory>")
        print("\nExample:")
        print("  python plot_physical_results.py ./trained_models/candidate_classifier_color_uq_run_20250926_225200/feature_importance_analysis/csv_reports/")
        sys.exit(1)
    
    csv_dir = Path(sys.argv[1])
    
    if not csv_dir.exists():
        print(f"Error: Directory does not exist: {csv_dir}")
        sys.exit(1)
    
    # Check for required CSV files
    importance_csv = csv_dir / 'shap_physical_importance_data.csv'
    breakdown_csv = csv_dir / 'shap_physical_breakdown_data.csv'
    
    if not importance_csv.exists():
        print(f"Error: Required file not found: {importance_csv}")
        sys.exit(1)
    
    if not breakdown_csv.exists():
        print(f"Error: Required file not found: {breakdown_csv}")
        sys.exit(1)
    
    print(f"Found CSV files in: {csv_dir}")
    print(f"  - {importance_csv.name}")
    print(f"  - {breakdown_csv.name}")
    
    # Create output directory for plots
    output_dir = csv_dir.parent / 'local_plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Generate physical importance plot
    try:
        importance_output = output_dir / 'shap_physical_importance.png'
        plot_physical_importance(importance_csv, importance_output)
    except Exception as e:
        print(f"Error generating importance plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate physical breakdown plot
    try:
        breakdown_output = output_dir / 'shap_physical_breakdown.png'
        plot_physical_breakdown(breakdown_csv, breakdown_output)
    except Exception as e:
        print(f"Error generating breakdown plot: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nPlots saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()