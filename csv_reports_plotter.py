#!/usr/bin/env python3
"""
CSV-only plotting script for SHAP physical importance results.

This script works with the CSV files in csv_reports/ directory and recreates the 
shap_physical_importance.png and shap_physical_breakdown.png plots locally.

This script should be placed in the same directory as the CSV files or run from there.

Usage (from csv_reports directory):
    python csv_reports_plotter.py
    
Or from parent directory:
    python csv_reports_plotter.py csv_reports/
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_physical_importance(csv_path, output_path=None):
    """Recreate shap_physical_importance.png from CSV data."""
    df = pd.read_csv(csv_path)
    
    groups = df['group_title'].tolist()
    importances = df['total_importance'].tolist()
    colors = df['color'].tolist()
    feature_counts = df['feature_count'].tolist()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(groups)), importances, color=colors, alpha=0.7, edgecolor='black')
    
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
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


def plot_physical_breakdown(csv_path, output_path=None):
    """Recreate shap_physical_breakdown.png from CSV data."""
    df = pd.read_csv(csv_path)
    
    # Group by feature group
    groups = df['group_name'].unique()
    group_data = {}
    
    for group in groups:
        group_df = df[df['group_name'] == group].copy()
        group_df = group_df.sort_values('importance', ascending=False)
        group_data[group] = group_df
    
    n_groups = len(groups)
    
    # Calculate adaptive height
    total_height = 0
    for group in groups:
        n_features = len(group_data[group])
        group_height = max(3, n_features * 0.4 + 1.5)
        total_height += group_height
    
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, max(total_height, 6)))
    if n_groups == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        group_df = group_data[group]
        
        physical_names = group_df['physical_feature_name'].tolist()
        importances = group_df['importance'].tolist()
        group_title = group_df['group_title'].iloc[0]
        group_description = group_df['group_description'].iloc[0]
        group_color = group_df['group_color'].iloc[0]
        
        if len(physical_names) > 0:
            bars = ax.barh(range(len(physical_names)), importances, 
                          color=group_color, alpha=0.7, edgecolor='black')
            
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
            
            value_fontsize = max(6, label_fontsize - 2)
            for bar, importance in zip(bars, importances):
                width = bar.get_width()
                ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.3f}', ha='left', va='center', fontsize=value_fontsize)
    
    plt.tight_layout(pad=2.0, h_pad=3.0)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig


def main():
    """Main function."""
    # Determine working directory
    if len(sys.argv) > 1:
        csv_dir = Path(sys.argv[1])
    else:
        csv_dir = Path('.')
    
    # Look for CSV files
    importance_csv = csv_dir / 'shap_physical_importance_data.csv'
    breakdown_csv = csv_dir / 'shap_physical_breakdown_data.csv'
    
    if not importance_csv.exists():
        print(f"Error: {importance_csv} not found")
        print(f"Current directory: {Path.cwd()}")
        print("Available files:")
        for f in csv_dir.glob('*.csv'):
            print(f"  {f.name}")
        sys.exit(1)
    
    if not breakdown_csv.exists():
        print(f"Error: {breakdown_csv} not found")
        sys.exit(1)
    
    print("Generating plots from CSV data...")
    print(f"Working directory: {csv_dir}")
    
    # Generate plots
    try:
        importance_output = csv_dir / 'shap_physical_importance.png'
        plot_physical_importance(importance_csv, importance_output)
        
        breakdown_output = csv_dir / 'shap_physical_breakdown.png'
        plot_physical_breakdown(breakdown_csv, breakdown_output)
        
        print("\nSuccess! Both plots generated.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()