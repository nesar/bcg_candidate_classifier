"""
Visualization utilities for feature importance analysis.

This module provides comprehensive plotting functions for:
- SHAP summary plots and waterfall plots
- Feature importance rankings
- Individual sample explanations
- Feature group comparisons
- Partial dependence plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ImportancePlotter:
    """
    Comprehensive plotting utilities for feature importance analysis.
    """
    
    def __init__(self, figsize=(12, 8), style='whitegrid', palette='viridis'):
        """
        Initialize plotter with default styling.
        
        Args:
            figsize: Default figure size
            style: Seaborn style
            palette: Color palette
        """
        self.figsize = figsize
        self.style = style
        self.palette = palette
        
        # Set plotting style
        plt.style.use('default')
        sns.set_style(self.style)
        sns.set_palette(self.palette)
    
    def plot_feature_ranking(self, ranking_df, top_n=20, method_name="Feature Importance",
                           save_path=None, show_values=True):
        """
        Plot feature importance ranking as horizontal bar chart.
        
        Args:
            ranking_df: DataFrame with columns ['feature_name', 'importance', 'rank']
            top_n: Number of top features to show
            method_name: Name of the importance method
            save_path: Path to save plot
            show_values: Whether to show values on bars
            
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Select top features
        top_features = ranking_df.head(top_n).copy()
        top_features = top_features.sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature_name'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Features - {method_name}')
        
        # Add value labels on bars
        if show_values:
            for i, (idx, row) in enumerate(top_features.iterrows()):
                ax.text(row['importance'] + 0.01 * max(top_features['importance']), 
                       i, f'{row["importance"]:.3f}', 
                       va='center', fontsize=9)
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_method_comparison(self, importance_results, top_n=15, save_path=None):
        """
        Compare feature importance across different methods.
        
        Args:
            importance_results: Dictionary of results from different methods
            top_n: Number of top features to show
            save_path: Path to save plot
            
        Returns:
            matplotlib figure
        """
        # Prepare data for comparison
        comparison_data = []
        feature_names = None
        
        for method, results in importance_results.items():
            if method == 'shap':
                importance = results['mean_abs_shap']
            elif method in ['permutation', 'gradient']:
                importance = results['importance']
            else:
                continue
                
            if feature_names is None:
                feature_names = results['feature_names']
            
            # Normalize importance scores (0-1 scale)
            importance_norm = importance / np.max(importance)
            
            for i, (fname, imp) in enumerate(zip(feature_names, importance_norm)):
                comparison_data.append({
                    'feature_name': fname,
                    'importance': imp,
                    'method': method.title()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Get top features across all methods
        avg_importance = comparison_df.groupby('feature_name')['importance'].mean()
        top_features = avg_importance.nlargest(top_n).index.tolist()
        
        # Filter data
        plot_data = comparison_df[comparison_df['feature_name'].isin(top_features)]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, max(8, len(top_features) * 0.4)))
        
        sns.barplot(data=plot_data, y='feature_name', x='importance', 
                   hue='method', ax=ax, orient='h')
        
        ax.set_xlabel('Normalized Importance Score')
        ax.set_ylabel('Feature Name')
        ax.set_title(f'Feature Importance Comparison - Top {top_n} Features')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_groups(self, group_scores, save_path=None):
        """
        Plot feature group importance comparison.
        
        Args:
            group_scores: Dictionary of group importance scores
            save_path: Path to save plot
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        groups = list(group_scores.keys())
        mean_scores = [group_scores[g]['mean_importance'] for g in groups]
        sum_scores = [group_scores[g]['sum_importance'] for g in groups]
        feature_counts = [group_scores[g]['feature_count'] for g in groups]
        
        # Mean importance plot
        bars1 = ax1.bar(groups, mean_scores)
        ax1.set_title('Average Feature Importance by Group')
        ax1.set_ylabel('Mean Importance Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars1, mean_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(mean_scores),
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Total importance plot
        bars2 = ax2.bar(groups, sum_scores, alpha=0.7)
        ax2.set_title('Total Feature Importance by Group')
        ax2.set_ylabel('Total Importance Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add feature count labels
        for bar, total, count in zip(bars2, sum_scores, feature_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(sum_scores),
                    f'{total:.2f}\n({count} features)', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_importance_distribution(self, importance_scores, feature_names, 
                                   save_path=None, bins=30):
        """
        Plot distribution of feature importance scores.
        
        Args:
            importance_scores: Array of importance scores
            feature_names: List of feature names
            save_path: Path to save plot
            bins: Number of histogram bins
            
        Returns:
            matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram of importance scores
        ax1.hist(importance_scores, bins=bins, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Importance Score')
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Distribution of Feature Importance Scores')
        ax1.axvline(np.mean(importance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importance_scores):.3f}')
        ax1.axvline(np.median(importance_scores), color='orange', linestyle='--',
                   label=f'Median: {np.median(importance_scores):.3f}')
        ax1.legend()
        
        # Cumulative importance plot
        sorted_importance = np.sort(importance_scores)[::-1]
        cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
        
        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                marker='o', markersize=3)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance (Normalized)')
        ax2.set_title('Cumulative Feature Importance')
        ax2.grid(True, alpha=0.3)
        
        # Add reference lines
        ax2.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% importance')
        ax2.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% importance')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_correlation_with_importance(self, X, importance_scores, 
                                               feature_names, save_path=None):
        """
        Plot correlation between features and their importance scores.
        
        Args:
            X: Input feature matrix
            importance_scores: Feature importance scores
            feature_names: List of feature names
            save_path: Path to save plot
            
        Returns:
            matplotlib figure
        """
        # Compute feature correlations
        feature_corr = np.corrcoef(X.T)
        
        # Get top features
        top_indices = np.argsort(importance_scores)[-15:]
        top_names = [feature_names[i] for i in top_indices]
        top_corr = feature_corr[np.ix_(top_indices, top_indices)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Correlation heatmap of top features
        sns.heatmap(top_corr, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=top_names, yticklabels=top_names,
                   ax=ax1, fmt='.2f')
        ax1.set_title('Correlation Matrix - Top 15 Important Features')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # Scatter plot: feature variance vs importance
        feature_vars = np.var(X, axis=0)
        ax2.scatter(feature_vars, importance_scores, alpha=0.6)
        ax2.set_xlabel('Feature Variance')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Feature Variance vs. Importance')
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(feature_vars, importance_scores)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_shap_summary_plot(shap_values, X, feature_names, save_path=None, 
                           plot_type='bar', max_display=20):
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        save_path: Path to save plot
        plot_type: Type of plot ('bar', 'beeswarm', 'violin')
        max_display: Maximum number of features to display
        
    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for SHAP plots")
    
    # Validate dimensions
    if len(shap_values) != len(X):
        print(f"Warning: SHAP values ({len(shap_values)}) and X ({len(X)}) have different sample counts")
        min_samples = min(len(shap_values), len(X))
        shap_values = shap_values[:min_samples]
        X = X[:min_samples]
    
    if shap_values.shape[1] > len(feature_names):
        print(f"Warning: SHAP values have {shap_values.shape[1]} features but only {len(feature_names)} feature names")
        feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), shap_values.shape[1])]
    elif shap_values.shape[1] < len(feature_names):
        print(f"Warning: Truncating feature names from {len(feature_names)} to {shap_values.shape[1]}")
        feature_names = feature_names[:shap_values.shape[1]]
    
    # Create plot
    plt.figure(figsize=(12, max(8, max_display * 0.4)))
    
    try:
        if plot_type == 'bar':
            shap.summary_plot(shap_values, X, feature_names=feature_names, 
                             plot_type='bar', max_display=max_display, show=False)
        elif plot_type == 'beeswarm':
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                             max_display=max_display, show=False)
        elif plot_type == 'violin':
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                             plot_type='violin', max_display=max_display, show=False)
    except (ValueError, RuntimeError) as e:
        print(f"Warning: SHAP plot failed ({e}), creating fallback plot")
        # Clear current plot and create fallback bar plot with feature importance
        plt.clf()
        
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[-max_display:][::-1]
        
        plt.barh(range(len(top_indices)), importance[top_indices])
        plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'SHAP Feature Importance ({plot_type} plot failed)')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_individual_explanation_plot(shap_values, X, feature_names, 
                                     sample_idx=0, save_path=None):
    """
    Create SHAP waterfall plot for individual sample explanation.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix  
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        save_path: Path to save plot
        
    Returns:
        matplotlib figure
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for SHAP plots")
    
    # Ensure matplotlib is available globally for SHAP
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Make plt available globally for SHAP's internal use
    import sys
    sys.modules['__main__'].plt = plt
    globals()['plt'] = plt
    
    plt.figure(figsize=(12, 8))
    
    # Create waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=np.mean(shap_values),
            data=X[sample_idx],
            feature_names=feature_names
        ),
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def create_feature_ranking_plot(ranking_df, top_n=20, method_name="Feature Importance",
                              save_path=None):
    """
    Create standalone feature ranking plot.
    
    Args:
        ranking_df: DataFrame with feature rankings
        top_n: Number of top features to show
        method_name: Name of method for title
        save_path: Path to save plot
        
    Returns:
        matplotlib figure
    """
    plotter = ImportancePlotter()
    return plotter.plot_feature_ranking(ranking_df, top_n, method_name, save_path)


def create_comprehensive_report(importance_results, X, feature_names, 
                              output_dir, sample_indices=None):
    """
    Create comprehensive visual report of feature importance analysis.
    
    Args:
        importance_results: Results from FeatureImportanceAnalyzer
        X: Input feature matrix
        feature_names: List of feature names
        output_dir: Directory to save all plots
        sample_indices: Specific sample indices for individual explanations
        
    Returns:
        Dictionary of created plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = ImportancePlotter()
    plot_paths = {}
    
    # 1. Feature ranking plots for each method
    for method, results in importance_results.items():
        if method == 'shap':
            importance = results['mean_abs_shap']
        else:
            importance = results['importance']
            
        ranking_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        save_path = output_dir / f'{method}_feature_ranking.png'
        plotter.plot_feature_ranking(ranking_df, method_name=method.title(), 
                                    save_path=save_path)
        plot_paths[f'{method}_ranking'] = save_path
    
    # 2. Method comparison plot
    if len(importance_results) > 1:
        save_path = output_dir / 'method_comparison.png'
        plotter.plot_method_comparison(importance_results, save_path=save_path)
        plot_paths['method_comparison'] = save_path
    
    # 3. Feature groups analysis
    from .feature_importance import create_default_feature_groups, FeatureGroupAnalyzer
    
    feature_groups = create_default_feature_groups()
    group_analyzer = FeatureGroupAnalyzer(feature_groups)
    
    for method, results in importance_results.items():
        if method == 'shap':
            importance = results['mean_abs_shap']
        else:
            importance = results['importance']
            
        group_scores = group_analyzer.compute_group_importance(importance, feature_names)
        
        save_path = output_dir / f'{method}_feature_groups.png'
        plotter.plot_feature_groups(group_scores, save_path=save_path)
        plot_paths[f'{method}_groups'] = save_path
    
    # 4. Distribution plots
    for method, results in importance_results.items():
        if method == 'shap':
            importance = results['mean_abs_shap']
        else:
            importance = results['importance']
            
        save_path = output_dir / f'{method}_distribution.png'
        plotter.plot_importance_distribution(importance, feature_names, save_path=save_path)
        plot_paths[f'{method}_distribution'] = save_path
    
    # 5. Correlation analysis
    for method, results in importance_results.items():
        if method == 'shap':
            importance = results['mean_abs_shap']
        else:
            importance = results['importance']
            
        save_path = output_dir / f'{method}_correlation.png'
        plotter.plot_feature_correlation_with_importance(
            X, importance, feature_names, save_path=save_path
        )
        plot_paths[f'{method}_correlation'] = save_path
    
    # 6. SHAP-specific plots
    if 'shap' in importance_results and SHAP_AVAILABLE:
        shap_results = importance_results['shap']
        
        # Summary plot
        save_path = output_dir / 'shap_summary_bar.png'
        create_shap_summary_plot(shap_results['shap_values'], X, feature_names,
                               save_path=save_path, plot_type='bar')
        plot_paths['shap_summary_bar'] = save_path
        
        save_path = output_dir / 'shap_summary_beeswarm.png'
        create_shap_summary_plot(shap_results['shap_values'], X, feature_names,
                               save_path=save_path, plot_type='beeswarm')
        plot_paths['shap_summary_beeswarm'] = save_path
        
        # Individual explanations
        if sample_indices is None:
            sample_indices = [0, len(X)//2, len(X)-1]  # First, middle, last
        
        for i, idx in enumerate(sample_indices):
            if idx < len(X):
                save_path = output_dir / f'shap_individual_sample_{idx}.png'
                create_individual_explanation_plot(
                    shap_results['shap_values'], X, feature_names,
                    sample_idx=idx, save_path=save_path
                )
                plot_paths[f'shap_individual_{idx}'] = save_path
    
    print(f"Comprehensive report saved to: {output_dir}")
    print(f"Generated {len(plot_paths)} plots")
    
    return plot_paths