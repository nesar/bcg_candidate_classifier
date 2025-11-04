#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of BCG Classifier Evaluation Results

This script performs in-depth analysis of the BCG classifier evaluation results,
examining correlations between RedMapper probabilities and ML performance,
rank distributions, uncertainty quantification, and detection patterns.

Author: Claude
Date: 2025-09-28
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp, mannwhitneyu
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import consistent plot configuration
from plot_config import setup_plot_style, COLORS, FONTS, SIZES

warnings.filterwarnings('ignore')

# Apply consistent plot style
setup_plot_style()
plt.ioff()  # Turn off interactive mode


class BCGEvaluationAnalyzer:
    """Comprehensive analyzer for BCG classifier evaluation results."""
    
    def __init__(self, eval_results_path, prob_analysis_path, output_dir):
        """
        Initialize analyzer with evaluation data.
        
        Args:
            eval_results_path: Path to evaluation_results.csv
            prob_analysis_path: Path to probability_analysis.csv  
            output_dir: Directory to save analysis plots
        """
        self.eval_results = pd.read_csv(eval_results_path)
        self.prob_analysis = pd.read_csv(prob_analysis_path)
        self.output_dir = output_dir
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Data preprocessing
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess and enrich the datasets."""
        print("Preprocessing evaluation data...")
        
        # Add derived columns to evaluation results
        self.eval_results['perfect_prediction'] = (self.eval_results['distance_error'] == 0.0)
        self.eval_results['good_prediction'] = (self.eval_results['distance_error'] <= 10.0)
        self.eval_results['failed_prediction'] = (self.eval_results['distance_error'] > 50.0)
        
        # Categorize RedMapper probabilities
        self.eval_results['redmapper_category'] = pd.cut(
            self.eval_results['bcg_prob'], 
            bins=[0, 0.5, 0.8, 0.95, 1.0],
            labels=['Low (0-0.5)', 'Medium (0.5-0.8)', 'High (0.8-0.95)', 'Very High (0.95-1.0)']
        )
        
        # Categorize ML confidence
        self.eval_results['ml_confidence_category'] = pd.cut(
            self.eval_results['max_probability'],
            bins=[0, 0.4, 0.6, 0.8, 1.0],
            labels=['Low (0-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (0.8-1.0)']
        )
        
        # Add rank categories
        self.eval_results['rank_category'] = self.eval_results['bcg_rank'].apply(
            lambda x: 'Rank 1' if x == 1 else 'Rank 2-3' if x <= 3 else 'Rank 4+'
        )
        
        # Enhanced probability analysis
        self.prob_analysis['rank'] = self.prob_analysis.groupby('sample_name')['probability'].rank(
            method='dense', ascending=False
        ).astype(int)
        
        print(f"Loaded {len(self.eval_results)} clusters and {len(self.prob_analysis)} candidates")
        print(f"Perfect predictions: {self.eval_results['perfect_prediction'].sum()}")
        print(f"Good predictions (≤10px): {self.eval_results['good_prediction'].sum()}")
        print(f"Failed predictions (>50px): {self.eval_results['failed_prediction'].sum()}")
        
    def analyze_redmapper_ml_correlation(self):
        """Analyze correlation between RedMapper probabilities and ML performance."""
        print("\n=== RedMapper-ML Correlation Analysis ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RedMapper vs Machine Learning Performance Correlation Analysis', fontsize=16, y=0.98)
        
        # 1. Direct probability correlation
        ax = axes[0, 0]
        x = self.eval_results['bcg_prob']
        y = self.eval_results['max_probability']

        ax.scatter(x, y, alpha=0.6, s=30)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2, label='Perfect Correlation')

        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "b-", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.3f})')

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        ax.set_xlabel('RedMapper BCG Probability')
        ax.set_ylabel('ML Predictive Confidence')
        ax.set_title(f'RedMapper vs ML Confidence Correlation\nPearson r={pearson_r:.3f} (p={pearson_p:.3e})\nSpearman ρ={spearman_r:.3f} (p={spearman_p:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Rank performance vs RedMapper probability
        ax = axes[0, 1]
        rank_data = []
        redmapper_bins = np.linspace(0, 1, 11)
        bin_centers = (redmapper_bins[:-1] + redmapper_bins[1:]) / 2
        
        for i in range(len(redmapper_bins)-1):
            mask = (self.eval_results['bcg_prob'] >= redmapper_bins[i]) & \
                   (self.eval_results['bcg_prob'] < redmapper_bins[i+1])
            if mask.sum() > 0:
                rank_1_fraction = (self.eval_results[mask]['bcg_rank'] == 1).mean()
                rank_1_3_fraction = (self.eval_results[mask]['bcg_rank'] <= 3).mean()
                rank_data.append([bin_centers[i], rank_1_fraction, rank_1_3_fraction, mask.sum()])
        
        rank_df = pd.DataFrame(rank_data, columns=['redmapper_prob', 'rank_1_frac', 'rank_1_3_frac', 'count'])
        
        ax.plot(rank_df['redmapper_prob'], rank_df['rank_1_frac'], 'o-', 
                linewidth=2, markersize=8, label='Rank 1 Performance')
        ax.plot(rank_df['redmapper_prob'], rank_df['rank_1_3_frac'], 's-', 
                linewidth=2, markersize=8, label='Rank 1-3 Performance')
        
        ax.set_xlabel('RedMapper BCG Probability')
        ax.set_ylabel('Success Fraction')
        ax.set_title('ML Rank Performance vs RedMapper Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # 3. Distance error vs RedMapper probability
        ax = axes[0, 2]
        
        # Create violin plot for different RedMapper categories
        violin_data = []
        categories = []
        for cat in self.eval_results['redmapper_category'].cat.categories:
            mask = self.eval_results['redmapper_category'] == cat
            if mask.sum() > 0:
                violin_data.append(self.eval_results[mask]['distance_error'])
                categories.append(cat)
        
        positions = range(len(categories))
        parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_ylabel('Distance Error (pixels)')
        ax.set_title('Distance Error Distribution\nby RedMapper Confidence')
        ax.grid(True, alpha=0.3)
        
        # 4. Detection statistics vs RedMapper
        ax = axes[1, 0]

        detection_stats = self.eval_results.groupby('redmapper_category').agg({
            'n_detections': ['mean', 'std'],
            'max_probability': ['mean', 'std'],
            'n_candidates': ['mean', 'std']
        }).round(3)

        x_pos = range(len(detection_stats))
        ax.errorbar(x_pos, detection_stats['n_detections']['mean'],
                   yerr=detection_stats['n_detections']['std'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5, label='Detections')

        ax2 = ax.twinx()
        ax2.errorbar(x_pos, detection_stats['max_probability']['mean'],
                    yerr=detection_stats['max_probability']['std'],
                    fmt='s-', linewidth=2, markersize=8, capsize=5, color='red', label='ML Max Confidence')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(detection_stats.index, rotation=45)
        ax.set_xlabel('RedMapper Category')
        ax.set_ylabel('Number of Detections', color='blue')
        ax2.set_ylabel('ML Predictive Confidence', color='red')
        ax.set_title('Detection Statistics by RedMapper Category')
        ax.grid(True, alpha=0.3)
        
        # 5. Uncertainty patterns
        ax = axes[1, 1]

        uncertainty_stats = self.eval_results.groupby('redmapper_category').agg({
            'max_uncertainty': ['mean', 'std'],
            'avg_uncertainty': ['mean', 'std']
        }).round(4)

        x_pos = range(len(uncertainty_stats))
        ax.errorbar(x_pos, uncertainty_stats['max_uncertainty']['mean'],
                   yerr=uncertainty_stats['max_uncertainty']['std'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5, label='Max ML Uncertainty')
        ax.errorbar(x_pos, uncertainty_stats['avg_uncertainty']['mean'],
                   yerr=uncertainty_stats['avg_uncertainty']['std'],
                   fmt='s-', linewidth=2, markersize=8, capsize=5, label='Avg ML Uncertainty')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(uncertainty_stats.index, rotation=45)
        ax.set_xlabel('RedMapper Category')
        ax.set_ylabel('ML Uncertainty Estimate')
        ax.set_title('ML Uncertainty Patterns by RedMapper Category')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Success rate matrix
        ax = axes[1, 2]
        
        # Create cross-tabulation of RedMapper vs ML confidence categories
        crosstab = pd.crosstab(self.eval_results['redmapper_category'], 
                              self.eval_results['ml_confidence_category'], 
                              values=self.eval_results['perfect_prediction'], 
                              aggfunc='mean', normalize=False)
        
        im = ax.imshow(crosstab.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(crosstab.index)):
            for j in range(len(crosstab.columns)):
                if not np.isnan(crosstab.iloc[i, j]):
                    text = ax.text(j, i, f'{crosstab.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(range(len(crosstab.columns)))
        ax.set_yticks(range(len(crosstab.index)))
        ax.set_xticklabels(crosstab.columns, rotation=45)
        ax.set_yticklabels(crosstab.index)
        ax.set_xlabel('ML Confidence Category')
        ax.set_ylabel('RedMapper Category')
        ax.set_title('Perfect Prediction Success Rate')
        
        plt.colorbar(im, ax=ax, label='Success Rate')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/redmapper_ml_correlation_analysis.png')
        plt.close()
        
        # Statistical summary
        print(f"RedMapper vs ML Predictive Confidence:")
        print(f"  Pearson correlation: {pearson_r:.4f} (p-value: {pearson_p:.2e})")
        print(f"  Spearman correlation: {spearman_r:.4f} (p-value: {spearman_p:.2e})")
        
        # Perfect prediction rates by category
        print("\nPerfect prediction rates by RedMapper category:")
        for cat in self.eval_results['redmapper_category'].cat.categories:
            mask = self.eval_results['redmapper_category'] == cat
            if mask.sum() > 0:
                rate = self.eval_results[mask]['perfect_prediction'].mean()
                count = mask.sum()
                print(f"  {cat}: {rate:.3f} ({count} samples)")
    
    def analyze_rank_distributions(self):
        """Analyze distribution of BCG ranks and their characteristics."""
        print("\n=== Rank Distribution Analysis ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BCG Rank Distribution and Performance Analysis', fontsize=16, y=0.98)
        
        # 1. Overall rank distribution
        ax = axes[0, 0]
        rank_counts = self.eval_results['bcg_rank'].value_counts().sort_index()
        ranks = rank_counts.index
        counts = rank_counts.values
        
        bars = ax.bar(ranks, counts, alpha=0.7, edgecolor='black')
        
        # Add percentage labels
        total = counts.sum()
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({100*count/total:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('BCG Rank')
        ax.set_ylabel('Number of Clusters')
        ax.set_title(f'Overall BCG Rank Distribution\n(Total: {total} clusters)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statistics
        median_rank = self.eval_results['bcg_rank'].median()
        mean_rank = self.eval_results['bcg_rank'].mean()
        ax.axvline(mean_rank, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rank:.2f}')
        ax.axvline(median_rank, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_rank:.1f}')
        ax.legend()
        
        # 2. Rank vs ML probability distribution
        ax = axes[0, 1]

        rank_prob_data = []
        for rank in sorted(self.eval_results['bcg_rank'].unique()):
            if rank <= 10:  # Focus on top 10 ranks
                mask = self.eval_results['bcg_rank'] == rank
                rank_prob_data.append(self.eval_results[mask]['max_probability'].values)

        bp = ax.boxplot(rank_prob_data, positions=range(1, len(rank_prob_data)+1),
                       patch_artist=True, notch=True)

        # Color boxes by rank
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('BCG Rank')
        ax.set_ylabel('ML Predictive Confidence')
        ax.set_title('ML Confidence Distribution by Rank')
        ax.grid(True, alpha=0.3)
        
        # 3. Rank performance by RedMapper category
        ax = axes[0, 2]
        
        rank_performance = {}
        for cat in self.eval_results['redmapper_category'].cat.categories:
            mask = self.eval_results['redmapper_category'] == cat
            if mask.sum() > 0:
                ranks = self.eval_results[mask]['bcg_rank']
                rank_performance[cat] = {
                    'rank_1': (ranks == 1).mean(),
                    'rank_1_3': (ranks <= 3).mean(),
                    'rank_1_5': (ranks <= 5).mean(),
                    'count': len(ranks)
                }
        
        categories = list(rank_performance.keys())
        rank_1_rates = [rank_performance[cat]['rank_1'] for cat in categories]
        rank_1_3_rates = [rank_performance[cat]['rank_1_3'] for cat in categories]
        rank_1_5_rates = [rank_performance[cat]['rank_1_5'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, rank_1_rates, width, label='Rank 1', alpha=0.8)
        ax.bar(x, rank_1_3_rates, width, label='Rank 1-3', alpha=0.8)
        ax.bar(x + width, rank_1_5_rates, width, label='Rank 1-5', alpha=0.8)
        
        ax.set_xlabel('RedMapper Category')
        ax.set_ylabel('Success Rate')
        ax.set_title('Rank Performance by RedMapper Category')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Probability distribution for each rank
        ax = axes[1, 0]
        
        # Get probability data for top candidates by rank from prob_analysis
        rank_prob_dist = {}
        for rank in range(1, 6):  # Top 5 ranks
            rank_data = self.prob_analysis[self.prob_analysis['rank'] == rank]['probability']
            if len(rank_data) > 0:
                rank_prob_dist[f'Rank {rank}'] = rank_data
        
        if rank_prob_dist:
            # Create violin plot
            data_list = list(rank_prob_dist.values())
            labels = list(rank_prob_dist.keys())
            
            parts = ax.violinplot(data_list, positions=range(len(labels)), 
                                showmeans=True, showmedians=True)
            
            # Color violins
            colors = plt.cm.viridis(np.linspace(0, 1, len(parts['bodies'])))
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('ML Predictive Confidence')
            ax.set_title('ML Confidence Distribution by Candidate Rank')
            ax.grid(True, alpha=0.3)
        
        # 5. Distance error vs rank
        ax = axes[1, 1]
        
        rank_error_data = []
        rank_labels = []
        for rank in sorted(self.eval_results['bcg_rank'].unique()):
            if rank <= 10:
                mask = self.eval_results['bcg_rank'] == rank
                errors = self.eval_results[mask]['distance_error']
                if len(errors) > 0:
                    rank_error_data.append(errors)
                    rank_labels.append(f'Rank {rank}')
        
        bp = ax.boxplot(rank_error_data, labels=rank_labels, patch_artist=True)
        
        # Color by rank quality
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('BCG Rank')
        ax.set_ylabel('Distance Error (pixels)')
        ax.set_title('Distance Error Distribution by Rank')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 6. Multiple detection analysis
        ax = axes[1, 2]
        
        # Analyze cases with multiple detections
        multi_detection_data = []
        n_det_values = sorted(self.eval_results['n_detections'].unique())
        
        for n_det in n_det_values:
            if n_det <= 10:  # Focus on reasonable numbers
                mask = self.eval_results['n_detections'] == n_det
                if mask.sum() > 0:
                    rank_1_rate = (self.eval_results[mask]['bcg_rank'] == 1).mean()
                    rank_1_3_rate = (self.eval_results[mask]['bcg_rank'] <= 3).mean()
                    count = mask.sum()
                    multi_detection_data.append([n_det, rank_1_rate, rank_1_3_rate, count])
        
        if multi_detection_data:
            multi_df = pd.DataFrame(multi_detection_data, 
                                  columns=['n_detections', 'rank_1_rate', 'rank_1_3_rate', 'count'])
            
            ax.scatter(multi_df['n_detections'], multi_df['rank_1_rate'], 
                      s=multi_df['count']*10, alpha=0.6, label='Rank 1', color='red')
            ax.scatter(multi_df['n_detections'], multi_df['rank_1_3_rate'], 
                      s=multi_df['count']*10, alpha=0.6, label='Rank 1-3', color='blue')
            
            ax.set_xlabel('Number of Detections (≥0.5)')
            ax.set_ylabel('Success Rate')
            ax.set_title('Performance vs Number of Detections\n(Bubble size ∝ sample count)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rank_distribution_analysis.png')
        plt.close()
        
        # Print summary statistics
        print(f"Rank 1 success rate: {(self.eval_results['bcg_rank'] == 1).mean():.3f}")
        print(f"Rank 1-3 success rate: {(self.eval_results['bcg_rank'] <= 3).mean():.3f}")
        print(f"Rank 1-5 success rate: {(self.eval_results['bcg_rank'] <= 5).mean():.3f}")
        print(f"Mean rank: {self.eval_results['bcg_rank'].mean():.2f}")
        print(f"Median rank: {self.eval_results['bcg_rank'].median():.1f}")
    
    def analyze_uncertainty_patterns(self):
        """Analyze uncertainty quantification patterns and their predictive power."""
        print("\n=== Uncertainty Quantification Analysis ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Uncertainty Quantification and Predictive Performance Analysis', fontsize=16, y=0.98)
        
        # 1. Uncertainty vs accuracy correlation
        ax = axes[0, 0]
        
        x = self.eval_results['max_uncertainty']
        y = self.eval_results['distance_error']
        
        # Create scatter plot with color coding for perfect predictions
        perfect_mask = self.eval_results['perfect_prediction']
        ax.scatter(x[~perfect_mask], y[~perfect_mask], alpha=0.6, s=30, color='red', label='Error > 0')
        ax.scatter(x[perfect_mask], y[perfect_mask], alpha=0.8, s=30, color='green', label='Perfect (Error = 0)')
        
        # Correlation analysis
        # Use log transform for distance error to handle zeros
        y_log = np.log1p(y)  # log(1 + x) to handle zeros
        corr_coef, corr_p = pearsonr(x, y_log)
        
        ax.set_xlabel('ML Uncertainty Estimate')
        ax.set_ylabel('Distance Error (pixels)')
        ax.set_title(f'ML Uncertainty vs Accuracy\nCorrelation: {corr_coef:.3f} (p={corr_p:.3e})')
        ax.set_yscale('symlog', linthresh=1)  # Symmetric log scale to handle zeros
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Uncertainty calibration analysis
        ax = axes[0, 1]
        
        # Bin by uncertainty and calculate actual error rates
        uncertainty_bins = np.percentile(self.eval_results['max_uncertainty'], np.linspace(0, 100, 11))
        bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
        
        calibration_data = []
        for i in range(len(uncertainty_bins)-1):
            mask = (self.eval_results['max_uncertainty'] >= uncertainty_bins[i]) & \
                   (self.eval_results['max_uncertainty'] < uncertainty_bins[i+1])
            if mask.sum() > 0:
                error_rate = (~self.eval_results[mask]['perfect_prediction']).mean()
                mean_uncertainty = self.eval_results[mask]['max_uncertainty'].mean()
                count = mask.sum()
                calibration_data.append([mean_uncertainty, error_rate, count])
        
        if calibration_data:
            calib_df = pd.DataFrame(calibration_data, columns=['uncertainty', 'error_rate', 'count'])
            
            ax.scatter(calib_df['uncertainty'], calib_df['error_rate'], 
                      s=calib_df['count']*5, alpha=0.7, color='blue')
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, linewidth=2, label='Perfect Calibration')
            
            # Add trend line
            if len(calib_df) > 1:
                z = np.polyfit(calib_df['uncertainty'], calib_df['error_rate'], 1)
                p = np.poly1d(z)
                ax.plot(calib_df['uncertainty'], p(calib_df['uncertainty']), 
                       "g-", alpha=0.8, linewidth=2, label=f'Actual (slope={z[0]:.2f})')
            
            ax.set_xlabel('Mean ML Uncertainty (binned)')
            ax.set_ylabel('Actual Error Rate')
            ax.set_title('ML Uncertainty Calibration\n(Bubble size ∝ sample count)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, None)
            ax.set_ylim(0, None)
        
        # 3. Uncertainty distribution by performance
        ax = axes[0, 2]
        
        perf_groups = {
            'Perfect (0px)': self.eval_results['perfect_prediction'],
            'Good (≤10px)': self.eval_results['good_prediction'] & ~self.eval_results['perfect_prediction'],
            'Poor (10-50px)': (~self.eval_results['good_prediction']) & (~self.eval_results['failed_prediction']),
            'Failed (>50px)': self.eval_results['failed_prediction']
        }
        
        uncertainty_data = []
        group_labels = []
        for label, mask in perf_groups.items():
            if mask.sum() > 0:
                uncertainty_data.append(self.eval_results[mask]['max_uncertainty'])
                group_labels.append(f'{label}\n(n={mask.sum()})')
        
        if uncertainty_data:
            bp = ax.boxplot(uncertainty_data, labels=group_labels, patch_artist=True)
            colors = ['green', 'yellow', 'orange', 'red'][:len(bp['boxes'])]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('ML Uncertainty Estimate')
        ax.set_title('ML Uncertainty Distribution by Performance')
        ax.grid(True, alpha=0.3)
        
        # 4. Confidence-uncertainty scatter
        ax = axes[1, 0]
        
        x = self.eval_results['max_probability']
        y = self.eval_results['max_uncertainty']
        colors = self.eval_results['distance_error']
        
        scatter = ax.scatter(x, y, c=colors, alpha=0.6, s=30, cmap='viridis_r')
        plt.colorbar(scatter, ax=ax, label='Distance Error (pixels)')

        ax.set_xlabel('ML Predictive Confidence')
        ax.set_ylabel('ML Uncertainty Estimate')
        ax.set_title('ML Confidence vs Uncertainty\n(Color = Distance Error)')
        ax.grid(True, alpha=0.3)
        
        # 5. Uncertainty vs number of detections
        ax = axes[1, 1]
        
        # Group by number of detections
        n_det_groups = {}
        for n_det in sorted(self.eval_results['n_detections'].unique()):
            if n_det <= 8:  # Focus on reasonable numbers
                mask = self.eval_results['n_detections'] == n_det
                if mask.sum() >= 5:  # Only include groups with sufficient samples
                    n_det_groups[f'{n_det} detections\n(n={mask.sum()})'] = \
                        self.eval_results[mask]['max_uncertainty']
        
        if n_det_groups:
            uncertainty_data = list(n_det_groups.values())
            group_labels = list(n_det_groups.keys())
            
            bp = ax.boxplot(uncertainty_data, labels=group_labels, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_ylabel('ML Uncertainty Estimate')
        ax.set_title('ML Uncertainty vs Number of Detections')
        ax.grid(True, alpha=0.3)
        
        # 6. Predictive value of uncertainty
        ax = axes[1, 2]
        
        # ROC-like analysis: Can uncertainty predict failures?
        from sklearn.metrics import roc_curve, auc
        
        # Use uncertainty to predict failure (distance_error > 10)
        y_true = self.eval_results['distance_error'] > 10
        y_scores = self.eval_results['max_uncertainty']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ML Uncertainty as Failure Predictor\n(Failure = Distance Error > 10px)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/uncertainty_analysis.png')
        plt.close()
        
        # Print summary statistics
        print("ML Uncertainty Analysis Summary:")
        print(f"Mean ML uncertainty estimate: {self.eval_results['max_uncertainty'].mean():.4f}")
        print(f"ML uncertainty-error correlation: {corr_coef:.4f} (p={corr_p:.3e})")
        print(f"ML uncertainty ROC AUC for failure prediction: {roc_auc:.3f}")
        print(f"Optimal ML uncertainty threshold for failure detection: {optimal_threshold:.3f}")
    
    def analyze_multiple_detections(self):
        """Analyze patterns in multiple BCG detections and candidate competition."""
        print("\n=== Multiple Detection and Candidate Competition Analysis ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multiple Detection Patterns and Candidate Competition Analysis', fontsize=16, y=0.98)
        
        # 1. Distribution of detection counts
        ax = axes[0, 0]
        
        detection_counts = self.eval_results['n_detections'].value_counts().sort_index()
        bars = ax.bar(detection_counts.index, detection_counts.values, alpha=0.7, edgecolor='black')
        
        # Add percentage labels
        total = detection_counts.sum()
        for bar, count in zip(bars, detection_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({100*count/total:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Number of Detections (≥0.5 threshold)')
        ax.set_ylabel('Number of Clusters')
        ax.set_title(f'Distribution of Detection Counts\n(Total: {total} clusters)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Multiple detection success analysis
        ax = axes[0, 1]
        
        multi_det_analysis = []
        for n_det in sorted(self.eval_results['n_detections'].unique()):
            if n_det <= 8:
                mask = self.eval_results['n_detections'] == n_det
                if mask.sum() > 0:
                    success_rates = {
                        'rank_1': (self.eval_results[mask]['bcg_rank'] == 1).mean(),
                        'rank_1_3': (self.eval_results[mask]['bcg_rank'] <= 3).mean(),
                        'perfect': self.eval_results[mask]['perfect_prediction'].mean(),
                        'count': mask.sum()
                    }
                    multi_det_analysis.append([n_det, success_rates['rank_1'], 
                                             success_rates['rank_1_3'], success_rates['perfect'], 
                                             success_rates['count']])
        
        if multi_det_analysis:
            multi_df = pd.DataFrame(multi_det_analysis, 
                                  columns=['n_detections', 'rank_1', 'rank_1_3', 'perfect', 'count'])
            
            x = multi_df['n_detections']
            ax.plot(x, multi_df['rank_1'], 'o-', linewidth=2, markersize=8, label='Rank 1')
            ax.plot(x, multi_df['rank_1_3'], 's-', linewidth=2, markersize=8, label='Rank 1-3')
            ax.plot(x, multi_df['perfect'], '^-', linewidth=2, markersize=8, label='Perfect')
            
            ax.set_xlabel('Number of Detections')
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rate vs Number of Detections')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        # 3. Candidate pool size analysis
        ax = axes[0, 2]
        
        candidate_bins = [0, 5, 10, 20, 30, 50, 100]
        candidate_categories = pd.cut(self.eval_results['n_candidates'], bins=candidate_bins)
        
        candidate_performance = []
        for cat in candidate_categories.cat.categories:
            mask = candidate_categories == cat
            if mask.sum() > 0:
                perf_data = {
                    'category': str(cat),
                    'rank_1': (self.eval_results[mask]['bcg_rank'] == 1).mean(),
                    'perfect': self.eval_results[mask]['perfect_prediction'].mean(),
                    'mean_max_prob': self.eval_results[mask]['max_probability'].mean(),
                    'count': mask.sum()
                }
                candidate_performance.append(perf_data)
        
        if candidate_performance:
            perf_df = pd.DataFrame(candidate_performance)
            
            x = range(len(perf_df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], perf_df['rank_1'], width, 
                  label='Rank 1 Rate', alpha=0.8)
            ax.bar([i + width/2 for i in x], perf_df['perfect'], width, 
                  label='Perfect Rate', alpha=0.8)
            
            ax.set_xlabel('Candidate Pool Size')
            ax.set_ylabel('Success Rate')
            ax.set_title('Performance vs Candidate Pool Size')
            ax.set_xticks(x)
            ax.set_xticklabels([cat[:10] + '...' if len(cat) > 10 else cat 
                               for cat in perf_df['category']], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Probability competition analysis
        ax = axes[1, 0]
        
        # Analyze probability gaps between top candidates
        prob_gaps = []
        cluster_info = []
        
        for cluster in self.prob_analysis['sample_name'].unique():
            cluster_data = self.prob_analysis[
                self.prob_analysis['sample_name'] == cluster
            ].sort_values('probability', ascending=False)
            
            if len(cluster_data) >= 2:
                top_prob = cluster_data.iloc[0]['probability']
                second_prob = cluster_data.iloc[1]['probability']
                prob_gap = top_prob - second_prob
                
                # Get cluster evaluation info
                cluster_eval = self.eval_results[self.eval_results['cluster_name'] == cluster]
                if len(cluster_eval) > 0:
                    bcg_rank = cluster_eval.iloc[0]['bcg_rank']
                    prob_gaps.append(prob_gap)
                    cluster_info.append(bcg_rank)
        
        if prob_gaps:
            # Create scatter plot colored by rank
            rank_colors = {1: 'green', 2: 'yellow', 3: 'orange'}
            default_color = 'red'
            
            for rank in [1, 2, 3]:
                mask = np.array(cluster_info) == rank
                if mask.sum() > 0:
                    ax.scatter(np.array(prob_gaps)[mask], [rank]*mask.sum(), 
                             alpha=0.6, s=50, color=rank_colors.get(rank, default_color),
                             label=f'Rank {rank} (n={mask.sum()})')
            
            # Add other ranks
            other_mask = np.array(cluster_info) > 3
            if other_mask.sum() > 0:
                ax.scatter(np.array(prob_gaps)[other_mask], 
                         np.array(cluster_info)[other_mask],
                         alpha=0.6, s=50, color=default_color, label=f'Rank >3 (n={other_mask.sum()})')
            
            ax.set_xlabel('Probability Gap (Top - Second)')
            ax.set_ylabel('BCG Rank')
            ax.set_title('Probability Competition vs BCG Rank')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. RedMapper vs ML detection overlap
        ax = axes[1, 1]
        
        # Create confusion matrix for high-confidence cases
        redmapper_high = self.eval_results['bcg_prob'] >= 0.8
        ml_detected = self.eval_results['n_detections'] >= 1
        ml_high_conf = self.eval_results['max_probability'] >= 0.6
        
        overlap_matrix = pd.crosstab(
            [redmapper_high, ml_detected], 
            [ml_high_conf], 
            margins=True, margins_name='Total'
        )
        
        # Create heatmap
        im = ax.imshow(overlap_matrix.iloc[:-1, :-1].values, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(overlap_matrix.shape[0]-1):
            for j in range(overlap_matrix.shape[1]-1):
                text = ax.text(j, i, overlap_matrix.iloc[i, j],
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('RedMapper vs ML Detection Overlap')
        ax.set_xlabel('ML High Confidence (≥0.6)')
        ax.set_ylabel('RedMapper High (≥0.8) & ML Detected (≥1)')
        
        # Set tick labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['RM:No, ML:No', 'RM:No, ML:Yes', 'RM:Yes, ML:No', 'RM:Yes, ML:Yes'])
        
        plt.colorbar(im, ax=ax)
        
        # 6. Detection threshold sensitivity
        ax = axes[1, 2]
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_analysis = []
        
        for thresh in thresholds:
            # Calculate number of detections at this threshold for each cluster
            cluster_detections = []
            cluster_ranks = []
            
            for cluster in self.prob_analysis['sample_name'].unique():
                cluster_data = self.prob_analysis[self.prob_analysis['sample_name'] == cluster]
                n_detections = (cluster_data['probability'] >= thresh).sum()
                
                # Get rank for this cluster
                cluster_eval = self.eval_results[self.eval_results['cluster_name'] == cluster]
                if len(cluster_eval) > 0:
                    rank = cluster_eval.iloc[0]['bcg_rank']
                    cluster_detections.append(n_detections)
                    cluster_ranks.append(rank)
            
            if cluster_detections:
                # Calculate success rates
                has_detection = np.array(cluster_detections) > 0
                rank_1_rate = np.mean(np.array(cluster_ranks)[has_detection] == 1) if has_detection.sum() > 0 else 0
                rank_1_3_rate = np.mean(np.array(cluster_ranks)[has_detection] <= 3) if has_detection.sum() > 0 else 0
                detection_rate = has_detection.mean()
                
                threshold_analysis.append([thresh, detection_rate, rank_1_rate, rank_1_3_rate])
        
        if threshold_analysis:
            thresh_df = pd.DataFrame(threshold_analysis, 
                                   columns=['threshold', 'detection_rate', 'rank_1_rate', 'rank_1_3_rate'])
            
            ax.plot(thresh_df['threshold'], thresh_df['detection_rate'], 
                   'o-', linewidth=2, label='Detection Rate')
            ax.plot(thresh_df['threshold'], thresh_df['rank_1_rate'], 
                   's-', linewidth=2, label='Rank 1 Success (given detection)')
            ax.plot(thresh_df['threshold'], thresh_df['rank_1_3_rate'], 
                   '^-', linewidth=2, label='Rank 1-3 Success (given detection)')
            
            ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Current Threshold')
            
            ax.set_xlabel('Detection Threshold')
            ax.set_ylabel('Rate')
            ax.set_title('Threshold Sensitivity Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.1, 0.95)
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/multiple_detection_analysis.png')
        plt.close()
        
        # Summary statistics
        print("Multiple Detection Analysis Summary:")
        print(f"Clusters with 0 detections: {(self.eval_results['n_detections'] == 0).sum()}")
        print(f"Clusters with 1 detection: {(self.eval_results['n_detections'] == 1).sum()}")
        print(f"Clusters with 2+ detections: {(self.eval_results['n_detections'] >= 2).sum()}")
        print(f"Mean candidate pool size: {self.eval_results['n_candidates'].mean():.1f}")
        if prob_gaps:
            print(f"Mean probability gap (top vs second): {np.mean(prob_gaps):.3f}")
    
    def analyze_failure_cases(self):
        """Analyze systematic patterns in prediction failures."""
        print("\n=== Failure Case Analysis ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Systematic Failure Pattern Analysis', fontsize=16, y=0.98)
        
        # Define failure categories
        perfect = self.eval_results['distance_error'] == 0
        good = (self.eval_results['distance_error'] > 0) & (self.eval_results['distance_error'] <= 10)
        poor = (self.eval_results['distance_error'] > 10) & (self.eval_results['distance_error'] <= 50)
        failed = self.eval_results['distance_error'] > 50
        
        # 1. Failure category distribution
        ax = axes[0, 0]
        
        categories = ['Perfect\n(0px)', 'Good\n(0-10px)', 'Poor\n(10-50px)', 'Failed\n(>50px)']
        counts = [perfect.sum(), good.sum(), poor.sum(), failed.sum()]
        colors = ['green', 'yellow', 'orange', 'red']
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({100*count/total:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Clusters')
        ax.set_title(f'Prediction Quality Distribution\n(Total: {total} clusters)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Failure characteristics by RedMapper confidence
        ax = axes[0, 1]
        
        failure_by_redmapper = []
        for cat in self.eval_results['redmapper_category'].cat.categories:
            mask = self.eval_results['redmapper_category'] == cat
            if mask.sum() > 0:
                cat_data = self.eval_results[mask]
                failure_rates = {
                    'perfect': (cat_data['distance_error'] == 0).mean(),
                    'good': ((cat_data['distance_error'] > 0) & (cat_data['distance_error'] <= 10)).mean(),
                    'poor': ((cat_data['distance_error'] > 10) & (cat_data['distance_error'] <= 50)).mean(),
                    'failed': (cat_data['distance_error'] > 50).mean(),
                    'count': len(cat_data)
                }
                failure_by_redmapper.append([cat, failure_rates['perfect'], 
                                           failure_rates['good'], failure_rates['poor'], 
                                           failure_rates['failed'], failure_rates['count']])
        
        if failure_by_redmapper:
            fail_df = pd.DataFrame(failure_by_redmapper, 
                                 columns=['category', 'perfect', 'good', 'poor', 'failed', 'count'])
            
            x = range(len(fail_df))
            width = 0.8
            
            ax.bar(x, fail_df['perfect'], width, label='Perfect', color='green', alpha=0.8)
            ax.bar(x, fail_df['good'], width, bottom=fail_df['perfect'], 
                  label='Good', color='yellow', alpha=0.8)
            ax.bar(x, fail_df['poor'], width, 
                  bottom=fail_df['perfect'] + fail_df['good'],
                  label='Poor', color='orange', alpha=0.8)
            ax.bar(x, fail_df['failed'], width,
                  bottom=fail_df['perfect'] + fail_df['good'] + fail_df['poor'],
                  label='Failed', color='red', alpha=0.8)
            
            ax.set_xlabel('RedMapper Category')
            ax.set_ylabel('Fraction')
            ax.set_title('Failure Distribution by RedMapper Category')
            ax.set_xticks(x)
            ax.set_xticklabels(fail_df['category'], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Extreme failure analysis
        ax = axes[0, 2]
        
        # Focus on complete failures (>50px error)
        extreme_failures = self.eval_results[failed]
        
        if len(extreme_failures) > 0:
            # Analyze characteristics of extreme failures
            char_analysis = {
                'RedMapper Prob': extreme_failures['bcg_prob'].values,
                'ML Max Prob': extreme_failures['max_probability'].values,
                'N Detections': extreme_failures['n_detections'].values,
                'N Candidates': extreme_failures['n_candidates'].values,
                'Max Uncertainty': extreme_failures['max_uncertainty'].values
            }
            
            # Create box plots comparing failures to successes
            success_data = self.eval_results[perfect | good]
            
            comparison_data = []
            labels = []
            for char_name, values in char_analysis.items():
                if char_name == 'RedMapper Prob':
                    comparison_data.append(success_data['bcg_prob'].values)
                    comparison_data.append(values)
                    labels.extend([f'{char_name}\nSuccess', f'{char_name}\nFailure'])
                elif char_name == 'ML Max Prob':
                    comparison_data.append(success_data['max_probability'].values)
                    comparison_data.append(values)
                    labels.extend([f'{char_name}\nSuccess', f'{char_name}\nFailure'])
                elif char_name == 'Max Uncertainty':
                    comparison_data.append(success_data['max_uncertainty'].values)
                    comparison_data.append(values)
                    labels.extend([f'{char_name}\nSuccess', f'{char_name}\nFailure'])
            
            if comparison_data:
                bp = ax.boxplot(comparison_data, labels=labels, patch_artist=True)
                
                # Color pairs
                colors = ['lightgreen', 'red'] * (len(comparison_data)//2)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
        
        ax.set_ylabel('Value')
        ax.set_title(f'Extreme Failures Characteristics\n({len(extreme_failures)} failures)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 4. Rank distribution in failures
        ax = axes[1, 0]
        
        failure_rank_data = []
        for failure_type, mask in [('Perfect', perfect), ('Good', good), ('Poor', poor), ('Failed', failed)]:
            if mask.sum() > 0:
                ranks = self.eval_results[mask]['bcg_rank']
                rank_counts = ranks.value_counts().sort_index()
                for rank, count in rank_counts.items():
                    failure_rank_data.append([failure_type, rank, count])
        
        if failure_rank_data:
            rank_df = pd.DataFrame(failure_rank_data, columns=['failure_type', 'rank', 'count'])
            
            # Pivot for stacked bar chart
            pivot_df = rank_df.pivot(index='rank', columns='failure_type', values='count').fillna(0)
            
            # Ensure we have all failure types
            for ftype in ['Perfect', 'Good', 'Poor', 'Failed']:
                if ftype not in pivot_df.columns:
                    pivot_df[ftype] = 0
            
            pivot_df = pivot_df[['Perfect', 'Good', 'Poor', 'Failed']]  # Ensure order
            
            pivot_df.plot(kind='bar', stacked=True, ax=ax, 
                         color=['green', 'yellow', 'orange', 'red'], alpha=0.8)
            
            ax.set_xlabel('BCG Rank')
            ax.set_ylabel('Count')
            ax.set_title('Rank Distribution by Failure Type')
            ax.legend(title='Failure Type')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Systematic bias analysis
        ax = axes[1, 1]
        
        # Look for systematic patterns in failures
        # Analyze correlation between different failure predictors
        failure_predictors = self.eval_results[['bcg_prob', 'max_probability', 'n_detections', 
                                              'n_candidates', 'max_uncertainty', 'avg_uncertainty']]
        failure_outcome = (self.eval_results['distance_error'] > 10).astype(int)
        
        correlations = []
        predictor_names = []
        for col in failure_predictors.columns:
            corr, p_val = pearsonr(failure_predictors[col], failure_outcome)
            correlations.append(corr)
            predictor_names.append(col.replace('_', ' ').title())
        
        bars = ax.barh(predictor_names, correlations, 
                      color=['red' if c < 0 else 'blue' for c in correlations], alpha=0.7)
        
        ax.axvline(0, color='black', linestyle='-', alpha=0.8)
        ax.set_xlabel('Correlation with Failure (Distance Error > 10px)')
        ax.set_title('Failure Prediction Correlations')
        ax.grid(True, alpha=0.3)
        
        # Add correlation values as text
        for bar, corr in zip(bars, correlations):
            width = bar.get_width()
            ax.text(width + 0.01 if width >= 0 else width - 0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{corr:.3f}', ha='left' if width >= 0 else 'right', 
                   va='center', fontweight='bold')
        
        # 6. RedMapper-ML disagreement analysis
        ax = axes[1, 2]
        
        # Define disagreement cases
        redmapper_confident = self.eval_results['bcg_prob'] >= 0.8
        ml_confident = self.eval_results['max_probability'] >= 0.6
        
        agreement_types = {
            'Both Confident': redmapper_confident & ml_confident,
            'RM Only': redmapper_confident & ~ml_confident,
            'ML Only': ~redmapper_confident & ml_confident,
            'Neither': ~redmapper_confident & ~ml_confident
        }
        
        agreement_performance = []
        for agreement_type, mask in agreement_types.items():
            if mask.sum() > 0:
                data = self.eval_results[mask]
                perf = {
                    'type': agreement_type,
                    'count': len(data),
                    'perfect_rate': (data['distance_error'] == 0).mean(),
                    'good_rate': (data['distance_error'] <= 10).mean(),
                    'failed_rate': (data['distance_error'] > 50).mean(),
                    'mean_error': data['distance_error'].mean(),
                    'median_error': data['distance_error'].median()
                }
                agreement_performance.append(perf)
        
        if agreement_performance:
            agree_df = pd.DataFrame(agreement_performance)
            
            x = range(len(agree_df))
            width = 0.25
            
            ax.bar([i - width for i in x], agree_df['perfect_rate'], width, 
                  label='Perfect Rate', alpha=0.8, color='green')
            ax.bar(x, agree_df['good_rate'], width, 
                  label='Good Rate (≤10px)', alpha=0.8, color='blue')
            ax.bar([i + width for i in x], agree_df['failed_rate'], width, 
                  label='Failed Rate (>50px)', alpha=0.8, color='red')
            
            ax.set_xlabel('Agreement Type')
            ax.set_ylabel('Rate')
            ax.set_title('Performance by RM-ML Agreement\n(Confident = RM≥0.8, ML≥0.6)')
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row['type']}\n(n={row['count']})" for _, row in agree_df.iterrows()], 
                              rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/failure_analysis.png')
        plt.close()
        
        # Statistical tests for significant differences
        print("Failure Analysis Summary:")
        print(f"Perfect predictions: {perfect.sum()} ({100*perfect.mean():.1f}%)")
        print(f"Good predictions (≤10px): {good.sum()} ({100*good.mean():.1f}%)")
        print(f"Poor predictions (10-50px): {poor.sum()} ({100*poor.mean():.1f}%)")
        print(f"Failed predictions (>50px): {failed.sum()} ({100*failed.mean():.1f}%)")
        
        # Test if high RedMapper confidence leads to better performance
        high_rm = self.eval_results['bcg_prob'] >= 0.8
        low_rm = self.eval_results['bcg_prob'] < 0.8
        
        if high_rm.sum() > 0 and low_rm.sum() > 0:
            high_rm_success = self.eval_results[high_rm]['perfect_prediction'].mean()
            low_rm_success = self.eval_results[low_rm]['perfect_prediction'].mean()
            
            from scipy.stats import chi2_contingency
            contingency = pd.crosstab(self.eval_results['bcg_prob'] >= 0.8, 
                                    self.eval_results['perfect_prediction'])
            chi2, p_value, _, _ = chi2_contingency(contingency)
            
            print(f"\nRedMapper confidence effect:")
            print(f"High RM confidence (≥0.8): {100*high_rm_success:.1f}% perfect")
            print(f"Low RM confidence (<0.8): {100*low_rm_success:.1f}% perfect")
            print(f"Chi-square test p-value: {p_value:.3e}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE BCG CLASSIFIER EVALUATION SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        total_clusters = len(self.eval_results)
        perfect_count = self.eval_results['perfect_prediction'].sum()
        good_count = self.eval_results['good_prediction'].sum()
        failed_count = self.eval_results['failed_prediction'].sum()
        
        print(f"\n1. OVERALL PERFORMANCE METRICS")
        print(f"   Total clusters analyzed: {total_clusters}")
        print(f"   Perfect predictions (0px error): {perfect_count} ({100*perfect_count/total_clusters:.1f}%)")
        print(f"   Good predictions (≤10px error): {good_count} ({100*good_count/total_clusters:.1f}%)")
        print(f"   Failed predictions (>50px error): {failed_count} ({100*failed_count/total_clusters:.1f}%)")
        print(f"   Mean distance error: {self.eval_results['distance_error'].mean():.2f} pixels")
        print(f"   Median distance error: {self.eval_results['distance_error'].median():.2f} pixels")
        
        # Rank performance
        rank_1 = (self.eval_results['bcg_rank'] == 1).sum()
        rank_1_3 = (self.eval_results['bcg_rank'] <= 3).sum()
        rank_1_5 = (self.eval_results['bcg_rank'] <= 5).sum()
        
        print(f"\n2. RANK PERFORMANCE")
        print(f"   Rank 1 success: {rank_1} ({100*rank_1/total_clusters:.1f}%)")
        print(f"   Rank 1-3 success: {rank_1_3} ({100*rank_1_3/total_clusters:.1f}%)")
        print(f"   Rank 1-5 success: {rank_1_5} ({100*rank_1_5/total_clusters:.1f}%)")
        print(f"   Mean rank: {self.eval_results['bcg_rank'].mean():.2f}")
        print(f"   Median rank: {self.eval_results['bcg_rank'].median():.1f}")
        
        # RedMapper correlation
        rm_ml_corr, rm_ml_p = pearsonr(self.eval_results['bcg_prob'],
                                      self.eval_results['max_probability'])

        print(f"\n3. REDMAPPER BCG PROBABILITY vs ML PREDICTIVE CONFIDENCE")
        print(f"   Correlation: {rm_ml_corr:.4f} (p={rm_ml_p:.2e})")
        
        # Performance by RedMapper confidence
        print(f"\n4. PERFORMANCE BY REDMAPPER CONFIDENCE")
        for cat in self.eval_results['redmapper_category'].cat.categories:
            mask = self.eval_results['redmapper_category'] == cat
            if mask.sum() > 0:
                perfect_rate = self.eval_results[mask]['perfect_prediction'].mean()
                rank_1_rate = (self.eval_results[mask]['bcg_rank'] == 1).mean()
                count = mask.sum()
                print(f"   {cat}: {100*perfect_rate:.1f}% perfect, {100*rank_1_rate:.1f}% rank 1 ({count} samples)")
        
        # Detection statistics
        print(f"\n5. DETECTION STATISTICS")
        print(f"   Mean detections per cluster: {self.eval_results['n_detections'].mean():.2f}")
        print(f"   Clusters with 0 detections: {(self.eval_results['n_detections'] == 0).sum()}")
        print(f"   Clusters with 1 detection: {(self.eval_results['n_detections'] == 1).sum()}")
        print(f"   Clusters with 2+ detections: {(self.eval_results['n_detections'] >= 2).sum()}")
        print(f"   Mean candidate pool size: {self.eval_results['n_candidates'].mean():.1f}")
        
        # Uncertainty analysis
        print(f"\n6. ML UNCERTAINTY QUANTIFICATION")
        print(f"   Mean ML uncertainty (max): {self.eval_results['max_uncertainty'].mean():.4f}")
        print(f"   Mean ML uncertainty (avg): {self.eval_results['avg_uncertainty'].mean():.4f}")

        # Failure prediction analysis
        uncertainty_error_corr, uncertainty_error_p = pearsonr(
            self.eval_results['max_uncertainty'],
            np.log1p(self.eval_results['distance_error'])
        )
        print(f"   ML uncertainty-error correlation: {uncertainty_error_corr:.4f} (p={uncertainty_error_p:.2e})")
        
        print(f"\n7. KEY FINDINGS")
        print(f"   • RedMapper BCG probability and ML predictive confidence show {'strong' if abs(rm_ml_corr) > 0.5 else 'moderate' if abs(rm_ml_corr) > 0.3 else 'weak'} correlation")
        print(f"   • {'High' if rank_1_3/total_clusters > 0.8 else 'Moderate' if rank_1_3/total_clusters > 0.6 else 'Low'} overall rank performance (top-3: {100*rank_1_3/total_clusters:.1f}%)")
        print(f"   • ML uncertainty estimate {'does' if abs(uncertainty_error_corr) > 0.2 else 'does not'} show predictive value for errors")

        high_rm_high_ml = ((self.eval_results['bcg_prob'] >= 0.8) &
                          (self.eval_results['max_probability'] >= 0.6)).sum()
        print(f"   • {high_rm_high_ml} clusters show high confidence from both RedMapper and ML")
        
        # Recommendations
        print(f"\n8. RECOMMENDATIONS")
        if rm_ml_corr < 0.5:
            print(f"   • Investigate systematic differences between RedMapper BCG probability and ML predictive confidence")
        if rank_1/total_clusters < 0.7:
            print(f"   • Consider ensemble methods or temperature calibration to improve rank-1 performance")
        if uncertainty_error_corr > 0.2:
            print(f"   • ML uncertainty estimates show promise for failure detection and active learning")
        if (self.eval_results['n_detections'] == 0).sum() > total_clusters * 0.1:
            print(f"   • Consider lowering detection threshold to reduce no-detection cases")
        
        print("="*80)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive BCG classifier evaluation analysis...")
        print(f"Output directory: {self.output_dir}")
        
        self.analyze_redmapper_ml_correlation()
        
        # Update todo status
        print("\n✓ RedMapper-ML correlation analysis completed")
        
        self.analyze_rank_distributions()
        print("✓ Rank distribution analysis completed")
        
        self.analyze_uncertainty_patterns()
        print("✓ Uncertainty quantification analysis completed")
        
        self.analyze_multiple_detections()
        print("✓ Multiple detection analysis completed")
        
        self.analyze_failure_cases()
        print("✓ Failure case analysis completed")
        
        self.generate_summary_report()
        print("✓ Summary report generated")
        
        print(f"\nAll analysis plots saved to: {self.output_dir}")
        print("Analysis pipeline completed successfully!")


def main():
    """Main function to run the analysis."""
    import sys
    import os

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_eval_results.py <evaluation_results.csv>")
        print("Example: python plot_eval_results.py /path/to/evaluation_results/evaluation_results.csv")
        sys.exit(1)

    # Get paths from command-line argument
    eval_results_path = sys.argv[1]

    # Derive other paths from the evaluation_results.csv location
    eval_results_dir = os.path.dirname(eval_results_path)
    prob_analysis_path = os.path.join(eval_results_dir, "probability_analysis.csv")
    output_dir = os.path.join(eval_results_dir, "analysis_plots")

    # Check if files exist
    if not os.path.exists(eval_results_path):
        print(f"Error: evaluation_results.csv not found at {eval_results_path}")
        sys.exit(1)
    if not os.path.exists(prob_analysis_path):
        print(f"Error: probability_analysis.csv not found at {prob_analysis_path}")
        sys.exit(1)

    # Create analyzer and run analysis
    analyzer = BCGEvaluationAnalyzer(eval_results_path, prob_analysis_path, output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()