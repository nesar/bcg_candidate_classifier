"""
Comparison Analysis: Pre-Purge vs Post-Purge BCG Classification Performance

This script compares model performance between:
- Pre-purge: October 13, 2024 run (candidate_classifier_color_uq_run_20251013_151226)
- Post-purge: November 21, 2024 run (candidate_classifier_color_uq_run_20251121_004114)

The analysis focuses on:
1. Rank-1 improvements and degradations
2. Rank movements (Rank-2/3 → Rank-1, etc.)
3. Failure → Success transitions
4. Clusters with no change

Usage:
    python compare_pre_post_purge_performance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set consistent plot style
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2
})


class PrePostPurgeComparison:
    """Compare BCG classification performance before and after purge."""

    def __init__(self, pre_purge_path, post_purge_path, output_dir=None):
        """
        Initialize comparison.

        Args:
            pre_purge_path: Path to pre-purge evaluation_results.csv
            post_purge_path: Path to post-purge evaluation_results.csv
            output_dir: Directory to save analysis results (default: current directory)
        """
        self.pre_purge_path = Path(pre_purge_path)
        self.post_purge_path = Path(post_purge_path)

        if output_dir is None:
            self.output_dir = Path.cwd() / "purge_comparison_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load data
        print("Loading evaluation results...")
        self.pre_df = pd.read_csv(self.pre_purge_path)
        self.post_df = pd.read_csv(self.post_purge_path)

        print(f"Pre-purge: {len(self.pre_df)} clusters")
        print(f"Post-purge: {len(self.post_df)} clusters")

        # Merge on cluster_name
        self.comparison_df = self._merge_and_compare()

    def _merge_and_compare(self):
        """Merge pre and post dataframes and add comparison columns."""
        # Rename columns to distinguish pre/post
        pre_cols = {
            'bcg_rank': 'pre_bcg_rank',
            'matches_any_target': 'pre_matches',
            'distance_error': 'pre_distance_error',
            'bcg_prob': 'pre_bcg_prob',
            'max_probability': 'pre_max_prob',
            'n_candidates': 'pre_n_candidates'
        }

        post_cols = {
            'bcg_rank': 'post_bcg_rank',
            'matches_any_target': 'post_matches',
            'distance_error': 'post_distance_error',
            'bcg_prob': 'post_bcg_prob',
            'max_probability': 'post_max_prob',
            'n_candidates': 'post_n_candidates'
        }

        pre_subset = self.pre_df[['cluster_name', 'z'] + list(pre_cols.keys())].rename(columns=pre_cols)
        post_subset = self.post_df[['cluster_name', 'z'] + list(post_cols.keys())].rename(columns=post_cols)

        # Merge
        merged = pd.merge(pre_subset, post_subset, on='cluster_name', how='outer', suffixes=('_pre', '_post'))

        # Handle missing z values
        merged['z'] = merged['z_pre'].fillna(merged['z_post'])
        merged = merged.drop(columns=['z_pre', 'z_post'])

        # Add comparison metrics
        merged['rank_change'] = merged['post_bcg_rank'] - merged['pre_bcg_rank']
        merged['improved'] = merged['rank_change'] < 0  # Lower rank is better
        merged['degraded'] = merged['rank_change'] > 0
        merged['unchanged'] = merged['rank_change'] == 0

        # Categorize improvements
        merged['category'] = 'unchanged'
        merged.loc[merged['improved'], 'category'] = 'improved'
        merged.loc[merged['degraded'], 'category'] = 'degraded'
        merged.loc[merged['pre_bcg_rank'].isna(), 'category'] = 'new_in_post'
        merged.loc[merged['post_bcg_rank'].isna(), 'category'] = 'missing_in_post'

        # Define success transitions
        def categorize_transition(row):
            """Categorize the type of rank transition."""
            if pd.isna(row['pre_bcg_rank']) or pd.isna(row['post_bcg_rank']):
                return 'data_missing'

            pre_rank = int(row['pre_bcg_rank'])
            post_rank = int(row['post_bcg_rank'])

            if pre_rank == post_rank:
                if pre_rank == 1:
                    return 'Rank-1 maintained'
                else:
                    return f'Rank-{pre_rank} maintained'
            elif post_rank < pre_rank:
                # Improvement
                if post_rank == 1:
                    if pre_rank == 2:
                        return 'Rank-2 → Rank-1'
                    elif pre_rank == 3:
                        return 'Rank-3 → Rank-1'
                    else:
                        return f'Rank-{pre_rank} → Rank-1'
                elif post_rank == 2:
                    return f'Rank-{pre_rank} → Rank-2'
                else:
                    return f'Rank-{pre_rank} → Rank-{post_rank}'
            else:
                # Degradation
                if pre_rank == 1:
                    if post_rank == 2:
                        return 'Rank-1 → Rank-2'
                    elif post_rank == 3:
                        return 'Rank-1 → Rank-3'
                    else:
                        return f'Rank-1 → Rank-{post_rank}'
                elif pre_rank == 2:
                    return f'Rank-2 → Rank-{post_rank}'
                else:
                    return f'Rank-{pre_rank} → Rank-{post_rank}'

        merged['transition'] = merged.apply(categorize_transition, axis=1)

        return merged

    def generate_summary_statistics(self):
        """Generate summary statistics of the comparison."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS: PRE-PURGE VS POST-PURGE")
        print("="*80)

        # Overall statistics
        total_compared = len(self.comparison_df[self.comparison_df['category'].isin(['improved', 'degraded', 'unchanged'])])
        n_improved = len(self.comparison_df[self.comparison_df['category'] == 'improved'])
        n_degraded = len(self.comparison_df[self.comparison_df['category'] == 'degraded'])
        n_unchanged = len(self.comparison_df[self.comparison_df['category'] == 'unchanged'])

        print(f"\nTotal clusters compared: {total_compared}")
        print(f"  Improved:   {n_improved:3d} ({n_improved/total_compared*100:5.1f}%)")
        print(f"  Degraded:   {n_degraded:3d} ({n_degraded/total_compared*100:5.1f}%)")
        print(f"  Unchanged:  {n_unchanged:3d} ({n_unchanged/total_compared*100:5.1f}%)")

        # Rank-1 specific analysis
        print("\n" + "-"*80)
        print("RANK-1 ANALYSIS")
        print("-"*80)

        pre_rank1 = len(self.comparison_df[self.comparison_df['pre_bcg_rank'] == 1])
        post_rank1 = len(self.comparison_df[self.comparison_df['post_bcg_rank'] == 1])

        print(f"Pre-purge Rank-1 successes:  {pre_rank1}")
        print(f"Post-purge Rank-1 successes: {post_rank1}")
        print(f"Net change: {post_rank1 - pre_rank1:+d}")

        # New Rank-1 successes (were not Rank-1 before)
        new_rank1 = self.comparison_df[
            (self.comparison_df['post_bcg_rank'] == 1) &
            (self.comparison_df['pre_bcg_rank'] != 1) &
            (self.comparison_df['pre_bcg_rank'].notna())
        ]
        print(f"\nNew Rank-1 successes (improved from other ranks): {len(new_rank1)}")

        # Lost Rank-1 (were Rank-1 before, now worse)
        lost_rank1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] != 1) &
            (self.comparison_df['post_bcg_rank'].notna())
        ]
        print(f"Lost Rank-1 status (degraded to other ranks): {len(lost_rank1)}")

        # Transition breakdown
        print("\n" + "-"*80)
        print("TRANSITION BREAKDOWN")
        print("-"*80)

        transition_counts = self.comparison_df['transition'].value_counts().sort_index()
        for transition, count in transition_counts.items():
            if transition != 'data_missing':
                print(f"{transition:30s}: {count:3d}")

        return {
            'total_compared': total_compared,
            'n_improved': n_improved,
            'n_degraded': n_degraded,
            'n_unchanged': n_unchanged,
            'pre_rank1': pre_rank1,
            'post_rank1': post_rank1,
            'new_rank1': len(new_rank1),
            'lost_rank1': len(lost_rank1),
            'transitions': transition_counts.to_dict()
        }

    def save_detailed_reports(self):
        """Save detailed CSV reports for different categories."""
        print("\n" + "="*80)
        print("SAVING DETAILED REPORTS")
        print("="*80)

        # All comparisons
        output_file = self.output_dir / "all_comparisons.csv"
        self.comparison_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

        # Improved clusters (all ranks)
        improved = self.comparison_df[self.comparison_df['category'] == 'improved'].sort_values('rank_change')
        output_file = self.output_dir / "improved_clusters.csv"
        improved.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(improved)} clusters)")

        # Degraded clusters (all ranks)
        degraded = self.comparison_df[self.comparison_df['category'] == 'degraded'].sort_values('rank_change', ascending=False)
        output_file = self.output_dir / "degraded_clusters.csv"
        degraded.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(degraded)} clusters)")

        # New Rank-1 successes
        new_rank1 = self.comparison_df[
            (self.comparison_df['post_bcg_rank'] == 1) &
            (self.comparison_df['pre_bcg_rank'] != 1) &
            (self.comparison_df['pre_bcg_rank'].notna())
        ].sort_values('pre_bcg_rank')
        output_file = self.output_dir / "new_rank1_successes.csv"
        new_rank1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(new_rank1)} clusters)")

        # Lost Rank-1 status
        lost_rank1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] != 1) &
            (self.comparison_df['post_bcg_rank'].notna())
        ].sort_values('post_bcg_rank')
        output_file = self.output_dir / "lost_rank1_status.csv"
        lost_rank1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(lost_rank1)} clusters)")

        # Rank-2 → Rank-1
        rank2_to_1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 2) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ]
        output_file = self.output_dir / "rank2_to_rank1.csv"
        rank2_to_1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(rank2_to_1)} clusters)")

        # Rank-3 → Rank-1
        rank3_to_1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 3) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ]
        output_file = self.output_dir / "rank3_to_rank1.csv"
        rank3_to_1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(rank3_to_1)} clusters)")

        # Maintained Rank-1
        maintained_rank1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ]
        output_file = self.output_dir / "maintained_rank1.csv"
        maintained_rank1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(maintained_rank1)} clusters)")

    def create_visualizations(self):
        """Create visualization plots."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        # 1. Transition sankey-style bar plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Overall category distribution
        ax = axes[0, 0]
        category_counts = self.comparison_df['category'].value_counts()
        valid_categories = ['improved', 'degraded', 'unchanged']
        category_counts = category_counts[category_counts.index.isin(valid_categories)]
        colors = {'improved': '#2ECC71', 'degraded': '#E74C3C', 'unchanged': '#95A5A6'}
        bar_colors = [colors[cat] for cat in category_counts.index]

        ax.bar(range(len(category_counts)), category_counts.values, color=bar_colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.index], fontsize=14)
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Overall Performance Changes', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        # Add value labels
        for i, v in enumerate(category_counts.values):
            ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Plot 2: Rank distribution comparison
        ax = axes[0, 1]
        ranks = [1, 2, 3, 4, 5, 6]
        pre_rank_counts = [len(self.comparison_df[self.comparison_df['pre_bcg_rank'] == r]) for r in ranks]
        post_rank_counts = [len(self.comparison_df[self.comparison_df['post_bcg_rank'] == r]) for r in ranks]

        x = np.arange(len(ranks))
        width = 0.35

        ax.bar(x - width/2, pre_rank_counts, width, label='Pre-purge', color='#3498DB', edgecolor='black', linewidth=1.2)
        ax.bar(x + width/2, post_rank_counts, width, label='Post-purge', color='#E67E22', edgecolor='black', linewidth=1.2)

        ax.set_xlabel('BCG Rank', fontsize=14)
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Rank Distribution: Pre vs Post Purge', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank-{r}' for r in ranks], fontsize=12)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)

        # Plot 3: Top transitions
        ax = axes[1, 0]
        transition_counts = self.comparison_df['transition'].value_counts()
        # Filter out 'data_missing' and get top 10
        transition_counts = transition_counts[transition_counts.index != 'data_missing'].head(10)

        # Color code: green for improvements, red for degradations, gray for maintained
        bar_colors = []
        for transition in transition_counts.index:
            if '→' in transition:
                parts = transition.split('→')
                pre_part = parts[0].strip()
                post_part = parts[1].strip()
                # Extract rank numbers
                pre_rank = int(pre_part.split('-')[1]) if 'Rank-' in pre_part else 999
                post_rank = int(post_part.split('-')[1]) if 'Rank-' in post_part else 999
                if post_rank < pre_rank:
                    bar_colors.append('#2ECC71')  # Green for improvement
                elif post_rank > pre_rank:
                    bar_colors.append('#E74C3C')  # Red for degradation
                else:
                    bar_colors.append('#95A5A6')  # Gray
            else:
                bar_colors.append('#95A5A6')  # Gray for maintained

        ax.barh(range(len(transition_counts)), transition_counts.values, color=bar_colors, edgecolor='black', linewidth=1.2)
        ax.set_yticks(range(len(transition_counts)))
        ax.set_yticklabels(transition_counts.index, fontsize=11)
        ax.set_xlabel('Number of Clusters', fontsize=14)
        ax.set_title('Top Rank Transitions', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.invert_yaxis()

        # Add value labels
        for i, v in enumerate(transition_counts.values):
            ax.text(v + 0.5, i, str(v), ha='left', va='center', fontsize=10, fontweight='bold')

        # Plot 4: Rank-1 success comparison
        ax = axes[1, 1]
        rank1_data = {
            'Pre-purge\nRank-1': len(self.comparison_df[self.comparison_df['pre_bcg_rank'] == 1]),
            'Post-purge\nRank-1': len(self.comparison_df[self.comparison_df['post_bcg_rank'] == 1]),
            'New\nRank-1': len(self.comparison_df[
                (self.comparison_df['post_bcg_rank'] == 1) &
                (self.comparison_df['pre_bcg_rank'] != 1) &
                (self.comparison_df['pre_bcg_rank'].notna())
            ]),
            'Lost\nRank-1': len(self.comparison_df[
                (self.comparison_df['pre_bcg_rank'] == 1) &
                (self.comparison_df['post_bcg_rank'] != 1) &
                (self.comparison_df['post_bcg_rank'].notna())
            ])
        }

        colors_rank1 = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C']
        ax.bar(range(len(rank1_data)), rank1_data.values(), color=colors_rank1, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(rank1_data)))
        ax.set_xticklabels(rank1_data.keys(), fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Rank-1 Success Analysis', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        # Add value labels
        for i, v in enumerate(rank1_data.values()):
            ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / "comparison_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

        # 2. Detailed transition matrix heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create transition matrix
        max_rank = 6
        transition_matrix = np.zeros((max_rank, max_rank))

        for _, row in self.comparison_df.iterrows():
            if pd.notna(row['pre_bcg_rank']) and pd.notna(row['post_bcg_rank']):
                pre_idx = int(row['pre_bcg_rank']) - 1
                post_idx = int(row['post_bcg_rank']) - 1
                if 0 <= pre_idx < max_rank and 0 <= post_idx < max_rank:
                    transition_matrix[pre_idx, post_idx] += 1

        # Plot heatmap
        sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'Rank-{i+1}' for i in range(max_rank)],
                   yticklabels=[f'Rank-{i+1}' for i in range(max_rank)],
                   cbar_kws={'label': 'Number of Clusters'},
                   linewidths=0.5, linecolor='black', ax=ax)

        ax.set_xlabel('Post-Purge Rank', fontsize=14, fontweight='bold')
        ax.set_ylabel('Pre-Purge Rank', fontsize=14, fontweight='bold')
        ax.set_title('Rank Transition Matrix: Pre-Purge → Post-Purge', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        output_file = self.output_dir / "transition_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*80)
        print("PRE-PURGE VS POST-PURGE PERFORMANCE COMPARISON")
        print("="*80)
        print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")

        # Generate summary statistics
        stats = self.generate_summary_statistics()

        # Save detailed reports
        self.save_detailed_reports()

        # Create visualizations
        self.create_visualizations()

        # Save summary statistics
        summary_file = self.output_dir / "summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write("PRE-PURGE VS POST-PURGE PERFORMANCE COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total clusters compared: {stats['total_compared']}\n")
            f.write(f"  Improved:   {stats['n_improved']:3d} ({stats['n_improved']/stats['total_compared']*100:5.1f}%)\n")
            f.write(f"  Degraded:   {stats['n_degraded']:3d} ({stats['n_degraded']/stats['total_compared']*100:5.1f}%)\n")
            f.write(f"  Unchanged:  {stats['n_unchanged']:3d} ({stats['n_unchanged']/stats['total_compared']*100:5.1f}%)\n\n")
            f.write("RANK-1 ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Pre-purge Rank-1 successes:  {stats['pre_rank1']}\n")
            f.write(f"Post-purge Rank-1 successes: {stats['post_rank1']}\n")
            f.write(f"Net change: {stats['post_rank1'] - stats['pre_rank1']:+d}\n")
            f.write(f"New Rank-1 successes: {stats['new_rank1']}\n")
            f.write(f"Lost Rank-1 status: {stats['lost_rank1']}\n\n")
            f.write("TRANSITION BREAKDOWN\n")
            f.write("-"*80 + "\n")
            for transition, count in sorted(stats['transitions'].items()):
                if transition != 'data_missing':
                    f.write(f"{transition:30s}: {count:3d}\n")

        print(f"\nSaved: {summary_file}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define paths
    pre_purge_path = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/oct13/candidate_classifier_color_uq_run_20251013_151226/evaluation_results/evaluation_results.csv"
    post_purge_path = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/nov21/candidate_classifier_color_uq_run_20251121_004114/evaluation_results/evaluation_results.csv"

    # Create output directory in current directory
    output_dir = Path.cwd() / "purge_comparison_analysis"

    # Run analysis
    comparison = PrePostPurgeComparison(pre_purge_path, post_purge_path, output_dir)
    comparison.run_full_analysis()
