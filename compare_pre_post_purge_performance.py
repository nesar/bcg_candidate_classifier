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
5. Visual comparisons for each transition type

Usage:
    python compare_pre_post_purge_performance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from PIL import Image
import os

# Set consistent plot style
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2
})


class PrePostPurgeComparison:
    """Compare BCG classification performance before and after purge."""

    def __init__(self, pre_purge_path, post_purge_path, pre_img_dir, post_img_dir, output_dir=None):
        """
        Initialize comparison.

        Args:
            pre_purge_path: Path to pre-purge evaluation_results.csv
            post_purge_path: Path to post-purge evaluation_results.csv
            pre_img_dir: Directory containing pre-purge prediction images
            post_img_dir: Directory containing post-purge prediction images
            output_dir: Directory to save analysis results (default: current directory)
        """
        self.pre_purge_path = Path(pre_purge_path)
        self.post_purge_path = Path(post_purge_path)
        self.pre_img_dir = Path(pre_img_dir)
        self.post_img_dir = Path(post_img_dir)

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

        # Merge and create comparison
        self.comparison_df = self._merge_and_compare()

        # Create image mapping for faster lookup
        self._create_image_mappings()

    def _create_image_mappings(self):
        """Create dictionaries mapping cluster names to image paths."""
        self.pre_images = {}
        self.post_images = {}

        # Pre-purge images
        for rank_file in self.pre_img_dir.glob("ProbabilisticTesting_prediction_sample_best_rank*_enhanced.png"):
            # Parse cluster name from the cluster column in pre_df
            pass

        # For now, we'll search by cluster name dynamically in the visualization function

    def _merge_and_compare(self):
        """Merge pre and post dataframes and add comparison columns."""
        # Rename columns to distinguish pre/post
        pre_cols = {
            'bcg_rank': 'pre_bcg_rank',
            'matches_any_target': 'pre_matches',
            'distance_error': 'pre_distance_error',
            'bcg_prob': 'pre_bcg_prob',
            'max_probability': 'pre_max_prob',
            'n_candidates': 'pre_n_candidates',
            'multi_target_rank': 'pre_multi_target_rank'
        }

        post_cols = {
            'bcg_rank': 'post_bcg_rank',
            'matches_any_target': 'post_matches',
            'distance_error': 'post_distance_error',
            'bcg_prob': 'post_bcg_prob',
            'max_probability': 'post_max_prob',
            'n_candidates': 'post_n_candidates',
            'multi_target_rank': 'post_multi_target_rank'
        }

        pre_subset = self.pre_df[['cluster_name', 'z'] + list(pre_cols.keys())].rename(columns=pre_cols)
        post_subset = self.post_df[['cluster_name', 'z'] + list(post_cols.keys())].rename(columns=post_cols)

        # Merge with outer join to capture all clusters
        merged = pd.merge(pre_subset, post_subset, on='cluster_name', how='outer', suffixes=('_pre', '_post'))

        # Handle missing z values
        merged['z'] = merged['z_pre'].fillna(merged['z_post'])
        merged = merged.drop(columns=['z_pre', 'z_post'])

        # Add comparison metrics - handle NaN values
        merged['rank_change'] = merged['post_bcg_rank'] - merged['pre_bcg_rank']
        merged['improved'] = merged['rank_change'] < 0  # Lower rank is better
        merged['degraded'] = merged['rank_change'] > 0
        merged['unchanged'] = merged['rank_change'] == 0

        # Categorize based on data availability and changes
        def categorize_cluster(row):
            """Categorize cluster based on pre/post availability and rank."""
            if pd.isna(row['pre_bcg_rank']) and pd.isna(row['post_bcg_rank']):
                return 'missing_both'
            elif pd.isna(row['pre_bcg_rank']):
                return 'new_in_post'
            elif pd.isna(row['post_bcg_rank']):
                return 'missing_in_post'
            elif row['rank_change'] < 0:
                return 'improved'
            elif row['rank_change'] > 0:
                return 'degraded'
            else:
                return 'unchanged'

        merged['category'] = merged.apply(categorize_cluster, axis=1)

        # Define success transitions with more detail
        def categorize_transition(row):
            """Categorize the type of rank transition."""
            if pd.isna(row['pre_bcg_rank']) and pd.isna(row['post_bcg_rank']):
                return 'missing_both'
            elif pd.isna(row['pre_bcg_rank']):
                # New cluster in post-purge
                post_rank = int(row['post_bcg_rank'])
                return f'New cluster → Rank-{post_rank}'
            elif pd.isna(row['post_bcg_rank']):
                # Missing in post-purge
                pre_rank = int(row['pre_bcg_rank'])
                return f'Rank-{pre_rank} → Missing'

            pre_rank = int(row['pre_bcg_rank'])
            post_rank = int(row['post_bcg_rank'])
            pre_matches = row['pre_matches']
            post_matches = row['post_matches']

            # Check for failure → success transitions
            if not pre_matches and post_matches:
                # Pre-purge was a failure (no correct detection), post-purge succeeded
                return f'Failure → Rank-{post_rank} (Success)'
            elif pre_matches and not post_matches:
                # Pre-purge succeeded, post-purge failed
                return f'Rank-{pre_rank} → Failure'
            elif not pre_matches and not post_matches:
                # Both failed
                return 'Failure → Failure'

            # Both succeeded - check rank changes
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

        # Overall dataset statistics
        print(f"\nTotal clusters in pre-purge dataset:  {len(self.pre_df)}")
        print(f"Total clusters in post-purge dataset: {len(self.post_df)}")
        print(f"Net change in dataset size: {len(self.post_df) - len(self.pre_df):+d}")

        # Comparable clusters (in both datasets)
        total_compared = len(self.comparison_df[self.comparison_df['category'].isin(['improved', 'degraded', 'unchanged'])])
        n_improved = len(self.comparison_df[self.comparison_df['category'] == 'improved'])
        n_degraded = len(self.comparison_df[self.comparison_df['category'] == 'degraded'])
        n_unchanged = len(self.comparison_df[self.comparison_df['category'] == 'unchanged'])

        print(f"\nComparable clusters (present in both): {total_compared}")
        print(f"  Improved:   {n_improved:3d} ({n_improved/total_compared*100:5.1f}%)")
        print(f"  Degraded:   {n_degraded:3d} ({n_degraded/total_compared*100:5.1f}%)")
        print(f"  Unchanged:  {n_unchanged:3d} ({n_unchanged/total_compared*100:5.1f}%)")

        # New and missing clusters
        n_new = len(self.comparison_df[self.comparison_df['category'] == 'new_in_post'])
        n_missing = len(self.comparison_df[self.comparison_df['category'] == 'missing_in_post'])
        print(f"\nClusters only in post-purge dataset: {n_new}")
        print(f"Clusters only in pre-purge dataset:  {n_missing}")

        # Rank-1 specific analysis
        print("\n" + "-"*80)
        print("RANK-1 ANALYSIS")
        print("-"*80)

        pre_rank1 = len(self.comparison_df[self.comparison_df['pre_bcg_rank'] == 1])
        post_rank1 = len(self.comparison_df[self.comparison_df['post_bcg_rank'] == 1])

        # Count Rank-1 in the original full datasets
        pre_rank1_total = len(self.pre_df[self.pre_df['bcg_rank'] == 1])
        post_rank1_total = len(self.post_df[self.post_df['bcg_rank'] == 1])

        print(f"Pre-purge Rank-1 successes (full dataset):  {pre_rank1_total}")
        print(f"Post-purge Rank-1 successes (full dataset): {post_rank1_total}")
        print(f"Net change in Rank-1 successes: {post_rank1_total - pre_rank1_total:+d}")
        print(f"\nBreakdown of {post_rank1_total - pre_rank1_total:+d} net Rank-1 change:")

        # For comparable clusters only
        new_rank1_from_improved = len(self.comparison_df[
            (self.comparison_df['post_bcg_rank'] == 1) &
            (self.comparison_df['pre_bcg_rank'] != 1) &
            (self.comparison_df['pre_bcg_rank'].notna())
        ])
        lost_rank1 = len(self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] != 1) &
            (self.comparison_df['post_bcg_rank'].notna())
        ])
        maintained_rank1 = len(self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ])

        # Rank-1 in new clusters (only in post-purge)
        new_rank1_from_new_clusters = len(self.comparison_df[
            (self.comparison_df['category'] == 'new_in_post') &
            (self.comparison_df['post_bcg_rank'] == 1)
        ])

        # Rank-1 lost in missing clusters (only in pre-purge)
        lost_rank1_from_missing = len(self.comparison_df[
            (self.comparison_df['category'] == 'missing_in_post') &
            (self.comparison_df['pre_bcg_rank'] == 1)
        ])

        print(f"  Maintained Rank-1 (in both datasets): {maintained_rank1}")
        print(f"  Gained Rank-1 from rank improvements: {new_rank1_from_improved:+d}")
        print(f"  Lost Rank-1 from rank degradations:  {-lost_rank1:+d}")
        print(f"  Rank-1 from new clusters:             {new_rank1_from_new_clusters:+d}")
        print(f"  Rank-1 from missing clusters:         {-lost_rank1_from_missing:+d}")
        print(f"  Total net change: {new_rank1_from_improved - lost_rank1 + new_rank1_from_new_clusters - lost_rank1_from_missing:+d}")

        # Failure → Success transitions
        print("\n" + "-"*80)
        print("FAILURE → SUCCESS TRANSITIONS")
        print("-"*80)

        failure_to_success = self.comparison_df[
            self.comparison_df['transition'].str.contains('Failure → Rank-', na=False)
        ]
        print(f"Total clusters moving from failure to success: {len(failure_to_success)}")

        for transition in sorted(failure_to_success['transition'].unique()):
            count = len(failure_to_success[failure_to_success['transition'] == transition])
            print(f"  {transition}: {count}")

        # Transition breakdown
        print("\n" + "-"*80)
        print("ALL TRANSITIONS")
        print("-"*80)

        transition_counts = self.comparison_df['transition'].value_counts().sort_index()
        for transition, count in transition_counts.items():
            print(f"{transition:40s}: {count:3d}")

        return {
            'total_compared': total_compared,
            'n_improved': n_improved,
            'n_degraded': n_degraded,
            'n_unchanged': n_unchanged,
            'n_new': n_new,
            'n_missing': n_missing,
            'pre_rank1_total': pre_rank1_total,
            'post_rank1_total': post_rank1_total,
            'new_rank1_from_improved': new_rank1_from_improved,
            'lost_rank1': lost_rank1,
            'maintained_rank1': maintained_rank1,
            'new_rank1_from_new_clusters': new_rank1_from_new_clusters,
            'lost_rank1_from_missing': lost_rank1_from_missing,
            'failure_to_success_count': len(failure_to_success),
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

        # New Rank-1 successes (from rank improvements)
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

        # Failure → Success transitions
        failure_to_success = self.comparison_df[
            self.comparison_df['transition'].str.contains('Failure → Rank-', na=False)
        ]
        output_file = self.output_dir / "failure_to_success.csv"
        failure_to_success.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(failure_to_success)} clusters)")

        # Specific rank transitions
        rank2_to_1 = self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 2) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ]
        output_file = self.output_dir / "rank2_to_rank1.csv"
        rank2_to_1.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(rank2_to_1)} clusters)")

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

        # New clusters
        new_clusters = self.comparison_df[self.comparison_df['category'] == 'new_in_post']
        output_file = self.output_dir / "new_clusters_post_purge.csv"
        new_clusters.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(new_clusters)} clusters)")

    def find_image_for_cluster(self, cluster_name, rank, image_dir, phase="ProbabilisticTesting"):
        """
        Find the prediction image for a given cluster and rank.

        Args:
            cluster_name: Name of the cluster
            rank: BCG rank (1, 2, 3, etc.)
            image_dir: Directory containing images
            phase: Phase name (default: "ProbabilisticTesting")

        Returns:
            Path to image file or None if not found
        """
        # Images are typically named like:
        # ProbabilisticTesting_prediction_sample_best_rank1_prediction_sample_1_enhanced.png

        # Strategy: Find all rank{rank} images, then match by cluster name in the corresponding eval results
        pattern = f"{phase}_prediction_sample_best_rank{rank}_*.png"
        matching_files = list(image_dir.glob(pattern))

        # For now, return None - we'll need to match by looking at the eval results
        # to map sample index to cluster name
        # This is a limitation - we need the eval results to have the sample order
        return None

    def create_2panel_comparison(self, cluster_name, pre_rank, post_rank, save_path):
        """
        Create a 2-panel side-by-side comparison image.

        Args:
            cluster_name: Name of the cluster
            pre_rank: Pre-purge rank
            post_rank: Post-purge rank
            save_path: Path to save the comparison image
        """
        # Find images - this is challenging without cluster name in filename
        # For now, skip if we can't find images
        # This would require more sophisticated matching

        # Create placeholder for now
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        axes[0].text(0.5, 0.5, f"Pre-purge\n{cluster_name}\nRank-{pre_rank}" if not pd.isna(pre_rank) else f"Pre-purge\n{cluster_name}\nNot in dataset",
                    ha='center', va='center', fontsize=16, transform=axes[0].transAxes)
        axes[0].axis('off')
        axes[0].set_title("Pre-Purge", fontsize=18, fontweight='bold')

        axes[1].text(0.5, 0.5, f"Post-purge\n{cluster_name}\nRank-{post_rank}" if not pd.isna(post_rank) else f"Post-purge\n{cluster_name}\nNot in dataset",
                    ha='center', va='center', fontsize=16, transform=axes[1].transAxes)
        axes[1].axis('off')
        axes[1].set_title("Post-Purge", fontsize=18, fontweight='bold')

        plt.suptitle(f"{cluster_name}: Rank {int(pre_rank) if not pd.isna(pre_rank) else 'N/A'} → Rank {int(post_rank) if not pd.isna(post_rank) else 'N/A'}",
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_transition_visualizations(self):
        """Create subdirectories with visual comparisons for each transition type."""
        print("\n" + "="*80)
        print("CREATING TRANSITION VISUALIZATIONS")
        print("="*80)

        # Get unique transitions
        transitions = self.comparison_df['transition'].value_counts()

        print(f"\nCreating visualization subdirectories for {len(transitions)} transition types...")

        for transition_type, count in transitions.items():
            if transition_type in ['missing_both']:
                continue

            # Create subdirectory for this transition type
            # Clean up transition name for directory
            dir_name = transition_type.replace('/', '_').replace(' ', '_').replace('→', 'to')
            trans_dir = self.output_dir / dir_name
            trans_dir.mkdir(exist_ok=True)

            # Get clusters with this transition
            clusters = self.comparison_df[self.comparison_df['transition'] == transition_type]

            print(f"\nProcessing: {transition_type} ({len(clusters)} clusters)")

            # Limit to first 20 for each transition type
            for idx, (_, row) in enumerate(clusters.head(20).iterrows()):
                cluster_name = row['cluster_name']
                pre_rank = row['pre_bcg_rank']
                post_rank = row['post_bcg_rank']

                # Create comparison image
                save_path = trans_dir / f"{cluster_name.replace(' ', '_')}_comparison.png"
                self.create_2panel_comparison(cluster_name, pre_rank, post_rank, save_path)

            print(f"  Created {min(len(clusters), 20)} comparison images in {trans_dir}")

    def create_visualizations(self):
        """Create overview visualization plots."""
        print("\n" + "="*80)
        print("CREATING OVERVIEW VISUALIZATIONS")
        print("="*80)

        # 1. Comprehensive 2x2 overview
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Plot 1: Overall category distribution (including new/missing clusters)
        ax = axes[0, 0]
        category_counts = self.comparison_df['category'].value_counts()
        category_order = ['improved', 'unchanged', 'degraded', 'new_in_post', 'missing_in_post']
        category_counts = category_counts.reindex([c for c in category_order if c in category_counts.index])

        colors = {
            'improved': '#2ECC71',
            'degraded': '#E74C3C',
            'unchanged': '#95A5A6',
            'new_in_post': '#3498DB',
            'missing_in_post': '#E67E22'
        }
        bar_colors = [colors.get(cat, '#95A5A6') for cat in category_counts.index]

        ax.bar(range(len(category_counts)), category_counts.values, color=bar_colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.index], fontsize=12, rotation=15, ha='right')
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Overall Performance Changes\n(Including New & Missing Clusters)', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        # Add value labels
        for i, v in enumerate(category_counts.values):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Plot 2: Rank distribution comparison (full datasets)
        ax = axes[0, 1]
        ranks = [1, 2, 3, 4, 5, 6]
        pre_rank_counts = [len(self.pre_df[self.pre_df['bcg_rank'] == r]) for r in ranks]
        post_rank_counts = [len(self.post_df[self.post_df['bcg_rank'] == r]) for r in ranks]

        x = np.arange(len(ranks))
        width = 0.35

        ax.bar(x - width/2, pre_rank_counts, width, label='Pre-purge', color='#3498DB', edgecolor='black', linewidth=1.2)
        ax.bar(x + width/2, post_rank_counts, width, label='Post-purge', color='#E67E22', edgecolor='black', linewidth=1.2)

        ax.set_xlabel('BCG Rank', fontsize=14)
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Rank Distribution: Pre vs Post Purge\n(Full Datasets)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Rank-{r}' for r in ranks], fontsize=12)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)

        # Plot 3: Top transitions (filter out missing_both and new/missing cluster transitions)
        ax = axes[1, 0]
        transition_counts = self.comparison_df['transition'].value_counts()

        # Filter for meaningful transitions
        meaningful_transitions = transition_counts[
            ~transition_counts.index.isin(['missing_both']) &
            ~transition_counts.index.str.contains('New cluster', na=False) &
            ~transition_counts.index.str.contains('Missing', na=False)
        ].head(12)

        # Color code: green for improvements, red for degradations, gray for maintained, blue for failure→success
        bar_colors = []
        for transition in meaningful_transitions.index:
            if 'Failure → Rank-' in transition:
                bar_colors.append('#3498DB')  # Blue for failure to success
            elif '→' in transition:
                parts = transition.split('→')
                pre_part = parts[0].strip()
                post_part = parts[1].strip()
                # Handle "Failure" as rank 999 for comparison
                if 'Failure' in pre_part:
                    pre_rank = 999
                elif 'Rank-' in pre_part:
                    pre_rank = int(pre_part.split('-')[1])
                else:
                    pre_rank = 999

                if 'Failure' in post_part:
                    post_rank = 999
                elif 'Rank-' in post_part:
                    post_rank = int(post_part.split('-')[1])
                else:
                    post_rank = 999

                if post_rank < pre_rank:
                    bar_colors.append('#2ECC71')  # Green for improvement
                elif post_rank > pre_rank:
                    bar_colors.append('#E74C3C')  # Red for degradation
                else:
                    bar_colors.append('#95A5A6')  # Gray
            else:
                bar_colors.append('#95A5A6')  # Gray for maintained

        ax.barh(range(len(meaningful_transitions)), meaningful_transitions.values, color=bar_colors, edgecolor='black', linewidth=1.2)
        ax.set_yticks(range(len(meaningful_transitions)))
        ax.set_yticklabels(meaningful_transitions.index, fontsize=10)
        ax.set_xlabel('Number of Clusters', fontsize=14)
        ax.set_title('Top Rank Transitions', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        ax.invert_yaxis()

        # Add value labels
        for i, v in enumerate(meaningful_transitions.values):
            ax.text(v + 0.5, i, str(v), ha='left', va='center', fontsize=10, fontweight='bold')

        # Plot 4: Rank-1 success breakdown
        ax = axes[1, 1]

        # Calculate Rank-1 components
        pre_rank1_total = len(self.pre_df[self.pre_df['bcg_rank'] == 1])
        post_rank1_total = len(self.post_df[self.post_df['bcg_rank'] == 1])

        maintained = len(self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] == 1)
        ])
        gained_from_ranks = len(self.comparison_df[
            (self.comparison_df['post_bcg_rank'] == 1) &
            (self.comparison_df['pre_bcg_rank'] != 1) &
            (self.comparison_df['pre_bcg_rank'].notna())
        ])
        lost_to_ranks = len(self.comparison_df[
            (self.comparison_df['pre_bcg_rank'] == 1) &
            (self.comparison_df['post_bcg_rank'] != 1) &
            (self.comparison_df['post_bcg_rank'].notna())
        ])
        from_new_clusters = len(self.comparison_df[
            (self.comparison_df['category'] == 'new_in_post') &
            (self.comparison_df['post_bcg_rank'] == 1)
        ])

        rank1_data = {
            'Pre-purge\nRank-1': pre_rank1_total,
            'Post-purge\nRank-1': post_rank1_total,
            'Maintained\nRank-1': maintained,
            'Gained from\nRank 2/3/etc': gained_from_ranks,
            'Lost to\nRank 2/3/etc': lost_to_ranks,
            'From new\nclusters': from_new_clusters
        }

        colors_rank1 = ['#3498DB', '#E67E22', '#95A5A6', '#2ECC71', '#E74C3C', '#9B59B6']
        ax.bar(range(len(rank1_data)), rank1_data.values(), color=colors_rank1, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(rank1_data)))
        ax.set_xticklabels(rank1_data.keys(), fontsize=11)
        ax.set_ylabel('Number of Clusters', fontsize=14)
        ax.set_title('Rank-1 Success Breakdown', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)

        # Add value labels
        for i, v in enumerate(rank1_data.values()):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / "comparison_overview.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

        # 2. Detailed transition matrix heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create transition matrix for comparable clusters only
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

        ax.set_xlabel('Post-Purge Rank', fontsize=16, fontweight='bold')
        ax.set_ylabel('Pre-Purge Rank', fontsize=16, fontweight='bold')
        ax.set_title('Rank Transition Matrix: Pre-Purge → Post-Purge\n(Comparable Clusters Only)', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)

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

        # Create transition visualizations
        self.create_transition_visualizations()

        # Save summary statistics to file
        summary_file = self.output_dir / "summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write("PRE-PURGE VS POST-PURGE PERFORMANCE COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Total clusters in pre-purge dataset:  {len(self.pre_df)}\n")
            f.write(f"Total clusters in post-purge dataset: {len(self.post_df)}\n")
            f.write(f"Net change in dataset size: {len(self.post_df) - len(self.pre_df):+d}\n\n")

            f.write(f"Comparable clusters (present in both): {stats['total_compared']}\n")
            f.write(f"  Improved:   {stats['n_improved']:3d} ({stats['n_improved']/stats['total_compared']*100:5.1f}%)\n")
            f.write(f"  Degraded:   {stats['n_degraded']:3d} ({stats['n_degraded']/stats['total_compared']*100:5.1f}%)\n")
            f.write(f"  Unchanged:  {stats['n_unchanged']:3d} ({stats['n_unchanged']/stats['total_compared']*100:5.1f}%)\n\n")

            f.write(f"Clusters only in post-purge dataset: {stats['n_new']}\n")
            f.write(f"Clusters only in pre-purge dataset:  {stats['n_missing']}\n\n")

            f.write("RANK-1 ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Pre-purge Rank-1 successes (full dataset):  {stats['pre_rank1_total']}\n")
            f.write(f"Post-purge Rank-1 successes (full dataset): {stats['post_rank1_total']}\n")
            f.write(f"Net change in Rank-1 successes: {stats['post_rank1_total'] - stats['pre_rank1_total']:+d}\n\n")

            f.write(f"Breakdown of {stats['post_rank1_total'] - stats['pre_rank1_total']:+d} net Rank-1 change:\n")
            f.write(f"  Maintained Rank-1 (in both datasets): {stats['maintained_rank1']}\n")
            f.write(f"  Gained Rank-1 from rank improvements: {stats['new_rank1_from_improved']:+d}\n")
            f.write(f"  Lost Rank-1 from rank degradations:  {-stats['lost_rank1']:+d}\n")
            f.write(f"  Rank-1 from new clusters:             {stats['new_rank1_from_new_clusters']:+d}\n")
            f.write(f"  Rank-1 from missing clusters:         {-stats['lost_rank1_from_missing']:+d}\n\n")

            f.write("FAILURE → SUCCESS TRANSITIONS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total clusters moving from failure to success: {stats['failure_to_success_count']}\n\n")

            f.write("ALL TRANSITIONS\n")
            f.write("-"*80 + "\n")
            for transition, count in sorted(stats['transitions'].items()):
                f.write(f"{transition:40s}: {count:3d}\n")

        print(f"\nSaved: {summary_file}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {self.output_dir}")


if __name__ == "__main__":
    # Define paths
    pre_purge_path = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/oct13/candidate_classifier_color_uq_run_20251013_151226/evaluation_results/evaluation_results.csv"
    post_purge_path = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/nov21/candidate_classifier_color_uq_run_20251121_004114/evaluation_results/evaluation_results.csv"

    # Image directories
    pre_img_dir = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/oct13/candidate_classifier_color_uq_run_20251013_151226/evaluation_results/physical_images"
    post_img_dir = "/Users/nesar/Projects/HEP/IMGmarker/best_runs/nov21/candidate_classifier_color_uq_run_20251121_004114/evaluation_results/physical_images"

    # Create output directory in current directory
    output_dir = Path.cwd() / "purge_comparison_analysis"

    # Run analysis
    comparison = PrePostPurgeComparison(pre_purge_path, post_purge_path, pre_img_dir, post_img_dir, output_dir)
    comparison.run_full_analysis()
