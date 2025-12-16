#!/usr/bin/env python3
"""
CCG Analysis Runner

This script performs the complete p_{CCG} (Cluster Central Galaxy probability) analysis:
1. Loads evaluation results from test.py
2. Loads detailed candidate data (if available)
3. Computes p_{CCG} for top candidates based on cluster member density
4. Generates diagnostic plots comparing p_{CCG} vs bar_p
5. Creates physical images with member overlays showing p_{CCG} values

Usage:
    python run_ccg_analysis.py --experiment_dir <path> --image_dir <path>

The analysis results are saved to:
    <experiment_dir>/evaluation_results/physical_images_with_members/
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccg_probability import (
    CCGProbabilityCalculator, load_rm_member_catalog,
    find_cluster_image, read_wcs_from_tif, pixel_to_radec,
    get_data_paths
)
from ccg_visualization import (
    plot_pccg_vs_barp_diagnostic, plot_pccg_summary_scatter,
    plot_cluster_with_members_pccg, plot_pccg_sectors,
    plot_pccg_completeness_purity, select_diverse_images
)


class CCGAnalysisRunner:
    """
    Runner class for complete p_{CCG} analysis.
    """

    def __init__(self, experiment_dir, image_dir, dataset_type='3p8arcmin',
                 radius_kpc=300.0, relative_threshold=5.0, top_n_candidates=3,
                 rm_member_dir=None, pmem_cutoff=0.2, use_adaptive_method=True,
                 dominance_fraction=0.4, min_member_fraction=0.05,
                 distribution_mode='proportional'):
        """
        Args:
            experiment_dir: Root experiment directory (e.g., trained_models/candidate_classifier_*)
            image_dir: Directory containing cluster images
            dataset_type: '2p2arcmin' or '3p8arcmin'
            radius_kpc: Physical radius for member counting (default 300 kpc)
            relative_threshold: Threshold for p_{CCG} dominance (legacy method)
            top_n_candidates: Number of top candidates to consider
            rm_member_dir: Directory with RedMapper member catalogs
            pmem_cutoff: Minimum pmem value to consider a member (default 0.2)
            use_adaptive_method: Use adaptive per-image criterion (default True)
            dominance_fraction: Fraction of total members for dominance (default 0.4)
                               A candidate with > 40% of cluster members is dominant
            min_member_fraction: Minimum fraction to be considered viable (default 0.05)
                                Candidates with < 5% of members get p_CCG = 0
            distribution_mode: How to distribute p_CCG: 'proportional' or 'equal'
        """
        self.experiment_dir = experiment_dir
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.radius_kpc = radius_kpc
        self.relative_threshold = relative_threshold
        self.top_n_candidates = top_n_candidates
        self.rm_member_dir = rm_member_dir or get_data_paths()['rm_member_dir']
        self.pmem_cutoff = pmem_cutoff
        self.use_adaptive_method = use_adaptive_method
        self.dominance_fraction = dominance_fraction
        self.min_member_fraction = min_member_fraction
        self.distribution_mode = distribution_mode

        # Set up paths
        self.eval_dir = os.path.join(experiment_dir, 'evaluation_results')
        self.output_dir = os.path.join(self.eval_dir, 'physical_images_with_members')

        # Initialize calculator
        self.calculator = CCGProbabilityCalculator(
            radius_kpc=radius_kpc,
            relative_threshold=relative_threshold,
            use_weighted_counts=True,
            rm_member_dir=self.rm_member_dir,
            pmem_cutoff=pmem_cutoff,
            use_adaptive_method=use_adaptive_method,
            dominance_fraction=dominance_fraction,
            min_member_fraction=min_member_fraction,
            distribution_mode=distribution_mode
        )

        # Results storage
        self.detailed_results = []
        self.summary_df = None

    def load_evaluation_data(self):
        """Load evaluation results and additional data."""
        eval_csv = os.path.join(self.eval_dir, 'evaluation_results.csv')
        if not os.path.exists(eval_csv):
            raise FileNotFoundError(f"Evaluation results not found: {eval_csv}")

        self.eval_df = pd.read_csv(eval_csv)
        print(f"Loaded evaluation results: {len(self.eval_df)} samples")

        # Check for features file with full candidate data
        features_file = os.path.join(self.eval_dir, 'test_features.npz')
        if os.path.exists(features_file):
            self.features_data = np.load(features_file, allow_pickle=True)
            print(f"Loaded test features: {self.features_data['X'].shape}")
        else:
            self.features_data = None
            print("Note: test_features.npz not found, using evaluation results only")

        return self.eval_df

    def compute_pccg_for_all_clusters(self, max_clusters=None, verbose=True):
        """
        Compute p_{CCG} for all clusters in the evaluation results.

        Args:
            max_clusters: Maximum number of clusters to process (None for all)
            verbose: Print progress information

        Returns:
            DataFrame with p_{CCG} results
        """
        if not hasattr(self, 'eval_df'):
            self.load_evaluation_data()

        results_list = []
        n_processed = 0
        n_errors = 0

        # Get unique clusters
        if 'cluster_name' not in self.eval_df.columns:
            print("Error: cluster_name column not found in evaluation results")
            return pd.DataFrame()

        for idx, row in self.eval_df.iterrows():
            if max_clusters is not None and n_processed >= max_clusters:
                break

            cluster_name = row.get('cluster_name', 'unknown')
            if cluster_name == 'unknown' or pd.isna(cluster_name):
                continue

            redshift = row.get('z', np.nan)
            pred_x = row.get('pred_x', np.nan)
            pred_y = row.get('pred_y', np.nan)
            bar_p = row.get('max_probability', 1.0)
            bcg_rank = row.get('bcg_rank', None)

            # Skip invalid entries
            if np.isnan(pred_x) or np.isnan(pred_y):
                continue

            # Find image
            image_path = find_cluster_image(cluster_name, self.image_dir)
            if image_path is None:
                if verbose:
                    print(f"  Warning: Image not found for {cluster_name}")
                n_errors += 1
                continue

            # For now, we work with the single predicted candidate
            # In future, this could be extended to load all candidates from test_features.npz
            candidates_pixel = np.array([[pred_x, pred_y]])
            candidate_probs = np.array([bar_p])

            # Compute p_{CCG}
            result = self.calculator.compute_for_cluster(
                cluster_name, candidates_pixel, candidate_probs,
                image_path=image_path, redshift=redshift,
                top_n_candidates=1
            )

            # Get target info
            true_x = row.get('true_x', np.nan)
            true_y = row.get('true_y', np.nan)
            bcg_prob = row.get('bcg_prob', np.nan)

            # Store detailed result
            detailed = {
                'cluster_name': cluster_name,
                'redshift': redshift,
                'candidates_pixel': candidates_pixel,
                'candidate_probs': candidate_probs,
                'p_ccg': result['p_ccg'],
                'member_counts': result['member_counts'],
                'weighted_counts': result['weighted_counts'],
                'member_fractions': result.get('member_fractions', np.array([])),
                'radius_kpc': self.radius_kpc,
                'members_in_fov': result['members_in_fov'],
                'total_weighted_members': result.get('total_weighted_members', 0),
                'target_coords': (true_x, true_y) if not np.isnan(true_x) else None,
                'target_prob': bcg_prob,
                'error': result.get('error')
            }
            self.detailed_results.append(detailed)

            # Store summary result
            member_frac = result.get('member_fractions', np.array([]))
            summary = {
                'cluster_name': cluster_name,
                'z': redshift,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'true_x': true_x,
                'true_y': true_y,
                'bar_p': bar_p,
                'bcg_rank': bcg_rank,
                'p_ccg': result['p_ccg'][0] if len(result['p_ccg']) > 0 else np.nan,
                'n_members': result['member_counts'][0] if len(result['member_counts']) > 0 else 0,
                'weighted_members': result['weighted_counts'][0] if len(result['weighted_counts']) > 0 else 0,
                'member_fraction': member_frac[0] if len(member_frac) > 0 else np.nan,
                'members_in_fov': result['members_in_fov'],
                'total_weighted_members': result.get('total_weighted_members', 0),
                'radius_kpc': self.radius_kpc,
                'error': result.get('error')
            }
            results_list.append(summary)

            n_processed += 1

            if verbose and n_processed % 50 == 0:
                print(f"  Processed {n_processed} clusters...")

        self.summary_df = pd.DataFrame(results_list)

        if verbose:
            print(f"\nProcessed {n_processed} clusters")
            print(f"Errors: {n_errors}")
            if len(self.summary_df) > 0:
                valid_pccg = ~self.summary_df['p_ccg'].isna()
                print(f"Valid p_CCG results: {valid_pccg.sum()}")

        return self.summary_df

    def generate_diagnostic_plots(self):
        """Generate diagnostic plots comparing p_{CCG} and bar_p."""
        if self.summary_df is None or len(self.summary_df) == 0:
            print("No results to plot. Run compute_pccg_for_all_clusters first.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Generate comprehensive diagnostic plots
        plot_pccg_vs_barp_diagnostic(
            self.summary_df, self.output_dir, self.dataset_type
        )

        # Generate summary scatter plot
        plot_pccg_summary_scatter(self.summary_df, self.output_dir)

        # Generate sectors plot (like diagnostic_plots_sectors.png)
        plot_pccg_sectors(self.summary_df, self.output_dir, self.dataset_type)

        # Generate completeness/purity plots (like completeness_purity_plots.png)
        plot_pccg_completeness_purity(self.summary_df, self.output_dir, self.dataset_type)

    def generate_physical_images(self, n_images=20, selection='diverse'):
        """
        Generate physical images with member overlays and p_{CCG} values.

        Args:
            n_images: Number of images to generate
            selection: How to select images:
                - 'diverse': Mix of high/low p_CCG vs bar_p agreement (best matches, mismatches)
                - 'disagreement': Focus on cases where p_CCG != bar_p
                - 'random': Random selection
        """
        if not self.detailed_results:
            print("No detailed results. Run compute_pccg_for_all_clusters first.")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Import candidate detection function for showing all BCG candidates
        try:
            from utils.candidate_based_bcg import find_bcg_candidates
            can_detect_candidates = True
        except ImportError:
            print("  Warning: Could not import find_bcg_candidates, will only show top prediction")
            can_detect_candidates = False

        # Use the diverse selection function for best mix of examples
        if selection == 'diverse':
            selected = select_diverse_images(self.detailed_results, n_images)
        else:
            # Fallback to old method for other selection types
            valid_results = [r for r in self.detailed_results
                            if r.get('error') is None and
                            len(r.get('p_ccg', [])) > 0 and
                            not np.isnan(r.get('redshift', np.nan))]

            if selection == 'disagreement':
                sorted_by_disagreement = sorted(
                    valid_results,
                    key=lambda r: abs(r['p_ccg'][0] - r['candidate_probs'][0]),
                    reverse=True
                )
                selected = sorted_by_disagreement[:n_images]
            else:  # random
                np.random.shuffle(valid_results)
                selected = valid_results[:n_images]

        if len(selected) == 0:
            print("No valid results for image generation")
            return

        print(f"Generating {len(selected)} physical images with members...")
        print(f"  Selection strategy: {selection}")
        print(f"  pmem cutoff: {self.pmem_cutoff}")

        n_generated = 0
        for result in selected:
            cluster_name = result['cluster_name']
            image_path = find_cluster_image(cluster_name, self.image_dir)

            if image_path is None:
                continue

            save_path = os.path.join(self.output_dir, f'{cluster_name}_pccg.png')

            # Detect ALL BCG candidates from the image (like ProbabilisticTesting plots)
            all_candidates = None
            if can_detect_candidates:
                try:
                    from PIL import Image as pillow_img
                    pil_image = pillow_img.open(image_path)
                    pil_image.seek(0)
                    image_array = np.array(pil_image)
                    pil_image.close()

                    # Ensure image is in correct format (convert 16-bit to 8-bit if needed)
                    if image_array.dtype == np.uint16:
                        image_array = (image_array / 256).astype(np.uint8)
                    elif image_array.dtype != np.uint8:
                        # Normalize to 0-255 range
                        img_min, img_max = image_array.min(), image_array.max()
                        if img_max > img_min:
                            image_array = ((image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            image_array = np.zeros_like(image_array, dtype=np.uint8)

                    # Use default candidate detection parameters
                    all_candidates, _ = find_bcg_candidates(
                        image_array,
                        min_distance=8,
                        threshold_rel=0.1,
                        exclude_border=0,
                        max_candidates=50
                    )

                    if all_candidates is not None and len(all_candidates) > 0:
                        print(f"    Detected {len(all_candidates)} BCG candidates for {cluster_name}")
                    else:
                        print(f"    No candidates detected for {cluster_name}")

                except Exception as e:
                    print(f"  Warning: Could not detect candidates for {cluster_name}: {e}")
                    all_candidates = None

            try:
                plot_cluster_with_members_pccg(
                    cluster_name=cluster_name,
                    image_path=image_path,
                    candidates_pixel=result['candidates_pixel'],
                    candidate_probs=result['candidate_probs'],
                    p_ccg_values=result['p_ccg'],
                    member_counts=result['member_counts'],
                    redshift=result['redshift'],
                    radius_kpc=self.radius_kpc,
                    wcs=None,
                    members_df=None,
                    rm_member_dir=self.rm_member_dir,
                    save_path=save_path,
                    dataset_type=self.dataset_type,
                    target_coords=result.get('target_coords'),
                    target_prob=result.get('target_prob'),
                    pmem_cutoff=self.pmem_cutoff,
                    all_candidates=all_candidates  # Pass all detected candidates
                )
                n_generated += 1
            except Exception as e:
                print(f"  Warning: Failed to generate image for {cluster_name}: {e}")

        print(f"Generated {n_generated} physical images to: {self.output_dir}")

    def save_results(self):
        """Save p_{CCG} results to CSV."""
        if self.summary_df is None:
            print("No results to save")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Save summary CSV
        csv_path = os.path.join(self.output_dir, 'p_ccg_results.csv')
        self.summary_df.to_csv(csv_path, index=False)
        print(f"Saved p_CCG results to: {csv_path}")

        # Print summary statistics
        valid_mask = ~self.summary_df['p_ccg'].isna()
        valid_df = self.summary_df[valid_mask]

        if len(valid_df) > 0:
            print("\n" + "="*60)
            print("p_{CCG} ANALYSIS SUMMARY")
            print("="*60)
            print(f"Total clusters processed: {len(self.summary_df)}")
            print(f"Valid p_CCG results: {len(valid_df)}")
            print(f"Search radius: {self.radius_kpc} kpc")
            print(f"p_mem cutoff: {self.pmem_cutoff}")
            print()
            print("Assignment method:")
            if self.use_adaptive_method:
                print(f"  Mode: ADAPTIVE (per-image member fractions)")
                print(f"  Dominance fraction: {self.dominance_fraction} ({self.dominance_fraction*100:.0f}% of cluster members)")
                print(f"  Min member fraction: {self.min_member_fraction} ({self.min_member_fraction*100:.0f}% threshold)")
                print(f"  Distribution mode: {self.distribution_mode}")
            else:
                print(f"  Mode: LEGACY (fixed relative threshold)")
                print(f"  Relative threshold: {self.relative_threshold}")
            print()

            # Agreement statistics
            if 'bar_p' in valid_df.columns:
                agree_high = ((valid_df['bar_p'] > 0.5) & (valid_df['p_ccg'] > 0.5)).sum()
                agree_low = ((valid_df['bar_p'] <= 0.5) & (valid_df['p_ccg'] <= 0.5)).sum()
                total_agree = agree_high + agree_low
                agree_pct = total_agree / len(valid_df) * 100

                print(f"Agreement (both >0.5 or both <=0.5): {agree_pct:.1f}%")

                corr = np.corrcoef(valid_df['bar_p'], valid_df['p_ccg'])[0, 1]
                print(f"Correlation (bar_p vs p_CCG): {corr:.3f}")

            # Member statistics
            if 'n_members' in valid_df.columns:
                print()
                print("Member count statistics:")
                print(f"  Mean: {valid_df['n_members'].mean():.1f}")
                print(f"  Median: {valid_df['n_members'].median():.0f}")
                print(f"  Min: {valid_df['n_members'].min():.0f}")
                print(f"  Max: {valid_df['n_members'].max():.0f}")

    def run_complete_analysis(self, n_images=20, max_clusters=None, verbose=True):
        """
        Run the complete p_{CCG} analysis pipeline.

        Args:
            n_images: Number of physical images to generate
            max_clusters: Maximum clusters to process (None for all)
            verbose: Print progress
        """
        print("="*60)
        print("CCG PROBABILITY ANALYSIS")
        print("="*60)
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Image directory: {self.image_dir}")
        print(f"Dataset type: {self.dataset_type}")
        print(f"Search radius: {self.radius_kpc} kpc")
        print()

        # Step 1: Load data
        print("Step 1: Loading evaluation data...")
        self.load_evaluation_data()

        # Step 2: Compute p_{CCG}
        print("\nStep 2: Computing p_{CCG}...")
        self.compute_pccg_for_all_clusters(max_clusters=max_clusters, verbose=verbose)

        # Step 3: Save results
        print("\nStep 3: Saving results...")
        self.save_results()

        # Step 4: Generate diagnostic plots
        print("\nStep 4: Generating diagnostic plots...")
        self.generate_diagnostic_plots()

        # Step 5: Generate physical images
        print(f"\nStep 5: Generating {n_images} physical images with members...")
        self.generate_physical_images(n_images=n_images, selection='diverse')

        print("\n" + "="*60)
        print("CCG ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")

        return self.summary_df


def run_ccg_analysis_from_experiment(experiment_dir, image_dir, dataset_type='3p8arcmin',
                                     radius_kpc=300.0, pmem_cutoff=0.2, n_images=20,
                                     use_adaptive_method=True, dominance_fraction=0.4,
                                     min_member_fraction=0.05, distribution_mode='proportional',
                                     relative_threshold=5.0):
    """
    Convenience function to run CCG analysis from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory
        image_dir: Path to image directory
        dataset_type: Dataset type
        radius_kpc: Search radius in kpc
        pmem_cutoff: Minimum pmem value to consider a member
        n_images: Number of images to generate
        use_adaptive_method: Use adaptive per-image criterion (default True)
        dominance_fraction: Fraction of total members for dominance (default 0.4)
        min_member_fraction: Minimum fraction to be considered viable (default 0.05)
        distribution_mode: 'proportional' or 'equal'
        relative_threshold: Threshold for p_{CCG} dominance (legacy method)

    Returns:
        DataFrame with p_{CCG} results
    """
    runner = CCGAnalysisRunner(
        experiment_dir=experiment_dir,
        image_dir=image_dir,
        dataset_type=dataset_type,
        radius_kpc=radius_kpc,
        relative_threshold=relative_threshold,
        pmem_cutoff=pmem_cutoff,
        use_adaptive_method=use_adaptive_method,
        dominance_fraction=dominance_fraction,
        min_member_fraction=min_member_fraction,
        distribution_mode=distribution_mode
    )

    return runner.run_complete_analysis(n_images=n_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run p_{CCG} analysis on BCG classification results"
    )

    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment directory containing evaluation_results/')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing cluster images')
    parser.add_argument('--dataset_type', type=str, default='3p8arcmin',
                       choices=['2p2arcmin', '3p8arcmin'],
                       help='Dataset type')
    parser.add_argument('--radius_kpc', type=float, default=300.0,
                       help='Physical radius for member counting (kpc)')
    parser.add_argument('--pmem_cutoff', type=float, default=0.2,
                       help='Minimum pmem value to consider a member')
    parser.add_argument('--n_images', type=int, default=20,
                       help='Number of physical images to generate')
    parser.add_argument('--max_clusters', type=int, default=None,
                       help='Maximum clusters to process (for testing)')

    # Adaptive method parameters (new)
    parser.add_argument('--use_adaptive', type=str, default='true',
                       choices=['true', 'false'],
                       help='Use adaptive per-image method (default: true)')
    parser.add_argument('--dominance_fraction', type=float, default=0.4,
                       help='Fraction of total members for dominance (default: 0.4 = 40%%)')
    parser.add_argument('--min_member_fraction', type=float, default=0.05,
                       help='Minimum fraction to be considered viable (default: 0.05 = 5%%)')
    parser.add_argument('--distribution_mode', type=str, default='proportional',
                       choices=['proportional', 'equal'],
                       help='How to distribute p_CCG among non-dominant candidates')

    # Legacy method parameter (only used if --use_adaptive=false)
    parser.add_argument('--relative_threshold', type=float, default=5.0,
                       help='Threshold for p_{CCG} dominance (legacy method only)')

    args = parser.parse_args()

    use_adaptive = args.use_adaptive.lower() == 'true'

    runner = CCGAnalysisRunner(
        experiment_dir=args.experiment_dir,
        image_dir=args.image_dir,
        dataset_type=args.dataset_type,
        radius_kpc=args.radius_kpc,
        relative_threshold=args.relative_threshold,
        pmem_cutoff=args.pmem_cutoff,
        use_adaptive_method=use_adaptive,
        dominance_fraction=args.dominance_fraction,
        min_member_fraction=args.min_member_fraction,
        distribution_mode=args.distribution_mode
    )

    results = runner.run_complete_analysis(
        n_images=args.n_images,
        max_clusters=args.max_clusters
    )

    print(f"\nFinal results shape: {results.shape}")
