#!/usr/bin/env python3
"""
PMEM Analysis: Explore RedMapper membership probabilities for BCG classification

This script analyzes the rm_member_catalogs data and compares it with:
1. Prior locations from prior_locations_more_info_PURGED_*.csv
2. BCG matched dataset from bcgs_3p8arcmin_clean_matched.csv

The goal is to understand how pmem values can improve BCG identification.

NOTE: File paths are configured for both local and remote (LCRC) execution.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

# =============================================================================
# Configuration - File paths for local vs cluster execution
# =============================================================================

# Detect environment based on path existence
LOCAL_BASE = "/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/bcgs"
CLUSTER_BASE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs"

if os.path.exists(LOCAL_BASE):
    DATA_BASE = LOCAL_BASE
    print(f"Running on LOCAL machine: {DATA_BASE}")
else:
    DATA_BASE = CLUSTER_BASE
    print(f"Running on CLUSTER: {DATA_BASE}")

# Data paths
RM_MEMBER_DIR = os.path.join(DATA_BASE, "rm_member_catalogs")
PRIORS_FILE = os.path.join(DATA_BASE, "prior_locations_more_info_PURGED_111925.csv")
BCG_MATCHED_FILE = os.path.join(DATA_BASE, "bcgs_3p8arcmin_clean_matched.csv")

# Output directory (relative to this script)
OUTPUT_DIR = Path(__file__).parent
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_rm_member_catalog(cluster_name):
    """Load RedMapper member catalog for a specific cluster."""
    filename = f"{cluster_name}_rm_members.csv"
    filepath = os.path.join(RM_MEMBER_DIR, filename)

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    df['cluster'] = cluster_name
    return df


def load_all_rm_members():
    """Load all RedMapper member catalogs into a single DataFrame."""
    all_members = []

    for filename in os.listdir(RM_MEMBER_DIR):
        if filename.endswith('_rm_members.csv'):
            cluster_name = filename.replace('_rm_members.csv', '')
            filepath = os.path.join(RM_MEMBER_DIR, filename)
            df = pd.read_csv(filepath)
            df['cluster'] = cluster_name
            all_members.append(df)

    if all_members:
        return pd.concat(all_members, ignore_index=True)
    return None


def load_priors():
    """Load prior locations data."""
    if not os.path.exists(PRIORS_FILE):
        print(f"Warning: Priors file not found: {PRIORS_FILE}")
        return None
    return pd.read_csv(PRIORS_FILE)


def load_bcg_matched():
    """Load BCG matched dataset."""
    if not os.path.exists(BCG_MATCHED_FILE):
        print(f"Warning: BCG matched file not found: {BCG_MATCHED_FILE}")
        return None
    return pd.read_csv(BCG_MATCHED_FILE)


# =============================================================================
# Analysis Functions
# =============================================================================

def match_coordinates(ra1, dec1, ra2, dec2, tol_arcsec=2.0):
    """
    Match coordinates between two catalogs using KD-tree.

    Args:
        ra1, dec1: First catalog coordinates
        ra2, dec2: Second catalog coordinates
        tol_arcsec: Matching tolerance in arcseconds

    Returns:
        idx1, idx2, sep: Matched indices and separations
    """
    tol_deg = tol_arcsec / 3600.0

    # Build KD-tree on second catalog
    coords2 = np.column_stack([ra2, dec2])
    tree = cKDTree(coords2)

    # Query for matches
    coords1 = np.column_stack([ra1, dec1])
    distances, indices = tree.query(coords1, k=1)

    # Filter by tolerance
    mask = distances <= tol_deg
    idx1 = np.where(mask)[0]
    idx2 = indices[mask]
    sep = distances[mask] * 3600  # Convert to arcsec

    return idx1, idx2, sep


def analyze_cluster(cluster_name, rm_members, priors, bcg_matched, verbose=True):
    """
    Analyze a single cluster: compare RM members with priors and BCG labels.

    Returns dict with analysis results.
    """
    results = {
        'cluster': cluster_name,
        'n_rm_members': 0,
        'n_priors': 0,
        'n_matched': 0,
        'matched_pmem_values': [],
        'unmatched_pmem_values': [],
        'bcg_pmem': None,
        'bcg_rank_by_pmem': None,
    }

    # Get RM members for this cluster
    cluster_rm = rm_members[rm_members['cluster'] == cluster_name].copy()
    if len(cluster_rm) == 0:
        return results
    results['n_rm_members'] = len(cluster_rm)

    # Get priors for this cluster
    cluster_priors = priors[priors['cluster'] == cluster_name].copy()
    results['n_priors'] = len(cluster_priors)

    if len(cluster_priors) == 0:
        return results

    # Match RM members to priors
    idx_rm, idx_prior, sep = match_coordinates(
        cluster_rm['ra'].values, cluster_rm['dec'].values,
        cluster_priors['ra'].values, cluster_priors['dec'].values,
        tol_arcsec=2.0
    )

    results['n_matched'] = len(idx_rm)

    if len(idx_rm) > 0:
        results['matched_pmem_values'] = cluster_rm.iloc[idx_rm]['pmem'].values.tolist()

        # Get unmatched pmem values
        unmatched_mask = ~cluster_rm.index.isin(cluster_rm.index[idx_rm])
        results['unmatched_pmem_values'] = cluster_rm[unmatched_mask]['pmem'].values.tolist()

    # Check if BCG is in RM members
    if bcg_matched is not None:
        cluster_bcg = bcg_matched[bcg_matched['Cluster name'] == cluster_name]
        if len(cluster_bcg) > 0:
            # Get the highest probability BCG
            best_bcg = cluster_bcg.loc[cluster_bcg['BCG Probability'].idxmax()]
            bcg_ra = best_bcg['BCG RA']
            bcg_dec = best_bcg['BCG Dec']

            # Find in RM members
            idx_bcg, idx_rm_match, sep_bcg = match_coordinates(
                [bcg_ra], [bcg_dec],
                cluster_rm['ra'].values, cluster_rm['dec'].values,
                tol_arcsec=2.0
            )

            if len(idx_bcg) > 0:
                bcg_pmem = cluster_rm.iloc[idx_rm_match[0]]['pmem']
                results['bcg_pmem'] = bcg_pmem

                # Rank by pmem (1 = highest)
                sorted_rm = cluster_rm.sort_values('pmem', ascending=False)
                rank = (sorted_rm['pmem'].values > bcg_pmem).sum() + 1
                results['bcg_rank_by_pmem'] = rank

    if verbose:
        print(f"\n{cluster_name}:")
        print(f"  RM members: {results['n_rm_members']}, Priors: {results['n_priors']}, Matched: {results['n_matched']}")
        if results['bcg_pmem'] is not None:
            print(f"  BCG pmem: {results['bcg_pmem']:.4f}, Rank by pmem: {results['bcg_rank_by_pmem']}")

    return results


def generate_summary_statistics(all_results):
    """Generate overall summary statistics from analysis results."""
    df = pd.DataFrame(all_results)

    # Filter to clusters with matches
    matched_df = df[df['n_matched'] > 0]

    summary = {
        'total_clusters_analyzed': len(df),
        'clusters_with_rm_data': len(df[df['n_rm_members'] > 0]),
        'clusters_with_matches': len(matched_df),
        'total_rm_members': df['n_rm_members'].sum(),
        'total_priors': df['n_priors'].sum(),
        'total_matched': df['n_matched'].sum(),
        'match_rate': df['n_matched'].sum() / df['n_priors'].sum() * 100 if df['n_priors'].sum() > 0 else 0,
    }

    # BCG statistics
    bcg_found = df[df['bcg_pmem'].notna()]
    if len(bcg_found) > 0:
        summary['clusters_bcg_in_rm'] = len(bcg_found)
        summary['bcg_pmem_mean'] = bcg_found['bcg_pmem'].mean()
        summary['bcg_pmem_median'] = bcg_found['bcg_pmem'].median()
        summary['bcg_pmem_std'] = bcg_found['bcg_pmem'].std()

        ranks = bcg_found['bcg_rank_by_pmem'].dropna()
        if len(ranks) > 0:
            summary['bcg_rank1_pct'] = (ranks == 1).sum() / len(ranks) * 100
            summary['bcg_top3_pct'] = (ranks <= 3).sum() / len(ranks) * 100
            summary['bcg_rank_mean'] = ranks.mean()

    return summary


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_pmem_distribution(all_results, output_path):
    """Plot overall pmem distribution and matched vs unmatched comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Collect all pmem values
    all_matched = []
    all_unmatched = []
    all_bcg_pmem = []

    for r in all_results:
        all_matched.extend(r['matched_pmem_values'])
        all_unmatched.extend(r['unmatched_pmem_values'])
        if r['bcg_pmem'] is not None:
            all_bcg_pmem.append(r['bcg_pmem'])

    # Plot 1: Overall pmem distribution
    all_pmem = all_matched + all_unmatched
    if all_pmem:
        axes[0].hist(all_pmem, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('pmem')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Overall pmem Distribution\n(All RM Members)')
        axes[0].axvline(np.median(all_pmem), color='red', linestyle='--', label=f'Median: {np.median(all_pmem):.3f}')
        axes[0].legend()

    # Plot 2: Matched vs Unmatched
    if all_matched and all_unmatched:
        axes[1].hist(all_matched, bins=30, alpha=0.7, label=f'Matched to priors (N={len(all_matched)})', edgecolor='black')
        axes[1].hist(all_unmatched, bins=30, alpha=0.5, label=f'Not in priors (N={len(all_unmatched)})', edgecolor='black')
        axes[1].set_xlabel('pmem')
        axes[1].set_ylabel('Count')
        axes[1].set_title('pmem: Matched vs Unmatched to Priors')
        axes[1].legend()

    # Plot 3: BCG pmem distribution
    if all_bcg_pmem:
        axes[2].hist(all_bcg_pmem, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[2].set_xlabel('pmem')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'BCG pmem Distribution\n(N={len(all_bcg_pmem)}, Median: {np.median(all_bcg_pmem):.3f})')
        axes[2].axvline(np.median(all_bcg_pmem), color='red', linestyle='--', label=f'Median: {np.median(all_bcg_pmem):.3f}')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bcg_rank_by_pmem(all_results, output_path):
    """Plot BCG ranking when sorted by pmem."""
    ranks = [r['bcg_rank_by_pmem'] for r in all_results if r['bcg_rank_by_pmem'] is not None]

    if not ranks:
        print("No BCG rank data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of ranks
    max_rank = max(ranks)
    bins = np.arange(0.5, min(max_rank + 1.5, 21.5), 1)
    axes[0].hist(ranks, bins=bins, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('BCG Rank (by pmem)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'BCG Rank Distribution\n(When RM members sorted by pmem)')
    axes[0].set_xticks(range(1, min(max_rank + 1, 21)))

    # Add statistics
    rank1_pct = sum(1 for r in ranks if r == 1) / len(ranks) * 100
    top3_pct = sum(1 for r in ranks if r <= 3) / len(ranks) * 100
    axes[0].text(0.95, 0.95, f'Rank 1: {rank1_pct:.1f}%\nTop 3: {top3_pct:.1f}%',
                 transform=axes[0].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Cumulative distribution
    sorted_ranks = np.sort(ranks)
    cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100
    axes[1].step(sorted_ranks, cumulative, where='post')
    axes[1].set_xlabel('BCG Rank (by pmem)')
    axes[1].set_ylabel('Cumulative %')
    axes[1].set_title('Cumulative Distribution of BCG Rank')
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(90, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlim(0, 20)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def read_wcs_from_tif(filepath):
    """Read WCS information from TIF files with WCS information embedded in header."""
    from astropy.wcs import WCS
    from astropy.io import fits
    from PIL import Image as pillow
    from PIL.TiffTags import TAGS
    from io import StringIO

    pil_image = pillow.open(filepath)
    pil_image.seek(0)

    meta_dict = {TAGS[key]: pil_image.tag[key] for key in pil_image.tag_v2}

    long_header_str = meta_dict['ImageDescription'][0]
    line_length = 80

    # Splitting the string into lines of 80 characters
    lines = [long_header_str[i:i+line_length] for i in range(0, len(long_header_str), line_length)]

    # Join the lines with newline characters to form a properly formatted header string
    corrected_header_str = "\n".join(lines)

    # Create a StringIO object from the corrected header string
    header_stream = StringIO(corrected_header_str)

    # Read the header using astropy.io.fits
    header = fits.Header.fromtextfile(header_stream)
    wcs = WCS(header)

    pil_image.close()
    return wcs


def find_cluster_image(cluster_name, image_dir):
    """Find the TIFF image file for a given cluster."""
    import glob

    # Look for image files matching the cluster name
    pattern = os.path.join(image_dir, f"{cluster_name}_*_grz.tif")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]
    return None


def plot_cluster_comparison(cluster_name, rm_members, priors, bcg_matched, output_path):
    """
    Create detailed comparison plot for a single cluster.
    Shows spatial distribution, pmem values, and highlights BCG.
    Uses pre-computed pixel coordinates from bcg_matched and proper WCS for RM members.
    """
    cluster_rm = rm_members[rm_members['cluster'] == cluster_name].copy()
    cluster_priors = priors[priors['cluster'] == cluster_name].copy()

    if len(cluster_rm) == 0 or len(cluster_priors) == 0:
        print(f"Insufficient data for cluster {cluster_name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get BCG info (all candidates for this cluster) - USE PRE-COMPUTED PIXEL COORDS
    all_bcg_candidates = []
    if bcg_matched is not None:
        cluster_bcg = bcg_matched[bcg_matched['Cluster name'] == cluster_name]
        for _, row in cluster_bcg.iterrows():
            all_bcg_candidates.append({
                'ra': row['BCG RA'],
                'dec': row['BCG Dec'],
                'prob': row['BCG Probability'],
                'delta_mstar_z': row['delta_mstar_z'],
                'x': row['x'],  # Pre-computed pixel x
                'y': row['y'],  # Pre-computed pixel y
            })
        # Sort by probability
        all_bcg_candidates.sort(key=lambda x: x['prob'], reverse=True)

    # Try to load actual image
    image_loaded = False
    wcs = None
    image_array = None
    img_height = 512  # Default

    # Detect environment for image directory
    LOCAL_IMAGE_DIR = "/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/bcgs/3p8arcmin"
    CLUSTER_IMAGE_DIR = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin"

    if os.path.exists(LOCAL_IMAGE_DIR):
        image_dir = LOCAL_IMAGE_DIR
    else:
        image_dir = CLUSTER_IMAGE_DIR

    image_path = find_cluster_image(cluster_name, image_dir)

    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image as pillow

            # Load image
            pil_image = pillow.open(image_path)
            image_array = np.array(pil_image)
            img_height = image_array.shape[0]

            # Load WCS
            wcs = read_wcs_from_tif(image_path)
            image_loaded = True

            pil_image.close()
        except Exception as e:
            print(f"Warning: Could not load image for {cluster_name}: {e}")
            image_loaded = False

    # =========================================================================
    # Plot 1: Image overlay with candidates
    # =========================================================================
    ax = axes[0]

    if image_loaded and wcs is not None:
        # Display actual image
        ax.imshow(image_array)

        # Convert RM members to pixel coordinates using WCS
        rm_x, rm_y = wcs.world_to_pixel_values(cluster_rm['ra'].values, cluster_rm['dec'].values)
        rm_y_corrected = img_height - rm_y  # Y-coordinate inversion

        # Plot RM members as circles with transparent fill, edge colored by pmem
        # Create custom colormap for edges
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Plot each member individually with edge color based on pmem
        for i in range(len(rm_x)):
            pmem_val = cluster_rm['pmem'].values[i]
            edge_color = cmap(norm(pmem_val))
            ax.scatter(rm_x[i], rm_y_corrected[i], marker='o', s=200,
                      facecolors='none', edgecolors=[edge_color],
                      linewidths=1.5, alpha=0.6, zorder=3)

        # Convert priors/candidates to pixel coordinates
        prior_x, prior_y = wcs.world_to_pixel_values(
            cluster_priors['ra'].values, cluster_priors['dec'].values)
        prior_y_corrected = img_height - prior_y

        # Match priors to RM members to get pmem values
        idx_prior_matched, idx_rm_matched, _ = match_coordinates(
            cluster_priors['ra'].values, cluster_priors['dec'].values,
            cluster_rm['ra'].values, cluster_rm['dec'].values,
            tol_arcsec=2.0
        )

        # Create pmem lookup for priors
        prior_pmem = {}
        for p_idx, rm_idx in zip(idx_prior_matched, idx_rm_matched):
            prior_pmem[p_idx] = cluster_rm.iloc[rm_idx]['pmem']

        # Plot candidates as gray squares
        for i in range(len(prior_x)):
            ax.scatter(prior_x[i], prior_y_corrected[i], marker='s', s=150,
                      facecolors='none', edgecolors='#808080', linewidths=1.5,
                      alpha=0.9, zorder=4)

        # Legend elements
        legend_elements = []

        # Add legend entries for Members (circles) and Candidates (squares)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markeredgecolor='#30a030', markersize=20,
                                        markeredgewidth=1.5, linestyle='None',
                                        markerfacecolor='None', label='Members'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                        markeredgecolor='#808080', markersize=10,
                                        markeredgewidth=1.5, linestyle='None',
                                        markerfacecolor='None', label='Candidates'))

        # Plot candidates using PRE-COMPUTED pixel coordinates
        # Find ALL top candidates (those with max p_RM probability)
        if len(all_bcg_candidates) > 0:
            max_prob = all_bcg_candidates[0]['prob']  # Already sorted by prob
            top_candidates = [c for c in all_bcg_candidates if abs(c['prob'] - max_prob) < 0.01]

            # Also find top member (highest pmem among candidates)
            top_member_pmem = None
            top_member_idx = None
            for i, bcg in enumerate(all_bcg_candidates):
                idx_b, idx_r, _ = match_coordinates(
                    [bcg['ra']], [bcg['dec']],
                    cluster_rm['ra'].values, cluster_rm['dec'].values,
                    tol_arcsec=2.0
                )
                if len(idx_r) > 0:
                    pmem = cluster_rm.iloc[idx_r[0]]['pmem']
                    if top_member_pmem is None or pmem > top_member_pmem:
                        top_member_pmem = pmem
                        top_member_idx = i

            # Plot ALL top candidates (red squares)
            for i, bcg in enumerate(top_candidates):
                bcg_x = bcg['x']
                bcg_y = bcg['y']
                ax.scatter(bcg_x, bcg_y, marker='s', s=800,
                          facecolors='none', edgecolors='#FF0000', linewidths=3,
                          zorder=10)
                # Add probability label
                ax.text(bcg_x + 12, bcg_y - 5, f'{bcg["prob"]:.2f}',
                       fontsize=10, color='black',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            # Add legend for top candidates
            n_top = len(top_candidates)
            label_text = f'Top candidate{"s" if n_top > 1 else ""} ($p_{{\\mathrm{{RM}}}}$: {max_prob:.2f})'
            if n_top > 1:
                label_text = f'Top {n_top} candidates ($p_{{\\mathrm{{RM}}}}$: {max_prob:.2f})'
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                            markeredgecolor='#FF0000', markersize=12,
                                            markeredgewidth=3, linestyle='None',
                                            markerfacecolor='None', label=label_text))

            # Plot top member (cyan circle) - the one with highest pmem
            if top_member_idx is not None and top_member_pmem is not None:
                top_member = all_bcg_candidates[top_member_idx]
                ax.scatter(top_member['x'], top_member['y'], marker='o', s=900,
                          facecolors='none', edgecolors='#59F5ED', linewidths=3,
                          zorder=9)
                label_text = f'Top member ($p_{{\\mathrm{{mem}}}}$: {top_member_pmem:.2f})'
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markeredgecolor='#59F5ED', markersize=12,
                                                markeredgewidth=3, linestyle='None',
                                                markerfacecolor='None', label=label_text))

        # Add colorbar for pmem
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('$p_{\\mathrm{mem}}$', fontsize=10)

        # Add cluster name in corner
        ax.text(0.02, 0.98, f'{cluster_name}', transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xlim(0, image_array.shape[1])
        ax.set_ylim(image_array.shape[0], 0)  # Inverted y-axis
        ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.8)
        ax.axis('off')

    else:
        # Fallback to RA/Dec plot if image not available
        scatter = ax.scatter(cluster_rm['ra'], cluster_rm['dec'],
                            c=cluster_rm['pmem'], cmap='viridis',
                            s=100, alpha=0.5, edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax, label='pmem')

        ax.scatter(cluster_priors['ra'], cluster_priors['dec'],
                  marker='s', s=80, facecolors='none', edgecolors='gray',
                  label='Priors', linewidths=1.5)

        for bcg in all_bcg_candidates[:2]:
            ax.scatter(bcg['ra'], bcg['dec'], marker='o', s=300,
                      facecolors='none', edgecolors='red', linewidths=2,
                      label=f'BCG (P={bcg["prob"]:.2f})', zorder=10)

        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title(f'{cluster_name}\nSpatial Distribution')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')

    # =========================================================================
    # Plot 2: p_mem histogram with top candidates marked
    # =========================================================================
    ax = axes[1]
    ax.hist(cluster_rm['pmem'], bins=20, alpha=0.7, edgecolor='black', color='steelblue')

    # Mark top candidates' pmem values
    colors_hist = ['red', 'orange', 'green', 'purple']
    for bcg_idx, bcg in enumerate(all_bcg_candidates[:4]):
        idx_bcg, idx_rm_match, _ = match_coordinates(
            [bcg['ra']], [bcg['dec']],
            cluster_rm['ra'].values, cluster_rm['dec'].values,
            tol_arcsec=2.0
        )
        if len(idx_bcg) > 0:
            bcg_pmem = cluster_rm.iloc[idx_rm_match[0]]['pmem']
            ax.axvline(bcg_pmem, color=colors_hist[bcg_idx % len(colors_hist)],
                      linestyle='--', linewidth=2,
                      label=f'Cand{bcg_idx+1} $p_{{mem}}$: {bcg_pmem:.2f}')

    ax.set_xlabel('$p_{\\mathrm{mem}}$')
    ax.set_ylabel('Count')
    ax.set_title(f'$p_{{mem}}$ Distribution\n(N={len(cluster_rm)} members)')
    ax.legend(fontsize=8)

    # =========================================================================
    # Plot 3: p_RM vs p_mem for candidates that have BOTH probabilities
    # =========================================================================
    ax = axes[2]

    # Only show candidates that have both p_RM (from bcg_matched) and p_mem (from rm_members)
    matched_prm = []  # p_RM values (from bcg_matched)
    matched_pmem = []  # p_mem values (from rm_members)

    # Add BCG candidates that match to RM members
    for bcg in all_bcg_candidates:
        idx_b, idx_r, _ = match_coordinates(
            [bcg['ra']], [bcg['dec']],
            cluster_rm['ra'].values, cluster_rm['dec'].values,
            tol_arcsec=2.0
        )
        if len(idx_r) > 0:
            matched_prm.append(bcg['prob'])
            matched_pmem.append(cluster_rm.iloc[idx_r[0]]['pmem'])

    # Count priors matched to members but NOT in bcg_candidates (for info only)
    n_other_matched = 0
    for i in range(len(cluster_priors)):
        if i in prior_pmem:
            prior_ra = cluster_priors.iloc[i]['ra']
            prior_dec = cluster_priors.iloc[i]['dec']
            is_bcg_candidate = False
            for bcg in all_bcg_candidates:
                if abs(bcg['ra'] - prior_ra) < 0.001 and abs(bcg['dec'] - prior_dec) < 0.001:
                    is_bcg_candidate = True
                    break
            if not is_bcg_candidate:
                n_other_matched += 1

    if matched_prm:
        # Plot candidates with both p_RM and p_mem
        ax.scatter(matched_pmem, matched_prm, s=150, c='red', edgecolors='black',
                  linewidths=2, zorder=5, label=f'Candidates ({len(matched_prm)})')

        # Add labels for each candidate
        for i, (pm, pr) in enumerate(zip(matched_pmem, matched_prm)):
            ax.annotate(f'{i+1}', (pm, pr), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, fontweight='bold')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='1:1 line')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('$p_{\\mathrm{mem}}$ (Members)')
        ax.set_ylabel('$p_{\\mathrm{RM}}$ (Candidates)')

        # Show info about other matched priors (without p_RM)
        title = f'$p_{{RM}}$ vs $p_{{mem}}$\n({len(matched_prm)} candidates'
        if n_other_matched > 0:
            title += f', {n_other_matched} other priors matched'
        title += ')'
        ax.set_title(title)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, f'No candidates with\nboth $p_{{RM}}$ and $p_{{mem}}$\n({n_other_matched} priors matched)',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('$p_{RM}$ vs $p_{mem}$')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('$p_{\\mathrm{mem}}$ (Members)')
        ax.set_ylabel('$p_{\\mathrm{RM}}$ (Candidates)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# p_RM vs p_mem Analysis (Candidates vs Members)
# =============================================================================

def analyze_prm_vs_pmem(rm_members, bcg_matched, tol_arcsec=2.0):
    """
    Compare p_RM (candidate probability) with p_mem (member probability).

    Returns DataFrame with matched candidates containing both probabilities.
    """
    if bcg_matched is None or rm_members is None:
        return None

    matched_data = []

    for cluster_name in bcg_matched['Cluster name'].unique():
        cluster_bcgs = bcg_matched[bcg_matched['Cluster name'] == cluster_name]
        cluster_rm = rm_members[rm_members['cluster'] == cluster_name]

        if len(cluster_rm) == 0:
            continue

        for _, bcg_row in cluster_bcgs.iterrows():
            bcg_ra = bcg_row['BCG RA']
            bcg_dec = bcg_row['BCG Dec']
            p_rm = bcg_row['BCG Probability']  # p_RM = candidate probability
            delta_mstar_z = bcg_row['delta_mstar_z']
            cluster_z = bcg_row['Cluster z']

            # Find matching RM member
            idx_bcg, idx_rm, sep = match_coordinates(
                [bcg_ra], [bcg_dec],
                cluster_rm['ra'].values, cluster_rm['dec'].values,
                tol_arcsec=tol_arcsec
            )

            if len(idx_bcg) > 0:
                pmem = cluster_rm.iloc[idx_rm[0]]['pmem']
                separation = sep[0]

                matched_data.append({
                    'cluster': cluster_name,
                    'ra': bcg_ra,
                    'dec': bcg_dec,
                    'p_rm': p_rm,  # Candidate probability
                    'p_mem': pmem,  # Member probability
                    'delta_mstar_z': delta_mstar_z,
                    'cluster_z': cluster_z,
                    'separation_arcsec': separation,
                })

    if matched_data:
        return pd.DataFrame(matched_data)
    return None


def plot_prm_vs_pmem(matched_df, output_dir):
    """
    Create comprehensive plots comparing p_RM (candidates) with p_mem (members).
    """
    if matched_df is None or len(matched_df) == 0:
        print("No matched data for p_RM vs p_mem comparison")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # =========================================================================
    # Plot 1: Scatter plot of p_RM vs p_mem
    # =========================================================================
    ax = axes[0, 0]
    scatter = ax.scatter(matched_df['p_mem'], matched_df['p_rm'],
                         c=matched_df['cluster_z'], cmap='viridis',
                         alpha=0.5, s=20, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Cluster Redshift')

    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='1:1 line')

    # Correlation
    corr = np.corrcoef(matched_df['p_mem'], matched_df['p_rm'])[0, 1]
    ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\nN = {len(matched_df)}',
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('$p_{\\mathrm{mem}}$ (Members)')
    ax.set_ylabel('$p_{\\mathrm{RM}}$ (Candidates)')
    ax.set_title('$p_{RM}$ vs $p_{mem}$\n(colored by redshift)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    # =========================================================================
    # Plot 2: 2D Histogram / Density plot
    # =========================================================================
    ax = axes[0, 1]
    h = ax.hist2d(matched_df['p_mem'], matched_df['p_rm'],
                  bins=30, cmap='Blues', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='1:1 line')
    ax.set_xlabel('$p_{\\mathrm{mem}}$ (Members)')
    ax.set_ylabel('$p_{\\mathrm{RM}}$ (Candidates)')
    ax.set_title('Density: $p_{RM}$ vs $p_{mem}$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')

    # =========================================================================
    # Plot 3: Distribution comparison (histograms)
    # =========================================================================
    ax = axes[0, 2]
    bins = np.linspace(0, 1, 31)
    ax.hist(matched_df['p_mem'], bins=bins, alpha=0.6, label='$p_{mem}$', edgecolor='black')
    ax.hist(matched_df['p_rm'], bins=bins, alpha=0.6, label='$p_{RM}$', edgecolor='black')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution Comparison')
    ax.legend()

    # Add statistics
    pmem_median = matched_df['p_mem'].median()
    prm_median = matched_df['p_rm'].median()
    ax.axvline(pmem_median, color='C0', linestyle='--', alpha=0.7)
    ax.axvline(prm_median, color='C1', linestyle='--', alpha=0.7)
    ax.text(0.95, 0.95, f'$p_{{mem}}$ median: {pmem_median:.3f}\n$p_{{RM}}$ median: {prm_median:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # =========================================================================
    # Plot 4: Agreement analysis - where do they agree/disagree?
    # =========================================================================
    ax = axes[1, 0]

    # Define agreement categories
    high_both = (matched_df['p_mem'] > 0.8) & (matched_df['p_rm'] > 0.8)
    low_both = (matched_df['p_mem'] < 0.5) & (matched_df['p_rm'] < 0.5)
    high_pmem_low_prm = (matched_df['p_mem'] > 0.8) & (matched_df['p_rm'] < 0.5)
    low_pmem_high_prm = (matched_df['p_mem'] < 0.5) & (matched_df['p_rm'] > 0.8)

    categories = ['High Both\n($p_{mem}$>0.8, $p_{RM}$>0.8)',
                  'Low Both\n($p_{mem}$<0.5, $p_{RM}$<0.5)',
                  'High $p_{mem}$, Low $p_{RM}$',
                  'Low $p_{mem}$, High $p_{RM}$']
    counts = [high_both.sum(), low_both.sum(), high_pmem_low_prm.sum(), low_pmem_high_prm.sum()]
    colors = ['green', 'gray', 'orange', 'red']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Agreement/Disagreement Analysis')
    ax.tick_params(axis='x', rotation=15)

    # Add percentage labels
    total = len(matched_df)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # Plot 5: Residuals (p_RM - p_mem) vs redshift
    # =========================================================================
    ax = axes[1, 1]
    residuals = matched_df['p_rm'] - matched_df['p_mem']
    scatter = ax.scatter(matched_df['cluster_z'], residuals,
                         c=matched_df['p_mem'], cmap='viridis',
                         alpha=0.5, s=20, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='$p_{mem}$')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Cluster Redshift')
    ax.set_ylabel('$p_{RM}$ - $p_{mem}$')
    ax.set_title('Residuals vs Redshift')

    # Add trend line
    z_vals = matched_df['cluster_z'].values
    mask = ~np.isnan(residuals) & ~np.isnan(z_vals)
    if mask.sum() > 10:
        z = np.polyfit(z_vals[mask], residuals[mask], 1)
        p = np.poly1d(z)
        z_range = np.linspace(z_vals.min(), z_vals.max(), 100)
        ax.plot(z_range, p(z_range), 'r-', alpha=0.7, label=f'Trend: slope={z[0]:.3f}')
        ax.legend()

    # =========================================================================
    # Plot 6: Residuals vs delta_mstar_z
    # =========================================================================
    ax = axes[1, 2]
    scatter = ax.scatter(matched_df['delta_mstar_z'], residuals,
                         c=matched_df['cluster_z'], cmap='viridis',
                         alpha=0.5, s=20, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Cluster Redshift')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('delta_mstar_z')
    ax.set_ylabel('$p_{RM}$ - $p_{mem}$')
    ax.set_title('Residuals vs delta_mstar_z')

    plt.tight_layout()
    output_path = output_dir / "prm_vs_pmem.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return matched_df


def print_prm_vs_pmem_summary(matched_df):
    """Print detailed summary statistics for p_RM vs p_mem comparison."""
    if matched_df is None or len(matched_df) == 0:
        return

    print("\n" + "=" * 70)
    print("p_RM (CANDIDATES) vs p_mem (MEMBERS) COMPARISON")
    print("=" * 70)

    print(f"\nTotal candidates matched to members: {len(matched_df)}")
    print(f"Unique clusters: {matched_df['cluster'].nunique()}")

    print("\n--- Basic Statistics ---")
    print(f"  p_RM (Candidates): mean={matched_df['p_rm'].mean():.3f}, "
          f"median={matched_df['p_rm'].median():.3f}, "
          f"std={matched_df['p_rm'].std():.3f}")
    print(f"  p_mem (Members):   mean={matched_df['p_mem'].mean():.3f}, "
          f"median={matched_df['p_mem'].median():.3f}, "
          f"std={matched_df['p_mem'].std():.3f}")

    print("\n--- Correlation ---")
    corr = np.corrcoef(matched_df['p_mem'], matched_df['p_rm'])[0, 1]
    print(f"  Pearson correlation: {corr:.4f}")

    # Spearman correlation
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(matched_df['p_mem'], matched_df['p_rm'])
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")

    print("\n--- Agreement Analysis ---")
    high_both = (matched_df['p_mem'] > 0.8) & (matched_df['p_rm'] > 0.8)
    low_both = (matched_df['p_mem'] < 0.5) & (matched_df['p_rm'] < 0.5)
    high_pmem_low_prm = (matched_df['p_mem'] > 0.8) & (matched_df['p_rm'] < 0.5)
    low_pmem_high_prm = (matched_df['p_mem'] < 0.5) & (matched_df['p_rm'] > 0.8)

    total = len(matched_df)
    print(f"  High agreement (both > 0.8): {high_both.sum()} ({high_both.sum()/total*100:.1f}%)")
    print(f"  Low agreement (both < 0.5):  {low_both.sum()} ({low_both.sum()/total*100:.1f}%)")
    print(f"  High p_mem, Low p_RM:        {high_pmem_low_prm.sum()} ({high_pmem_low_prm.sum()/total*100:.1f}%)")
    print(f"  Low p_mem, High p_RM:        {low_pmem_high_prm.sum()} ({low_pmem_high_prm.sum()/total*100:.1f}%)")

    print("\n--- Threshold Analysis ---")
    for thresh in [0.5, 0.7, 0.9, 0.95, 0.99]:
        pmem_above = (matched_df['p_mem'] >= thresh).sum()
        prm_above = (matched_df['p_rm'] >= thresh).sum()
        both_above = ((matched_df['p_mem'] >= thresh) & (matched_df['p_rm'] >= thresh)).sum()
        print(f"  >= {thresh}: p_mem={pmem_above} ({pmem_above/total*100:.1f}%), "
              f"p_RM={prm_above} ({prm_above/total*100:.1f}%), "
              f"both={both_above} ({both_above/total*100:.1f}%)")

    # Cases where they strongly disagree
    print("\n--- Strong Disagreements (|diff| > 0.5) ---")
    disagreements = matched_df[np.abs(matched_df['p_rm'] - matched_df['p_mem']) > 0.5]
    print(f"  Total: {len(disagreements)} ({len(disagreements)/total*100:.1f}%)")
    if len(disagreements) > 0:
        print(f"  Mean redshift of disagreements: {disagreements['cluster_z'].mean():.3f}")
        print(f"  Mean delta_mstar_z of disagreements: {disagreements['delta_mstar_z'].mean():.3f}")


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    """Run the full pmem analysis."""
    print("=" * 70)
    print("PMEM Analysis: RedMapper Membership Probabilities for BCG Classification")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")
    rm_members = load_all_rm_members()
    priors = load_priors()
    bcg_matched = load_bcg_matched()

    if rm_members is None:
        print("ERROR: Could not load RM member catalogs")
        return

    print(f"Loaded {len(rm_members)} RM member entries")
    print(f"Number of clusters with RM data: {rm_members['cluster'].nunique()}")

    if priors is not None:
        print(f"Loaded {len(priors)} prior entries from {priors['cluster'].nunique()} clusters")

    if bcg_matched is not None:
        print(f"Loaded {len(bcg_matched)} BCG matched entries from {bcg_matched['Cluster name'].nunique()} clusters")

    # Get common clusters
    rm_clusters = set(rm_members['cluster'].unique())
    prior_clusters = set(priors['cluster'].unique()) if priors is not None else set()
    common_clusters = rm_clusters & prior_clusters
    print(f"\nClusters in common: {len(common_clusters)}")

    # Analyze each cluster
    print("\n--- Analyzing Clusters ---")
    all_results = []

    for cluster in sorted(common_clusters):
        result = analyze_cluster(cluster, rm_members, priors, bcg_matched, verbose=False)
        all_results.append(result)

    # Generate summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = generate_summary_statistics(all_results)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR / "cluster_analysis_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved cluster results to: {results_file}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_pmem_distribution(all_results, PLOTS_DIR / "pmem_distribution.png")
    plot_bcg_rank_by_pmem(all_results, PLOTS_DIR / "bcg_rank_by_pmem.png")

    # Detailed plots for example clusters: 3 good, 3 bad, 6 random
    import random
    random.seed(42)  # For reproducibility

    # Good clusters: high match rate, BCG found with high pmem (rank 1)
    good_clusters = [r['cluster'] for r in all_results
                     if r['n_matched'] > 5
                     and r['bcg_pmem'] is not None
                     and r['bcg_pmem'] > 0.9
                     and r['bcg_rank_by_pmem'] == 1]

    # Bad clusters: low match rate OR BCG not found OR low pmem OR high rank
    bad_clusters = [r['cluster'] for r in all_results
                    if (r['n_matched'] <= 3
                        or r['bcg_pmem'] is None
                        or (r['bcg_pmem'] is not None and r['bcg_pmem'] < 0.5)
                        or (r['bcg_rank_by_pmem'] is not None and r['bcg_rank_by_pmem'] > 3))]

    # Get remaining clusters for random selection
    selected_good = good_clusters[:3] if len(good_clusters) >= 3 else good_clusters
    selected_bad = bad_clusters[:3] if len(bad_clusters) >= 3 else bad_clusters

    already_selected = set(selected_good + selected_bad)
    remaining_clusters = [r['cluster'] for r in all_results
                         if r['cluster'] not in already_selected
                         and r['n_matched'] > 0
                         and r['bcg_pmem'] is not None]

    # Select 6 random clusters (or fewer if not enough available)
    n_random = min(6, len(remaining_clusters))
    selected_random = random.sample(remaining_clusters, n_random) if remaining_clusters else []

    print(f"\n--- Detailed Plots for Example Clusters ---")
    print(f"  Good clusters (high pmem, rank 1): {selected_good}")
    print(f"  Bad clusters (low pmem/match or high rank): {selected_bad}")
    print(f"  Random clusters: {selected_random}")

    all_example_clusters = selected_good + selected_bad + selected_random

    for cluster in all_example_clusters:
        # Add prefix to indicate category
        if cluster in selected_good:
            prefix = "good"
        elif cluster in selected_bad:
            prefix = "bad"
        else:
            prefix = "random"

        output_path = PLOTS_DIR / f"cluster_detail_{prefix}_{cluster.replace('SPT-CLJ', '')}.png"
        plot_cluster_comparison(cluster, rm_members, priors, bcg_matched, output_path)

    # =========================================================================
    # p_RM vs p_mem comparison (Candidates vs Members)
    # =========================================================================
    print("\n--- p_RM vs p_mem Analysis (Candidates vs Members) ---")
    prm_pmem_df = analyze_prm_vs_pmem(rm_members, bcg_matched)

    if prm_pmem_df is not None:
        # Save the matched data
        prm_pmem_file = OUTPUT_DIR / "prm_vs_pmem.csv"
        prm_pmem_df.to_csv(prm_pmem_file, index=False)
        print(f"Saved p_RM vs p_mem data to: {prm_pmem_file}")

        # Generate plots
        plot_prm_vs_pmem(prm_pmem_df, PLOTS_DIR)

        # Print summary
        print_prm_vs_pmem_summary(prm_pmem_df)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return all_results, summary, prm_pmem_df


if __name__ == "__main__":
    all_results, summary, prm_pmem_df = main()
