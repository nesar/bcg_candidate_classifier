#!/usr/bin/env python3
"""
CCG Probability Calculator

This module computes p_{CCG} (probability of being a Cluster Central Galaxy) based on
cluster member density around top-ranked BCG candidates. This is a post-processing
calculation performed after the model predictions (\bar{p}) are computed.

The idea is that true CCGs should be located near the center of the cluster, which
should be indicated by a higher density of cluster members in their vicinity.

Algorithm:
- For each top-ranked candidate (Rank-1, Rank-2, Rank-3, etc.), count the number of
  cluster members within a specified physical radius (default 300 kpc).
- Assign p_{CCG} based on the relative member counts:
  - C1: If only 1 top candidate, p_{CCG} = 1
  - C2: Count n_{mem}_j for each top candidate j
  - C3: If one candidate has vastly higher n_{mem}, p_{CCG} = 1 for that candidate, 0 for rest
  - C4: If 2-3 candidates have similar n_{mem}, distribute probability (0.5 or 0.33)
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# ============================================================================
# Cosmology for physical distance calculations
# ============================================================================
# Standard cosmology parameters (Planck 2018)
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


# ============================================================================
# Path Configuration
# ============================================================================
def get_data_paths():
    """Get data paths based on environment (local vs cluster)."""
    LOCAL_BASE = "/Users/nesar/Projects/HEP/IMGmarker/data/bcgs/bcgs"
    CLUSTER_BASE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs"

    if os.path.exists(LOCAL_BASE):
        data_base = LOCAL_BASE
    else:
        data_base = CLUSTER_BASE

    return {
        'base': data_base,
        'rm_member_dir': os.path.join(data_base, "rm_member_catalogs"),
        'image_dir_2p2': os.path.join(data_base, "2p2arcmin"),
        'image_dir_3p8': os.path.join(data_base, "3p8arcmin"),
    }


# ============================================================================
# WCS and Coordinate Functions
# ============================================================================
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

    lines = [long_header_str[i:i+line_length] for i in range(0, len(long_header_str), line_length)]
    corrected_header_str = "\n".join(lines)
    header_stream = StringIO(corrected_header_str)
    header = fits.Header.fromtextfile(header_stream)
    wcs = WCS(header)

    pil_image.close()
    return wcs


def pixel_to_radec(x, y, wcs, img_height=512):
    """Convert pixel coordinates to RA/Dec using WCS."""
    # Correct for y-axis inversion in image coordinates
    y_corrected = img_height - y
    ra, dec = wcs.all_pix2world(x, y_corrected, 0)
    return ra, dec


def radec_to_pixel(ra, dec, wcs, img_height=512):
    """Convert RA/Dec to pixel coordinates using WCS."""
    x, y = wcs.all_world2pix(ra, dec, 0)
    y_corrected = img_height - y  # Correct for y-axis inversion
    return x, y_corrected


def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    """Calculate angular separation in arcseconds between two coordinates."""
    # Convert to radians
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    # Haversine formula
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad

    a = np.sin(delta_dec/2)**2 + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(delta_ra/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Convert to arcseconds
    sep_arcsec = np.degrees(c) * 3600
    return sep_arcsec


def angular_to_physical_kpc(sep_arcsec, redshift):
    """Convert angular separation (arcsec) to physical distance (kpc) at given redshift."""
    if redshift <= 0 or np.isnan(redshift):
        return np.nan

    # Get angular diameter distance at this redshift
    d_A = COSMO.angular_diameter_distance(redshift)

    # Convert arcsec to radians
    sep_rad = sep_arcsec * (1 * u.arcsec).to(u.rad).value

    # Physical distance in kpc
    physical_kpc = (sep_rad * d_A.to(u.kpc).value)

    return physical_kpc


def physical_to_angular_arcsec(physical_kpc, redshift):
    """Convert physical distance (kpc) to angular separation (arcsec) at given redshift."""
    if redshift <= 0 or np.isnan(redshift):
        return np.nan

    # Get angular diameter distance at this redshift
    d_A = COSMO.angular_diameter_distance(redshift).to(u.kpc).value

    # Angular separation in radians
    sep_rad = physical_kpc / d_A

    # Convert to arcsec
    sep_arcsec = sep_rad * (1 * u.rad).to(u.arcsec).value

    return sep_arcsec


# ============================================================================
# Member Loading Functions
# ============================================================================
def load_rm_member_catalog(cluster_name, rm_member_dir=None):
    """Load RedMapper member catalog for a specific cluster."""
    if rm_member_dir is None:
        rm_member_dir = get_data_paths()['rm_member_dir']

    filename = f"{cluster_name}_rm_members.csv"
    filepath = os.path.join(rm_member_dir, filename)

    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    df['cluster'] = cluster_name
    return df


def find_cluster_image(cluster_name, image_dir):
    """Find the TIFF image file for a given cluster."""
    import glob

    pattern = os.path.join(image_dir, f"{cluster_name}_*_grz.tif")
    matches = glob.glob(pattern)

    if matches:
        return matches[0]
    return None


# ============================================================================
# Member Counting Functions
# ============================================================================
def count_members_around_candidate(candidate_ra, candidate_dec, members_df, radius_kpc, redshift):
    """
    Count cluster members within a physical radius of a candidate.

    Args:
        candidate_ra: RA of candidate (degrees)
        candidate_dec: Dec of candidate (degrees)
        members_df: DataFrame with 'ra', 'dec', 'pmem' columns
        radius_kpc: Search radius in kpc
        redshift: Cluster redshift for distance conversion

    Returns:
        n_members: Number of members within radius
        member_indices: Indices of members within radius
        weighted_count: Sum of pmem values for members within radius
    """
    if members_df is None or len(members_df) == 0:
        return 0, [], 0.0

    if redshift <= 0 or np.isnan(redshift):
        return 0, [], 0.0

    # Convert radius to angular separation
    radius_arcsec = physical_to_angular_arcsec(radius_kpc, redshift)

    if np.isnan(radius_arcsec):
        return 0, [], 0.0

    # Calculate angular separations to all members
    member_ras = members_df['ra'].values
    member_decs = members_df['dec'].values

    separations = angular_separation_arcsec(
        candidate_ra, candidate_dec,
        member_ras, member_decs
    )

    # Find members within radius
    within_radius = separations <= radius_arcsec
    n_members = np.sum(within_radius)
    member_indices = np.where(within_radius)[0]

    # Weighted count by pmem
    if 'pmem' in members_df.columns:
        weighted_count = members_df.iloc[member_indices]['pmem'].sum()
    else:
        weighted_count = float(n_members)

    return n_members, member_indices, weighted_count


def count_members_around_candidates_batch(candidates_radec, members_df, radius_kpc, redshift):
    """
    Count members around multiple candidates efficiently.

    Args:
        candidates_radec: Array of shape (N, 2) with [ra, dec] for each candidate
        members_df: DataFrame with cluster members
        radius_kpc: Search radius in kpc
        redshift: Cluster redshift

    Returns:
        member_counts: Array of member counts for each candidate
        weighted_counts: Array of pmem-weighted counts for each candidate
    """
    n_candidates = len(candidates_radec)
    member_counts = np.zeros(n_candidates)
    weighted_counts = np.zeros(n_candidates)

    for i, (ra, dec) in enumerate(candidates_radec):
        n_mem, _, weighted = count_members_around_candidate(
            ra, dec, members_df, radius_kpc, redshift
        )
        member_counts[i] = n_mem
        weighted_counts[i] = weighted

    return member_counts, weighted_counts


# ============================================================================
# p_{CCG} Assignment Functions
# ============================================================================
def assign_p_ccg(member_counts, relative_threshold=2.0):
    """
    Assign p_{CCG} values based on member counts.

    Rules:
    - C1: If only 1 candidate, p_{CCG} = 1
    - C2: Count n_{mem}_j for each candidate
    - C3: If one candidate has vastly higher n_{mem} (> relative_threshold times others),
          p_{CCG} = 1 for that candidate, 0 for rest
    - C4: If 2+ candidates have similar n_{mem}, distribute probability equally

    Args:
        member_counts: Array of member counts for each top candidate
        relative_threshold: Threshold for "vastly higher" determination

    Returns:
        p_ccg: Array of p_{CCG} values for each candidate
    """
    n_candidates = len(member_counts)

    if n_candidates == 0:
        return np.array([])

    if n_candidates == 1:
        # C1: Only 1 candidate
        return np.array([1.0])

    # Sort member counts to identify dominant candidates
    sorted_indices = np.argsort(member_counts)[::-1]  # Descending order
    sorted_counts = member_counts[sorted_indices]

    p_ccg = np.zeros(n_candidates)

    # Check if highest count is vastly higher than others
    max_count = sorted_counts[0]
    second_max = sorted_counts[1] if n_candidates > 1 else 0

    if max_count == 0:
        # No members found around any candidate - equal probability
        p_ccg[:] = 1.0 / n_candidates
        return p_ccg

    # C3: Check if dominant candidate exists
    if second_max == 0 or (max_count / max(second_max, 1e-10) >= relative_threshold):
        # One candidate dominates
        p_ccg[sorted_indices[0]] = 1.0
        return p_ccg

    # C4: Multiple candidates have similar counts
    # Find candidates with counts >= max_count / relative_threshold
    similar_threshold = max_count / relative_threshold
    similar_mask = member_counts >= similar_threshold
    n_similar = np.sum(similar_mask)

    if n_similar >= 2:
        # Distribute probability among similar candidates
        p_ccg[similar_mask] = 1.0 / n_similar
    else:
        # Only top candidate is significant
        p_ccg[sorted_indices[0]] = 1.0

    return p_ccg


def assign_p_ccg_weighted(member_counts, weighted_counts, relative_threshold=2.0,
                          use_weighted=True):
    """
    Assign p_{CCG} using optional pmem-weighted counts.

    Args:
        member_counts: Array of raw member counts
        weighted_counts: Array of pmem-weighted member counts
        relative_threshold: Threshold for dominance determination
        use_weighted: Whether to use pmem-weighted counts

    Returns:
        p_ccg: Array of p_{CCG} values
    """
    if use_weighted:
        return assign_p_ccg(weighted_counts, relative_threshold)
    else:
        return assign_p_ccg(member_counts, relative_threshold)


# ============================================================================
# Main CCG Probability Computation
# ============================================================================
class CCGProbabilityCalculator:
    """
    Calculator for p_{CCG} based on cluster member density.
    """

    def __init__(self, radius_kpc=300.0, relative_threshold=5.0,
                 use_weighted_counts=True, rm_member_dir=None, pmem_cutoff=0.2):
        """
        Args:
            radius_kpc: Physical radius in kpc for member counting (default 300 kpc)
            relative_threshold: Threshold for determining dominant candidate (default 5.0)
            use_weighted_counts: Use pmem-weighted member counts
            rm_member_dir: Directory containing RedMapper member catalogs
            pmem_cutoff: Minimum pmem value to consider a member (default 0.2)
        """
        self.radius_kpc = radius_kpc
        self.relative_threshold = relative_threshold
        self.use_weighted_counts = use_weighted_counts
        self.rm_member_dir = rm_member_dir or get_data_paths()['rm_member_dir']
        self.pmem_cutoff = pmem_cutoff

    def compute_for_cluster(self, cluster_name, candidates_pixel, candidate_probs,
                           wcs=None, image_path=None, redshift=None,
                           top_n_candidates=3):
        """
        Compute p_{CCG} for candidates in a single cluster.

        Args:
            cluster_name: Name of the cluster (e.g., 'SPT-CLJ0001.5-1555')
            candidates_pixel: Array of shape (N, 2) with [x, y] pixel coordinates
            candidate_probs: Array of \bar{p} probabilities for each candidate
            wcs: WCS object for coordinate transformation (optional if image_path provided)
            image_path: Path to cluster image file (used to read WCS if wcs not provided)
            redshift: Cluster redshift
            top_n_candidates: Number of top candidates to consider (default 3)

        Returns:
            dict with keys:
                'p_ccg': Array of p_{CCG} values for top candidates
                'member_counts': Array of member counts
                'weighted_counts': Array of pmem-weighted counts
                'top_indices': Indices of top candidates in original array
                'candidates_radec': RA/Dec of top candidates
                'radius_kpc': Search radius used
                'members_in_fov': Total members loaded for this cluster
        """
        result = {
            'p_ccg': np.array([]),
            'member_counts': np.array([]),
            'weighted_counts': np.array([]),
            'top_indices': np.array([]),
            'candidates_radec': np.array([]),
            'radius_kpc': self.radius_kpc,
            'members_in_fov': 0,
            'cluster_name': cluster_name,
            'redshift': redshift,
            'error': None
        }

        if len(candidates_pixel) == 0:
            result['error'] = 'no_candidates'
            return result

        # Load WCS if not provided
        if wcs is None:
            if image_path is None:
                result['error'] = 'no_wcs_or_image_path'
                return result
            try:
                wcs = read_wcs_from_tif(image_path)
            except Exception as e:
                result['error'] = f'wcs_load_failed: {str(e)}'
                return result

        # Load cluster members
        members_df = load_rm_member_catalog(cluster_name, self.rm_member_dir)
        if members_df is None or len(members_df) == 0:
            result['error'] = 'no_members_found'
            result['p_ccg'] = np.ones(min(len(candidates_pixel), top_n_candidates)) / min(len(candidates_pixel), top_n_candidates)
            result['top_indices'] = np.argsort(candidate_probs)[-top_n_candidates:][::-1]
            return result

        # Apply pmem cutoff - filter out low probability members
        if 'pmem' in members_df.columns and self.pmem_cutoff > 0:
            members_df = members_df[members_df['pmem'] >= self.pmem_cutoff].copy()
            if len(members_df) == 0:
                result['error'] = f'no_members_above_pmem_cutoff_{self.pmem_cutoff}'
                result['p_ccg'] = np.ones(min(len(candidates_pixel), top_n_candidates)) / min(len(candidates_pixel), top_n_candidates)
                result['top_indices'] = np.argsort(candidate_probs)[-top_n_candidates:][::-1]
                return result

        result['members_in_fov'] = len(members_df)
        result['pmem_cutoff'] = self.pmem_cutoff

        # Check redshift
        if redshift is None or np.isnan(redshift) or redshift <= 0:
            result['error'] = 'invalid_redshift'
            result['p_ccg'] = np.ones(min(len(candidates_pixel), top_n_candidates)) / min(len(candidates_pixel), top_n_candidates)
            result['top_indices'] = np.argsort(candidate_probs)[-top_n_candidates:][::-1]
            return result

        # Get top candidates by probability
        n_top = min(top_n_candidates, len(candidates_pixel))
        top_indices = np.argsort(candidate_probs)[-n_top:][::-1]
        top_candidates = candidates_pixel[top_indices]

        result['top_indices'] = top_indices

        # Convert pixel to RA/Dec
        candidates_radec = []
        for x, y in top_candidates:
            ra, dec = pixel_to_radec(x, y, wcs)
            candidates_radec.append([ra, dec])
        candidates_radec = np.array(candidates_radec)
        result['candidates_radec'] = candidates_radec

        # Count members around each candidate
        member_counts, weighted_counts = count_members_around_candidates_batch(
            candidates_radec, members_df, self.radius_kpc, redshift
        )

        result['member_counts'] = member_counts
        result['weighted_counts'] = weighted_counts

        # Assign p_{CCG}
        p_ccg = assign_p_ccg_weighted(
            member_counts, weighted_counts,
            self.relative_threshold, self.use_weighted_counts
        )

        result['p_ccg'] = p_ccg

        return result

    def compute_for_evaluation_results(self, evaluation_csv, image_dir, dataset_type='3p8arcmin',
                                       output_dir=None, top_n_candidates=3):
        """
        Compute p_{CCG} for all clusters in evaluation results.

        Args:
            evaluation_csv: Path to evaluation_results.csv from test.py
            image_dir: Directory containing cluster images
            dataset_type: '2p2arcmin' or '3p8arcmin'
            output_dir: Directory to save results (optional)
            top_n_candidates: Number of top candidates to consider

        Returns:
            DataFrame with p_{CCG} results for all clusters
        """
        # Load evaluation results
        eval_df = pd.read_csv(evaluation_csv)

        results_list = []

        for idx, row in eval_df.iterrows():
            cluster_name = row.get('cluster_name', 'unknown')
            redshift = row.get('z', np.nan)

            # Skip if missing critical info
            if cluster_name == 'unknown':
                continue

            # Find image path
            image_path = find_cluster_image(cluster_name, image_dir)
            if image_path is None:
                print(f"Warning: Image not found for {cluster_name}")
                continue

            # Get candidate info from evaluation results
            # Note: evaluation_results.csv contains the predicted BCG position
            # For full p_{CCG} computation, we need all candidate positions
            # This requires loading from the original test features or re-running prediction

            # For now, we'll compute using the predicted position as Rank-1
            pred_x = row.get('pred_x', np.nan)
            pred_y = row.get('pred_y', np.nan)

            if np.isnan(pred_x) or np.isnan(pred_y):
                continue

            # Single candidate case from evaluation results
            candidates_pixel = np.array([[pred_x, pred_y]])
            candidate_probs = np.array([row.get('max_probability', 1.0)])

            result = self.compute_for_cluster(
                cluster_name, candidates_pixel, candidate_probs,
                image_path=image_path, redshift=redshift,
                top_n_candidates=1
            )

            results_list.append({
                'cluster_name': cluster_name,
                'z': redshift,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'bar_p': candidate_probs[0],
                'p_ccg': result['p_ccg'][0] if len(result['p_ccg']) > 0 else np.nan,
                'n_members': result['member_counts'][0] if len(result['member_counts']) > 0 else 0,
                'weighted_members': result['weighted_counts'][0] if len(result['weighted_counts']) > 0 else 0,
                'members_in_fov': result['members_in_fov'],
                'radius_kpc': result['radius_kpc'],
                'error': result.get('error')
            })

        results_df = pd.DataFrame(results_list)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'p_ccg_results.csv')
            results_df.to_csv(output_path, index=False)
            print(f"Saved p_CCG results to: {output_path}")

        return results_df


def compute_p_ccg_from_test_data(test_output_dir, image_dir, dataset_type='3p8arcmin',
                                  radius_kpc=300.0, relative_threshold=2.0,
                                  top_n_candidates=3, rm_member_dir=None):
    """
    Main function to compute p_{CCG} from test.py output data.

    This function loads the full candidate data saved by test.py and computes
    p_{CCG} for all top candidates in each cluster.

    Args:
        test_output_dir: Directory containing test.py outputs (evaluation_results/)
        image_dir: Directory containing cluster images
        dataset_type: '2p2arcmin' or '3p8arcmin'
        radius_kpc: Physical radius for member counting (default 300 kpc)
        relative_threshold: Threshold for p_{CCG} assignment
        top_n_candidates: Number of top candidates to consider
        rm_member_dir: Directory with RedMapper member catalogs

    Returns:
        DataFrame with comprehensive p_{CCG} results
    """
    calculator = CCGProbabilityCalculator(
        radius_kpc=radius_kpc,
        relative_threshold=relative_threshold,
        use_weighted_counts=True,
        rm_member_dir=rm_member_dir
    )

    evaluation_csv = os.path.join(test_output_dir, 'evaluation_results.csv')

    if not os.path.exists(evaluation_csv):
        raise FileNotFoundError(f"Evaluation results not found: {evaluation_csv}")

    results_df = calculator.compute_for_evaluation_results(
        evaluation_csv, image_dir, dataset_type,
        output_dir=test_output_dir, top_n_candidates=top_n_candidates
    )

    return results_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Compute p_{CCG} from test results")
    parser.add_argument('--test_output_dir', type=str, required=True,
                       help='Directory containing test.py outputs')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing cluster images')
    parser.add_argument('--dataset_type', type=str, default='3p8arcmin',
                       choices=['2p2arcmin', '3p8arcmin'])
    parser.add_argument('--radius_kpc', type=float, default=300.0,
                       help='Physical radius for member counting (kpc)')
    parser.add_argument('--relative_threshold', type=float, default=2.0,
                       help='Threshold for p_{CCG} assignment')
    parser.add_argument('--top_n', type=int, default=3,
                       help='Number of top candidates to consider')

    args = parser.parse_args()

    results = compute_p_ccg_from_test_data(
        args.test_output_dir, args.image_dir, args.dataset_type,
        args.radius_kpc, args.relative_threshold, args.top_n
    )

    print(f"\nComputed p_CCG for {len(results)} clusters")
    print(results.head(10))
