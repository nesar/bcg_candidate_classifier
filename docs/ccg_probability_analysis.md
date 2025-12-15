# CCG Probability Analysis (p_{CCG})

## Overview

The CCG (Cluster Central Galaxy) probability analysis provides an independent estimate of which BCG candidate is most likely to be the true central galaxy of a cluster. While the ML-based probability `bar_p` uses visual and photometric features, `p_{CCG}` uses cluster member density to verify if a candidate is located at the center of the cluster member distribution.

## Motivation

The central galaxy of a cluster should be located near the gravitational center, which is typically where the cluster members are most concentrated. By counting the number of cluster members around each BCG candidate, we can provide an independent verification of the ML prediction.

This approach addresses cases where:
- Multiple candidates have similar `bar_p` values
- The ML model is uncertain between candidates
- We want to identify potential merger systems or off-center BCGs

## Algorithm

### Step 1: Identify Top Candidates

For each cluster image, identify the top-ranked candidates (Rank-1, Rank-2, Rank-3, etc.) based on their ML probabilities (`bar_p`).

### Step 2: Count Cluster Members

For each top candidate:
1. Convert pixel coordinates to RA/Dec using the image WCS
2. Load RedMapper cluster members from the `rm_member_catalogs/` directory
3. Convert the search radius from physical (kpc) to angular (arcsec) using:
   - Cosmology: Flat ΛCDM with H₀=70, Ω_m=0.3
   - Angular diameter distance at cluster redshift

4. Count members within the search radius:
   - `n_mem`: Raw member count
   - `weighted_count`: Sum of p_mem values (membership probability weighted)

### Step 3: Assign p_{CCG}

Apply the following rules:

| Rule | Condition | p_{CCG} Assignment |
|------|-----------|-------------------|
| C1 | Only 1 top candidate | p_{CCG} = 1.0 |
| C2 | Multiple candidates | Count n_mem for each |
| C3 | One candidate dominates (n_mem >> others) | Dominant: p_{CCG}=1.0, others: p_{CCG}=0.0 |
| C4 | 2-3 candidates with similar n_mem | Distribute equally (0.5 or 0.33) |

The "dominance" threshold is configurable (default: 2.0x). A candidate is considered dominant if its member count is ≥2x higher than the second-highest.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius_kpc` | 300.0 | Physical search radius in kpc |
| `relative_threshold` | 2.0 | Threshold for determining dominance |
| `top_n_candidates` | 3 | Number of top candidates to consider |
| `use_weighted_counts` | True | Use p_mem-weighted counts vs raw counts |

### Radius Selection

The default radius of 300 kpc is motivated by:
- Typical cluster core radii: 100-400 kpc
- BCG effective radii: 10-50 kpc
- Balance between local (near BCG) and cluster-scale membership

Users can adjust this from 200-400 kpc based on cluster properties.

## Output Files

### Directory Structure

```
trained_models/candidate_classifier_*/
└── evaluation_results/
    └── physical_images_with_members/
        ├── p_ccg_results.csv          # Main results table
        ├── pccg_diagnostic_plots.png  # 6-panel diagnostic figure
        ├── pccg_diagnostic_plots.pdf  # PDF version
        ├── pccg_vs_barp_scatter.png   # Summary scatter plot
        └── <cluster_name>_pccg.png    # Individual cluster images
```

### p_ccg_results.csv Columns

| Column | Description |
|--------|-------------|
| `cluster_name` | Cluster identifier |
| `z` | Cluster redshift |
| `pred_x`, `pred_y` | Predicted BCG pixel coordinates |
| `true_x`, `true_y` | True BCG pixel coordinates |
| `bar_p` | ML probability |
| `bcg_rank` | Rank of true BCG among candidates |
| `p_ccg` | Member-density probability |
| `n_members` | Raw member count within radius |
| `weighted_members` | p_mem-weighted member count |
| `members_in_fov` | Total members loaded for cluster |
| `radius_kpc` | Search radius used |
| `error` | Any error messages |

## Diagnostic Plots

### 6-Panel Diagnostic Figure

1. **p_{CCG} vs bar_p Scatter**: Shows correlation between ML and member-based probabilities, colored by redshift
2. **Distribution Comparison**: Histograms of both probability types
3. **Member Count Distribution**: Distribution of members found within search radius
4. **p_{CCG} vs Member Count**: Relationship between member density and assigned probability
5. **Agreement Analysis**: Bar chart showing agreement/disagreement categories
6. **Mean Probabilities by Redshift**: How both probabilities vary with redshift

### Physical Images with Members

Each cluster image shows:
- All detected cluster members (circles colored by p_mem)
- Top BCG candidates (colored by rank, labeled with probabilities)
- Search radius circles (dashed gray)
- Both `bar_p` and `p_{CCG}` values in labels
- Target BCG (if available)

## Usage

### From enhanced_full_run.py (Recommended)

The analysis is integrated into the main workflow:

```bash
python enhanced_full_run.py
```

When prompted, accept the default (Y) to run CCG analysis.

### Standalone

```bash
python run_ccg_analysis.py \
    --experiment_dir trained_models/candidate_classifier_20241201_123456 \
    --image_dir /path/to/images/3p8arcmin \
    --dataset_type 3p8arcmin \
    --radius_kpc 300.0 \
    --relative_threshold 2.0 \
    --n_images 20
```

### Python API

```python
from run_ccg_analysis import CCGAnalysisRunner

runner = CCGAnalysisRunner(
    experiment_dir="trained_models/candidate_classifier_...",
    image_dir="/path/to/images",
    dataset_type="3p8arcmin",
    radius_kpc=300.0,
    relative_threshold=2.0
)

results_df = runner.run_complete_analysis(n_images=20)
```

## Interpretation

### High Agreement

When `bar_p` and `p_{CCG}` both give high probability to the same candidate:
- **Interpretation**: Strong confidence in BCG identification
- **Action**: Use as high-confidence training/validation samples

### Disagreement Cases

When `bar_p` is high but `p_{CCG}` is low (or vice versa):
- **Possible causes**:
  - Cluster merger with displaced BCG
  - Multiple BCG candidates (multiple nuclei)
  - Projection effects
  - Member catalog incompleteness
- **Action**: Manual inspection recommended

### Low Both

When both probabilities are low:
- **Interpretation**: Uncertain identification
- **Action**: May need additional data or analysis

## Dependencies

- `astropy`: Cosmology and WCS handling
- `scipy`: KD-tree for coordinate matching
- `numpy`, `pandas`: Data handling
- `matplotlib`: Visualization
- `PIL`: Image loading

## Data Requirements

1. **Cluster Images**: TIF files with WCS headers in `ImageDescription` tag
2. **RedMapper Member Catalogs**: CSV files in `rm_member_catalogs/` with columns:
   - `ra`, `dec`: Member coordinates
   - `pmem`: Membership probability
3. **Evaluation Results**: `evaluation_results.csv` from test.py

## References

The member-based approach is inspired by:
- RedMapper cluster finding algorithm (Rykoff et al. 2014)
- BCG identification studies using member distributions (Lauer et al. 2014)
- Cluster center determination methods (George et al. 2012)

## Troubleshooting

### "no_members_found" Error

- Check that the cluster name matches files in `rm_member_catalogs/`
- Verify cluster name format: `SPT-CLJ0001.5-1555`

### "wcs_load_failed" Error

- Ensure TIF files have valid WCS in the `ImageDescription` tag
- Check that images match the expected format

### Low Member Counts

- Try increasing `radius_kpc` (e.g., 400 kpc)
- Check cluster redshift validity
- Verify member catalog completeness

## Version History

- **v1.0** (December 2024): Initial implementation with C1-C4 rules
