# BCG Candidate Classifier - Updates & Changes

This document consolidates all recent updates, improvements, and changes to the BCG candidate classifier codebase.

---

## Plot Styling Standardization (Nov 2024)

All plotting routines now use consistent styling via `plot_config.py`:

- **Fonts**: Title 20pt, Labels 18pt, Ticks 14pt, Serif family
- **Sizes**: Linewidth 2.0, DPI 300, Markersize 8
- **Colors**: Consistent palette (completeness=green, purity=blue, train=blue, validation=red)
- **Updated files**: `utils/diagnostic_plots.py`, `plot_sectors_hardcoded.py`, `plot_physical_results.py`, `plot_completeness_purity.py`, `plot_rank_histograms.py`, `plot_eval_results.py`, `train.py`

**Usage**:
```python
from plot_config import setup_plot_style, COLORS, FONTS, SIZES
setup_plot_style()
plt.plot(x, y, color=COLORS['primary'], linewidth=SIZES['linewidth'])
```

---

## Completeness & Purity Analysis (Oct 2024)

**New script**: `plot_completeness_purity.py`
- Plots completeness and purity as functions of redshift (z) and delta_mstar_z
- Handles both single-prediction and multi-detection (UQ) models
- Includes error bars based on binomial statistics
- Outputs: `completeness_purity_plots.png/pdf`

**Key metrics**:
- **Completeness**: Fraction of true BCGs detected
- **Purity**: Fraction of detections that are correct
- For single-prediction models: purity = completeness
- For UQ models: purity accounts for multiple detections per image

---

## Sector & Rank Visualization (Oct 2024)

**Enhanced donut charts** in `plot_sectors_hardcoded.py`:
- Shows distribution of BCG detections by rank (Rank 1, Rank 2, Rank 3, Rest)
- Handles both single-target and multi-target scenarios
- Outputs: `diagnostic_plots_sectors.png/pdf`

**Rank histograms** in `plot_rank_histograms.py`:
- r_center: Radial distance from image center
- p_RM: RedMapper BCG probabilities
- Color-coded by rank category

---

## Terminology & Scientific Integrity (Sep-Oct 2024)

**Key changes**:
- "Confidence" → "Predictive Confidence" (model outputs)
- "Uncertainty" → "Uncertainty Estimate" (epistemic uncertainty from ensemble)
- Clear distinction between:
  - **Predictive confidence**: P(BCG | candidate) from model
  - **Uncertainty estimate**: Variance/disagreement in ensemble predictions
  - **RedMapper probability**: External catalog probability (p_RM)

**Updated labels across**:
- All plotting scripts
- Column names in CSV outputs
- Axis labels and legends
- Documentation

---

## Literature Comparison Analysis (Oct 2024)

**New module**: `analysis/literature_comparison/`
- Generates 9 publication-quality comparison plots
- Benchmarks against 4 recent papers (2022-2025):
  - Janulewicz+ 2025: Neural Networks for BCG ID
  - Chu+ 2025: ML for Rubin-LSST
  - Tian+ 2024: COSMIC Cluster Finding
  - Marini+ 2022: ICL and BCG Identification

**Outputs**: Learning curves, accuracy comparisons, feature rankings, error distributions, redshift performance

---

## Feature Importance Analysis (Sep 2024)

**New module**: `analysis/feature_importance.py` + `analysis/importance_plots.py`
- SHAP values for feature attribution
- Gradient-based importance
- Permutation importance
- Feature group analysis (morphological, color, contextual, auxiliary)
- Individual sample explanations

**Usage**:
```python
from analysis.run_analysis import BCGAnalysisRunner
runner = BCGAnalysisRunner(model_path, data_path, output_dir)
results = runner.run_complete_analysis()
```

---

## Quick Reference

### Main Scripts
- `train.py` - Train BCG classifier
- `test.py` - Evaluate classifier
- `enhanced_full_run.py` - Complete workflow (train → test → analyze → visualize)

### Key Plotting Scripts
- `plot_sectors_hardcoded.py` - Rank distribution donut charts
- `plot_completeness_purity.py` - Performance vs physical properties
- `plot_rank_histograms.py` - Spatial and probability distributions
- `plot_physical_results.py` - Feature importance by physical groups
- `plot_eval_results.py` - Comprehensive evaluation analysis
- `utils/diagnostic_plots.py` - Multi-panel diagnostic plots

### Important Files
- `plot_config.py` - Centralized plot styling configuration
- `evaluation_results.csv` - Detailed per-cluster evaluation metrics
- `probability_analysis.csv` - Per-candidate probability outputs

### Dataset Types
- `bcg_2p2arcmin` - 2.2 arcmin field of view (512×512 px)
- `bcg_3p8arcmin` - 3.8 arcmin field of view (512×512 px)

---

## Common Commands

```bash
# Run full workflow
python enhanced_full_run.py

# Train only
python train.py --image_dir <dir> --truth_table <csv> --dataset_type bcg_3p8arcmin

# Test only
python test.py --model_path <model> --scaler_path <scaler> --dataset_type bcg_3p8arcmin

# Generate specific plots
python plot_sectors_hardcoded.py <evaluation_results.csv>
python plot_completeness_purity.py <evaluation_results.csv> --bcg_csv <bcg_catalog.csv>
python plot_rank_histograms.py <evaluation_results.csv> --dataset_type 3p8arcmin
```

---

## Git Workflow

To sync changes to remote cluster:
```bash
git add .
git commit -m "Description of changes"
git push

# On remote cluster
git pull
python enhanced_full_run.py
```

---

*Last updated: November 2024*
