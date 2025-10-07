# Comprehensive BCG Classification Analysis

This directory contains scripts for generating comprehensive analysis reports and visualizations for BCG classification experiments, with comparisons to recent literature (2022-2025).

## Scripts

### `comprehensive_analysis.py`

A generic analysis script that works with any experiment directory from `enhanced_full_run.py` or similar training scripts.

#### Features

- **Automatic Data Loading**: Detects and loads all available CSV files from experiment directories
- **Comprehensive Statistics**: Training performance, detection statistics, error analysis, feature importance
- **Literature Comparison**: Compares results with 4 recent papers (2022-2025)
- **Visualization Suite**: Generates 9 publication-quality plots
- **Generic & Portable**: Works with any experiment directory structure

#### Usage

```bash
# Run with specific experiment directory
python comprehensive_analysis.py /path/to/experiment_dir

# Run with latest experiment (auto-detects from ../best_runs/oct6 or oct5)
python comprehensive_analysis.py
```

#### Required Files

The script requires these CSV files in the experiment directory:
- `training_data.csv` - Training metrics (epochs, loss, accuracy)
- `evaluation_results/evaluation_results.csv` - Per-cluster evaluation results
- `evaluation_results/probability_analysis.csv` - Per-candidate probability data

#### Optional Files

If available, the script will also use:
- `feature_importance_analysis/csv_reports/shap_physical_breakdown_data.csv` - Per-feature SHAP values
- `feature_importance_analysis/csv_reports/shap_physical_importance_data.csv` - Feature group importance

#### Output

All outputs are saved to `<experiment_dir>/literature_analysis/`:

1. **plot1_learning_curve.png** - Training/validation loss curves (Janulewicz et al. 2025 Fig. 4 style)
2. **plot2_accuracy.png** - Training/validation accuracy evolution
3. **plot3_top_features.png** - Top 15 most important features (COSMIC Fig. 4 style)
4. **plot4_feature_groups.png** - Feature group importance
5. **plot5_error_cdf.png** - Cumulative error distribution (Janulewicz et al. 2025 Fig. 6 style)
6. **plot6_error_histogram.png** - Distance error histogram
7. **plot7_probability_dist.png** - BCG probability distribution (Chu et al. 2025 Fig. 7 style)
8. **plot8_uncertainty_scatter.png** - Uncertainty vs probability scatter plot
9. **plot9_redshift_performance.png** - Performance by redshift (Janulewicz et al. 2025 Fig. 10 style)

#### Console Output

The script prints:
- **Training Performance**: Final/best accuracy, loss metrics
- **Detection Statistics**: Total candidates, detections, confidence levels
- **Error Analysis**: Average/median error, accuracy within 25/50/100 kpc thresholds
- **Top Features**: Top 10 SHAP-ranked features (if available)
- **Feature Groups**: Importance breakdown by feature category
- **Redshift Performance**: Performance metrics binned by redshift
- **Literature Comparison**: Detailed comparison with 4 recent papers

## Literature References

The analysis compares with:

1. **Janulewicz et al. (2025)** - arXiv:2502.00104
   - "Using Neural Networks to Automate BCG Identification"
   - R² ≈ 0.94 on simulations, ~80-90% accuracy within 25 kpc

2. **Chu et al. (2025)** - arXiv:2503.15945
   - "Preparing for Rubin-LSST - BCG Detection with ML"
   - ResNet: 81%, Autoencoder: 95% accuracy

3. **Tian et al. (2024)** - arXiv:2410.20083
   - "COSMIC: Galaxy Cluster Finding Algorithm Using ML"
   - XGBoost with feature importance, 90% accuracy

4. **Marini et al. (2022)** - arXiv:2203.03360
   - "ML to identify ICL and BCG in simulated clusters"
   - Random Forest P=0.78-0.95, R=0.85-0.92

## Running on HPC Systems

The script is designed to work with experiments from both local systems and HPC:

```bash
# On HPC after transferring results
python comprehensive_analysis.py /path/to/hpc/experiment/results

# Or set up to auto-detect from your HPC results directory
# (modify the search_paths in find_latest_experiment() function)
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn

## Example Output Structure

```
experiment_dir/
├── training_data.csv
├── evaluation_results/
│   ├── evaluation_results.csv
│   └── probability_analysis.csv
├── feature_importance_analysis/
│   └── csv_reports/
│       ├── shap_physical_breakdown_data.csv
│       └── shap_physical_importance_data.csv
└── literature_analysis/           ← Created by this script
    ├── plot1_learning_curve.png
    ├── plot2_accuracy.png
    ├── plot3_top_features.png
    ├── plot4_feature_groups.png
    ├── plot5_error_cdf.png
    ├── plot6_error_histogram.png
    ├── plot7_probability_dist.png
    ├── plot8_uncertainty_scatter.png
    └── plot9_redshift_performance.png
```

## Example Run

```bash
$ python comprehensive_analysis.py /Users/nesar/Projects/HEP/IMGmarker/best_runs/oct6/candidate_classifier_color_uq_run_20251006_172315

Analyzing experiment: /Users/nesar/Projects/HEP/IMGmarker/best_runs/oct6/candidate_classifier_color_uq_run_20251006_172315
----------------------------------------------------------------------

Loading data...
✓ Loaded 128 rows from training_data.csv
✓ Loaded 385 rows from evaluation_results.csv
✓ Loaded 7709 rows from probability_analysis.csv
✓ Loaded 40 rows from shap_physical_breakdown_data.csv
✓ Loaded 4 rows from shap_physical_importance_data.csv

======================================================================
BCG CLASSIFICATION ANALYSIS REPORT
Comparison with Recent Literature (2022-2025)
======================================================================

📈 TRAINING PERFORMANCE
----------------------------------------------------------------------
  Total Epochs: 128
  Final Training Accuracy: 81.07%
  Final Validation Accuracy: 80.08%
  Best Validation Accuracy: 81.25%
...

✓ Results align with state-of-the-art BCG detection methods!
======================================================================
```
