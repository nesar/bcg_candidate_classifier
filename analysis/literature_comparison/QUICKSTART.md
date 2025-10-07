# Quick Start Guide

## Run Analysis on Oct 6 Experiment

```bash
cd /Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/analysis/literature_comparison

python comprehensive_analysis.py /Users/nesar/Projects/HEP/IMGmarker/best_runs/oct6/candidate_classifier_color_uq_run_20251006_172315
```

## Run on Any Experiment

```bash
# Option 1: Specify experiment directory
python comprehensive_analysis.py /path/to/your/experiment

# Option 2: Auto-detect latest experiment
python comprehensive_analysis.py
```

## For HPC Users

After copying your experiment results from HPC to local:

```bash
# Copy results from HPC (run on your local machine)
scp -r username@hpc:/path/to/experiment_results /Users/nesar/Projects/HEP/IMGmarker/best_runs/

# Run analysis
python comprehensive_analysis.py /Users/nesar/Projects/HEP/IMGmarker/best_runs/your_experiment
```

## Expected Output

The script will:
1. Load all CSV data files from the experiment
2. Print comprehensive statistics to console
3. Create 9 publication-quality plots in `experiment_dir/literature_analysis/`
4. Compare your results with 4 recent papers (2022-2025)

## Minimal Requirements

Your experiment directory must contain:
- `training_data.csv`
- `evaluation_results/evaluation_results.csv`
- `evaluation_results/probability_analysis.csv`

Optional (for feature importance analysis):
- `feature_importance_analysis/csv_reports/shap_physical_breakdown_data.csv`
- `feature_importance_analysis/csv_reports/shap_physical_importance_data.csv`

## Example Output Location

After running, find your plots at:
```
/path/to/experiment/literature_analysis/
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
