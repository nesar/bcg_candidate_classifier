# BCG Classification Feature Importance Analysis

This module provides comprehensive post-analysis tools for understanding feature importance and model behavior in Brightest Cluster Galaxy (BCG) candidate classification models.

## Features

### 🎯 Global Feature Importance Analysis
- **SHAP (SHapley Additive exPlanations)**: Model-agnostic explanations with rigorous theoretical foundation
- **Permutation Importance**: Feature importance through performance degradation when features are shuffled
- **Gradient-based Importance**: Neural network specific importance using gradient information

### 🔍 Individual Sample Analysis
- **Local SHAP Explanations**: Understanding individual predictions
- **Feature Contribution Analysis**: How each feature contributes to specific predictions
- **Sample Similarity Analysis**: Finding similar samples and comparing their predictions
- **Prediction Boundary Visualization**: How predictions change when features are varied

### 📊 Comprehensive Visualizations
- **Feature Ranking Plots**: Ranked importance of all features
- **Method Comparison**: Compare importance across different methods
- **Feature Group Analysis**: Analysis by feature categories (morphological, color, contextual, etc.)
- **Distribution Analysis**: Statistical distributions of importance scores
- **Individual Explanation Plots**: Detailed visualizations for specific samples

### 📈 Advanced Analysis
- **Feature Group Categorization**: Automatic grouping of features by type
- **Uncertainty Quantification**: Analysis of prediction uncertainty (for probabilistic models)
- **Correlation Analysis**: Relationship between features and their importance

## Installation

### Required Dependencies

```bash
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Deep learning
torch>=1.10.0
torchvision>=0.11.0

# Feature importance analysis
shap>=0.41.0  # Optional but recommended
```

### Install SHAP (Recommended)

SHAP provides the most comprehensive feature importance analysis:

```bash
# Standard installation
pip install shap

# Or with conda
conda install -c conda-forge shap
```

**Note**: SHAP is optional. If not installed, the framework will work with permutation and gradient-based methods only.

## Quick Start

### 1. Basic Analysis

```python
from analysis.feature_importance import FeatureImportanceAnalyzer
from analysis.feature_utils import create_bcg_feature_names

# Load your trained model and test data
model = load_your_model()  # Your trained BCG classifier
X_test, y_test = load_your_data()  # Your test dataset

# Create feature names
feature_names = create_bcg_feature_names(
    use_color_features=True,
    use_auxiliary_features=True
)

# Initialize analyzer
analyzer = FeatureImportanceAnalyzer(
    model=model,
    feature_names=feature_names,
    device='cuda',  # or 'cpu'
    probabilistic=False  # Set to True for UQ models
)

# Run analysis
results = analyzer.analyze_feature_importance(
    X_test, y_test,
    methods=['permutation', 'shap', 'gradient'],
    n_repeats=10
)

# Get feature rankings
for method in results:
    ranking_df = analyzer.get_feature_ranking(results, method)
    print(f"\nTop 10 features ({method}):")
    print(ranking_df.head(10)[['feature_name', 'importance']])
```

### 2. Individual Sample Analysis

```python
from analysis.individual_analysis import IndividualSampleAnalyzer

# Initialize individual analyzer
individual_analyzer = IndividualSampleAnalyzer(
    model=model,
    feature_names=feature_names,
    device='cuda',
    probabilistic=False
)

# Analyze specific sample
sample_idx = 0
explanation = individual_analyzer.explain_sample(
    X_test[sample_idx],
    background_data=X_test[:100]  # Background for SHAP
)

# Print results
pred_info = explanation['prediction']
print(f"Predicted class: {pred_info['predicted_class']}")
print(f"Confidence: {pred_info['prediction_confidence']:.4f}")

# Create visualization
fig = individual_analyzer.plot_sample_explanation(
    explanation,
    save_path='sample_explanation.png'
)
```

### 3. Complete Analysis Pipeline

```python
from analysis.run_analysis import BCGAnalysisRunner

# Create configuration
config = {
    'model_path': 'path/to/your/model.pth',
    'data_path': 'path/to/your/test_data.npz',
    'model_type': 'deterministic',  # or 'probabilistic'
    'output_dir': 'analysis_results',
    'analysis_methods': ['permutation', 'shap', 'gradient'],
    'analysis_samples': 1000
}

# Run complete analysis
runner = BCGAnalysisRunner(**config)
results = runner.run_complete_analysis()

print(f"Analysis complete! Results in: {results['output_directory']}")
```

### 4. Command Line Usage

```bash
# Create default configuration
python analysis/run_analysis.py --create_config

# Run analysis with configuration file
python analysis/run_analysis.py --config analysis_config.yaml

# Run analysis with direct parameters
python analysis/run_analysis.py \
    --model_path models/bcg_classifier.pth \
    --data_path data/test_features.npz \
    --output_dir results \
    --methods permutation shap \
    --samples 1000
```

## Feature Types

The analysis framework automatically categorizes BCG classification features into groups:

### 🔬 Morphological Features
- **Intensity Statistics**: mean, std, max, min, median, skewness of patch intensity
- **Shape Features**: concentration ratio, eccentricity
- **Gradient Features**: gradient magnitude statistics
- **Geometric Moments**: second-order intensity moments

### 🗺️ Contextual Features
- **Position Features**: relative x, y position, distance from center
- **Environmental Features**: brightness rank, candidate density, background level
- **Directional Features**: intensity sampling in cardinal directions

### 🌈 Color Features (Optional)
- **Color Ratios**: R/G, R/B ratios approximating photometric colors
- **Red-sequence Indicators**: scores targeting red-sequence galaxies
- **Spatial Color Variation**: standard deviations of color ratios
- **Color Gradients**: correlations between color channel gradients
- **PCA Features**: dimensionality-reduced color feature space

### 🔭 Auxiliary Features (Optional)
- **Redshift**: photometric redshift estimates
- **Stellar Mass**: Δm*_z stellar mass indicators

## Output Structure

The analysis generates comprehensive reports in the specified output directory:

```
analysis_results/
├── raw_results/
│   ├── global_importance.pkl      # Raw importance results
│   ├── individual_analysis.pkl    # Individual sample analyses
│   └── group_analysis.pkl         # Feature group results
├── csv_reports/
│   ├── permutation_feature_ranking.csv
│   ├── shap_feature_ranking.csv
│   └── gradient_feature_ranking.csv
├── plots/
│   ├── permutation_feature_ranking.png
│   ├── shap_summary_bar.png
│   ├── method_comparison.png
│   ├── feature_groups.png
│   └── ... (many more plots)
├── individual_plots/
│   ├── sample_0_explanation.png
│   ├── sample_500_explanation.png
│   └── ...
└── analysis_summary.txt           # Human-readable summary
```

## Advanced Usage

### Custom Feature Groups

```python
from analysis.feature_importance import FeatureGroupAnalyzer

# Define custom feature groups
custom_groups = {
    'intensity_features': ['patch_mean', 'patch_std', 'patch_max'],
    'position_features': ['x_relative', 'y_relative', 'r_center'],
    'color_features': ['rg_ratio_mean', 'rb_ratio_mean', 'color_magnitude']
}

group_analyzer = FeatureGroupAnalyzer(custom_groups)
group_scores = group_analyzer.compute_group_importance(
    importance_scores, feature_names
)
```

### Probabilistic Model Analysis

```python
# For models with uncertainty quantification
analyzer = FeatureImportanceAnalyzer(
    model=uq_model,
    feature_names=feature_names,
    probabilistic=True  # Enable uncertainty analysis
)

# Individual analysis includes uncertainty metrics
explanation = individual_analyzer.explain_sample(sample)
pred_info = explanation['prediction']

print(f"Epistemic uncertainty: {pred_info['epistemic_uncertainty']}")
print(f"Aleatoric uncertainty: {pred_info['aleatoric_uncertainty']}")
print(f"Is uncertain: {pred_info['is_uncertain']}")
```

### Prediction Boundary Analysis

```python
from analysis.individual_analysis import analyze_prediction_boundary

# Analyze how predictions change when varying a specific feature
fig = analyze_prediction_boundary(
    model=model,
    sample=X_test[0],
    feature_names=feature_names,
    feature_idx=10,  # Index of feature to vary
    n_points=100,
    save_path='boundary_analysis.png'
)
```

## Examples

See `analysis/example_usage.py` for comprehensive examples covering:

1. **Basic Analysis**: Simple feature importance analysis
2. **Visualization**: Creating various plots and reports
3. **Individual Analysis**: Explaining specific predictions
4. **Feature Groups**: Analyzing feature categories
5. **Custom Workflows**: Building specialized analysis pipelines

Run examples:

```bash
cd analysis/
python example_usage.py
```

## Performance Considerations

### SHAP Analysis
- **Deep learning models**: Use `DeepExplainer` for faster computation
- **Large datasets**: Sample background data (100-1000 samples typically sufficient)
- **Multi-class**: SHAP analysis focuses on predicted class by default

### Permutation Importance
- **Computational cost**: O(n_features × n_repeats × inference_time)
- **Parallelization**: Uses all CPU cores by default (`n_jobs=-1`)
- **Sample size**: Use subset of data for faster analysis

### Memory Usage
- **Large models**: Consider using CPU for analysis even if model was trained on GPU
- **Batch processing**: Individual analysis processes samples one at a time

## Integration with BCG Pipeline

The analysis framework integrates seamlessly with the existing BCG classification pipeline:

1. **Model Compatibility**: Works with both deterministic and probabilistic classifiers
2. **Feature Consistency**: Automatically generates feature names matching the extraction pipeline
3. **Data Formats**: Supports numpy arrays, PyTorch tensors, and custom datasets

## Troubleshooting

### Common Issues

**1. SHAP Installation Problems**
```bash
# Try different installation methods
pip install shap --no-binary shap
conda install -c conda-forge shap

# Verify installation
python -c "import shap; print('SHAP installed successfully')"
```

**2. Memory Issues with Large Models**
```python
# Use CPU for analysis
analyzer = FeatureImportanceAnalyzer(model, feature_names, device='cpu')

# Reduce sample size
results = analyzer.analyze_feature_importance(X[:500], y[:500])
```

**3. Feature Name Mismatches**
```python
# Validate feature names
from analysis.feature_utils import validate_feature_names
validation = validate_feature_names(your_feature_names, expected_count=58)
if not validation['valid']:
    print("Issues:", validation['warnings'])
```

**4. Slow Analysis**
```python
# Use faster methods only
results = analyzer.analyze_feature_importance(
    X, y, 
    methods=['permutation'],  # Skip SHAP and gradient
    n_repeats=5  # Reduce repeats
)
```

## Contributing

To extend the analysis framework:

1. **New Importance Methods**: Add methods to `FeatureImportanceAnalyzer`
2. **Custom Visualizations**: Extend `ImportancePlotter` class
3. **Feature Types**: Update `feature_utils.py` for new feature categories
4. **Analysis Types**: Create new analyzer classes following existing patterns

## References

- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- **Permutation Importance**: Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- **BCG Classification**: See main project documentation for astronomical background.

## Support

For issues or questions:

1. Check the examples in `example_usage.py`
2. Review the troubleshooting section
3. Examine the generated analysis reports for insights
4. Consult the main project documentation for BCG-specific details