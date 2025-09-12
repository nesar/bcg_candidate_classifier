"""
Example usage of the BCG classification feature importance analysis framework.

This script demonstrates how to use the various components of the analysis
framework to understand model behavior and feature importance.
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Import analysis modules
from feature_importance import FeatureImportanceAnalyzer
from importance_plots import ImportancePlotter, create_comprehensive_report
from individual_analysis import IndividualSampleAnalyzer
from feature_utils import create_bcg_feature_names, print_feature_summary
from run_analysis import BCGAnalysisRunner

# Import your model classes (adjust paths as needed)
import sys
sys.path.append('..')
from ml_models.candidate_classifier import BCGCandidateClassifier


def create_synthetic_data(n_samples=1000, n_features=58):
    """Create synthetic data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic features with different importance levels
    X = np.random.randn(n_samples, n_features)
    
    # Make some features more predictive
    important_features = [0, 5, 10, 15, 20]  # Morphological features
    moderate_features = [25, 30, 35]         # Contextual features
    
    # Create target based on important features
    y = np.zeros(n_samples)
    
    # Important features have strong effect
    for feat_idx in important_features:
        y += 2 * X[:, feat_idx]
    
    # Moderate features have weaker effect
    for feat_idx in moderate_features:
        y += 0.5 * X[:, feat_idx]
    
    # Add noise and convert to binary classification
    y += 0.2 * np.random.randn(n_samples)
    y = (y > np.median(y)).astype(int)
    
    return X, y


def create_synthetic_model(n_features=58):
    """Create a simple model for demonstration."""
    model = BCGCandidateClassifier(
        input_size=n_features,
        hidden_sizes=[128, 64, 32],
        num_classes=2
    )
    
    # Initialize weights randomly
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    return model


def example_basic_analysis():
    """Example of basic feature importance analysis."""
    print("=== Example 1: Basic Feature Importance Analysis ===\n")
    
    # Create synthetic data and model
    X, y = create_synthetic_data()
    model = create_synthetic_model()
    feature_names = create_bcg_feature_names()
    
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {np.sum(y)} positive, {np.sum(1-y)} negative")
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        model=model,
        feature_names=feature_names[:X.shape[1]],
        device='cpu',
        probabilistic=False
    )
    
    # Run analysis (without SHAP for speed in demo)
    print("\nRunning feature importance analysis...")
    results = analyzer.analyze_feature_importance(
        X, y, 
        methods=['permutation', 'gradient'],
        n_repeats=5  # Fewer repeats for demo
    )
    
    # Display top features
    for method in results:
        print(f"\n--- {method.upper()} Results ---")
        ranking_df = analyzer.get_feature_ranking(results, method)
        
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(ranking_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature_name']:<25} : {row['importance']:.4f}")
    
    return analyzer, results, X, y


def example_visualization():
    """Example of visualization capabilities."""
    print("\n=== Example 2: Visualization Capabilities ===\n")
    
    # Use results from previous example
    analyzer, results, X, y = example_basic_analysis()
    
    # Create plotter
    plotter = ImportancePlotter()
    
    # Create output directory
    output_dir = Path('example_plots')
    output_dir.mkdir(exist_ok=True)
    
    print("Creating visualization plots...")
    
    # 1. Feature ranking plots
    for method in results:
        ranking_df = analyzer.get_feature_ranking(results, method)
        
        fig = plotter.plot_feature_ranking(
            ranking_df, 
            top_n=15,
            method_name=method.title(),
            save_path=output_dir / f'{method}_ranking.png'
        )
        plt.close(fig)
    
    # 2. Method comparison
    if len(results) > 1:
        fig = plotter.plot_method_comparison(
            results, 
            top_n=12,
            save_path=output_dir / 'method_comparison.png'
        )
        plt.close(fig)
    
    # 3. Distribution analysis
    for method in results:
        if method == 'shap':
            importance = results[method]['mean_abs_shap']
        else:
            importance = results[method]['importance']
        
        fig = plotter.plot_importance_distribution(
            importance, 
            analyzer.feature_names,
            save_path=output_dir / f'{method}_distribution.png'
        )
        plt.close(fig)
    
    print(f"Plots saved to: {output_dir}")


def example_individual_analysis():
    """Example of individual sample analysis."""
    print("\n=== Example 3: Individual Sample Analysis ===\n")
    
    # Create data and model
    X, y = create_synthetic_data()
    model = create_synthetic_model()
    feature_names = create_bcg_feature_names()[:X.shape[1]]
    
    # Initialize individual analyzer
    individual_analyzer = IndividualSampleAnalyzer(
        model=model,
        feature_names=feature_names,
        device='cpu',
        probabilistic=False
    )
    
    # Analyze a few samples
    sample_indices = [0, 100, 500]
    
    for idx in sample_indices:
        print(f"\n--- Analyzing Sample {idx} ---")
        
        sample = X[idx]
        explanation = individual_analyzer.explain_sample(
            sample,
            background_data=X[:50]  # Use first 50 samples as background
        )
        
        # Print summary
        pred_info = explanation['prediction']
        print(f"Predicted class: {pred_info['predicted_class']}")
        print(f"Confidence: {pred_info['prediction_confidence']:.4f}")
        print(f"True label: {y[idx]}")
        
        # Top contributing features
        local_importance = explanation['local_importance']
        top_features = local_importance['top_positive'][:5]
        
        print("Top contributing features:")
        for feat_name, contribution in top_features:
            print(f"  {feat_name}: {contribution:.4f}")
        
        # Create visualization
        output_dir = Path('example_individual')
        output_dir.mkdir(exist_ok=True)
        
        fig = individual_analyzer.plot_sample_explanation(
            explanation,
            save_path=output_dir / f'sample_{idx}_explanation.png'
        )
        plt.close(fig)
    
    print(f"Individual analysis plots saved to: {output_dir}")


def example_feature_groups():
    """Example of feature group analysis."""
    print("\n=== Example 4: Feature Group Analysis ===\n")
    
    # Create data
    X, y = create_synthetic_data()
    model = create_synthetic_model()
    feature_names = create_bcg_feature_names()[:X.shape[1]]
    
    # Print feature summary
    print_feature_summary(feature_names)
    
    # Run importance analysis
    analyzer = FeatureImportanceAnalyzer(
        model=model,
        feature_names=feature_names,
        device='cpu',
        probabilistic=False
    )
    
    results = analyzer.analyze_feature_importance(
        X, y, 
        methods=['permutation', 'gradient'],
        n_repeats=3
    )
    
    # Feature group analysis
    from feature_importance import FeatureGroupAnalyzer, create_default_feature_groups
    from feature_utils import get_feature_groups_mapping
    
    # Get feature groups
    feature_groups = get_feature_groups_mapping(feature_names)
    group_analyzer = FeatureGroupAnalyzer(feature_groups)
    
    print("\n--- Feature Group Analysis ---")
    
    for method in results:
        if method == 'shap':
            importance = results[method]['mean_abs_shap']
        else:
            importance = results[method]['importance']
        
        group_scores = group_analyzer.compute_group_importance(importance, feature_names)
        
        print(f"\n{method.upper()} - Group Importance:")
        sorted_groups = sorted(group_scores.items(), 
                             key=lambda x: x[1]['mean_importance'], 
                             reverse=True)
        
        for group_name, scores in sorted_groups:
            print(f"  {group_name:15s}: {scores['mean_importance']:.4f} (avg), "
                  f"{scores['sum_importance']:.4f} (total), "
                  f"{scores['feature_count']:2d} features")


def example_comprehensive_analysis():
    """Example of running comprehensive analysis."""
    print("\n=== Example 5: Comprehensive Analysis ===\n")
    
    # Create configuration
    config = {
        'model_path': None,  # Will use synthetic model
        'data_path': None,   # Will use synthetic data
        'model_type': 'deterministic',
        'output_dir': 'example_comprehensive',
        'analysis_methods': ['permutation', 'gradient'],
        'analysis_samples': 500,
        'permutation_repeats': 5
    }
    
    # Create synthetic data and save
    X, y = create_synthetic_data()
    feature_names = create_bcg_feature_names()[:X.shape[1]]
    
    # Since we can't easily integrate with the full pipeline for demo,
    # let's show what the analysis would produce
    print("This example would run the complete analysis pipeline including:")
    print("1. Model loading and data preparation")
    print("2. Global feature importance analysis")
    print("3. Individual sample explanations")
    print("4. Feature group analysis")
    print("5. Comprehensive visualization reports")
    print("6. Summary report generation")
    print()
    print("To run the full analysis on real data, use:")
    print("  python analysis/run_analysis.py --config analysis_config.yaml")
    print("  or")
    print("  python analysis/run_analysis.py --model_path model.pth --data_path data.npz")


def example_custom_analysis():
    """Example of custom analysis workflow."""
    print("\n=== Example 6: Custom Analysis Workflow ===\n")
    
    # This shows how to build custom analysis workflows
    X, y = create_synthetic_data()
    model = create_synthetic_model()
    feature_names = create_bcg_feature_names()[:X.shape[1]]
    
    print("Custom workflow: Combining different analysis types...")
    
    # 1. Quick permutation importance
    analyzer = FeatureImportanceAnalyzer(model, feature_names, device='cpu')
    perm_results = analyzer.analyze_feature_importance(
        X[:100], y[:100],  # Subset for speed
        methods=['permutation'],
        n_repeats=3
    )
    
    # 2. Focus on top features for detailed analysis
    ranking_df = analyzer.get_feature_ranking(perm_results, 'permutation')
    top_features = ranking_df.head(10)['feature_name'].tolist()
    
    print("Top 10 features for detailed analysis:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 3. Individual analysis on interesting samples
    individual_analyzer = IndividualSampleAnalyzer(model, feature_names, device='cpu')
    
    # Find samples with different prediction confidences
    predictions = []
    for i in range(min(50, len(X))):
        pred_info = individual_analyzer.get_prediction_info(X[i:i+1])
        predictions.append((i, pred_info['prediction_confidence']))
    
    # Sort by confidence
    predictions.sort(key=lambda x: x[1])
    
    print("\nAnalyzing samples with different confidence levels:")
    low_conf_idx = predictions[0][0]  # Lowest confidence
    high_conf_idx = predictions[-1][0]  # Highest confidence
    
    for name, idx in [("Low confidence", low_conf_idx), ("High confidence", high_conf_idx)]:
        explanation = individual_analyzer.explain_sample(X[idx], background_data=X[:20])
        pred_info = explanation['prediction']
        
        print(f"\n{name} sample (index {idx}):")
        print(f"  Predicted class: {pred_info['predicted_class']}")
        print(f"  Confidence: {pred_info['prediction_confidence']:.4f}")
        print(f"  True label: {y[idx]}")
        
        # Top features for this sample
        local_importance = explanation['local_importance']
        top_pos = local_importance['top_positive'][:3]
        print(f"  Top contributing features: {[f[0] for f in top_pos]}")
    
    print("\nCustom analysis complete!")


def main():
    """Run all examples."""
    print("BCG Classification Feature Importance Analysis - Examples")
    print("="*60)
    
    # Make sure plots directory exists
    Path('example_plots').mkdir(exist_ok=True)
    
    try:
        # Run examples
        example_basic_analysis()
        example_visualization()
        example_individual_analysis()
        example_feature_groups()
        example_comprehensive_analysis()
        example_custom_analysis()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("\nGenerated files:")
        print("- example_plots/ : Basic visualization examples")
        print("- example_individual/ : Individual sample analysis")
        
        print("\nTo run real analysis:")
        print("1. Train your BCG classification model")
        print("2. Prepare your test dataset")
        print("3. Run: python analysis/run_analysis.py --model_path YOUR_MODEL.pth --data_path YOUR_DATA.npz")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()