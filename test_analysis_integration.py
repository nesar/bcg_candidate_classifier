#!/usr/bin/env python3
"""
Test script to verify the feature importance analysis integration works.
This creates synthetic data and runs a quick analysis to ensure everything is working.
"""

import numpy as np
import torch
import os
from pathlib import Path
import tempfile

def create_synthetic_test():
    """Create synthetic data and model for testing."""
    print("Creating synthetic test data and model...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 40  # Smaller for quick test
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 5] + 0.3 * X[:, 10] > 0).astype(int)  # Simple synthetic labels
    
    # Create a simple model
    from ml_models.candidate_classifier import BCGCandidateClassifier
    model = BCGCandidateClassifier(feature_dim=n_features, hidden_dims=[32, 16])
    
    # Initialize weights randomly but deterministically
    torch.manual_seed(42)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    return X, y, model


def test_analysis_components():
    """Test individual analysis components."""
    print("\n=== Testing Analysis Components ===")
    
    X, y, model = create_synthetic_test()
    
    # Test feature name generation
    from analysis.feature_utils import create_bcg_feature_names
    feature_names = create_bcg_feature_names(
        use_color_features=False,  # Simplified for test
        use_auxiliary_features=False,
        color_pca_components=0
    )
    
    # Ensure we have the right number of feature names
    if len(feature_names) < X.shape[1]:
        # Add generic feature names to match
        for i in range(len(feature_names), X.shape[1]):
            feature_names.append(f'feature_{i}')
    elif len(feature_names) > X.shape[1]:
        # Truncate to match
        feature_names = feature_names[:X.shape[1]]
    
    print(f"✓ Feature names created: {len(feature_names)} features")
    
    # Test feature importance analyzer
    from analysis.feature_importance import FeatureImportanceAnalyzer
    analyzer = FeatureImportanceAnalyzer(
        model=model,
        feature_names=feature_names,
        device='cpu',
        probabilistic=False
    )
    
    print("✓ Feature importance analyzer created")
    
    # Test analysis (permutation only for speed)
    try:
        results = analyzer.analyze_feature_importance(
            X, y, 
            methods=['permutation'],  # Skip SHAP for speed
            n_repeats=3
        )
        print("✓ Feature importance analysis completed")
        
        # Test ranking
        ranking_df = analyzer.get_feature_ranking(results, 'permutation')
        print(f"✓ Feature ranking created: top feature is '{ranking_df.iloc[0]['feature_name']}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_analysis_runner():
    """Test the full analysis runner."""
    print("\n=== Testing Full Analysis Runner ===")
    
    X, y, model = create_synthetic_test()
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save test data
        data_path = os.path.join(temp_dir, "test_data.npz")
        np.savez(data_path, X=X, y=y)
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dim': X.shape[1],
            'hidden_dims': [32, 16]  # Match the model we created
        }, model_path)
        
        # Test analysis runner
        try:
            from analysis.run_analysis import BCGAnalysisRunner
            
            config = {
                'model_path': model_path,
                'data_path': data_path,
                'model_type': 'deterministic',
                'output_dir': os.path.join(temp_dir, 'analysis_results'),
                'analysis_methods': ['permutation'],  # Fast method only
                'analysis_samples': 50,  # Small sample for speed
                'features': {
                    'use_color': False,
                    'use_auxiliary': False,
                    'color_pca_components': 0
                }
            }
            
            runner = BCGAnalysisRunner(**config)
            results = runner.run_complete_analysis()
            
            print("✓ Full analysis runner completed successfully")
            print(f"✓ Results saved to: {results['output_directory']}")
            
            # Check if key files were created
            analysis_dir = Path(results['output_directory'])
            
            expected_files = [
                'analysis_summary.txt',
                'csv_reports/permutation_feature_ranking.csv'
            ]
            
            for file_path in expected_files:
                full_path = analysis_dir / file_path
                if full_path.exists():
                    print(f"✓ Created: {file_path}")
                else:
                    print(f"✗ Missing: {file_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Full runner failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING BCG FEATURE IMPORTANCE ANALYSIS INTEGRATION")
    print("=" * 60)
    
    # Test imports
    print("\n=== Testing Imports ===")
    try:
        from analysis.feature_importance import FeatureImportanceAnalyzer
        from analysis.run_analysis import BCGAnalysisRunner
        from analysis.feature_utils import create_bcg_feature_names
        print("✓ All analysis modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test components
    component_success = test_analysis_components()
    
    # Test full runner
    runner_success = test_full_analysis_runner()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if component_success and runner_success:
        print("🎉 ALL TESTS PASSED!")
        print("\nThe feature importance analysis integration is working correctly.")
        print("\nTo use in enhanced_full_run.py:")
        print("1. Make sure you have seaborn installed: pip install seaborn")
        print("2. Optional: Install SHAP for comprehensive analysis: pip install shap")
        print("3. Run: python enhanced_full_run.py")
        print("4. When prompted, choose 'Y' for feature importance analysis")
        
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above and fix any issues.")
        
    return component_success and runner_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)