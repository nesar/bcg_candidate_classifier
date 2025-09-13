#!/usr/bin/env python3
"""
Quick test of feature importance analysis with example data.
This runs a simplified version to verify everything is working.
"""

import numpy as np
import tempfile
import os

def main():
    """Run a quick analysis test."""
    print("=" * 50)
    print("QUICK BCG ANALYSIS TEST")
    print("=" * 50)
    
    # Install dependencies check
    try:
        import seaborn
        print("✓ seaborn available")
    except ImportError:
        print("✗ seaborn missing - install with: pip install seaborn")
        return False
    
    try:
        import shap
        print("✓ SHAP available")
        use_shap = True
    except ImportError:
        print("⚠️ SHAP missing - install with: pip install shap (continuing without SHAP)")
        use_shap = False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Create synthetic BCG-like data
        print("Creating synthetic BCG data...")
        np.random.seed(42)
        
        # Simulate BCG features
        n_samples = 200
        n_features = 24  # Match expected BCG features
        
        X = np.random.randn(n_samples, n_features)
        
        # Make some features more predictive (simulate real BCG patterns)
        # Feature 0: brightness (most important for BCG detection)
        X[:, 0] = X[:, 0] + np.random.uniform(0.5, 2.0, n_samples)
        
        # Feature 5: concentration (important morphological feature)
        X[:, 5] = X[:, 5] + np.random.uniform(0.3, 1.0, n_samples) 
        
        # Feature 10: position feature (contextual)
        X[:, 10] = X[:, 10] + np.random.uniform(0.2, 0.8, n_samples)
        
        # Create labels based on important features
        y = ((X[:, 0] > 0.5) & 
             (X[:, 5] > 0.3) & 
             (X[:, 10] > 0.1)).astype(int)
        
        print(f"✓ Created data: {n_samples} samples, {n_features} features")
        print(f"✓ Class distribution: {np.sum(y)} positive, {np.sum(1-y)} negative")
        
        # Save data
        data_path = os.path.join(temp_dir, "bcg_test_data.npz")
        np.savez(data_path, X=X, y=y)
        
        # Create a dummy model file (analysis will handle missing model gracefully)
        model_path = os.path.join(temp_dir, "dummy_model.pth")
        with open(model_path, 'w') as f:
            f.write("dummy")  # Placeholder
        
        # Run analysis with permutation only (fastest method)
        print("\nRunning feature importance analysis...")
        
        try:
            from analysis.feature_importance import FeatureImportanceAnalyzer
            from analysis.feature_utils import create_bcg_feature_names
            from ml_models.candidate_classifier import BCGCandidateClassifier
            
            # Create a simple model for analysis
            import torch
            torch.manual_seed(42)
            model = BCGCandidateClassifier(feature_dim=n_features, hidden_dims=[32, 16])
            
            # Get feature names
            feature_names = create_bcg_feature_names(
                use_color_features=False,
                use_auxiliary_features=False
            )[:n_features]
            
            # Pad with generic names if needed
            while len(feature_names) < n_features:
                feature_names.append(f'feature_{len(feature_names)}')
            
            print(f"✓ Feature names: {len(feature_names)} names for {n_features} features")
            
            # Run analysis
            analyzer = FeatureImportanceAnalyzer(
                model=model,
                feature_names=feature_names,
                device='cpu',
                probabilistic=False
            )
            
            methods = ['permutation']
            if use_shap:
                methods.append('gradient')  # Skip SHAP for speed but include gradient
            
            results = analyzer.analyze_feature_importance(
                X, y,
                methods=methods,
                n_repeats=3
            )
            
            print("✓ Analysis completed!")
            
            # Show results
            for method in results:
                ranking_df = analyzer.get_feature_ranking(results, method)
                print(f"\n📊 {method.upper()} - Top 10 Important Features:")
                
                for i, (_, row) in enumerate(ranking_df.head(10).iterrows()):
                    print(f"  {i+1:2d}. {row['feature_name']:<20} : {row['importance']:.4f}")
            
            # Create simple plots
            from analysis.importance_plots import ImportancePlotter
            
            plotter = ImportancePlotter()
            output_dir = os.path.join(temp_dir, "plots")
            os.makedirs(output_dir, exist_ok=True)
            
            for method in results:
                ranking_df = analyzer.get_feature_ranking(results, method)
                fig = plotter.plot_feature_ranking(
                    ranking_df,
                    top_n=15,
                    method_name=method.title(),
                    save_path=os.path.join(output_dir, f'{method}_ranking.png')
                )
                print(f"✓ Created plot: {method}_ranking.png")
            
            print(f"\n🎉 SUCCESS! Quick analysis completed.")
            print(f"📊 This confirms the analysis framework is working correctly.")
            print(f"📈 You can now run analysis on your real BCG models!")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 50)
        print("✅ FRAMEWORK IS READY!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Train your BCG model using enhanced_full_run.py")
        print("2. When prompted, choose 'Y' for feature importance analysis")
        print("3. Or use: python run_standalone_analysis.py")
        print("\nThe analysis integration is working correctly!")
    else:
        print("\n❌ There are still issues to resolve.")
    