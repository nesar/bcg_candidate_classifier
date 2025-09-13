#!/usr/bin/env python3
"""
Standalone script to run feature importance analysis on your BCG model.
Use this after you have trained a model and have test results.
"""

import os
# Fix NUMEXPR warning - MUST be set before ANY numpy/sklearn imports
os.environ['NUMEXPR_MAX_THREADS'] = '64'

import sys
from pathlib import Path

def main():
    """Run standalone feature importance analysis."""
    print("=" * 60)
    print("BCG FEATURE IMPORTANCE ANALYSIS - STANDALONE")
    print("=" * 60)
    
    # Configuration
    print("Please provide the following paths:")
    
    # Model path
    while True:
        model_path = input("Path to trained model (.pth file): ").strip()
        if os.path.exists(model_path):
            break
        print(f"File not found: {model_path}")
    
    # Scaler path (optional)
    scaler_path = input("Path to scaler (.pkl file, or press Enter to skip): ").strip()
    if scaler_path and not os.path.exists(scaler_path):
        print(f"Warning: Scaler file not found: {scaler_path}")
        scaler_path = None
    
    # Test data path
    print("\nTest data options:")
    print("1. evaluation_results.csv from test script")
    print("2. test_features.npz")
    print("3. Custom data file")
    
    while True:
        data_path = input("Path to test data: ").strip()
        if os.path.exists(data_path):
            break
        print(f"File not found: {data_path}")
    
    # Output directory
    output_dir = input("Output directory (default: ./analysis_results): ").strip()
    if not output_dir:
        output_dir = "./analysis_results"
    
    # Model type
    print("\nModel configuration:")
    model_type = input("Model type (deterministic/probabilistic, default: deterministic): ").strip()
    if not model_type:
        model_type = "deterministic"
    
    probabilistic = model_type.lower() == "probabilistic"
    
    # Feature configuration
    print("\nFeature configuration:")
    use_color = input("Does model use color features? (y/N): ").strip().lower() in ['y', 'yes']
    use_auxiliary = input("Does model use auxiliary features (redshift, delta_m)? (y/N): ").strip().lower() in ['y', 'yes']
    
    # Analysis methods
    print("\nAnalysis methods:")
    print("Available: permutation, gradient")
    print("Note: SHAP requires 'pip install shap'")
    
    methods_input = input("Methods (comma-separated, default: permutation,gradient): ").strip()
    if not methods_input:
        methods = ['permutation', 'gradient']
    else:
        methods = [m.strip() for m in methods_input.split(',')]
    
    # Number of samples
    samples_input = input("Number of samples to analyze (default: 1000): ").strip()
    analysis_samples = int(samples_input) if samples_input else 1000
    
    print("\n" + "=" * 60)
    print("RUNNING ANALYSIS")
    print("=" * 60)
    
    # Prepare data if needed
    if data_path.endswith('.csv'):
        print("Converting CSV data to analysis format...")
        from analysis.prepare_analysis_data import convert_evaluation_csv_to_analysis_format
        
        data_dir = Path(data_path).parent
        converted_path = data_dir / "analysis_data.npz"
        
        try:
            convert_evaluation_csv_to_analysis_format(data_path, converted_path)
            data_path = str(converted_path)
            print(f"✓ Data converted: {data_path}")
        except Exception as e:
            print(f"✗ Data conversion failed: {e}")
            print("Continuing with CSV data...")
    
    # Run analysis
    try:
        from analysis.run_analysis import BCGAnalysisRunner
        
        config = {
            'model_path': model_path,
            'data_path': data_path,
            'model_type': model_type,
            'probabilistic_model': probabilistic,
            'output_dir': output_dir,
            'analysis_methods': methods,
            'analysis_samples': analysis_samples,
            'features': {
                'use_color': use_color,
                'use_auxiliary': use_auxiliary,
                'color_pca_components': 8
            }
        }
        
        print("\nStarting BCG feature importance analysis...")
        runner = BCGAnalysisRunner(**config)
        results = runner.run_complete_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nResults saved to: {results['output_directory']}")
        print("\nGenerated files:")
        print(f"📊 Feature rankings: {results['output_directory']}/csv_reports/")
        print(f"📈 Importance plots: {results['output_directory']}/plots/")
        print(f"🔍 Individual explanations: {results['output_directory']}/individual_plots/")
        print(f"📋 Summary report: {results['output_directory']}/analysis_summary.txt")
        
        # Show key insights
        try:
            summary_file = Path(results['output_directory']) / 'analysis_summary.txt'
            if summary_file.exists():
                print(f"\n📋 KEY INSIGHTS (from {summary_file}):")
                with open(summary_file, 'r') as f:
                    lines = f.readlines()
                
                # Find and print top features section
                in_top_features = False
                for line in lines:
                    if 'Top 10 Most Important Features:' in line:
                        in_top_features = True
                        print(line.strip())
                    elif in_top_features and line.strip():
                        if line.startswith('  '):
                            print(line.strip())
                        else:
                            break
        except:
            pass
        
        print(f"\n🎯 To explore results:")
        print(f"   - Open plots in: {results['output_directory']}/plots/")
        print(f"   - Read summary: {results['output_directory']}/analysis_summary.txt")
        print(f"   - Check individual explanations: {results['output_directory']}/individual_plots/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print("\nAnalysis complete!" if success else "\nAnalysis failed - check errors above.")
    sys.exit(0 if success else 1)