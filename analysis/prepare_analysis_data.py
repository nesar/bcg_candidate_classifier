"""
Helper script to prepare data for feature importance analysis.

This script can be used to convert evaluation results to the format
needed by the analysis framework if the test script doesn't automatically
save the required data format.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import argparse


def convert_evaluation_csv_to_analysis_format(csv_path, output_path):
    """
    Convert evaluation results CSV to format needed for analysis.
    
    Args:
        csv_path: Path to evaluation_results.csv
        output_path: Path to save analysis-ready data (.npz)
    """
    print(f"Loading evaluation results from: {csv_path}")
    
    # Load evaluation results
    df = pd.read_csv(csv_path)
    
    # Extract features (assuming they're in the CSV)
    feature_cols = [col for col in df.columns if col.startswith('feature_') 
                   or col in ['patch_mean', 'patch_std', 'concentration', 'eccentricity']]
    
    if not feature_cols:
        # Try to identify feature columns by excluding known non-feature columns
        non_feature_cols = {
            'image_id', 'true_x', 'true_y', 'predicted_x', 'predicted_y',
            'distance_error', 'correct', 'prediction_score', 'prediction_confidence',
            'prediction_class', 'true_class', 'sample_index'
        }
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found in CSV. Unable to prepare analysis data.")
    
    print(f"Found {len(feature_cols)} feature columns")
    
    # Extract features and labels
    X = df[feature_cols].values
    
    # Extract labels (assuming binary classification)
    if 'true_class' in df.columns:
        y = df['true_class'].values
    elif 'correct' in df.columns:
        y = df['correct'].values.astype(int)
    else:
        print("Warning: No labels found. Using dummy labels.")
        y = np.zeros(len(X))
    
    # Save in analysis format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, X=X, y=y, feature_names=feature_cols)
    
    print(f"Analysis data saved to: {output_path}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Features: {feature_cols}")
    
    return output_path


def prepare_analysis_data_from_test_results(test_output_dir, output_file="analysis_data.npz"):
    """
    Prepare analysis data from test results directory.
    
    Args:
        test_output_dir: Directory containing test results
        output_file: Name of output file
        
    Returns:
        Path to created analysis data file
    """
    test_dir = Path(test_output_dir)
    
    # Look for evaluation results
    evaluation_csv = test_dir / "evaluation_results.csv"
    
    if evaluation_csv.exists():
        output_path = test_dir / output_file
        return convert_evaluation_csv_to_analysis_format(evaluation_csv, output_path)
    
    # Look for other data files
    test_features = test_dir / "test_features.npz"
    if test_features.exists():
        print(f"Found existing test features file: {test_features}")
        return test_features
    
    raise FileNotFoundError(f"No suitable data files found in: {test_output_dir}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Prepare data for BCG feature importance analysis"
    )
    
    parser.add_argument('--test_output_dir', type=str, required=True,
                       help='Directory containing test results')
    parser.add_argument('--output_file', type=str, default='analysis_data.npz',
                       help='Output file name (default: analysis_data.npz)')
    
    args = parser.parse_args()
    
    try:
        output_path = prepare_analysis_data_from_test_results(
            args.test_output_dir, 
            args.output_file
        )
        print(f"Analysis data prepared successfully: {output_path}")
        
    except Exception as e:
        print(f"Error preparing analysis data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())