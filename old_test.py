#!/usr/bin/env python3
"""
Candidate-Based BCG Classifier Testing Script

This script evaluates trained candidate-based BCG classifiers and
provides detailed analysis and visualizations of predictions.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from data.data_read import prepare_dataframe, BCGDataset
from ml_models.candidate_classifier import BCGCandidateClassifier
from utils.candidate_based_bcg import predict_bcg_from_candidates
from utils.viz_bcg import show_predictions_with_candidates, show_failures


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, random_seed=42):
    """Split dataset into train/validation/test sets (same as training)."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    n_samples = len(dataset)
    indices = torch.randperm(n_samples).tolist()
    
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def load_trained_model(model_path, scaler_path, feature_dim):
    """Load trained model and feature scaler."""
    # Load model
    model = BCGCandidateClassifier(feature_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    feature_scaler = joblib.load(scaler_path)
    
    return model, feature_scaler


def evaluate_model(model, scaler, test_dataset, candidate_params, original_dataframe=None, dataset_type='SPT3G_1500d'):
    """Evaluate model on test set and compute metrics."""
    print(f"Evaluating model on {len(test_dataset)} test images...")
    
    predictions = []
    targets = []
    distances = []
    candidate_counts = []
    failed_predictions = []
    all_candidates_list = []
    all_scores_list = []
    test_images = []
    sample_metadata = []  # Store additional info from truth table
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        image = sample['image']
        true_bcg = sample['BCG']
        filename = sample.get('filename', f'sample_{i}')
        
        # Store image for visualization
        test_images.append(image)
        
        # Extract metadata from original dataframe if available
        metadata = {'filename': filename}
        if original_dataframe is not None:
            # Extract cluster name (remove .tif extension and any suffix)
            cluster_name = filename.replace('.tif', '').split('_')[0]
            metadata['cluster_name'] = cluster_name
            
            # Find corresponding row in original dataframe by cluster name
            cluster_col = 'Cluster name' if 'Cluster name' in original_dataframe.columns else 'cluster_name'
            if cluster_col in original_dataframe.columns:
                matching_rows = original_dataframe[original_dataframe[cluster_col] == cluster_name]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    
                    # Add redshift if available
                    if 'z' in row:
                        metadata['z'] = row['z']
                    
                    # Look for BCG probability columns (various possible names)
                    prob_cols = [col for col in row.index if 'prob' in col.lower()]
                    if prob_cols:
                        metadata['bcg_prob'] = row[prob_cols[0]]
        
        # Make prediction
        predicted_bcg, all_candidates, scores = predict_bcg_from_candidates(
            image, model, scaler, **candidate_params
        )
        
        if predicted_bcg is None:
            # No candidates found
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': 'no_candidates',
                'true_bcg': true_bcg
            })
            # Add empty entries to maintain list consistency
            all_candidates_list.append(np.array([]).reshape(0, 2))
            all_scores_list.append(np.array([]))
            sample_metadata.append(metadata)
            continue
        
        # Compute distance error
        distance = np.sqrt(np.sum((np.array(predicted_bcg) - true_bcg)**2))
        distances.append(distance)
        candidate_counts.append(len(all_candidates))
        
        # Store results
        predictions.append(predicted_bcg)
        targets.append(true_bcg)
        all_candidates_list.append(all_candidates)
        all_scores_list.append(scores)
        sample_metadata.append(metadata)
        
        # Check for potential failure cases
        if distance > 50:  # Large error threshold
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': 'large_error',
                'predicted': predicted_bcg,
                'true_bcg': true_bcg,
                'distance': distance,
                'candidates': all_candidates,
                'scores': scores
            })
    
    # Compute metrics
    distances = np.array(distances)
    success_rates = {}
    
    for threshold in [10, 20, 30, 50]:
        success_rate = np.mean(distances <= threshold) if len(distances) > 0 else 0
        success_rates[f'success_rate_{threshold}px'] = success_rate
    
    metrics = {
        'n_predictions': len(predictions),
        'n_failed': len(failed_predictions),
        'mean_distance': np.mean(distances) if len(distances) > 0 else float('inf'),
        'median_distance': np.median(distances) if len(distances) > 0 else float('inf'),
        'std_distance': np.std(distances) if len(distances) > 0 else 0,
        'min_distance': np.min(distances) if len(distances) > 0 else float('inf'),
        'max_distance': np.max(distances) if len(distances) > 0 else 0,
        'mean_candidates': np.mean(candidate_counts) if len(candidate_counts) > 0 else 0,
        **success_rates
    }
    
    return (predictions, targets, distances, failed_predictions, metrics,
            all_candidates_list, all_scores_list, test_images, sample_metadata)


def print_evaluation_report(metrics, failed_predictions):
    """Print detailed evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total predictions: {metrics['n_predictions']}")
    print(f"Failed predictions: {metrics['n_failed']}")
    print(f"Average candidates per image: {metrics['mean_candidates']:.1f}")
    print()
    
    if metrics['n_predictions'] > 0:
        print("Distance Metrics:")
        print(f"  Mean error: {metrics['mean_distance']:.2f} pixels")
        print(f"  Median error: {metrics['median_distance']:.2f} pixels")
        print(f"  Std deviation: {metrics['std_distance']:.2f} pixels")
        print(f"  Min error: {metrics['min_distance']:.2f} pixels")
        print(f"  Max error: {metrics['max_distance']:.2f} pixels")
        print()
        
        print("Success Rates:")
        for key, value in metrics.items():
            if 'success_rate' in key:
                threshold = key.split('_')[-1]
                print(f"  Within {threshold}: {value*100:.1f}%")
        print()
    
    if failed_predictions:
        print("Failed Prediction Analysis:")
        failure_reasons = {}
        for failure in failed_predictions:
            reason = failure['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} cases")


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("CANDIDATE-BASED BCG CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Images: {args.image_dir}")
    print()
    
    # Load original truth table for metadata
    print("Loading original truth table...")
    original_df = pd.read_csv(args.truth_table)
    
    # Load processed dataset
    print("Loading processed dataset...")
    dataframe = prepare_dataframe(args.image_dir, args.truth_table, args.dataset_type)
    print(f"Found {len(dataframe)} samples in dataset")
    
    # Create BCG dataset
    dataset = BCGDataset(args.image_dir, dataframe)
    
    # Split dataset (use same random seed as training)
    train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
    print(f"Using test split: {len(test_subset)} samples")
    
    # Get feature dimension from a dummy sample to load model
    # For now, we'll use a fixed feature dimension (this should match training)
    feature_dim = 30  # This matches the feature extraction in candidate_based_bcg.py
    
    # Load trained model
    print("Loading trained model...")
    model, scaler = load_trained_model(args.model_path, args.scaler_path, feature_dim)
    
    # Set up candidate parameters
    candidate_params = {
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'exclude_border': args.exclude_border,
        'max_candidates': args.max_candidates
    }
    
    # Evaluate model
    (predictions, targets, distances, failures, metrics, 
     all_candidates_list, all_scores_list, test_images, sample_metadata) = evaluate_model(
        model, scaler, test_subset, candidate_params, original_df, args.dataset_type
    )
    
    # Print results
    print_evaluation_report(metrics, failures)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Show sample predictions
    if args.show_samples > 0 and len(predictions) > 0:
        print(f"\nShowing {min(args.show_samples, len(predictions))} sample predictions...")
        
        # Sort by distance for best examples
        sorted_indices = np.argsort(distances)[:args.show_samples]
        
        # Prepare data for visualization
        sample_images = [test_images[i] for i in sorted_indices]
        sample_targets = [targets[i] for i in sorted_indices]
        sample_predictions = [predictions[i] for i in sorted_indices]
        sample_candidates = [all_candidates_list[i] for i in sorted_indices]
        sample_scores = [all_scores_list[i] for i in sorted_indices]
        
        show_predictions_with_candidates(
            sample_images, sample_targets, sample_predictions,
            sample_candidates, sample_scores,
            indices=range(len(sample_images)),
            save_dir=args.output_dir,
            phase="CandidateBasedTesting"
        )
    
    # Show failure cases
    if args.show_failures and len(distances) > 0:
        print(f"\nShowing worst prediction failures...")
        
        # Get worst cases
        worst_indices = np.argsort(distances)[-args.show_failures:]
        
        failure_images = [test_images[i] for i in worst_indices]
        failure_targets = [targets[i] for i in worst_indices]
        failure_predictions = [predictions[i] for i in worst_indices]
        
        show_failures(
            failure_images, failure_targets, failure_predictions,
            threshold=20, max_failures=args.show_failures,
            save_dir=args.output_dir,
            phase="CandidateBasedTesting"
        )
    
    # Save detailed results
    if args.save_results and len(predictions) > 0:
        results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
        
        # Create base results dictionary
        results_data = {
            'pred_x': [pred[0] for pred in predictions],
            'pred_y': [pred[1] for pred in predictions], 
            'true_x': [target[0] for target in targets],
            'true_y': [target[1] for target in targets],
            'distance_error': distances,
            'n_candidates': [len(cand) for cand in all_candidates_list]
        }
        
        # Add metadata columns
        if sample_metadata:
            # Extract cluster names
            cluster_names = [meta.get('cluster_name', 'unknown') for meta in sample_metadata]
            results_data['cluster_name'] = cluster_names
            
            # Extract redshifts if available
            if any('z' in meta for meta in sample_metadata):
                redshifts = [meta.get('z', np.nan) for meta in sample_metadata]
                results_data['z'] = redshifts
            
            # Extract BCG probabilities if available
            if any('bcg_prob' in meta for meta in sample_metadata):
                bcg_probs = [meta.get('bcg_prob', np.nan) for meta in sample_metadata]
                results_data['bcg_prob'] = bcg_probs
        
        results_df = pd.DataFrame(results_data)
        
        # Reorder columns to put metadata first
        cols = ['cluster_name'] if 'cluster_name' in results_df.columns else []
        if 'z' in results_df.columns:
            cols.append('z')
        if 'bcg_prob' in results_df.columns:
            cols.append('bcg_prob')
        cols.extend(['pred_x', 'pred_y', 'true_x', 'true_y', 'distance_error', 'n_candidates'])
        
        # Only include columns that exist
        cols = [col for col in cols if col in results_df.columns]
        results_df = results_df[cols]
        
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Results include {len(results_df.columns)} columns: {', '.join(results_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Candidate-Based BCG Classifier")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--scaler_path', type=str, required=True,
                       help='Path to feature scaler (.pkl file)')
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing .tif image files')
    parser.add_argument('--truth_table', type=str, required=True,
                       help='Path to CSV file with BCG coordinates')
    parser.add_argument('--dataset_type', type=str, default='SPT3G_1500d',
                       choices=['SPT3G_1500d', 'megadeep500'],
                       help='Type of dataset')
    
    # Candidate finding arguments (should match training)
    parser.add_argument('--min_distance', type=int, default=15,
                       help='Minimum distance between candidates')
    parser.add_argument('--threshold_rel', type=float, default=0.12,
                       help='Relative threshold for candidate detection')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=25,
                       help='Maximum candidates per image')
    
    # Visualization arguments
    parser.add_argument('--show_samples', type=int, default=5,
                       help='Number of sample predictions to visualize')
    parser.add_argument('--show_failures', type=int, default=3,
                       help='Number of failure cases to visualize')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to CSV')
    
    args = parser.parse_args()
    main(args)