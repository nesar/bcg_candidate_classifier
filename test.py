#!/usr/bin/env python3
"""
Enhanced BCG Classifier Testing Script

This script evaluates trained BCG classifiers with:
1. Uncertainty quantification and probabilistic outputs
2. Detection threshold analysis
3. Enhanced visualizations with probability information
"""

import os
# Fix NUMEXPR warning
os.environ['NUMEXPR_MAX_THREADS'] = '64'

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import maximum_filter, zoom

from data.data_read import prepare_dataframe, BCGDataset
from ml_models.candidate_classifier import BCGCandidateClassifier
from utils.candidate_based_bcg import extract_patch_features, extract_context_features
from utils.viz_bcg import show_failures

# NEW: BCG dataset support
from data.data_read_bcgs import create_bcg_datasets, BCGDataset as NewBCGDataset
from data.candidate_dataset_bcgs import (create_bcg_candidate_dataset_from_loader, 
                                        create_desprior_candidate_dataset_from_files,
                                        collate_bcg_candidate_samples)
from ml_models.uq_classifier import BCGProbabilisticClassifier



# ============================================================================
# Use BCGProbabilisticClassifier from ml_models.uq_classifier
# ============================================================================


# ============================================================================
# ENHANCED PREDICTION FUNCTIONS
# ============================================================================

def predict_bcg_with_probabilities(image, model, feature_scaler=None, 
                                 detection_threshold=0.1, return_all_candidates=False, additional_features=None, **candidate_kwargs):
    """Predict BCG candidates with calibrated probabilities and uncertainty.
    
    Args:
        additional_features: Additional features to append to visual features (e.g., redshift, delta_mstar_z)
    """
    
    # Find candidates using standard method
    from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
    all_candidates, intensities = find_bcg_candidates(image, **candidate_kwargs)
    
    if len(all_candidates) == 0:
        return {
            'best_bcg': None,
            'all_candidates': np.array([]),
            'probabilities': np.array([]),
            'uncertainties': np.array([]),
            'detections': np.array([]),
            'detection_probabilities': np.array([])
        }
    
    features, _ = extract_candidate_features(image, all_candidates)
    
    # Append additional features if provided (e.g., from BCG dataset)
    if additional_features is not None and len(features) > 0:
        # Replicate additional features for each candidate
        additional_features_repeated = np.tile(additional_features, (len(features), 1))
        features = np.concatenate([features, additional_features_repeated], axis=1)
    
    # Scale features
    if feature_scaler is not None:
        scaled_features = feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(scaled_features)
    else:
        features_tensor = torch.FloatTensor(features)
    
    # Get probabilities and uncertainties
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict_with_uncertainty') and hasattr(model, 'temperature'):
            # This is a probabilistic model with UQ trained with ranking loss
            # Debug: Check raw logits first
            raw_logits = model(features_tensor).squeeze(-1)
            print(f"DEBUG: Raw logits range [{torch.min(raw_logits):.4f}, {torch.max(raw_logits):.4f}], mean={torch.mean(raw_logits):.4f}")
            print(f"DEBUG: Temperature: {model.temperature.item():.4f}")
            
            probabilities, uncertainties = model.predict_with_uncertainty(features_tensor)
            probabilities = probabilities.numpy()
            uncertainties = uncertainties.numpy()
            print(f"DEBUG: Final probabilities range [{np.min(probabilities):.10f}, {np.max(probabilities):.10f}]")
            
            # If still getting zeros, try different approach
            if np.max(probabilities) < 1e-6:
                print("DEBUG: Probabilities still near zero, trying raw sigmoid")
                probabilities = torch.sigmoid(raw_logits).numpy()
                uncertainties = np.zeros_like(probabilities)
                print(f"DEBUG: Raw sigmoid range [{np.min(probabilities):.10f}, {np.max(probabilities):.10f}]")
        elif hasattr(model, 'temperature'):
            # This is a probabilistic model without MC dropout - trained with ranking loss
            logits = model.forward_with_temperature(features_tensor).squeeze(-1)
            probabilities = torch.sigmoid(logits).numpy()
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
        else:
            # This is a traditional classifier - use raw scores for ranking, convert to probs for display
            scores = model(features_tensor).squeeze(-1)
            probabilities = torch.sigmoid(scores).numpy()
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
    
    # Find detections above threshold
    detection_mask = probabilities >= detection_threshold
    detections = all_candidates[detection_mask]
    detection_probabilities = probabilities[detection_mask]
    
    # Find best BCG (highest probability)
    if len(probabilities) > 0:
        best_idx = np.argmax(probabilities)
        best_bcg = tuple(all_candidates[best_idx])
    else:
        best_bcg = None
    
    results = {
        'best_bcg': best_bcg,
        'all_candidates': all_candidates,
        'probabilities': probabilities,
        'uncertainties': uncertainties,
        'detections': detections,
        'detection_probabilities': detection_probabilities
    }
    
    return results


# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def show_enhanced_predictions(images, targets, predictions, all_candidates_list, 
                            all_scores_list, all_probabilities_list=None,
                            indices=None, save_dir=None, phase=None, use_uq=False,
                            metadata_list=None, detection_threshold=0.5):
    """Enhanced visualization with probability information, adaptive candidate display, and probability labels."""
    from utils.viz_bcg import show_predictions_with_candidates
    
    # Use the enhanced visualization function from viz_bcg
    show_predictions_with_candidates(
        images=images,
        targets=targets, 
        predictions=predictions,
        all_candidates_list=all_candidates_list,
        candidate_scores_list=all_scores_list,
        indices=indices,
        save_dir=save_dir,
        phase=phase,
        probabilities_list=all_probabilities_list,
        detection_threshold=detection_threshold,
        use_uq=use_uq
    )


def plot_probability_analysis(all_probabilities_list, all_uncertainties_list, 
                            distances, save_dir=None):
    """Plot probability and uncertainty analysis."""
    if not all_probabilities_list or not any(len(p) > 0 for p in all_probabilities_list):
        return
    
    # Collect all probabilities and uncertainties
    all_probs = []
    all_uncs = []
    best_probs = []  # Probability of best candidate
    best_uncs = []   # Uncertainty of best candidate
    
    for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
        if len(probs) > 0:
            all_probs.extend(probs)
            best_probs.append(np.max(probs))
            
            if len(uncs) > 0:
                all_uncs.extend(uncs)
                best_idx = np.argmax(probs)
                best_uncs.append(uncs[best_idx])
    
    if not all_probs:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Probability distribution
    axes[0, 0].hist(all_probs, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('BCG Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of All Candidate Probabilities')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Best candidate probabilities
    axes[0, 1].hist(best_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Best Candidate Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Best Candidate Probabilities')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty analysis
    if all_uncs:
        axes[1, 0].hist(all_uncs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Uncertainty')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of All Candidate Uncertainties')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Probability vs Uncertainty scatter
        if len(best_probs) == len(best_uncs):
            scatter = axes[1, 1].scatter(best_probs, best_uncs, c=distances[:len(best_probs)], 
                                       cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Best Candidate Probability')
            axes[1, 1].set_ylabel('Best Candidate Uncertainty')
            axes[1, 1].set_title('Probability vs Uncertainty (colored by distance error)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Distance Error (pixels)')
    else:
        # If no uncertainties, just show probability vs distance
        if len(best_probs) <= len(distances):
            axes[1, 0].scatter(best_probs, distances[:len(best_probs)], alpha=0.6)
            axes[1, 0].set_xlabel('Best Candidate Probability')
            axes[1, 0].set_ylabel('Distance Error (pixels)')
            axes[1, 0].set_title('Probability vs Distance Error')
            axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].text(0.5, 0.5, 'No uncertainty\ninformation available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Uncertainty Analysis')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'probability_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability analysis saved to: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# MAIN TESTING FUNCTIONS
# ============================================================================

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


def load_trained_model(model_path, scaler_path, feature_dim, use_uq=False):
    """Load trained model and feature scaler."""
    # Load appropriate model type
    if use_uq:
        model = BCGProbabilisticClassifier(feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2)
    else:
        model = BCGCandidateClassifier(feature_dim)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    feature_scaler = joblib.load(scaler_path)
    
    return model, feature_scaler


def evaluate_enhanced_model(model, scaler, test_dataset, candidate_params, 
                          original_dataframe=None, dataset_type='SPT3G_1500d',
                          use_uq=False, detection_threshold=0.1,
                          use_desprior_candidates=False):
    """Evaluate enhanced model with UQ capabilities."""
    print(f"Evaluating {'probabilistic' if use_uq else 'deterministic'} model on {len(test_dataset)} test images...")
    
    predictions = []
    targets = []
    distances = []
    candidate_counts = []
    failed_predictions = []
    all_candidates_list = []
    all_scores_list = []
    test_images = []
    sample_metadata = []
    
    # UQ-specific tracking
    all_probabilities_list = []
    all_uncertainties_list = []
    detection_counts = []
    
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
            cluster_name = filename.replace('.tif', '').split('_')[0]
            metadata['cluster_name'] = cluster_name
            
            cluster_col = 'Cluster name' if 'Cluster name' in original_dataframe.columns else 'cluster_name'
            if cluster_col in original_dataframe.columns:
                matching_rows = original_dataframe[original_dataframe[cluster_col] == cluster_name]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    if 'z' in row:
                        metadata['z'] = row['z']
                    prob_cols = [col for col in row.index if 'prob' in col.lower()]
                    if prob_cols:
                        metadata['bcg_prob'] = row[prob_cols[0]]
        
        # Extract additional features if using BCG data
        additional_features = None
        if args.use_bcg_data and args.use_additional_features:
            if 'additional_features' in sample:
                additional_features = sample['additional_features']
                if hasattr(additional_features, 'numpy'):
                    additional_features = additional_features.numpy()
                elif torch.is_tensor(additional_features):
                    additional_features = additional_features.numpy()
            else:
                # Fallback: extract additional features directly from sample
                if 'cluster_z' in sample and 'delta_mstar_z' in sample:
                    cluster_z = sample['cluster_z']
                    delta_mstar_z = sample['delta_mstar_z']
                    if hasattr(cluster_z, 'numpy'):
                        cluster_z = cluster_z.numpy()
                    elif torch.is_tensor(cluster_z):
                        cluster_z = cluster_z.numpy()
                    if hasattr(delta_mstar_z, 'numpy'):
                        delta_mstar_z = delta_mstar_z.numpy()
                    elif torch.is_tensor(delta_mstar_z):
                        delta_mstar_z = delta_mstar_z.numpy()
                    
                    additional_features = np.array([cluster_z, delta_mstar_z])
        
        # Check if additional features are available for BCG data
        
        # Make prediction with appropriate method
        if use_uq:
            results = predict_bcg_with_probabilities(
                image, model, scaler, 
                detection_threshold=detection_threshold,
                additional_features=additional_features,
                **candidate_params
            )
            
            predicted_bcg = results['best_bcg']
            all_candidates = results['all_candidates']
            scores = results['probabilities']  # These are probabilities, not raw scores
            probabilities = results['probabilities']
            uncertainties = results['uncertainties']
            detections = results['detections']
            
            # Track UQ metrics
            all_probabilities_list.append(probabilities)
            all_uncertainties_list.append(uncertainties)
            detection_counts.append(len(detections))
            
        else:
            # Use traditional method
            if use_desprior_candidates:
                # Use DESprior candidates from BCG dataset
                from data.candidate_dataset_bcgs import create_desprior_candidate_dataset_from_files
                
                # For testing, we need to extract DESprior candidates for this specific image
                # This is a simplified approach - in practice, you'd want to cache this
                # filename is already available from the loop variable
                
                # Import required modules
                import pandas as pd
                from data.data_read_bcgs import BCGDataset
                
                # Load DESprior candidates for this specific image/cluster
                if dataset_type == 'bcg_2p2arcmin':
                    candidates_csv = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_2p2arcmin_clean_matched.csv'
                else:  # bcg_3p8arcmin
                    candidates_csv = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_3p8arcmin_clean_matched.csv'
                
                try:
                    candidates_df = pd.read_csv(candidates_csv)
                    file_candidates = candidates_df[candidates_df['filename'] == filename]
                    
                    if len(file_candidates) == 0:
                        predicted_bcg = None
                        scores = np.array([])
                        all_candidates = np.array([])
                    else:
                        # Extract coordinates and candidate features
                        all_candidates = file_candidates[['x', 'y']].values
                        candidate_specific_features = file_candidates[['delta_mstar', 'starflag']].values
                        
                        # Extract visual features and combine with candidate features
                        from utils.candidate_based_bcg import extract_candidate_features
                        visual_features, _ = extract_candidate_features(image, all_candidates, include_context=True)
                        
                        # Combine visual features with candidate-specific features
                        combined_features = np.hstack([visual_features, candidate_specific_features])
                        
                        # Add additional features if available (redshift, delta_mstar_z)
                        if additional_features is not None:
                            # Replicate additional features for each candidate
                            additional_features_repeated = np.tile(additional_features, (len(combined_features), 1))
                            combined_features = np.concatenate([combined_features, additional_features_repeated], axis=1)
                        
                        if scaler is not None:
                            scaled_features = scaler.transform(combined_features)
                            features_tensor = torch.FloatTensor(scaled_features)
                        else:
                            features_tensor = torch.FloatTensor(combined_features)
                        
                        with torch.no_grad():
                            scores = model(features_tensor).squeeze(-1).numpy()
                        
                        best_idx = np.argmax(scores)
                        predicted_bcg = tuple(all_candidates[best_idx])
                        
                except Exception as e:
                    print(f"Warning: Failed to load DESprior candidates for {filename}: {e}")
                    predicted_bcg = None
                    scores = np.array([])
                    all_candidates = np.array([])
                    
            else:
                from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
                all_candidates, intensities = find_bcg_candidates(image, **candidate_params)
                
                if len(all_candidates) == 0:
                    predicted_bcg = None
                    scores = np.array([])
                else:
                    features, _ = extract_candidate_features(image, all_candidates)
                    
                    if scaler is not None:
                        scaled_features = scaler.transform(features)
                        features_tensor = torch.FloatTensor(scaled_features)
                    else:
                        features_tensor = torch.FloatTensor(features)
                    
                    with torch.no_grad():
                        scores = model(features_tensor).squeeze(-1).numpy()
                    
                    best_idx = np.argmax(scores)
                    predicted_bcg = tuple(all_candidates[best_idx])
            
            # No UQ information available
            probabilities = np.array([])
            uncertainties = np.array([])
            all_probabilities_list.append(probabilities)
            all_uncertainties_list.append(uncertainties)
            detection_counts.append(len(all_candidates) if len(all_candidates) > 0 else 0)
        
        if predicted_bcg is None:
            # No candidates found
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': 'no_candidates',
                'true_bcg': true_bcg
            })
            # Add placeholder entries to maintain list consistency
            predictions.append((np.nan, np.nan))  # Placeholder prediction
            targets.append(true_bcg)  # Still add the true target
            distances.append(np.inf)  # Infinite distance for failed predictions
            candidate_counts.append(0)  # No candidates found
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
        if distance > 100:  # Large error threshold
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
    
    # Add UQ-specific metrics
    if use_uq:
        metrics.update({
            'mean_detections': np.mean(detection_counts) if len(detection_counts) > 0 else 0,
            'detection_threshold': detection_threshold,
            'mean_probability': np.mean([np.mean(p) for p in all_probabilities_list if len(p) > 0]),
            'mean_uncertainty': np.mean([np.mean(u) for u in all_uncertainties_list if len(u) > 0])
        })
    
    return (predictions, targets, distances, failed_predictions, metrics,
            all_candidates_list, all_scores_list, test_images, sample_metadata,
            all_probabilities_list, all_uncertainties_list)


def print_enhanced_evaluation_report(metrics, failed_predictions, use_uq=False):
    """Print detailed evaluation report with UQ information."""
    print("\n" + "="*60)
    print("ENHANCED EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total predictions: {metrics['n_predictions']}")
    print(f"Failed predictions: {metrics['n_failed']}")
    print(f"Average candidates per image: {metrics['mean_candidates']:.1f}")
    
    if use_uq:
        print(f"Average detections per image: {metrics['mean_detections']:.1f}")
        print(f"Detection threshold: {metrics['detection_threshold']:.3f}")
        print(f"Average probability: {metrics.get('mean_probability', 0):.3f}")
        print(f"Average uncertainty: {metrics.get('mean_uncertainty', 0):.3f}")
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
    print("ENHANCED BCG CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Images: {args.image_dir}")
    
    if args.use_uq:
        print(f"Uncertainty quantification: threshold={args.detection_threshold}")
    print()
    
    # Load original truth table for metadata
    print("Loading original truth table...")
    original_df = pd.read_csv(args.truth_table)
    
    if args.use_bcg_data:
        # Use new BCG dataset
        print("Loading new BCG dataset...")
        print(f"Dataset type: {args.bcg_arcmin_type}")
        if args.z_range:
            print(f"Redshift filter: {args.z_range}")
        if args.delta_mstar_z_range:
            print(f"Delta M* z filter: {args.delta_mstar_z_range}")
        
        # Create train and test datasets using the new BCG data reader
        # During testing, we never use RedMapper probabilities as input features
        # (that would be cheating - we want to predict without knowing the answer)
        train_dataset, test_dataset = create_bcg_datasets(
            dataset_type=args.bcg_arcmin_type,
            split_ratio=0.8,  # 80% train, 20% test
            z_range=args.z_range,
            delta_mstar_z_range=args.delta_mstar_z_range,
            include_additional_features=args.use_additional_features,
            include_redmapper_probs=False  # Never use RedMapper probs during testing
        )
        
        # Use test split for evaluation
        test_subset = test_dataset
        dataset = test_dataset  # Set dataset for feature dimension analysis
        print(f"Found {len(test_dataset)} samples in test split")
    else:
        # Load processed dataset (original approach)
        print("Loading processed dataset...")
        dataframe = prepare_dataframe(args.image_dir, args.truth_table, args.dataset_type)
        print(f"Found {len(dataframe)} samples in dataset")
        
        # Create BCG dataset
        dataset = BCGDataset(args.image_dir, dataframe)
        
        # Split dataset (use same random seed as training)
        train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
        print(f"Using test split: {len(test_subset)} samples")
    
    # Determine feature dimension by analyzing a sample
    # This ensures we get the correct dimension regardless of options
    print("Determining feature dimension from a sample...")
    sample_image = dataset[0]['image']
    if hasattr(sample_image, 'numpy'):
        sample_image = sample_image.numpy()
    elif torch.is_tensor(sample_image):
        sample_image = sample_image.numpy()
    
    # Get candidate parameters for feature extraction
    candidate_params_sample = {
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'exclude_border': args.exclude_border,
        'max_candidates': args.max_candidates
    }
    
    # Determine feature dimension from sample data
    print("Determining feature dimension from sample data...")
    
    if args.use_desprior_candidates:
        # For DESprior candidates: extract actual features to get correct dimension
        try:
            if args.dataset_type == 'bcg_2p2arcmin':
                candidates_csv = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_2p2arcmin_clean_matched.csv'
            else:
                candidates_csv = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_3p8arcmin_clean_matched.csv'
            
            candidates_df = pd.read_csv(candidates_csv)
            first_file = candidates_df['filename'].iloc[0]
            file_candidates = candidates_df[candidates_df['filename'] == first_file]
            
            if len(file_candidates) > 0:
                candidates = file_candidates[['x', 'y']].values
                candidate_features = file_candidates[['delta_mstar', 'starflag']].values
                
                # Extract visual features
                from utils.candidate_based_bcg import extract_candidate_features
                visual_features, _ = extract_candidate_features(sample_image, candidates, include_context=True)
                combined_features = np.hstack([visual_features, candidate_features])
                base_feature_dim = combined_features.shape[1]
                print(f"Determined DESprior feature dimension: {base_feature_dim}")
            else:
                base_feature_dim = 32  # Fallback
        except Exception as e:
            print(f"Warning: Could not determine DESprior feature dim: {e}")
            base_feature_dim = 32  # Fallback
    else:
        # For regular candidate detection
        from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
        candidates, _ = find_bcg_candidates(sample_image, **candidate_params_sample)
        if len(candidates) > 0:
            features, _ = extract_candidate_features(sample_image, candidates, include_context=True)
            base_feature_dim = features.shape[1] if len(features) > 0 else 30
        else:
            base_feature_dim = 30  # Default for single-scale
    
    # Adjust feature dimension for BCG dataset additional features
    if args.use_bcg_data and args.use_additional_features:
        print(f"Base feature dimension: {base_feature_dim}")
        print("Adding additional features from BCG dataset: +2 (redshift, delta_mstar_z)")
        base_feature_dim += 2
    
    print(f"Final feature dimension: {base_feature_dim}")
    
    # Load trained model
    print("Loading trained model...")
    model, scaler = load_trained_model(args.model_path, args.scaler_path, 
                                     base_feature_dim, use_uq=args.use_uq)
    
    # Set up candidate parameters
    candidate_params = {
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'exclude_border': args.exclude_border,
        'max_candidates': args.max_candidates
    }
    
    
    # Evaluate model
    results = evaluate_enhanced_model(
        model, scaler, test_subset, candidate_params, original_df, args.dataset_type,
        use_uq=args.use_uq, 
        detection_threshold=args.detection_threshold,
        use_desprior_candidates=args.use_desprior_candidates
    )
    
    (predictions, targets, distances, failures, metrics, 
     all_candidates_list, all_scores_list, test_images, sample_metadata,
     all_probabilities_list, all_uncertainties_list) = results
    
    # Print results
    print_enhanced_evaluation_report(metrics, failures, use_uq=args.use_uq)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # UQ-specific analysis
    if args.use_uq and len(predictions) > 0:
        print("\nGenerating probability and uncertainty analysis...")
        plot_probability_analysis(all_probabilities_list, all_uncertainties_list, 
                                distances, save_dir=args.output_dir)
    
    # Show sample predictions with enhanced visualization
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
        sample_metadata_list = [sample_metadata[i] for i in sorted_indices]
        
        # Get probabilities if available
        if args.use_uq:
            sample_probabilities = [all_probabilities_list[i] for i in sorted_indices]
        else:
            sample_probabilities = None
        
        phase_name = "EnhancedTesting"
        if args.use_uq:
            phase_name = "ProbabilisticTesting"
        
        show_enhanced_predictions(
            sample_images, sample_targets, sample_predictions,
            sample_candidates, sample_scores, sample_probabilities,
            indices=range(len(sample_images)),
            save_dir=args.output_dir,
            phase=phase_name,
            use_uq=args.use_uq,
            metadata_list=sample_metadata_list,
            detection_threshold=args.detection_threshold
        )
    
    # Show failure cases
    if args.show_failures and len(distances) > 0:
        print(f"\nShowing worst prediction failures...")
        
        # Handle both finite and infinite distances for failures
        finite_distances = np.array(distances)
        finite_mask = np.isfinite(finite_distances)
        infinite_mask = ~finite_mask
        
        failure_indices = []
        
        # First, add cases with infinite distances (complete failures)
        infinite_indices = np.where(infinite_mask)[0]
        failure_indices.extend(infinite_indices[:args.show_failures])
        
        # Then add worst finite distances if we need more samples
        if len(failure_indices) < args.show_failures and np.any(finite_mask):
            finite_indices = np.where(finite_mask)[0]
            finite_distances_only = finite_distances[finite_mask]
            worst_finite = np.argsort(finite_distances_only)[-max(0, args.show_failures - len(failure_indices)):]
            failure_indices.extend(finite_indices[worst_finite])
        
        # Limit to requested number
        failure_indices = failure_indices[:args.show_failures]
        
        if len(failure_indices) > 0:
            failure_images = [test_images[i] for i in failure_indices]
            failure_targets = [targets[i] for i in failure_indices]
            failure_predictions = [predictions[i] for i in failure_indices]
            failure_metadata_list = [sample_metadata[i] for i in failure_indices]
            failure_candidates_list = [all_candidates_list[i] for i in failure_indices]
            failure_scores_list = [all_scores_list[i] for i in failure_indices]
            failure_probabilities_list = [all_probabilities_list[i] for i in failure_indices] if all_probabilities_list else None
        
            phase_name = "CandidateBasedTesting"
            if args.use_uq:
                phase_name = "ProbabilisticTesting"
            
            show_failures(
                failure_images, failure_targets, failure_predictions,
                threshold=20, max_failures=args.show_failures,
                save_dir=args.output_dir,
                phase=phase_name,
                metadata_list=failure_metadata_list,
                all_candidates_list=failure_candidates_list,
                candidate_scores_list=failure_scores_list,
                probabilities_list=failure_probabilities_list,
                detection_threshold=args.detection_threshold,
                use_uq=args.use_uq
            )
        else:
            print("No failure cases to show")
    
    # Save detailed results
    if args.save_results and len(predictions) > 0:
        results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
        
        # Create enhanced results dictionary
        results_data = {
            'pred_x': [pred[0] for pred in predictions],
            'pred_y': [pred[1] for pred in predictions], 
            'true_x': [target[0] for target in targets],
            'true_y': [target[1] for target in targets],
            'distance_error': distances,
            'n_candidates': [len(cand) for cand in all_candidates_list]
        }
        
        # Add UQ-specific columns
        if args.use_uq:
            max_probabilities = []
            avg_probabilities = []
            max_uncertainties = []
            avg_uncertainties = []
            n_detections = []
            
            for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
                if len(probs) > 0:
                    max_probabilities.append(np.max(probs))
                    avg_probabilities.append(np.mean(probs))
                    n_detections.append(np.sum(probs >= args.detection_threshold))
                else:
                    max_probabilities.append(np.nan)
                    avg_probabilities.append(np.nan)
                    n_detections.append(0)
                
                if len(uncs) > 0:
                    max_uncertainties.append(np.max(uncs))
                    avg_uncertainties.append(np.mean(uncs))
                else:
                    max_uncertainties.append(np.nan)
                    avg_uncertainties.append(np.nan)
            
            results_data.update({
                'max_probability': max_probabilities,
                'avg_probability': avg_probabilities,
                'max_uncertainty': max_uncertainties,
                'avg_uncertainty': avg_uncertainties,
                'n_detections': n_detections,
                'detection_threshold': [args.detection_threshold] * len(predictions)
            })
        
        # Add metadata columns
        if sample_metadata:
            cluster_names = [meta.get('cluster_name', 'unknown') for meta in sample_metadata]
            results_data['cluster_name'] = cluster_names
            
            if any('z' in meta for meta in sample_metadata):
                redshifts = [meta.get('z', np.nan) for meta in sample_metadata]
                results_data['z'] = redshifts
            
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
        
        # Add coordinate and error columns
        cols.extend(['pred_x', 'pred_y', 'true_x', 'true_y', 'distance_error'])
        
        # Add UQ columns
        if args.use_uq:
            cols.extend(['max_probability', 'avg_probability', 'n_detections', 
                        'detection_threshold', 'max_uncertainty', 'avg_uncertainty'])
        
        # Add remaining columns
        cols.extend(['n_candidates'])
        
        # Only include columns that exist
        cols = [col for col in cols if col in results_df.columns]
        results_df = results_df[cols]
        
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Results include {len(results_df.columns)} columns: {', '.join(results_df.columns)}")
        
        # Save UQ-specific analysis if enabled
        if args.use_uq:
            uq_analysis_file = os.path.join(args.output_dir, 'probability_analysis.csv')
            
            # Compile probability analysis
            prob_analysis_data = []
            for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
                sample_name = sample_metadata[i].get('cluster_name', f'sample_{i}') if i < len(sample_metadata) else f'sample_{i}'
                
                for j, prob in enumerate(probs):
                    unc = uncs[j] if j < len(uncs) else np.nan
                    is_detection = prob >= args.detection_threshold
                    is_best = j == np.argmax(probs) if len(probs) > 0 else False
                    
                    prob_analysis_data.append({
                        'sample_name': sample_name,
                        'candidate_idx': j,
                        'probability': prob,
                        'uncertainty': unc,
                        'is_detection': is_detection,
                        'is_best_candidate': is_best,
                        'distance_error': distances[i] if i < len(distances) else np.nan
                    })
            
            if prob_analysis_data:
                prob_df = pd.DataFrame(prob_analysis_data)
                prob_df.to_csv(uq_analysis_file, index=False)
                print(f"Probability analysis saved to: {uq_analysis_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Enhanced BCG Classifier")
    
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
                       choices=['SPT3G_1500d', 'megadeep500', 'bcg_2p2arcmin', 'bcg_3p8arcmin'],
                       help='Type of dataset')
    
    # Candidate finding arguments (should match training)
    parser.add_argument('--min_distance', type=int, default=8,
                       help='Minimum distance between candidates (reduced for higher precision)')
    parser.add_argument('--threshold_rel', type=float, default=0.1,
                       help='Relative threshold for candidate detection (lowered for more candidates)')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=50,
                       help='Maximum candidates per image (increased for better coverage)')
    
    
    parser.add_argument('--use_uq', action='store_true',
                       help='Enable uncertainty quantification with probabilistic outputs')
    parser.add_argument('--detection_threshold', type=float, default=0.1,
                       help='Probability threshold for BCG detection (0.0-1.0)')
    
    # BCG dataset arguments
    parser.add_argument('--use_bcg_data', action='store_true',
                       help='Use new BCG dataset (2.2 or 3.8 arcmin)')
    parser.add_argument('--bcg_arcmin_type', type=str, default='2p2arcmin',
                       choices=['2p2arcmin', '3p8arcmin'],
                       help='BCG image scale (2.2 or 3.8 arcmin)')
    parser.add_argument('--z_range', type=str, default=None,
                       help='Redshift filter range as "z_min,z_max" (e.g. "0.3,0.7")')
    parser.add_argument('--delta_mstar_z_range', type=str, default=None,
                       help='Delta M* z filter range as "min,max" (e.g. "-2.0,-1.0")')
    parser.add_argument('--use_additional_features', action='store_true',
                       help='Include redshift and delta_mstar_z as additional features')
    parser.add_argument('--use_redmapper_probs', action='store_true',
                       help='Load RedMapper BCG probabilities for evaluation (not used as input features)')
    parser.add_argument('--use_desprior_candidates', action='store_true',
                       help='Use DESprior candidates instead of automatic detection')
    parser.add_argument('--candidate_delta_mstar_range', type=str, default=None,
                       help='Filter DESprior candidates by delta_mstar range as "min,max"')
    
    # Visualization arguments
    parser.add_argument('--show_samples', type=int, default=5,
                       help='Number of sample predictions to visualize')
    parser.add_argument('--show_failures', type=int, default=20,
                       help='Number of failure cases to visualize')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to CSV')
    
    args = parser.parse_args()
    
    
    # Validate detection threshold
    if args.use_uq:
        args.detection_threshold = max(0.0, min(1.0, args.detection_threshold))
    
    # Parse BCG dataset arguments
    if args.use_bcg_data:
        # Parse range arguments
        if args.z_range:
            z_min, z_max = map(float, args.z_range.split(','))
            args.z_range = (z_min, z_max)
        
        if args.delta_mstar_z_range:
            delta_min, delta_max = map(float, args.delta_mstar_z_range.split(','))
            args.delta_mstar_z_range = (delta_min, delta_max)
        
        if args.candidate_delta_mstar_range:
            delta_min, delta_max = map(float, args.candidate_delta_mstar_range.split(','))
            args.candidate_delta_mstar_range = (delta_min, delta_max)
    
    main(args)