#!/usr/bin/env python3
"""
Enhanced BCG Classifier Testing Script

This script evaluates trained BCG classifiers with:
1. Uncertainty quantification and probabilistic outputs
2. Detection threshold analysis
3. Enhanced visualizations with probability information
"""

import os
# Fix threading issues for HPC systems - set before any numpy/sklearn imports
os.environ['NUMEXPR_MAX_THREADS'] = '128'
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent numpy threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threading limit

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

def calculate_bcg_rank(true_bcg, all_candidates, scores, distance_threshold=10.0):
    """Calculate the rank of the true BCG among all candidates.
    
    Args:
        true_bcg: True BCG coordinates (x, y)
        all_candidates: Array of candidate coordinates
        scores: Scores/probabilities for each candidate
        distance_threshold: Distance threshold to consider a candidate as matching the true BCG
    
    Returns:
        rank: 1-indexed rank of true BCG (1 = best, 2 = second best, etc.)
              Returns None if true BCG not found within threshold
    """
    if len(all_candidates) == 0 or len(scores) == 0:
        return None
    
    # Calculate distances from each candidate to true BCG
    distances_to_true = np.sqrt(np.sum((all_candidates - np.array(true_bcg))**2, axis=1))
    
    # Find candidates within distance threshold of true BCG
    matching_candidates = distances_to_true <= distance_threshold
    
    if not np.any(matching_candidates):
        return None  # True BCG not found among candidates
    
    # Get the best matching candidate (closest to true BCG)
    best_match_idx = np.argmin(distances_to_true)
    
    # Sort candidates by score (descending order)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Find rank of the best matching candidate
    rank = np.where(sorted_indices == best_match_idx)[0][0] + 1  # 1-indexed
    
    return rank

def predict_bcg_with_probabilities(image, model, feature_scaler=None, 
                                 detection_threshold=0.1, return_all_candidates=False, additional_features=None, 
                                 use_desprior_candidates=False, filename=None, dataset_type=None, 
                                 use_color_features=False, color_extractor=None, desprior_csv_path=None, **candidate_kwargs):
    """Predict BCG candidates with calibrated probabilities and uncertainty.
    
    Args:
        additional_features: Additional features to append to visual features (e.g., redshift, delta_mstar_z)
        use_desprior_candidates: Whether to use DESprior candidates instead of automatic detection
        filename: Image filename (required if use_desprior_candidates=True)
        dataset_type: Dataset type (required if use_desprior_candidates=True)
    """
    
    # Find candidates using appropriate method
    if use_desprior_candidates:
        # Use DESprior candidates from BCG dataset
        if desprior_csv_path is None:
            raise ValueError("desprior_csv_path must be provided when use_desprior_candidates=True")
        
        try:
            candidates_df = pd.read_csv(desprior_csv_path)
            file_candidates = candidates_df[candidates_df['filename'] == filename]
            
            if len(file_candidates) == 0:
                all_candidates = np.array([])
            else:
                # Extract coordinates and candidate features
                all_candidates = file_candidates[['x', 'y']].values
                candidate_specific_features = file_candidates[['delta_mstar', 'starflag']].values
                
                # Extract visual features and combine with candidate features
                from utils.candidate_based_bcg import extract_candidate_features
                visual_features, _ = extract_candidate_features(
                    image, all_candidates, patch_size=candidate_kwargs.get('patch_size', 64), 
                    include_context=True, include_color=use_color_features, 
                    color_extractor=color_extractor
                )
                
                # Combine visual features with candidate-specific features
                features = np.hstack([visual_features, candidate_specific_features])
                
                # NOTE: DESprior candidates already include all necessary features including additional features
                # Do not add additional features again as they are already included in the feature extraction
                
        except Exception as e:
            print(f"Warning: Failed to load DESprior candidates for {filename}: {e}")
            all_candidates = np.array([])
            
    else:
        # Use automatic candidate detection
        from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
        all_candidates, intensities = find_bcg_candidates(image, **candidate_kwargs)
        
        if len(all_candidates) > 0:
            features, _ = extract_candidate_features(
                image, all_candidates, patch_size=candidate_kwargs.get('patch_size', 64),
                include_context=True, include_color=use_color_features, 
                color_extractor=color_extractor
            )
            
            # Append additional features if provided (e.g., from BCG dataset)
            if additional_features is not None and len(features) > 0:
                # Replicate additional features for each candidate
                additional_features_repeated = np.tile(additional_features, (len(features), 1))
                features = np.concatenate([features, additional_features_repeated], axis=1)
    
    if len(all_candidates) == 0:
        return {
            'best_bcg': None,
            'all_candidates': np.array([]),
            'probabilities': np.array([]),
            'uncertainties': np.array([]),
            'detections': np.array([]),
            'detection_probabilities': np.array([]),
            'best_features': None  # No features when no candidates
        }
    
    # Scale features
    if feature_scaler is not None:
        scaled_features = feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(scaled_features)
    else:
        scaled_features = features
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
            
            # Ensure arrays are at least 1D
            probabilities = np.atleast_1d(probabilities)
            uncertainties = np.atleast_1d(uncertainties)
            
            print(f"DEBUG: Final probabilities range [{np.min(probabilities):.10f}, {np.max(probabilities):.10f}]")
            
            # If still getting zeros, try different approach
            if np.max(probabilities) < 1e-6:
                print("DEBUG: Probabilities still near zero, trying raw sigmoid")
                probabilities = np.atleast_1d(torch.sigmoid(raw_logits).numpy())
                uncertainties = np.zeros_like(probabilities)
                print(f"DEBUG: Raw sigmoid range [{np.min(probabilities):.10f}, {np.max(probabilities):.10f}]")
        elif hasattr(model, 'temperature'):
            # This is a probabilistic model without MC dropout - trained with ranking loss
            logits = model.forward_with_temperature(features_tensor).squeeze(-1)
            probabilities = np.atleast_1d(torch.sigmoid(logits).numpy())
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
        else:
            # This is a traditional classifier - use raw scores for ranking, convert to probs for display
            scores = model(features_tensor).squeeze(-1)
            probabilities = np.atleast_1d(torch.sigmoid(scores).numpy())
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
    
    # Find detections above threshold
    detection_mask = probabilities >= detection_threshold
    detections = all_candidates[detection_mask]
    detection_probabilities = probabilities[detection_mask]
    
    # Find best BCG (highest probability)
    if len(probabilities) > 0:
        best_idx = np.argmax(probabilities)
        best_bcg = tuple(all_candidates[best_idx])
        # FEATURE ANALYSIS: Get best candidate's features
        best_features = scaled_features[best_idx]
    else:
        best_bcg = None
        best_features = None
    
    results = {
        'best_bcg': best_bcg,
        'all_candidates': all_candidates,
        'probabilities': probabilities,
        'uncertainties': uncertainties,
        'detections': detections,
        'detection_probabilities': detection_probabilities,
        'best_features': best_features  # Add features to return
    }
    
    return results


# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def show_enhanced_predictions(images, targets, predictions, all_candidates_list, 
                            all_scores_list, all_probabilities_list=None,
                            indices=None, save_dir=None, phase=None, use_uq=False,
                            metadata_list=None, detection_threshold=0.5, dataset_type="bcg_2p2arcmin"):
    """Enhanced visualization with probability information, adaptive candidate display, and probability labels."""
    from utils.viz_bcg import show_predictions_with_candidates, show_predictions_with_candidates_enhanced
    
    # Use both original and enhanced visualization functions
    # First create original plots
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
        use_uq=use_uq,
        metadata_list=metadata_list
    )
    
    # Then create enhanced plots in physical_images subdirectory
    show_predictions_with_candidates_enhanced(
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
        use_uq=use_uq,
        metadata_list=metadata_list,
        dataset_type=dataset_type
    )


def plot_probability_analysis(all_probabilities_list, all_uncertainties_list, 
                            distances, save_dir=None):
    """Plot probability and uncertainty analysis."""
    if not all_probabilities_list or not any(len(p) > 0 for p in all_probabilities_list):
        return
    
    # Set style consistent with plot_physical_results.py
    plt.rcParams.update({"text.usetex":False,"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":1.2})
    
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
    axes[0, 0].set_xlabel('BCG Probability', fontsize=18)
    axes[0, 0].set_ylabel('Count', fontsize=18)
    axes[0, 0].set_title('Distribution of All Candidate Probabilities', fontsize=18)
    axes[0, 0].tick_params(axis='both', labelsize=18)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Best candidate probabilities
    axes[0, 1].hist(best_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Best Candidate Probability', fontsize=18)
    axes[0, 1].set_ylabel('Count', fontsize=18)
    axes[0, 1].set_title('Distribution of Best Candidate Probabilities', fontsize=18)
    axes[0, 1].tick_params(axis='both', labelsize=18)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty analysis
    if all_uncs:
        axes[1, 0].hist(all_uncs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Uncertainty', fontsize=18)
        axes[1, 0].set_ylabel('Count', fontsize=18)
        axes[1, 0].set_title('Distribution of All Candidate Uncertainties', fontsize=18)
        axes[1, 0].tick_params(axis='both', labelsize=18)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Probability vs Uncertainty scatter
        if len(best_probs) == len(best_uncs):
            scatter = axes[1, 1].scatter(best_probs, best_uncs, c=distances[:len(best_probs)], 
                                       cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Best Candidate Probability', fontsize=18)
            axes[1, 1].set_ylabel('Best Candidate Uncertainty', fontsize=18)
            axes[1, 1].set_title('Probability vs Uncertainty (colored by distance error)', fontsize=18)
            axes[1, 1].tick_params(axis='both', labelsize=18)
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Distance Error (pixels)')
    else:
        # If no uncertainties, just show probability vs distance
        if len(best_probs) <= len(distances):
            axes[1, 0].scatter(best_probs, distances[:len(best_probs)], alpha=0.6)
            axes[1, 0].set_xlabel('Best Candidate Probability', fontsize=18)
            axes[1, 0].set_ylabel('Distance Error (pixels)', fontsize=18)
            axes[1, 0].set_title('Probability vs Distance Error', fontsize=18)
            axes[1, 0].tick_params(axis='both', labelsize=18)
            axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].text(0.5, 0.5, 'No uncertainty\ninformation available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=18)
        axes[1, 1].set_title('Uncertainty Analysis', fontsize=18)
    
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


def load_trained_model(model_path, scaler_path, feature_dim, use_uq=False, use_color_features=False):
    """Load trained model, feature scaler, and optional color extractor."""
    # Load appropriate model type
    if use_uq:
        model = BCGProbabilisticClassifier(feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2)
    else:
        model = BCGCandidateClassifier(feature_dim)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    feature_scaler = joblib.load(scaler_path)
    
    # Load color extractor if available (lazy import to avoid NUMEXPR issues)
    color_extractor = None
    if use_color_features:
        try:
            from utils.color_features import ColorFeatureExtractor
            
            # Try to load color extractor from the same directory as the model
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            color_extractor_path = os.path.join(model_dir, f"{model_name}_color_extractor.pkl")
            
            if os.path.exists(color_extractor_path):
                try:
                    color_extractor = joblib.load(color_extractor_path)
                    print(f"Loaded color extractor from: {color_extractor_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load required color extractor from {color_extractor_path}: {e}. "
                        f"Color features cannot be used without the properly trained color extractor. "
                        f"Either re-run without --use_color_features or ensure the color extractor file exists."
                    )
            else:
                raise FileNotFoundError(
                    f"Color extractor not found at: {color_extractor_path}. "
                    f"Color features require the trained color extractor file. "
                    f"Either re-run without --use_color_features or ensure the color extractor was saved during training."
                )
        except ImportError as e:
            raise ImportError(
                f"Failed to import ColorFeatureExtractor: {e}. "
                f"Color features cannot be used without the color feature module. "
                f"Either re-run without --use_color_features or install the required dependencies."
            )
    
    return model, feature_scaler, color_extractor


def evaluate_enhanced_model(model, scaler, test_dataset, candidate_params, 
                          original_dataframe=None, dataset_type='SPT3G_1500d',
                          use_uq=False, detection_threshold=0.1,
                          use_desprior_candidates=False, use_color_features=False, 
                          color_extractor=None, desprior_csv_path=None):
    """Evaluate enhanced model with UQ capabilities."""
    print(f"Evaluating {'probabilistic' if use_uq else 'deterministic'} model on {len(test_dataset)} test images...")
    print("Using rank-based evaluation (top-k candidate success tracking)...")
    
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
    
    # Rank-based evaluation tracking
    bcg_ranks = []
    
    # FEATURE ANALYSIS: Collect features for post-analysis
    all_features_list = []  # Store all features for analysis
    sample_labels = []      # Store labels for analysis (rank-based success)
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        image = sample['image']
        true_bcg = sample['BCG']
        filename = sample.get('filename', f'sample_{i}')
        
        # Store image for visualization
        test_images.append(image)
        
        # Extract metadata from original dataframe if available
        metadata = {'filename': filename}
        
        # For BCG data, extract redshift directly from sample
        if args.use_bcg_data and 'cluster_z' in sample:
            cluster_z = sample['cluster_z']
            if hasattr(cluster_z, 'numpy'):
                cluster_z = cluster_z.numpy()
            elif torch.is_tensor(cluster_z):
                cluster_z = cluster_z.numpy()
            metadata['z'] = float(cluster_z)
        
        if original_dataframe is not None:
            cluster_name = filename.replace('.tif', '').split('_')[0]
            metadata['cluster_name'] = cluster_name

            cluster_col = 'Cluster name' if 'Cluster name' in original_dataframe.columns else 'cluster_name'
            if cluster_col in original_dataframe.columns:
                matching_rows = original_dataframe[original_dataframe[cluster_col] == cluster_name]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    # Try multiple redshift column names
                    for z_col in ['z', 'Cluster z', 'redshift']:
                        if z_col in row and not pd.isna(row[z_col]):
                            metadata['z'] = row[z_col]
                            break
                    prob_cols = [col for col in row.index if 'prob' in col.lower()]
                    if prob_cols:
                        metadata['bcg_prob'] = row[prob_cols[0]]
                    # Extract RA/Dec for coordinate system
                    if 'BCG RA' in row:
                        metadata['bcg_ra'] = row['BCG RA']
                    if 'BCG Dec' in row:
                        metadata['bcg_dec'] = row['BCG Dec']

                    # Extract ALL BCG candidates for this cluster (for multiple RedMapper candidates)
                    all_bcg_candidates = []
                    for _, bcg_row in matching_rows.iterrows():
                        bcg_info = {}
                        # Get coordinates
                        if 'BCG RA' in bcg_row and 'BCG Dec' in bcg_row:
                            bcg_info['ra'] = bcg_row['BCG RA']
                            bcg_info['dec'] = bcg_row['BCG Dec']
                        # Get probability
                        prob_cols = [col for col in bcg_row.index if 'prob' in col.lower()]
                        if prob_cols and not pd.isna(bcg_row[prob_cols[0]]):
                            bcg_info['prob'] = bcg_row[prob_cols[0]]
                        # Get pixel coordinates (x, y)
                        if 'x' in bcg_row and 'y' in bcg_row:
                            bcg_info['x'] = bcg_row['x']
                            bcg_info['y'] = bcg_row['y']

                        # Only add if we have essential information
                        if bcg_info:
                            all_bcg_candidates.append(bcg_info)

                    if len(all_bcg_candidates) > 0:
                        metadata['all_bcg_candidates'] = all_bcg_candidates
        
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
                use_desprior_candidates=use_desprior_candidates,
                filename=filename,
                dataset_type=dataset_type,
                use_color_features=use_color_features,
                color_extractor=color_extractor,
                desprior_csv_path=desprior_csv_path,
                **candidate_params
            )
            
            predicted_bcg = results['best_bcg']
            all_candidates = results['all_candidates']
            scores = results['probabilities']  # These are probabilities, not raw scores
            probabilities = results['probabilities']
            uncertainties = results['uncertainties']
            detections = results['detections']
            best_features = results['best_features']  # Get the best candidate's features
            
            # FEATURE ANALYSIS: Store features for UQ case
            all_features_list.append(best_features)
            
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
                from data.data_read_bcgs import BCGDataset

                # Load DESprior candidates for this specific image/cluster
                if desprior_csv_path is None:
                    raise ValueError("desprior_csv_path must be provided when use_desprior_candidates=True")
                
                try:
                    candidates_df = pd.read_csv(desprior_csv_path)
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
                        visual_features, _ = extract_candidate_features(
                            image, all_candidates, patch_size=candidate_params.get('patch_size', 64), 
                            include_context=True, include_color=use_color_features, 
                            color_extractor=color_extractor
                        )
                        
                        # Combine visual features with candidate-specific features
                        combined_features = np.hstack([visual_features, candidate_specific_features])
                        
                        # NOTE: DESprior candidates use their own feature set and don't include additional BCG features
                        
                        if scaler is not None:
                            scaled_features = scaler.transform(combined_features)
                            features_tensor = torch.FloatTensor(scaled_features)
                        else:
                            scaled_features = combined_features
                            features_tensor = torch.FloatTensor(combined_features)
                        
                        with torch.no_grad():
                            scores = model(features_tensor).squeeze(-1).numpy()
                        
                        best_idx = np.argmax(scores)
                        predicted_bcg = tuple(all_candidates[best_idx])
                        
                        # FEATURE ANALYSIS: Store the best candidate's features
                        all_features_list.append(scaled_features[best_idx])
                        
                except Exception as e:
                    print(f"Warning: Failed to load DESprior candidates for {filename}: {e}")
                    predicted_bcg = None
                    scores = np.array([])
                    all_candidates = np.array([])
                    # FEATURE ANALYSIS: Add placeholder for failed cases
                    all_features_list.append(None)
                    
            else:
                from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
                all_candidates, intensities = find_bcg_candidates(image, **candidate_params)
                
                if len(all_candidates) == 0:
                    predicted_bcg = None
                    scores = np.array([])
                    # FEATURE ANALYSIS: Add placeholder for no candidates case
                    all_features_list.append(None)
                else:
                    features, _ = extract_candidate_features(
                        image, all_candidates, patch_size=candidate_params.get('patch_size', 64),
                        include_context=True, include_color=use_color_features, 
                        color_extractor=color_extractor
                    )
                    
                    # Append additional features if provided (e.g., from BCG dataset)
                    if additional_features is not None and len(features) > 0:
                        # Replicate additional features for each candidate
                        additional_features_repeated = np.tile(additional_features, (len(features), 1))
                        features = np.concatenate([features, additional_features_repeated], axis=1)
                    
                    if scaler is not None:
                        scaled_features = scaler.transform(features)
                        features_tensor = torch.FloatTensor(scaled_features)
                    else:
                        scaled_features = features
                        features_tensor = torch.FloatTensor(features)
                    
                    with torch.no_grad():
                        scores = model(features_tensor).squeeze(-1).numpy()
                    
                    best_idx = np.argmax(scores)
                    predicted_bcg = tuple(all_candidates[best_idx])
                    
                    # FEATURE ANALYSIS: Store the best candidate's features
                    all_features_list.append(scaled_features[best_idx])
            
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
            bcg_ranks.append(None)  # No rank when no candidates found
            # FEATURE ANALYSIS: Add label for failed case (failure = 0)
            sample_labels.append(0)
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
        
        # Calculate rank of true BCG among all candidates
        bcg_rank = calculate_bcg_rank(true_bcg, all_candidates, scores, distance_threshold=10.0)
        bcg_ranks.append(bcg_rank)
        
        # FEATURE ANALYSIS: Generate label based on rank and distance
        # Success criteria: rank 1 (best candidate) OR distance <= 20 pixels
        label = 1 if (bcg_rank == 1 or distance <= 20.0) else 0
        sample_labels.append(label)
        
        # Check for potential failure cases - only consider it a failure if:
        # 1. Distance is large AND true BCG is not in top-3 candidates
        # 2. Or if true BCG is not found among candidates at all
        is_failure = False
        failure_reason = None
        
        if bcg_rank is None:  # True BCG not found among candidates
            is_failure = True
            failure_reason = 'bcg_not_detected'
        elif distance > 100 and bcg_rank > 3:  # Large distance error AND not in top-3
            is_failure = True
            failure_reason = 'large_error_low_rank'
        elif distance > 100:  # Large distance error but in top-3 (could be acceptable)
            failure_reason = 'large_error_good_rank'  # Log but don't treat as failure
        
        if is_failure:
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': failure_reason,
                'predicted': predicted_bcg,
                'true_bcg': true_bcg,
                'distance': distance,
                'rank': bcg_rank,
                'candidates': all_candidates,
                'scores': scores
            })
    
    # Compute metrics
    distances = np.array(distances)
    success_rates = {}
    
    for threshold in [10, 20, 30, 50]:
        success_rate = np.mean(distances <= threshold) if len(distances) > 0 else 0
        success_rates[f'success_rate_{threshold}px'] = success_rate
    
    # Calculate rank-based success metrics
    valid_ranks = [rank for rank in bcg_ranks if rank is not None]
    rank_metrics = {}
    
    if len(valid_ranks) > 0:
        # Count successes by rank (top-k accuracy)
        rank_metrics['rank_1_success'] = len([r for r in valid_ranks if r == 1]) / len(predictions) if len(predictions) > 0 else 0
        rank_metrics['rank_2_success'] = len([r for r in valid_ranks if r <= 2]) / len(predictions) if len(predictions) > 0 else 0
        rank_metrics['rank_3_success'] = len([r for r in valid_ranks if r <= 3]) / len(predictions) if len(predictions) > 0 else 0
        rank_metrics['rank_5_success'] = len([r for r in valid_ranks if r <= 5]) / len(predictions) if len(predictions) > 0 else 0
        rank_metrics['mean_rank'] = np.mean(valid_ranks)
        rank_metrics['median_rank'] = np.median(valid_ranks)
    else:
        rank_metrics = {
            'rank_1_success': 0.0,
            'rank_2_success': 0.0,
            'rank_3_success': 0.0,
            'rank_5_success': 0.0,
            'mean_rank': float('inf'),
            'median_rank': float('inf')
        }
    
    metrics = {
        'n_predictions': len(predictions),
        'n_failed': len(failed_predictions),
        'mean_distance': np.mean(distances) if len(distances) > 0 else float('inf'),
        'median_distance': np.median(distances) if len(distances) > 0 else float('inf'),
        'std_distance': np.std(distances) if len(distances) > 0 else 0,
        'min_distance': np.min(distances) if len(distances) > 0 else float('inf'),
        'max_distance': np.max(distances) if len(distances) > 0 else 0,
        'mean_candidates': np.mean(candidate_counts) if len(candidate_counts) > 0 else 0,
        **success_rates,
        **rank_metrics
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
            all_probabilities_list, all_uncertainties_list, bcg_ranks,
            all_features_list, sample_labels)


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
        
        print("Distance-based Success Rates:")
        for key, value in metrics.items():
            if 'success_rate' in key:
                threshold = key.split('_')[-1]
                print(f"  Within {threshold}: {value*100:.1f}%")
        print()
        
        # Add rank-based success rates
        if 'rank_1_success' in metrics:
            print("Rank-based Success Rates:")
            print(f"  Best candidate (Rank 1): {metrics['rank_1_success']*100:.1f}%")
            print(f"  Top-2 candidates (Rank ≤2): {metrics['rank_2_success']*100:.1f}%") 
            print(f"  Top-3 candidates (Rank ≤3): {metrics['rank_3_success']*100:.1f}%")
            print(f"  Top-5 candidates (Rank ≤5): {metrics['rank_5_success']*100:.1f}%")
            if metrics['mean_rank'] != float('inf'):
                print(f"  Mean rank: {metrics['mean_rank']:.2f}")
                print(f"  Median rank: {metrics['median_rank']:.1f}")
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
            include_redmapper_probs=False,  # Never use RedMapper probs during testing
            image_dir=args.image_dir,  # Pass the image directory from command line
            csv_path=args.bcg_csv_path  # Pass custom BCG CSV path if provided
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
            if args.desprior_csv_path is None:
                raise ValueError("--desprior_csv_path must be provided when using DESprior candidates")
            
            candidates_df = pd.read_csv(args.desprior_csv_path)
            first_file = candidates_df['filename'].iloc[0]
            file_candidates = candidates_df[candidates_df['filename'] == first_file]
            
            if len(file_candidates) > 0:
                candidates = file_candidates[['x', 'y']].values
                candidate_features = file_candidates[['delta_mstar', 'starflag']].values
                
                # Extract visual features (without color for dimension estimation)
                from utils.candidate_based_bcg import extract_candidate_features
                visual_features, _ = extract_candidate_features(
                    sample_image, candidates, patch_size=args.patch_size, 
                    include_context=True, include_color=False,  # Don't use color for dimension estimation
                    color_extractor=None
                )
                combined_features = np.hstack([visual_features, candidate_features])
                base_feature_dim = combined_features.shape[1]
                print(f"Determined DESprior feature dimension: {base_feature_dim}")
            else:
                raise ValueError("No DESprior candidates found for feature dimension determination. Cannot proceed without real feature data.")
        except Exception as e:
            raise RuntimeError(f"Failed to determine DESprior feature dimension: {e}. Cannot proceed without accurate feature dimensions.")
    else:
        # For regular candidate detection
        from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
        candidates, _ = find_bcg_candidates(sample_image, **candidate_params_sample)
        if len(candidates) > 0:
            features, _ = extract_candidate_features(
                sample_image, candidates, patch_size=args.patch_size, 
                include_context=True, include_color=False,  # Don't use color for dimension estimation
                color_extractor=None
            )
            if len(features) > 0:
                base_feature_dim = features.shape[1]
            else:
                raise ValueError("No candidates found for feature dimension determination. Cannot proceed without real feature data.")
        else:
            raise ValueError("No candidates could be detected for feature dimension determination. Cannot proceed without real candidate data.")
    
    # Adjust feature dimension for BCG dataset additional features
    print(f"Base feature dimension (without color): {base_feature_dim}")
    
    # Add color features if enabled
    if args.use_color_features:
        print("Model was trained with color features - adding color feature dimensions")
        # Color features add: 8 color ratios + 8 PCA components + other color stats = ~20+ features
        color_feature_count = 20  # Estimated from ColorFeatureExtractor
        base_feature_dim += color_feature_count
        print(f"Added {color_feature_count} color features")
    
    # Note: DESprior candidate-specific features (delta_mstar, starflag) are already included in base_feature_dim
    
    # Add additional BCG features if enabled
    if args.use_bcg_data and args.use_additional_features:
        print("Adding additional features from BCG dataset: +2 (redshift, delta_mstar_z)")
        base_feature_dim += 2
    
    print(f"Final feature dimension: {base_feature_dim}")
    
    # Load trained model and determine actual feature dimension from the saved model
    print("Loading trained model...")
    print("Determining actual feature dimension from saved model...")
    
    # Read the model checkpoint to get the actual input dimension
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get actual feature dimension from first layer
    if 'network.0.weight' in state_dict:
        actual_feature_dim = state_dict['network.0.weight'].shape[1]
        print(f"Model expects {actual_feature_dim} features")
        
        if base_feature_dim != actual_feature_dim:
            print(f"⚠️  WARNING: Calculated dimension ({base_feature_dim}) != model's actual dimension ({actual_feature_dim})")
            print("   Using model's actual dimension for compatibility...")
            base_feature_dim = actual_feature_dim
    else:
        print("⚠️  Could not determine feature dimension from model, using calculated dimension")
    
    if args.use_color_features:
        print("Color features enabled - loading color extractor...")
    model, scaler, color_extractor = load_trained_model(args.model_path, args.scaler_path,
                                                       base_feature_dim, use_uq=args.use_uq,
                                                       use_color_features=args.use_color_features)

    # Print model architecture
    print("\n" + "="*80)
    print("LOADED MODEL ARCHITECTURE")
    print("="*80)
    print(f"Model type: {'BCGProbabilisticClassifier' if args.use_uq else 'BCGCandidateClassifier'}")
    print(f"Input dimension: {base_feature_dim}")
    print(f"Model structure:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*80 + "\n")

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
        use_desprior_candidates=args.use_desprior_candidates,
        use_color_features=args.use_color_features,
        color_extractor=color_extractor,
        desprior_csv_path=args.desprior_csv_path
    )
    
    (predictions, targets, distances, failures, metrics, 
     all_candidates_list, all_scores_list, test_images, sample_metadata,
     all_probabilities_list, all_uncertainties_list, bcg_ranks,
     all_features_list, sample_labels) = results
    
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
        if args.use_uq:
            # For UQ models, show rank-based sample categories
            print(f"\nGenerating rank-based sample visualizations...")
            
            # Organize samples by rank
            rank_categories = {
                'rank1': [],
                'rank2': [], 
                'rank3': [],
                'rest': []
            }
            
            for i in range(len(predictions)):
                rank = bcg_ranks[i]
                if rank == 1:
                    rank_categories['rank1'].append(i)
                elif rank == 2:
                    rank_categories['rank2'].append(i)
                elif rank == 3:
                    rank_categories['rank3'].append(i)
                else:
                    rank_categories['rest'].append(i)  # Includes rank > 3 and None
            
            # Generate visualizations for each rank category
            samples_per_category = args.show_samples  # Use full requested number for each category
            
            for category, indices in rank_categories.items():
                if len(indices) == 0:
                    continue
                    
                # Select samples for this category (prefer better scores within the category)
                if len(indices) > samples_per_category:
                    # Sort by best probability/score within this rank category
                    category_scores = []
                    for idx in indices:
                        if len(all_probabilities_list[idx]) > 0:
                            category_scores.append(np.max(all_probabilities_list[idx]))
                        elif len(all_scores_list[idx]) > 0:
                            category_scores.append(np.max(all_scores_list[idx]))
                        else:
                            category_scores.append(0)
                    
                    # Select top samples within this category
                    best_in_category = np.argsort(category_scores)[-samples_per_category:]
                    selected_indices = [indices[i] for i in best_in_category]
                else:
                    selected_indices = indices
                
                if len(selected_indices) == 0:
                    continue
                
                # Prepare data for this category
                sample_images = [test_images[i] for i in selected_indices]
                sample_targets = [targets[i] for i in selected_indices]
                sample_predictions = [predictions[i] for i in selected_indices]
                sample_candidates = [all_candidates_list[i] for i in selected_indices]
                sample_scores = [all_scores_list[i] for i in selected_indices]
                sample_metadata_list = [sample_metadata[i] for i in selected_indices]
                sample_probabilities = [all_probabilities_list[i] for i in selected_indices]
                
                # Create category-specific phase name
                phase_name = f"ProbabilisticTesting_prediction_sample_best_{category}"
                
                print(f"  Generating {len(selected_indices)} samples for {category} (ranks {1 if category=='rank1' else 2 if category=='rank2' else 3 if category=='rank3' else '>3 or None'})")
                
                show_enhanced_predictions(
                    sample_images, sample_targets, sample_predictions,
                    sample_candidates, sample_scores, sample_probabilities,
                    indices=range(len(sample_images)),
                    save_dir=args.output_dir,
                    phase=phase_name,
                    use_uq=args.use_uq,
                    metadata_list=sample_metadata_list,
                    detection_threshold=args.detection_threshold,
                    dataset_type=args.dataset_type
                )
        
        else:
            # For non-UQ models, use traditional distance-based sorting
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
            sample_probabilities = None
            
            phase_name = "EnhancedTesting"
            
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
    if args.show_failures and len(distances) > 0 and not args.use_uq:
        # Only show failure cases for non-UQ models
        # For UQ models, failures are covered in the 'rest' category
        print(f"\nShowing worst prediction failures...")
        
        # For non-UQ models, use traditional distance-based failure detection
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
        
        # FEATURE ANALYSIS: Save collected features for analysis
        features_file = os.path.join(args.output_dir, 'test_features.npz')
        print(f"Saving {len(all_features_list)} samples of features for analysis...")
        
        # Calculate multi-target matching (for images with multiple BCG candidates)
        matches_any_target = []
        n_targets_per_image = []
        min_distance_to_any_target = []
        multi_target_bcg_ranks = []

        for i, (pred, meta) in enumerate(zip(predictions, sample_metadata)):
            all_bcg_candidates = meta.get('all_bcg_candidates', [])
            n_targets = len(all_bcg_candidates) if len(all_bcg_candidates) > 1 else 1
            n_targets_per_image.append(n_targets)

            # Check if prediction matches ANY target (within threshold)
            if len(all_bcg_candidates) > 1:
                # Multiple targets - check distance to all and find best rank
                min_dist = float('inf')
                best_rank_for_any_target = None

                for bcg_cand in all_bcg_candidates:
                    if 'x' in bcg_cand and 'y' in bcg_cand:
                        bcg_pos = np.array([bcg_cand['x'], bcg_cand['y']])
                        pred_pos = np.array(pred)
                        dist = np.sqrt(np.sum((pred_pos - bcg_pos)**2))
                        min_dist = min(min_dist, dist)

                        # Calculate rank for this target
                        if i < len(all_candidates_list) and i < len(all_scores_list):
                            candidates = all_candidates_list[i]
                            scores = all_scores_list[i] if all_scores_list else []
                            if len(candidates) > 0 and len(scores) > 0:
                                rank = calculate_bcg_rank(bcg_pos, candidates, scores, distance_threshold=10.0)
                                if rank is not None:
                                    if best_rank_for_any_target is None or rank < best_rank_for_any_target:
                                        best_rank_for_any_target = rank

                min_distance_to_any_target.append(min_dist)
                matches_any_target.append(min_dist <= 10.0)
                multi_target_bcg_ranks.append(best_rank_for_any_target)
            else:
                # Single target - use the original distance and rank
                min_distance_to_any_target.append(distances[i])
                matches_any_target.append(distances[i] <= 10.0)
                multi_target_bcg_ranks.append(bcg_ranks[i])

        # Create enhanced results dictionary
        results_data = {
            'pred_x': [pred[0] for pred in predictions],
            'pred_y': [pred[1] for pred in predictions],
            'true_x': [target[0] for target in targets],
            'true_y': [target[1] for target in targets],
            'distance_error': distances,
            'n_candidates': [len(cand) for cand in all_candidates_list],
            'bcg_rank': bcg_ranks,
            'n_targets': n_targets_per_image,
            'matches_any_target': matches_any_target,
            'min_dist_to_any_target': min_distance_to_any_target,
            'multi_target_rank': multi_target_bcg_ranks
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

            # Always include redshift column (use NaN for missing values)
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

        # Add multi-target matching columns
        cols.extend(['n_targets', 'matches_any_target', 'min_dist_to_any_target', 'multi_target_rank'])

        # Add rank-based evaluation column
        cols.extend(['bcg_rank'])
        
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
        
        # FEATURE ANALYSIS: Save features and labels for analysis
        valid_features = []
        valid_labels = []
        valid_indices = []
        
        for i, (features, label) in enumerate(zip(all_features_list, sample_labels)):
            if features is not None:  # Skip failed cases with no features
                valid_features.append(features)
                valid_labels.append(label)
                valid_indices.append(i)
        
        if len(valid_features) > 0:
            # Convert to numpy arrays
            features_array = np.array(valid_features)
            labels_array = np.array(valid_labels)
            indices_array = np.array(valid_indices)
            
            # Save features, labels, and indices
            np.savez(features_file, 
                    X=features_array, 
                    y=labels_array, 
                    sample_indices=indices_array)
            
            print(f"Feature analysis data saved to: {features_file}")
            print(f"Features: {features_array.shape}, Labels: {labels_array.shape}")
            print(f"Success rate: {np.mean(labels_array):.3f} ({np.sum(labels_array)}/{len(labels_array)} samples)")
        else:
            print("Warning: No valid features collected for analysis")
        
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
    parser.add_argument('--patch_size', type=int, default=64,
                       help='Size of square patches extracted around candidates (e.g., 64, 128, 256)')
    
    
    # Color features arguments
    parser.add_argument('--use_color_features', action='store_true',
                       help='Enable color feature extraction from RGB patches for red-sequence detection')
    
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
    parser.add_argument('--desprior_csv_path', type=str, default=None,
                       help='Path to DESprior candidates CSV file')
    parser.add_argument('--bcg_csv_path', type=str, default=None,
                       help='Path to BCG CSV file (overrides default path selection)')
    
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