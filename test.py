#!/usr/bin/env python3
"""
Enhanced BCG Classifier Testing Script

This script evaluates trained BCG classifiers with:
1. Multi-scale candidate detection
2. Uncertainty quantification and probabilistic outputs
3. Detection threshold analysis
4. Enhanced visualizations with probability information
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
# MULTI-SCALE CANDIDATE DETECTION (COPY FROM TRAIN.PY)
# ============================================================================

def find_multiscale_bcg_candidates(image, scales=[0.5, 1.0, 1.5], 
                                  base_min_distance=15, threshold_rel=0.12, 
                                  exclude_border=0, max_candidates_per_scale=10):
    """Find candidates at multiple scales to capture objects of different sizes."""
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        grayscale = image.copy()
    
    all_candidates = []
    all_intensities = []
    all_scales = []
    
    for scale in scales:
        # Adjust parameters based on scale
        min_distance = max(int(base_min_distance * scale), 5)
        filter_size = max(int(3 * scale), 3)
        
        # Find local maxima with scale-adjusted filter
        local_max_mask = (grayscale == maximum_filter(grayscale, size=filter_size))
        
        # Apply threshold
        threshold_abs = threshold_rel * grayscale.max()
        local_max_mask &= (grayscale > threshold_abs)
        
        # Exclude border
        if exclude_border > 0:
            local_max_mask[:exclude_border, :] = False
            local_max_mask[-exclude_border:, :] = False
            local_max_mask[:, :exclude_border] = False
            local_max_mask[:, -exclude_border:] = False
        
        # Extract coordinates and intensities
        y_coords, x_coords = np.where(local_max_mask)
        if len(y_coords) == 0:
            continue
        
        candidates = np.column_stack((x_coords, y_coords))
        intensities = grayscale[y_coords, x_coords]
        
        # Sort by intensity (brightest first)
        sort_indices = np.argsort(intensities)[::-1]
        candidates = candidates[sort_indices]
        intensities = intensities[sort_indices]
        
        # Apply non-maximum suppression
        selected_candidates = []
        selected_intensities = []
        
        for candidate, intensity in zip(candidates, intensities):
            # Check distance to previously selected candidates
            too_close = False
            
            # Check against candidates from this scale
            for selected_candidate in selected_candidates:
                distance = np.sqrt(np.sum((candidate - selected_candidate)**2))
                if distance < min_distance:
                    too_close = True
                    break
            
            # Check against candidates from other scales
            if not too_close:
                for prev_candidate in all_candidates:
                    distance = np.sqrt(np.sum((candidate - prev_candidate[:2])**2))
                    min_scale_distance = min(min_distance, base_min_distance * prev_candidate[2])
                    if distance < min_scale_distance:
                        too_close = True
                        break
            
            if not too_close:
                selected_candidates.append(candidate)
                selected_intensities.append(intensity)
                
                if len(selected_candidates) >= max_candidates_per_scale:
                    break
        
        # Add scale information and store
        for candidate, intensity in zip(selected_candidates, selected_intensities):
            all_candidates.append(np.append(candidate, scale))
            all_intensities.append(intensity)
            all_scales.append(scale)
    
    if not all_candidates:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to arrays
    candidates_with_scale = np.array(all_candidates)
    intensities_array = np.array(all_intensities)
    scales_array = np.array(all_scales)
    
    # Compute adaptive patch sizes based on scale
    base_patch_size = 64
    patch_sizes = (base_patch_size * scales_array).astype(int)
    patch_sizes = np.clip(patch_sizes, 32, 128)
    
    return candidates_with_scale, intensities_array, patch_sizes


def extract_multiscale_candidate_features(image, candidate_coords_with_scale, patch_sizes, include_context=True):
    """Extract features for multi-scale candidates with adaptive patch sizes."""
    if len(candidate_coords_with_scale) == 0:
        return np.array([]), np.array([])
    
    # Ensure image is 3D
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    H, W = image.shape[:2]
    features_list = []
    patches_list = []
    
    for i, (candidate_info, patch_size) in enumerate(zip(candidate_coords_with_scale, patch_sizes)):
        x, y, scale = candidate_info[0], candidate_info[1], candidate_info[2]
        x, y = int(x), int(y)
        
        half_patch = patch_size // 2
        
        # Extract patch around candidate
        x_min = max(0, x - half_patch)
        x_max = min(W, x + half_patch)
        y_min = max(0, y - half_patch)
        y_max = min(H, y + half_patch)
        
        patch = image[y_min:y_max, x_min:x_max]
        
        # Resize to consistent size for feature extraction
        base_patch_size = 64
        if patch.shape[0] != base_patch_size or patch.shape[1] != base_patch_size:
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                scale_y = base_patch_size / patch.shape[0]
                scale_x = base_patch_size / patch.shape[1]
                patch = zoom(patch, (scale_y, scale_x, 1), order=1)
                patch = patch.astype(image.dtype)
            else:
                patch = np.zeros((base_patch_size, base_patch_size, image.shape[2]), dtype=image.dtype)
        
        patches_list.append(patch)
        
        # Extract features from patch
        patch_features = extract_patch_features(patch, x, y, image.shape[:2])
        
        # Add scale-specific features
        scale_features = np.array([
            scale,                    # Scale factor
            patch_size,              # Actual patch size used
            scale / np.mean([s[2] for s in candidate_coords_with_scale])  # Relative scale
        ])
        
        if include_context:
            context_features = extract_context_features(image, x, y, patch_size)
            patch_features = np.concatenate([patch_features, context_features, scale_features])
        else:
            patch_features = np.concatenate([patch_features, scale_features])
        
        features_list.append(patch_features)
    
    features = np.array(features_list) if features_list else np.array([])
    patches = np.array(patches_list) if patches_list else np.array([])
    
    return features, patches


# ============================================================================
# PROBABILISTIC CLASSIFIER (COPY FROM TRAIN.PY)
# ============================================================================

class BCGProbabilisticClassifier(nn.Module):
    """Probabilistic BCG classifier that outputs calibrated probabilities."""
    
    def __init__(self, feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(BCGProbabilisticClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_dim = feature_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer - logits for binary classification (BCG vs non-BCG)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, features):
        """Forward pass to get logits."""
        logits = self.network(features)
        # Apply temperature scaling
        logits = logits / self.temperature
        return logits
    
    def predict_probabilities(self, features):
        """Predict calibrated probabilities for being BCG."""
        logits = self.forward(features)
        probabilities = torch.sigmoid(logits)
        return probabilities
    
    def predict_with_uncertainty(self, features, n_samples=10):
        """Predict with epistemic uncertainty using Monte Carlo Dropout."""
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(features)
                raw_probs = torch.sigmoid(logits)
                # Normalize probabilities to sum to 1 across all candidates
                prob_sum = raw_probs.sum()
                normalized_probs = raw_probs / prob_sum if prob_sum > 0 else raw_probs
                predictions.append(normalized_probs)
        
        self.eval()  # Return to eval mode
        
        predictions = torch.stack(predictions)  # (n_samples, n_candidates, 1)
        
        mean_probs = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)  # Epistemic uncertainty
        
        return mean_probs.squeeze(-1), uncertainty.squeeze(-1)


# ============================================================================
# ENHANCED PREDICTION FUNCTIONS
# ============================================================================

def predict_bcg_with_probabilities(image, model, feature_scaler=None, 
                                 detection_threshold=0.5, use_multiscale=False, 
                                 return_all_candidates=False, **candidate_kwargs):
    """Predict BCG candidates with calibrated probabilities and uncertainty."""
    
    # Find candidates using appropriate method
    if use_multiscale:
        # Map candidate_kwargs to multiscale function parameters
        multiscale_params = {
            'scales': candidate_kwargs.get('scales', [0.5, 1.0, 1.5]),
            'base_min_distance': candidate_kwargs.get('min_distance', 15),
            'threshold_rel': candidate_kwargs.get('threshold_rel', 0.12),
            'exclude_border': candidate_kwargs.get('exclude_border', 30),
            'max_candidates_per_scale': candidate_kwargs.get('max_candidates_per_scale', 10)
        }
        candidates_with_scale, intensities, patch_sizes = find_multiscale_bcg_candidates(
            image, **multiscale_params
        )
        if len(candidates_with_scale) == 0:
            return {
                'best_bcg': None,
                'all_candidates': np.array([]),
                'probabilities': np.array([]),
                'uncertainties': np.array([]),
                'detections': np.array([]),
                'detection_probabilities': np.array([])
            }
        
        # Extract features
        features, _ = extract_multiscale_candidate_features(image, candidates_with_scale, patch_sizes)
        all_candidates = candidates_with_scale[:, :2]  # Extract x, y coordinates
        
    else:
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
            # This is a probabilistic model with UQ
            probabilities, uncertainties = model.predict_with_uncertainty(features_tensor)
            probabilities = probabilities.numpy()
            uncertainties = uncertainties.numpy()
            # Additional normalization to ensure probabilities sum to 1 (in case of numerical errors)
            probabilities = probabilities / np.sum(probabilities) if np.sum(probabilities) > 0 else probabilities
        elif hasattr(model, 'temperature'):
            # This is a probabilistic model without MC dropout
            logits = model(features_tensor).squeeze(-1)
            raw_probabilities = torch.sigmoid(logits).numpy()
            # Normalize probabilities to sum to 1 across all candidates
            probabilities = raw_probabilities / np.sum(raw_probabilities) if np.sum(raw_probabilities) > 0 else raw_probabilities
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
        else:
            # This is a traditional classifier, convert scores to probabilities
            scores = model(features_tensor).squeeze(-1)
            # Convert scores to probabilities using softmax
            probabilities = torch.softmax(scores, dim=0).numpy()
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
        model = BCGProbabilisticClassifier(feature_dim)
    else:
        model = BCGCandidateClassifier(feature_dim)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    feature_scaler = joblib.load(scaler_path)
    
    return model, feature_scaler


def evaluate_enhanced_model(model, scaler, test_dataset, candidate_params, 
                          original_dataframe=None, dataset_type='SPT3G_1500d',
                          use_multiscale=False, use_uq=False, detection_threshold=0.5,
                          use_desprior_candidates=False):
    """Evaluate enhanced model with multiscale and UQ capabilities."""
    print(f"Evaluating {'probabilistic' if use_uq else 'deterministic'} model on {len(test_dataset)} test images...")
    if use_multiscale:
        print("Using multi-scale candidate detection")
    
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
        
        # Make prediction with appropriate method
        if use_uq:
            results = predict_bcg_with_probabilities(
                image, model, scaler, 
                detection_threshold=detection_threshold,
                use_multiscale=use_multiscale,
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
            if use_multiscale:
                # Map candidate_params to multiscale function parameters
                multiscale_params = {
                    'scales': candidate_params.get('scales', [0.5, 1.0, 1.5]),
                    'base_min_distance': candidate_params.get('min_distance', 15),
                    'threshold_rel': candidate_params.get('threshold_rel', 0.12),
                    'exclude_border': candidate_params.get('exclude_border', 30),
                    'max_candidates_per_scale': candidate_params.get('max_candidates_per_scale', 10)
                }
                candidates_with_scale, intensities, patch_sizes = find_multiscale_bcg_candidates(
                    image, **multiscale_params
                )
                if len(candidates_with_scale) == 0:
                    predicted_bcg = None
                    all_candidates = np.array([])
                    scores = np.array([])
                else:
                    all_candidates = candidates_with_scale[:, :2]
                    features, _ = extract_multiscale_candidate_features(
                        image, candidates_with_scale, patch_sizes
                    )
                    
                    if scaler is not None:
                        scaled_features = scaler.transform(features)
                        features_tensor = torch.FloatTensor(scaled_features)
                    else:
                        features_tensor = torch.FloatTensor(features)
                    
                    with torch.no_grad():
                        scores = model(features_tensor).squeeze(-1).numpy()
                    
                    best_idx = np.argmax(scores)
                    predicted_bcg = tuple(all_candidates[best_idx])
            elif use_desprior_candidates:
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
    
    if args.use_multiscale:
        print(f"Multi-scale: scales={args.scales}")
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
    
    if args.use_multiscale:
        if args.use_desprior_candidates:
            # Multiscale + DESprior combination - features depend on multiscale output + candidate features
            # This might vary, so we need to determine it properly
            base_feature_dim = 35  # Approximate: multiscale features + candidate features
        else:
            multiscale_params = {
                'scales': args.scales,
                'base_min_distance': args.min_distance,
                'threshold_rel': args.threshold_rel,
                'exclude_border': args.exclude_border,
                'max_candidates_per_scale': args.max_candidates_per_scale
            }
            candidates_with_scale, _, patch_sizes = find_multiscale_bcg_candidates(
                sample_image, **multiscale_params
            )
            if len(candidates_with_scale) > 0:
                features, _ = extract_multiscale_candidate_features(
                    sample_image, candidates_with_scale, patch_sizes, include_context=True
                )
                base_feature_dim = features.shape[1] if len(features) > 0 else 33
            else:
                base_feature_dim = 33  # Default for multiscale
    else:
        if args.use_desprior_candidates:
            # For DESprior candidates, we need to account for additional features (delta_mstar, starflag)
            # Visual features (30) + candidate features (2) = 32 total
            base_feature_dim = 32
        else:
            from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
            candidates, _ = find_bcg_candidates(sample_image, **candidate_params_sample)
            if len(candidates) > 0:
                features, _ = extract_candidate_features(sample_image, candidates, include_context=True)
                base_feature_dim = features.shape[1] if len(features) > 0 else 30
            else:
                base_feature_dim = 30  # Default for single-scale
    
    print(f"Detected feature dimension: {base_feature_dim}")
    
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
    
    if args.use_multiscale:
        candidate_params.update({
            'scales': args.scales,
            'max_candidates_per_scale': args.max_candidates_per_scale
        })
    
    # Evaluate model
    results = evaluate_enhanced_model(
        model, scaler, test_subset, candidate_params, original_df, args.dataset_type,
        use_multiscale=args.use_multiscale, use_uq=args.use_uq, 
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
        if args.use_multiscale:
            phase_name = "MultiscaleTesting"
        if args.use_uq:
            phase_name = "ProbabilisticTesting"
        if args.use_multiscale and args.use_uq:
            phase_name = "MultiscaleProbabilisticTesting"
        
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
        
            phase_name = "CandidateBasedTesting"  # Changed name to match your request
            if args.use_multiscale:
                phase_name = "MultiscaleTesting"
            if args.use_uq:
                phase_name = "ProbabilisticTesting"
            if args.use_multiscale and args.use_uq:
                phase_name = "MultiscaleProbabilisticTesting"
            
            show_failures(
                failure_images, failure_targets, failure_predictions,
                threshold=20, max_failures=args.show_failures,
                save_dir=args.output_dir,
                phase=phase_name,
                metadata_list=failure_metadata_list
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
    parser.add_argument('--min_distance', type=int, default=15,
                       help='Minimum distance between candidates')
    parser.add_argument('--threshold_rel', type=float, default=0.12,
                       help='Relative threshold for candidate detection')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=25,
                       help='Maximum candidates per image')
    
    # Enhanced feature arguments
    parser.add_argument('--use_multiscale', action='store_true',
                       help='Enable multi-scale candidate detection')
    parser.add_argument('--scales', type=str, default='0.5,1.0,1.5',
                       help='Comma-separated scale factors for multiscale detection')
    parser.add_argument('--max_candidates_per_scale', type=int, default=10,
                       help='Maximum candidates per scale in multiscale mode')
    
    parser.add_argument('--use_uq', action='store_true',
                       help='Enable uncertainty quantification with probabilistic outputs')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
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
    
    # Parse scales if multiscale is enabled
    if args.use_multiscale:
        args.scales = [float(s.strip()) for s in args.scales.split(',')]
    else:
        args.scales = [1.0]
    
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