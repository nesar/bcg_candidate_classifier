"""
Multi-Scale Candidate Detection for BCG

This module extends the candidate detection to handle objects of different sizes
by using multiple scales and adaptive patch sizes.
"""

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from sklearn.preprocessing import StandardScaler
from utils.candidate_based_bcg import extract_patch_features, extract_context_features


def find_multiscale_bcg_candidates(image, scales=[0.5, 1.0, 1.5], 
                                  base_min_distance=15, threshold_rel=0.12, 
                                  exclude_border=30, max_candidates_per_scale=10):
    """
    Find candidates at multiple scales to capture objects of different sizes.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image or 3D RGB image
    scales : list
        Scale factors for candidate detection (relative to base parameters)
    base_min_distance : int
        Base minimum distance between candidates
    threshold_rel : float
        Relative threshold for candidate detection
    exclude_border : int
        Exclude candidates near image borders
    max_candidates_per_scale : int
        Maximum candidates per scale
        
    Returns:
    --------
    candidates : numpy.ndarray
        Array of shape (N, 3) containing (x, y, scale) coordinates of candidates
    intensities : numpy.ndarray
        Array of intensities at candidate locations
    patch_sizes : numpy.ndarray
        Adaptive patch sizes for each candidate
    """
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
            # Check distance to previously selected candidates (across all scales)
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
                    # Use minimum of current and previous scale distances
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
    # Ensure patch sizes are reasonable
    patch_sizes = np.clip(patch_sizes, 32, 128)
    
    return candidates_with_scale, intensities_array, patch_sizes


def extract_multiscale_candidate_features(image, candidate_coords_with_scale, patch_sizes, include_context=True):
    """
    Extract features for multi-scale candidates with adaptive patch sizes.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (H, W, C) or (H, W)
    candidate_coords_with_scale : numpy.ndarray
        Array of (x, y, scale) coordinates for candidates
    patch_sizes : numpy.ndarray
        Adaptive patch sizes for each candidate
    include_context : bool
        Whether to include contextual features
        
    Returns:
    --------
    features : numpy.ndarray
        Array of shape (N_candidates, feature_dim) containing features
    patches : numpy.ndarray
        Array of patches around candidates for visualization
    """
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
        
        # Pad if necessary to maintain consistent size (use base size for consistency)
        base_patch_size = 64
        if patch.shape[0] < base_patch_size or patch.shape[1] < base_patch_size:
            padded_patch = np.zeros((base_patch_size, base_patch_size, image.shape[2]), dtype=image.dtype)
            pad_y = (base_patch_size - patch.shape[0]) // 2
            pad_x = (base_patch_size - patch.shape[1]) // 2
            padded_patch[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded_patch
        elif patch.shape[0] > base_patch_size or patch.shape[1] > base_patch_size:
            # Resize larger patches to base size
            from scipy.ndimage import zoom
            scale_y = base_patch_size / patch.shape[0]
            scale_x = base_patch_size / patch.shape[1]
            patch = zoom(patch, (scale_y, scale_x, 1), order=1)
            patch = patch.astype(image.dtype)
        
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
            # Add contextual features with scale-adjusted patch size
            context_features = extract_context_features(image, x, y, patch_size)
            patch_features = np.concatenate([patch_features, context_features, scale_features])
        else:
            patch_features = np.concatenate([patch_features, scale_features])
        
        features_list.append(patch_features)
    
    features = np.array(features_list) if features_list else np.array([])
    patches = np.array(patches_list) if patches_list else np.array([])
    
    return features, patches


def predict_bcg_from_multiscale_candidates(image, model=None, feature_scaler=None, 
                                         use_multiscale=False, **candidate_kwargs):
    """
    Complete pipeline: find multi-scale candidates, extract features, predict best BCG.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    model : BCGCandidateClassifier or None
        Trained classifier model
    feature_scaler : StandardScaler or None
        Feature scaler
    use_multiscale : bool
        Whether to use multi-scale candidate detection
    **candidate_kwargs : dict
        Arguments for candidate finding
        
    Returns:
    --------
    best_bcg : tuple or None
        (x, y) coordinates of best BCG candidate
    all_candidates : numpy.ndarray
        All candidate coordinates (x, y, scale if multiscale)
    candidate_scores : numpy.ndarray
        Scores for all candidates
    """
    if use_multiscale:
        # Extract multiscale parameters
        scales = candidate_kwargs.pop('scales', [0.5, 1.0, 1.5])
        max_candidates_per_scale = candidate_kwargs.pop('max_candidates_per_scale', 10)
        
        # Find multiscale candidates
        candidates_with_scale, intensities, patch_sizes = find_multiscale_bcg_candidates(
            image, 
            scales=scales,
            max_candidates_per_scale=max_candidates_per_scale,
            **candidate_kwargs
        )
        
        if len(candidates_with_scale) == 0:
            return None, np.array([]), np.array([])
        
        # Extract features with adaptive patch sizes
        features, patches = extract_multiscale_candidate_features(
            image, 
            candidates_with_scale,
            patch_sizes,
            include_context=True
        )
        
        # Return coordinates without scale for compatibility
        candidates = candidates_with_scale[:, :2]
        
    else:
        # Use original single-scale approach
        from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
        
        candidates, intensities = find_bcg_candidates(image, **candidate_kwargs)
        
        if len(candidates) == 0:
            return None, np.array([]), np.array([])
        
        # Extract features
        features, patches = extract_candidate_features(image, candidates)
    
    if model is None or feature_scaler is None:
        # Fallback: return brightest candidate
        best_idx = 0  # Already sorted by brightness
        return tuple(candidates[best_idx]), candidates, intensities
    
    # Scale features
    scaled_features = feature_scaler.transform(features)
    
    # Predict scores
    model.eval()
    with torch.no_grad():
        feature_tensor = torch.FloatTensor(scaled_features)
        scores = model(feature_tensor).squeeze().numpy()
    
    # Find best candidate
    best_idx = np.argmax(scores)
    best_bcg = tuple(candidates[best_idx])
    
    return best_bcg, candidates, scores
