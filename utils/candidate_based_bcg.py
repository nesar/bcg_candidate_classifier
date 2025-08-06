"""
Candidate-Based BCG Detection

This module implements a candidate-based approach where:
1. Local maxima (bright spots) are found first
2. Features are extracted around each candidate location
3. A model ranks/classifies candidates to find the best BCG
"""

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from sklearn.preprocessing import StandardScaler


def find_bcg_candidates(image, min_distance=15, threshold_rel=0.15, exclude_border=30, max_candidates=20):
    """
    Find candidate BCG locations as local maxima in the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image or 3D RGB image
    min_distance : int
        Minimum distance between candidates (pixels)
    threshold_rel : float
        Relative threshold for candidate detection
    exclude_border : int
        Exclude candidates near image borders
    max_candidates : int
        Maximum number of candidates to return
        
    Returns:
    --------
    candidates : numpy.ndarray
        Array of shape (N, 2) containing (x, y) coordinates of candidates
    intensities : numpy.ndarray
        Array of intensities at candidate locations
    """
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        grayscale = image.copy()
    
    # Find local maxima using smaller window first to get more candidates
    local_max_mask = (grayscale == maximum_filter(grayscale, size=3))
    
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
        return np.array([]), np.array([])
    
    candidates = np.column_stack((x_coords, y_coords))
    intensities = grayscale[y_coords, x_coords]
    
    # Sort by intensity (brightest first)
    sort_indices = np.argsort(intensities)[::-1]
    candidates = candidates[sort_indices]
    intensities = intensities[sort_indices]
    
    # Apply non-maximum suppression to enforce min_distance between candidates
    if len(candidates) > 1:
        selected_candidates = []
        selected_intensities = []
        
        for candidate, intensity in zip(candidates, intensities):
            # Check if this candidate is too close to any already selected candidate
            too_close = False
            for selected_candidate in selected_candidates:
                distance = np.sqrt(np.sum((candidate - selected_candidate)**2))
                if distance < min_distance:
                    too_close = True
                    break
            
            # If not too close to existing candidates, add it
            if not too_close:
                selected_candidates.append(candidate)
                selected_intensities.append(intensity)
                
                # Stop if we have enough candidates
                if len(selected_candidates) >= max_candidates:
                    break
        
        candidates = np.array(selected_candidates) if selected_candidates else np.array([])
        intensities = np.array(selected_intensities) if selected_intensities else np.array([])
    else:
        # Only one candidate, just limit to max_candidates (should be 1)
        candidates = candidates[:max_candidates]
        intensities = intensities[:max_candidates]
    
    return candidates, intensities


def extract_candidate_features(image, candidate_coords, patch_size=64, include_context=True):
    """
    Extract features around each candidate location.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (H, W, C) or (H, W)
    candidate_coords : numpy.ndarray
        Array of (x, y) coordinates for candidates
    patch_size : int
        Size of patch to extract around each candidate
    include_context : bool
        Whether to include contextual features
        
    Returns:
    --------
    features : numpy.ndarray
        Array of shape (N_candidates, feature_dim) containing features
    patches : numpy.ndarray
        Array of patches around candidates for visualization
    """
    if len(candidate_coords) == 0:
        return np.array([]), np.array([])
    
    # Ensure image is 3D
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    H, W = image.shape[:2]
    half_patch = patch_size // 2
    
    features_list = []
    patches_list = []
    
    for x, y in candidate_coords:
        x, y = int(x), int(y)
        
        # Extract patch around candidate
        x_min = max(0, x - half_patch)
        x_max = min(W, x + half_patch)
        y_min = max(0, y - half_patch)
        y_max = min(H, y + half_patch)
        
        patch = image[y_min:y_max, x_min:x_max]
        
        # Pad if necessary to maintain consistent size
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            pad_y = (patch_size - patch.shape[0]) // 2
            pad_x = (patch_size - patch.shape[1]) // 2
            padded_patch[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded_patch
        
        patches_list.append(patch)
        
        # Extract features from patch
        patch_features = extract_patch_features(patch, x, y, image.shape[:2])
        
        if include_context:
            # Add contextual features
            context_features = extract_context_features(image, x, y, patch_size)
            patch_features = np.concatenate([patch_features, context_features])
        
        features_list.append(patch_features)
    
    features = np.array(features_list) if features_list else np.array([])
    patches = np.array(patches_list) if patches_list else np.array([])
    
    return features, patches


def extract_patch_features(patch, center_x, center_y, image_shape):
    """
    Extract features from a patch around a candidate.
    
    Parameters:
    -----------
    patch : numpy.ndarray
        Image patch around candidate
    center_x, center_y : int
        Coordinates of candidate in original image
    image_shape : tuple
        Shape of original image (H, W)
        
    Returns:
    --------
    features : numpy.ndarray
        Feature vector for this patch
    """
    features = []
    
    # Convert to grayscale for some features
    if len(patch.shape) == 3:
        gray_patch = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    else:
        gray_patch = patch
    
    # 1. Intensity statistics
    features.extend([
        np.mean(gray_patch),           # Mean intensity
        np.std(gray_patch),            # Standard deviation
        np.max(gray_patch),            # Maximum intensity
        np.min(gray_patch),            # Minimum intensity
        np.median(gray_patch),         # Median intensity
    ])
    
    # 2. Central vs peripheral intensity
    center = gray_patch.shape[0] // 2
    central_region = gray_patch[center-8:center+8, center-8:center+8]
    peripheral_mask = np.ones_like(gray_patch, dtype=bool)
    peripheral_mask[center-8:center+8, center-8:center+8] = False
    peripheral_region = gray_patch[peripheral_mask]
    
    features.extend([
        np.mean(central_region),       # Central intensity
        np.mean(peripheral_region),    # Peripheral intensity
        np.mean(central_region) / (np.mean(peripheral_region) + 1e-8),  # Central/peripheral ratio
    ])
    
    # 3. Gradient features (edge detection)
    grad_x = np.gradient(gray_patch, axis=1)
    grad_y = np.gradient(gray_patch, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        np.mean(gradient_magnitude),   # Mean gradient magnitude
        np.std(gradient_magnitude),    # Gradient magnitude std
        np.max(gradient_magnitude),    # Max gradient magnitude
    ])
    
    # 4. Position features (relative to image center)
    img_h, img_w = image_shape
    rel_x = (center_x - img_w/2) / (img_w/2)  # Normalized x position (-1 to 1)
    rel_y = (center_y - img_h/2) / (img_h/2)  # Normalized y position (-1 to 1)
    distance_from_center = np.sqrt(rel_x**2 + rel_y**2)
    
    features.extend([
        rel_x,                         # Relative x position
        rel_y,                         # Relative y position  
        distance_from_center,          # Distance from image center
    ])
    
    # 5. Shape/symmetry features
    # Compute moments to assess symmetry
    y_indices, x_indices = np.mgrid[:gray_patch.shape[0], :gray_patch.shape[1]]
    total_intensity = np.sum(gray_patch)
    
    if total_intensity > 0:
        # Centroid
        cx = np.sum(x_indices * gray_patch) / total_intensity
        cy = np.sum(y_indices * gray_patch) / total_intensity
        
        # Second moments
        mu20 = np.sum((x_indices - cx)**2 * gray_patch) / total_intensity
        mu02 = np.sum((y_indices - cy)**2 * gray_patch) / total_intensity  
        mu11 = np.sum((x_indices - cx) * (y_indices - cy) * gray_patch) / total_intensity
        
        # Eccentricity measure
        eccentricity = np.sqrt((mu20 - mu02)**2 + 4*mu11**2) / (mu20 + mu02 + 1e-8)
        
        features.extend([
            cx - gray_patch.shape[1]/2,    # Centroid offset x
            cy - gray_patch.shape[0]/2,    # Centroid offset y
            eccentricity,                  # Shape eccentricity
        ])
    else:
        features.extend([0, 0, 0])
    
    return np.array(features)


def extract_context_features(image, center_x, center_y, patch_size):
    """
    Extract contextual features around a candidate.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Full image
    center_x, center_y : int
        Candidate coordinates
    patch_size : int
        Size of local patch
        
    Returns:
    --------
    context_features : numpy.ndarray
        Contextual feature vector
    """
    features = []
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray_image = image
    
    H, W = gray_image.shape
    
    # 1. Multi-scale context (different radii around candidate)
    radii = [patch_size//2, patch_size, patch_size*2]
    
    for radius in radii:
        # Create circular mask
        y_grid, x_grid = np.mgrid[:H, :W]
        mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
        
        if np.any(mask):
            context_region = gray_image[mask]
            features.extend([
                np.mean(context_region),     # Mean intensity in radius
                np.std(context_region),      # Std intensity in radius
                np.sum(mask),                # Number of pixels in radius
            ])
        else:
            features.extend([0, 0, 0])
    
    # 2. Directional context (intensity in different directions)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
    for dx, dy in directions:
        # Sample points in this direction
        intensities = []
        for step in range(1, patch_size//2):
            x_sample = center_x + dx * step
            y_sample = center_y + dy * step
            if 0 <= x_sample < W and 0 <= y_sample < H:
                intensities.append(gray_image[y_sample, x_sample])
        
        if intensities:
            features.append(np.mean(intensities))  # Mean intensity in direction
        else:
            features.append(0)
    
    return np.array(features)


def predict_bcg_from_candidates(image, model=None, feature_scaler=None, **candidate_kwargs):
    """
    Complete pipeline: find candidates, extract features, predict best BCG.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    model : BCGCandidateClassifier or None
        Trained classifier model
    feature_scaler : StandardScaler or None
        Feature scaler
    **candidate_kwargs : dict
        Arguments for candidate finding
        
    Returns:
    --------
    best_bcg : tuple or None
        (x, y) coordinates of best BCG candidate
    all_candidates : numpy.ndarray
        All candidate coordinates
    candidate_scores : numpy.ndarray
        Scores for all candidates
    """
    # Find candidates
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