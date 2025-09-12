"""
Color Feature Extraction for BCG Classification

This module extracts color-based features from RGB patches to help distinguish
red-sequence cluster galaxies from bright white objects (stars, QSOs).

The color features complement the existing morphological and contextual features
by preserving information that is lost during grayscale conversion.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import convolve
from sklearn.decomposition import PCA


class ColorFeatureExtractor:
    """
    Extracts color-based features from RGB patches to identify red-sequence objects.
    
    Features include:
    1. Color ratios (g-r, r-i equivalents from RGB)
    2. Color gradients and spatial color variation
    3. Dimensionally reduced color information using PCA/filters
    """
    
    def __init__(self, use_pca_reduction=True, n_pca_components=8):
        """
        Initialize color feature extractor.
        
        Parameters:
        -----------
        use_pca_reduction : bool
            Whether to use PCA for dimensionality reduction of color features
        n_pca_components : int
            Number of PCA components to keep (default: 8)
        """
        self.use_pca_reduction = use_pca_reduction
        self.n_pca_components = n_pca_components
        self.pca_fitted = False
        self.pca_reducers = {}  # Separate PCA for different feature types
        
    def extract_color_features(self, rgb_patch):
        """
        Extract color features from an RGB patch.
        
        Parameters:
        -----------
        rgb_patch : numpy.ndarray
            RGB patch of shape (H, W, 3)
            
        Returns:
        --------
        color_features : numpy.ndarray
            Color feature vector
        """
        if len(rgb_patch.shape) != 3 or rgb_patch.shape[2] != 3:
            raise ValueError("Expected RGB patch of shape (H, W, 3)")
            
        features = []
        
        # 1. Basic color statistics
        basic_features = self._extract_basic_color_features(rgb_patch)
        features.extend(basic_features)
        
        # 2. Color ratios (red-sequence indicators) 
        ratio_features = self._extract_color_ratio_features(rgb_patch)
        features.extend(ratio_features)
        
        # 3. Spatial color variation
        spatial_features = self._extract_spatial_color_features(rgb_patch)
        features.extend(spatial_features)
        
        # 4. Color gradients
        gradient_features = self._extract_color_gradient_features(rgb_patch)
        features.extend(gradient_features)
        
        # 5. Convolution-based color features
        conv_features = self._extract_conv_color_features(rgb_patch)
        features.extend(conv_features)
        
        return np.array(features)
    
    def _extract_basic_color_features(self, rgb_patch):
        """Extract basic color statistics."""
        features = []
        
        R, G, B = rgb_patch[:, :, 0], rgb_patch[:, :, 1], rgb_patch[:, :, 2]
        
        # Mean values in each channel
        features.extend([np.mean(R), np.mean(G), np.mean(B)])
        
        # Standard deviation in each channel  
        features.extend([np.std(R), np.std(G), np.std(B)])
        
        # Channel-wise intensity ratios
        total_intensity = R + G + B + 1e-8
        features.extend([
            np.mean(R / total_intensity),  # Relative red contribution
            np.mean(G / total_intensity),  # Relative green contribution  
            np.mean(B / total_intensity),  # Relative blue contribution
        ])
        
        return features
    
    def _extract_color_ratio_features(self, rgb_patch):
        """Extract color ratio features that indicate red-sequence properties."""
        features = []
        
        R, G, B = rgb_patch[:, :, 0], rgb_patch[:, :, 1], rgb_patch[:, :, 2]
        
        # Approximate photometric color indices
        # These are rough approximations of g-r, r-i colors from RGB
        
        # R/G ratio (approximates r-g color)
        rg_ratio = np.mean(R / (G + 1e-8))
        features.append(rg_ratio)
        
        # (R-G)/(R+G) normalized color difference 
        rg_diff = np.mean((R - G) / (R + G + 1e-8))
        features.append(rg_diff)
        
        # R/B ratio  
        rb_ratio = np.mean(R / (B + 1e-8))
        features.append(rb_ratio)
        
        # (R-B)/(R+B) normalized color difference
        rb_diff = np.mean((R - B) / (R + B + 1e-8))
        features.append(rb_diff)
        
        # G/B ratio
        gb_ratio = np.mean(G / (B + 1e-8))
        features.append(gb_ratio)
        
        # Color magnitude: how "colorful" vs "white" the patch is
        # White objects have R≈G≈B, colored objects have different ratios
        color_magnitude = np.mean(np.sqrt((R-G)**2 + (G-B)**2 + (R-B)**2) / (R + G + B + 1e-8))
        features.append(color_magnitude)
        
        # Red-sequence indicator: higher values for redder objects
        red_sequence_score = np.mean((R - 0.5*(G + B)) / (R + G + B + 1e-8))
        features.append(red_sequence_score)
        
        return features
    
    def _extract_spatial_color_features(self, rgb_patch):
        """Extract features describing spatial variation of color."""
        features = []
        
        R, G, B = rgb_patch[:, :, 0], rgb_patch[:, :, 1], rgb_patch[:, :, 2]
        
        # Color uniformity: how uniform the colors are across the patch
        # More uniform = more likely to be a single galaxy vs blend of objects
        
        # Standard deviation of color ratios across the patch
        rg_ratios = R / (G + 1e-8)
        rb_ratios = R / (B + 1e-8) 
        
        features.extend([
            np.std(rg_ratios),  # Spatial variation in R/G
            np.std(rb_ratios),  # Spatial variation in R/B
        ])
        
        # Central vs peripheral color differences
        center = rgb_patch.shape[0] // 2
        central_radius = max(4, rgb_patch.shape[0] // 8)
        
        central_region = rgb_patch[center-central_radius:center+central_radius, 
                                  center-central_radius:center+central_radius]
        
        # Create peripheral mask
        peripheral_mask = np.ones(rgb_patch.shape[:2], dtype=bool)
        peripheral_mask[center-central_radius:center+central_radius, 
                       center-central_radius:center+central_radius] = False
        
        if np.any(peripheral_mask):
            peripheral_region = rgb_patch[peripheral_mask]
            
            # Compare central and peripheral color properties
            central_rg = np.mean(central_region[:, :, 0] / (central_region[:, :, 1] + 1e-8))
            peripheral_rg = np.mean(peripheral_region[:, 0] / (peripheral_region[:, 1] + 1e-8))
            
            features.append(central_rg - peripheral_rg)  # Central-peripheral R/G difference
        else:
            features.append(0.0)
        
        return features
    
    def _extract_color_gradient_features(self, rgb_patch):
        """Extract color gradient features."""
        features = []
        
        R, G, B = rgb_patch[:, :, 0], rgb_patch[:, :, 1], rgb_patch[:, :, 2]
        
        # Compute gradients in each channel
        for channel in [R, G, B]:
            grad_x = np.gradient(channel, axis=1)  
            grad_y = np.gradient(channel, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),   # Mean gradient magnitude
                np.std(gradient_magnitude),    # Gradient variation
            ])
        
        # Cross-channel gradient correlations
        # Objects with coherent color gradients vs noise
        r_grad_mag = np.sqrt(np.gradient(R, axis=1)**2 + np.gradient(R, axis=0)**2)
        g_grad_mag = np.sqrt(np.gradient(G, axis=1)**2 + np.gradient(G, axis=0)**2)
        b_grad_mag = np.sqrt(np.gradient(B, axis=1)**2 + np.gradient(B, axis=0)**2)
        
        # Correlation between channel gradients (coherent structure indicator)
        rg_grad_corr = np.corrcoef(r_grad_mag.flatten(), g_grad_mag.flatten())[0, 1]
        rb_grad_corr = np.corrcoef(r_grad_mag.flatten(), b_grad_mag.flatten())[0, 1]
        
        # Handle NaN correlations (when one channel has no variation)
        features.extend([
            rg_grad_corr if not np.isnan(rg_grad_corr) else 0.0,
            rb_grad_corr if not np.isnan(rb_grad_corr) else 0.0,
        ])
        
        return features
    
    def _extract_conv_color_features(self, rgb_patch):
        """Extract convolution-based color features for dimensionality reduction."""
        features = []
        
        # Define simple convolution kernels for color feature extraction
        # These act as learned filters that capture color patterns
        
        kernels = {
            'edge_3x3': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'smooth_3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            'laplacian': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        }
        
        for kernel_name, kernel in kernels.items():
            for channel_idx in range(3):  # R, G, B
                channel = rgb_patch[:, :, channel_idx]
                convolved = convolve(channel, kernel, mode='constant')
                
                features.extend([
                    np.mean(convolved),       # Mean response
                    np.std(convolved),        # Response variation
                    np.max(np.abs(convolved)) # Max absolute response
                ])
        
        return features
    
    def fit_pca_reduction(self, color_features_list):
        """
        Fit PCA reduction on a collection of color features.
        
        Parameters:
        -----------
        color_features_list : list of numpy.ndarray
            List of color feature vectors for PCA fitting
        """
        if not self.use_pca_reduction:
            return
            
        if len(color_features_list) == 0:
            return
            
        # Stack all features
        all_features = np.array(color_features_list)
        
        # Fit PCA
        self.pca_reducer = PCA(n_components=self.n_pca_components, random_state=42)
        self.pca_reducer.fit(all_features)
        self.pca_fitted = True
        
        print(f"Color features PCA fitted: {all_features.shape[1]} → {self.n_pca_components} dimensions")
        print(f"Explained variance ratio: {np.sum(self.pca_reducer.explained_variance_ratio_):.3f}")
    
    def reduce_color_features(self, color_features):
        """
        Apply PCA dimensionality reduction to color features.
        
        Parameters:
        -----------
        color_features : numpy.ndarray
            Color features to reduce
            
        Returns:
        --------
        reduced_features : numpy.ndarray
            Dimensionally reduced color features
        """
        if not self.use_pca_reduction or not self.pca_fitted:
            return color_features
            
        return self.pca_reducer.transform(color_features.reshape(1, -1)).flatten()


def extract_candidate_color_features(image, candidate_coords, patch_size=64, 
                                    color_extractor=None):
    """
    Extract color features for all candidates in an image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB image of shape (H, W, 3)
    candidate_coords : numpy.ndarray  
        Array of (x, y) coordinates for candidates
    patch_size : int
        Size of patch to extract around each candidate
    color_extractor : ColorFeatureExtractor
        Fitted color feature extractor
        
    Returns:
    --------
    color_features : numpy.ndarray
        Array of shape (N_candidates, color_feature_dim) containing color features
    """
    if len(candidate_coords) == 0:
        return np.array([])
    
    # Ensure we have an RGB image
    if len(image.shape) == 2:
        # Convert grayscale to RGB by duplicating channels
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] != 3:
        raise ValueError("Expected RGB image with 3 channels")
    
    if color_extractor is None:
        color_extractor = ColorFeatureExtractor()
    
    H, W = image.shape[:2]
    half_patch = patch_size // 2
    
    color_features_list = []
    
    for x, y in candidate_coords:
        x, y = int(x), int(y)
        
        # Extract RGB patch around candidate
        x_min = max(0, x - half_patch)
        x_max = min(W, x + half_patch)
        y_min = max(0, y - half_patch)  
        y_max = min(H, y + half_patch)
        
        rgb_patch = image[y_min:y_max, x_min:x_max]
        
        # Pad if necessary to maintain consistent size
        if rgb_patch.shape[0] < patch_size or rgb_patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
            pad_y = (patch_size - rgb_patch.shape[0]) // 2
            pad_x = (patch_size - rgb_patch.shape[1]) // 2
            padded_patch[pad_y:pad_y+rgb_patch.shape[0], pad_x:pad_x+rgb_patch.shape[1]] = rgb_patch
            rgb_patch = padded_patch
        
        # Extract color features from this patch
        color_features = color_extractor.extract_color_features(rgb_patch)
        
        # Apply PCA reduction if fitted
        if color_extractor.use_pca_reduction and color_extractor.pca_fitted:
            color_features = color_extractor.reduce_color_features(color_features)
        
        color_features_list.append(color_features)
    
    return np.array(color_features_list) if color_features_list else np.array([])