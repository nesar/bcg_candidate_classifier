"""
Utility functions for feature name generation and management.

This module provides utilities for creating consistent feature names
that match the feature extraction pipeline in the BCG classification system.
"""

import numpy as np
from typing import List, Dict, Optional


def create_bcg_feature_names(use_color_features=True, use_auxiliary_features=True, 
                            color_pca_components=8) -> List[str]:
    """
    Create feature names matching the BCG classification feature extraction pipeline.
    
    Args:
        use_color_features: Whether color features are included
        use_auxiliary_features: Whether auxiliary features (redshift, delta_m) are included
        color_pca_components: Number of PCA components for color features
        
    Returns:
        List of feature names in order
    """
    feature_names = []
    
    # 1. Intensity statistics (5 features)
    intensity_features = [
        'patch_mean', 'patch_std', 'patch_max', 'patch_min', 'patch_median'
    ]
    feature_names.extend(intensity_features)
    
    # 2. Central vs peripheral analysis (3 features)
    central_peripheral_features = [
        'central_mean',         # central region intensity
        'peripheral_mean',      # peripheral region intensity  
        'concentration_ratio'   # central vs peripheral ratio
    ]
    feature_names.extend(central_peripheral_features)
    
    # 3. Gradient features (3 features)
    gradient_features = [
        'gradient_mean', 'gradient_std', 'gradient_max'
    ]
    feature_names.extend(gradient_features)
    
    # 4. Position features (3 features)
    position_features = [
        'x_relative',           # normalized x position
        'y_relative',           # normalized y position  
        'r_center'              # distance from image center
    ]
    feature_names.extend(position_features)
    
    # 5. Shape/symmetry features (3 features)
    shape_features = [
        'centroid_offset_x',    # centroid offset x
        'centroid_offset_y',    # centroid offset y
        'eccentricity'          # departure from circular symmetry
    ]
    feature_names.extend(shape_features)
    
    # 6. Multi-scale context features (9 features)
    context_multiscale_features = []
    for scale in ['small', 'medium', 'large']:  # 3 radii
        context_multiscale_features.extend([
            f'context_{scale}_mean',      # mean intensity
            f'context_{scale}_std',       # std intensity  
            f'context_{scale}_pixels'     # number of pixels
        ])
    feature_names.extend(context_multiscale_features)
    
    # 7. Directional context features (4 features)
    directional_context_features = [
        'context_north_mean', 'context_east_mean', 
        'context_south_mean', 'context_west_mean'
    ]
    feature_names.extend(directional_context_features)
    
    # 8. Color features (if enabled)
    if use_color_features:
        # Basic color ratios
        color_ratio_features = [
            'rg_ratio_mean', 'rb_ratio_mean', 'color_magnitude'
        ]
        feature_names.extend(color_ratio_features)
        
        # Red-sequence indicator
        feature_names.append('red_sequence_score')
        
        # Spatial color variation
        color_variation_features = [
            'rg_ratio_std', 'rb_ratio_std'
        ]
        feature_names.extend(color_variation_features)
        
        # Color gradient correlations
        color_correlation_features = [
            'color_gradient_corr_rg', 'color_gradient_corr_rb'
        ]
        feature_names.extend(color_correlation_features)
        
        # Convolution-based color features (edge, smooth, laplacian responses)
        conv_features = []
        for channel in ['r', 'g', 'b']:
            for kernel in ['edge', 'smooth', 'laplacian']:
                conv_features.append(f'color_conv_{channel}_{kernel}')
        feature_names.extend(conv_features)
        
        # PCA-reduced color features
        pca_features = [f'color_pca_{i}' for i in range(color_pca_components)]
        feature_names.extend(pca_features)
    
    # 9. Auxiliary astronomical features (if enabled)
    if use_auxiliary_features:
        auxiliary_features = [
            'redshift_z',           # photometric redshift
            'delta_m_star_z'        # stellar mass indicator
        ]
        feature_names.extend(auxiliary_features)
    
    return feature_names


def get_feature_groups_mapping(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Create feature group mapping for analysis.
    Follows the updated classification scheme: Intensity Statistics, Morphology, Color Information, Auxiliary.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping group names to feature lists
    """
    groups = {
        'intensity_statistics': [],
        'morphology': [],
        'color_information': [],
        'auxiliary': []
    }
    
    # Intensity statistics features (surface brightness)
    intensity_keywords = [
        'patch_mean', 'patch_std', 'patch_max', 'patch_min', 'patch_median',
        'central_mean', 'peripheral_mean', 'concentration_ratio'
    ]
    
    # Morphological features (shape, structure, position, context)
    morphological_keywords = [
        'gradient', 'relative', 'center', 'centroid', 'eccentricity', 'context_'
    ]
    
    # Color information features
    color_keywords = [
        'ratio', 'color', 'red_sequence', 'pca'
    ]
    
    # Auxiliary features
    auxiliary_keywords = [
        'redshift', 'delta_m'
    ]
    
    # Classify features
    for feature_name in feature_names:
        classified = False
        
        # Check intensity statistics (exact match preferred)
        for keyword in intensity_keywords:
            if keyword in feature_name:
                groups['intensity_statistics'].append(feature_name)
                classified = True
                break
        
        if classified:
            continue
            
        # Check color
        for keyword in color_keywords:
            if keyword in feature_name:
                groups['color_information'].append(feature_name)
                classified = True
                break
        
        if classified:
            continue
            
        # Check auxiliary
        for keyword in auxiliary_keywords:
            if keyword in feature_name:
                groups['auxiliary'].append(feature_name)
                classified = True
                break
        
        if classified:
            continue
            
        # Check morphological (includes contextual features)
        for keyword in morphological_keywords:
            if keyword in feature_name:
                groups['morphology'].append(feature_name)
                classified = True
                break
        
        # If not classified, add to morphology as default
        if not classified:
            groups['morphology'].append(feature_name)
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    
    return groups


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get detailed descriptions of BCG classification features.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    descriptions = {
        # Intensity statistics
        'patch_mean': 'Mean intensity within 64x64 patch around candidate',
        'patch_std': 'Standard deviation of intensity within patch',
        'patch_max': 'Maximum intensity within patch',
        'patch_min': 'Minimum intensity within patch',
        'patch_median': 'Median intensity within patch',
        'patch_skew': 'Skewness of intensity distribution within patch',
        
        # Shape and concentration
        'concentration_ratio': 'Ratio of central (16x16) to peripheral intensity',
        'eccentricity': 'Departure from circular symmetry based on intensity moments',
        
        # Gradient features
        'gradient_mean': 'Mean gradient magnitude within patch',
        'gradient_std': 'Standard deviation of gradient magnitude',
        'gradient_max': 'Maximum gradient magnitude within patch',
        
        # Geometric moments
        'moment_m20': 'Second-order intensity moment in x-direction',
        'moment_m02': 'Second-order intensity moment in y-direction',  
        'moment_m11': 'Mixed second-order intensity moment',
        
        # Directional features
        'north_intensity_mean': 'Mean intensity in northward direction from candidate',
        'east_intensity_mean': 'Mean intensity in eastward direction from candidate',
        'south_intensity_mean': 'Mean intensity in southward direction from candidate',
        'west_intensity_mean': 'Mean intensity in westward direction from candidate',
        
        # Contextual features
        'x_relative': 'Normalized x-position within image (-1 to 1)',
        'y_relative': 'Normalized y-position within image (-1 to 1)',
        'r_center': 'Distance from image center (normalized)',
        'brightness_rank': 'Rank of candidate by brightness among all candidates',
        'candidate_density': 'Local density of other candidates nearby',
        'background_level': 'Local background intensity level',
        
        # Color features
        'rg_ratio_mean': 'Mean R/G color ratio within patch (approximates r-g color)',
        'rb_ratio_mean': 'Mean R/B color ratio within patch (approximates r-i color)',
        'color_magnitude': 'Departure from achromatic (white) colors',
        'red_sequence_score': 'Score indicating red-sequence galaxy characteristics',
        'rg_ratio_std': 'Standard deviation of R/G ratio (spatial color variation)',
        'rb_ratio_std': 'Standard deviation of R/B ratio (spatial color variation)',
        'color_gradient_corr_rg': 'Correlation between R and G channel gradients',
        'color_gradient_corr_rb': 'Correlation between R and B channel gradients',
        
        # Auxiliary features
        'redshift_z': 'Photometric redshift estimate',
        'delta_m_star_z': 'Magnitude difference from characteristic stellar mass at redshift z'
    }
    
    # Add PCA and convolution feature descriptions
    for i in range(20):  # Up to 20 PCA components
        descriptions[f'color_pca_{i}'] = f'Principal component {i+1} of color feature space'
    
    for channel in ['r', 'g', 'b']:
        for kernel in ['edge', 'smooth', 'laplacian']:
            desc_map = {
                'edge': 'edge detection response',
                'smooth': 'smoothing filter response', 
                'laplacian': 'Laplacian edge enhancement response'
            }
            descriptions[f'color_conv_{channel}_{kernel}'] = f'{channel.upper()}-channel {desc_map[kernel]}'
    
    return descriptions


def validate_feature_names(feature_names: List[str], expected_count: Optional[int] = None) -> Dict:
    """
    Validate feature names and check for consistency.
    
    Args:
        feature_names: List of feature names to validate
        expected_count: Expected number of features
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'feature_count': len(feature_names),
        'has_duplicates': len(feature_names) != len(set(feature_names)),
        'duplicates': [],
        'missing_groups': [],
        'warnings': []
    }
    
    # Check for duplicates
    if results['has_duplicates']:
        seen = set()
        for name in feature_names:
            if name in seen:
                results['duplicates'].append(name)
            seen.add(name)
        results['valid'] = False
    
    # Check expected count
    if expected_count and len(feature_names) != expected_count:
        results['warnings'].append(
            f"Expected {expected_count} features, got {len(feature_names)}"
        )
    
    # Check for standard feature groups
    groups = get_feature_groups_mapping(feature_names)
    expected_groups = ['luminosity_profile', 'morphology']
    
    for group in expected_groups:
        if group not in groups or len(groups[group]) == 0:
            results['missing_groups'].append(group)
    
    if results['missing_groups']:
        results['warnings'].append(
            f"Missing standard feature groups: {results['missing_groups']}"
        )
    
    return results


def print_feature_summary(feature_names: List[str]):
    """
    Print a summary of features organized by groups.
    
    Args:
        feature_names: List of feature names
    """
    print(f"=== BCG Classification Features Summary ===")
    print(f"Total Features: {len(feature_names)}\n")
    
    # Get feature groups
    groups = get_feature_groups_mapping(feature_names)
    descriptions = get_feature_descriptions()
    
    for group_name, features in groups.items():
        print(f"=== {group_name.upper()} FEATURES ({len(features)}) ===")
        
        for i, feature in enumerate(features, 1):
            desc = descriptions.get(feature, "No description available")
            print(f"{i:2d}. {feature:<25} : {desc}")
        
        print()
    
    # Validation
    validation = validate_feature_names(feature_names)
    if not validation['valid'] or validation['warnings']:
        print("=== VALIDATION RESULTS ===")
        if validation['duplicates']:
            print(f"⚠️  Duplicate features: {validation['duplicates']}")
        for warning in validation['warnings']:
            print(f"⚠️  {warning}")
        print()


if __name__ == "__main__":
    # Example usage
    print("Generating BCG feature names...")
    
    # Full feature set
    full_features = create_bcg_feature_names(
        use_color_features=True,
        use_auxiliary_features=True,
        color_pca_components=8
    )
    
    print_feature_summary(full_features)
    
    # Minimal feature set
    minimal_features = create_bcg_feature_names(
        use_color_features=False,
        use_auxiliary_features=False
    )
    
    print(f"\nMinimal feature set: {len(minimal_features)} features")
    print("Minimal features:", minimal_features[:10], "...")