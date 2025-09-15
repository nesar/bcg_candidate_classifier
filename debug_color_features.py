#!/usr/bin/env python3
"""
Debug Color Features Analysis

This script helps debug the color feature extraction and identify issues
with duplicate SHAP values and feature naming problems.
"""

import sys
import numpy as np
from analysis.physical_interpretation import PhysicalFeatureInterpreter

def test_color_feature_interpretation():
    """Test the enhanced color feature interpretation."""
    
    print("Testing Enhanced Color Feature Interpretation")
    print("=" * 50)
    
    # Create interpreter
    interpreter = PhysicalFeatureInterpreter()
    
    # Simulate typical feature names from your system
    feature_names = [
        # Morphological features
        'patch_mean', 'patch_std', 'concentration', 'eccentricity',
        
        # Color PCA components
        'color_pca_0', 'color_pca_1', 'color_pca_2', 'color_pca_3',
        'color_pca_4', 'color_pca_5', 'color_pca_6', 'color_pca_7',
        
        # Color ratios  
        'color_ratio_rg', 'color_ratio_rb', 'color_ratio_gb',
        
        # Color convolution features
        'color_conv_r_edge', 'color_conv_g_edge', 'color_conv_b_edge',
        'color_conv_r_smooth', 'color_conv_g_smooth', 'color_conv_b_smooth',
        
        # Other features
        'context_mean', 'context_std'
    ]
    
    # Simulate importance values - some identical (like your issue)
    importance_values = [
        0.15, 0.12, 0.20, 0.08,  # morphological
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # identical color PCA values
        0.03, 0.04, 0.02,  # color ratios
        0.01, 0.01, 0.01, 0.02, 0.02, 0.02,  # convolution features
        0.06, 0.04  # context
    ]
    
    print(f"Total features: {len(feature_names)}")
    print(f"Total importance values: {len(importance_values)}")
    
    # Test feature name mapping
    print("\n1. Testing Feature Name Mapping:")
    print("-" * 30)
    mapped_names = interpreter.map_feature_names(feature_names)
    for orig, mapped in zip(feature_names, mapped_names):
        print(f"  {orig:20} -> {mapped}")
    
    # Test validation
    print("\n2. Feature Validation Results:")
    print("-" * 30)
    validation = interpreter.validate_feature_names(feature_names, importance_values)
    
    print(f"  Total features: {validation['total_features']}")
    print(f"  Unique features: {validation['unique_features']}")
    print(f"  Color PCA components: {len(validation['color_pca_components'])}")
    print(f"  Unmapped features: {len(validation['unmapped_features'])}")
    print(f"  Identical value groups: {len(validation['identical_values'])}")
    
    if validation['warnings']:
        print("\n  Warnings:")
        for warning in validation['warnings']:
            print(f"    ⚠️  {warning}")
    
    # Test color feature summary
    print("\n3. Detailed Color Feature Analysis:")
    interpreter.print_color_feature_summary(feature_names, importance_values)
    
    # Test physical importance computation
    print("\n4. Testing Physical Importance Computation:")
    print("-" * 40)
    
    # Simulate SHAP results structure
    mock_shap_results = {
        'shap': {
            'mean_abs_shap': importance_values,
            'raw_shap_values': np.random.randn(100, len(feature_names))  # mock data
        }
    }
    
    physical_results = interpreter.compute_physical_importance(mock_shap_results, feature_names)
    
    if 'shap' in physical_results:
        print("\nPhysical group importance scores:")
        for group_name, details in physical_results['shap']['group_details'].items():
            print(f"  {group_name:20}: {details['total_importance']:.4f} ({details['feature_count']} features)")

if __name__ == "__main__":
    test_color_feature_interpretation()