"""
Physical Feature Interpretation for BCG Classification

This module provides mappings from technical feature names to physically meaningful
interpretations, allowing for more intuitive understanding of feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PhysicalFeatureInterpreter:
    """
    Interprets technical features in terms of physical/astronomical meanings.
    """
    
    def __init__(self):
        self.feature_groups = self._create_feature_groups()
        self.feature_mappings = self._create_feature_mappings()
    
    def validate_feature_names(self, feature_names: List[str], importance_values: Optional[List[float]] = None) -> Dict:
        """
        Validate feature names and identify potential issues.
        
        Args:
            feature_names: List of feature names
            importance_values: Optional list of importance values to check for duplicates
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_features': len(feature_names),
            'unique_features': len(set(feature_names)),
            'duplicates': [],
            'color_pca_components': [],
            'unmapped_features': [],
            'identical_values': [],
            'warnings': []
        }
        
        # Check for duplicate feature names
        seen_names = {}
        for i, name in enumerate(feature_names):
            if name in seen_names:
                validation_results['duplicates'].append({
                    'name': name,
                    'indices': [seen_names[name], i]
                })
            else:
                seen_names[name] = i
        
        # Check color PCA components
        color_pca_features = [name for name in feature_names if name.startswith('color_pca_')]
        validation_results['color_pca_components'] = color_pca_features
        
        # Check for unmapped features
        for name in feature_names:
            if name not in self.feature_mappings and not any([
                name.startswith('color_pca_'),
                name.startswith('color_conv_'),
                name.startswith('moment_'),
                name.startswith('context_'),
                name.startswith('feature_')
            ]):
                validation_results['unmapped_features'].append(name)
        
        # Check for identical importance values if provided
        if importance_values is not None:
            if len(importance_values) == len(feature_names):
                value_groups = {}
                for i, (name, value) in enumerate(zip(feature_names, importance_values)):
                    # Round to avoid floating point precision issues
                    rounded_value = round(value, 10)
                    if rounded_value not in value_groups:
                        value_groups[rounded_value] = []
                    value_groups[rounded_value].append((i, name))
                
                # Find groups with multiple features
                for value, features in value_groups.items():
                    if len(features) > 1:
                        feature_list = [f[1] for f in features]
                        
                        # Special analysis for morphology features
                        morphology_features = [f for f in feature_list if any(
                            morph_pattern in f for morph_pattern in ['moment_', 'concentration', 'eccentricity', 'gradient_']
                        )]
                        
                        validation_results['identical_values'].append({
                            'value': value,
                            'count': len(features),
                            'features': feature_list,
                            'morphology_count': len(morphology_features),
                            'morphology_features': morphology_features
                        })
        
        # Generate warnings
        if validation_results['duplicates']:
            validation_results['warnings'].append(f"Found {len(validation_results['duplicates'])} duplicate feature names")
        
        if len(validation_results['color_pca_components']) != 8:
            validation_results['warnings'].append(f"Expected 8 color PCA components, found {len(validation_results['color_pca_components'])}")
        
        if validation_results['identical_values']:
            identical_count = sum(group['count'] for group in validation_results['identical_values'])
            validation_results['warnings'].append(f"Found {identical_count} features with identical importance values")
        
        return validation_results
    
    def print_color_feature_summary(self, feature_names: List[str], importance_values: Optional[List[float]] = None):
        """
        Print a detailed summary of color features for debugging.
        
        Args:
            feature_names: List of feature names
            importance_values: Optional importance values
        """
        print("\n" + "="*60)
        print("COLOR FEATURE ANALYSIS SUMMARY")
        print("="*60)
        
        # Find all color-related features
        color_features = {
            'pca_components': [],
            'ratios': [],
            'convolution': [],
            'spatial': [],
            'gradient': [],
            'other': []
        }
        
        for i, name in enumerate(feature_names):
            importance = importance_values[i] if importance_values else None
            
            if name.startswith('color_pca_'):
                color_features['pca_components'].append((name, importance))
            elif 'ratio' in name and 'color' in name:
                color_features['ratios'].append((name, importance))
            elif name.startswith('color_conv_'):
                color_features['convolution'].append((name, importance))
            elif 'spatial' in name and 'color' in name:
                color_features['spatial'].append((name, importance))
            elif 'grad' in name and 'color' in name:
                color_features['gradient'].append((name, importance))
            elif 'color' in name:
                color_features['other'].append((name, importance))
        
        # Print summary for each category
        for category, features in color_features.items():
            if features:
                print(f"\n{category.upper().replace('_', ' ')} ({len(features)} features):")
                for name, importance in features:
                    mapped_name = self.map_feature_names([name])[0]
                    if importance is not None:
                        print(f"  {name:20} -> {mapped_name:35} (importance: {importance:.6f})")
                    else:
                        print(f"  {name:20} -> {mapped_name}")
        
        # Check for identical importance values among color features
        if importance_values:
            all_color_features = []
            for category_features in color_features.values():
                all_color_features.extend(category_features)
            
            color_values = [imp for _, imp in all_color_features if imp is not None]
            color_names = [name for name, imp in all_color_features if imp is not None]
            
            if color_values:
                validation = self.validate_feature_names(color_names, color_values)
                if validation['identical_values']:
                    print(f"\n⚠️  IDENTICAL COLOR FEATURE VALUES:")
                    for group in validation['identical_values']:
                        print(f"    Value {group['value']:.6f}: {group['features']}")
        
        print("="*60)
    
    def _create_feature_groups(self) -> Dict[str, Dict]:
        """Create physical feature groups from technical features."""
        return {
            'intensity_statistics': {
                'description': 'Surface brightness and luminosity distribution (8 features)',
                'technical_features': ['patch_mean', 'patch_std', 'patch_median', 'patch_max', 'patch_min',
                                     'central_mean', 'peripheral_mean', 'concentration_ratio'],
                'combination_method': 'weighted_sum',
                'color': '#FF6B6B'
            },
            'morphology': {
                'description': 'Galaxy shape, structure and environmental context (22 features)',
                'technical_features': ['gradient_mean', 'gradient_std', 'gradient_max',
                                     'x_relative', 'y_relative', 'r_center',
                                     'centroid_offset_x', 'centroid_offset_y', 'eccentricity',
                                     'context_small_mean', 'context_small_std', 'context_small_pixels',
                                     'context_medium_mean', 'context_medium_std', 'context_medium_pixels',
                                     'context_large_mean', 'context_large_std', 'context_large_pixels',
                                     'context_north_mean', 'context_east_mean', 'context_south_mean', 'context_west_mean'],
                'combination_method': 'weighted_sum',
                'color': '#4ECDC4'
            },
            'color_information': {
                'description': 'Red-sequence and color properties (25 features: 17 raw + 8 PCA from 54 features)',
                'technical_features': ['rg_ratio_mean', 'rb_ratio_mean', 'color_magnitude', 'red_sequence_score',
                                     'rg_ratio_std', 'rb_ratio_std', 'color_gradient_corr_rg', 'color_gradient_corr_rb',
                                     'color_conv_r_edge', 'color_conv_r_smooth', 'color_conv_r_laplacian',
                                     'color_conv_g_edge', 'color_conv_g_smooth', 'color_conv_g_laplacian',
                                     'color_conv_b_edge', 'color_conv_b_smooth', 'color_conv_b_laplacian',
                                     'color_pca_0', 'color_pca_1', 'color_pca_2', 'color_pca_3', 'color_pca_4',
                                     'color_pca_5', 'color_pca_6', 'color_pca_7'],
                'combination_method': 'weighted_sum',
                'color': '#45B7D1'
            },
            'auxiliary': {
                'description': 'Cosmological and physical parameters (2 features)',
                'technical_features': ['redshift_z', 'delta_m_star_z'],
                'combination_method': 'weighted_sum',
                'color': '#FECA57'
            }
        }
    
    def _create_feature_mappings(self) -> Dict[str, str]:
        """Create individual feature name mappings."""
        return {
            # Intensity statistics (8 features)
            'patch_mean': 'Mean Surface Brightness',
            'patch_std': 'Brightness Variability',
            'patch_median': 'Median Luminosity',
            'patch_max': 'Peak Brightness',
            'patch_min': 'Background Level',
            'central_mean': 'Central Region Brightness',
            'peripheral_mean': 'Peripheral Region Brightness',
            'concentration_ratio': 'Central Light Concentration Ratio',
            
            # Morphology (22 features)
            'gradient_mean': 'Mean Edge Strength',
            'gradient_std': 'Edge Structure Variability',
            'gradient_max': 'Maximum Edge Response',
            'x_relative': 'Normalized X Position',
            'y_relative': 'Normalized Y Position',
            'r_center': 'Distance from Image Center',
            'centroid_offset_x': 'Light Centroid Offset X',
            'centroid_offset_y': 'Light Centroid Offset Y',
            'eccentricity': 'Galaxy Ellipticity',
            'context_small_mean': 'Local Environment Brightness',
            'context_small_std': 'Local Environment Variability',
            'context_small_pixels': 'Local Environment Size',
            'context_medium_mean': 'Intermediate Environment Brightness',
            'context_medium_std': 'Intermediate Environment Variability',
            'context_medium_pixels': 'Intermediate Environment Size',
            'context_large_mean': 'Extended Environment Brightness',
            'context_large_std': 'Extended Environment Variability',
            'context_large_pixels': 'Extended Environment Size',
            'context_north_mean': 'Northern Environment Brightness',
            'context_east_mean': 'Eastern Environment Brightness',
            'context_south_mean': 'Southern Environment Brightness',
            'context_west_mean': 'Western Environment Brightness',
            
            # Color PCA Components - each represents different color aspects
            'color_pca_0': 'Red-Sequence Color (PC1)',
            'color_pca_1': 'Blue-Red Contrast (PC2)',
            'color_pca_2': 'Color Uniformity (PC3)',
            'color_pca_3': 'Spatial Color Gradient (PC4)',
            'color_pca_4': 'Multi-Band Color (PC5)',
            'color_pca_5': 'Color Asymmetry (PC6)',
            'color_pca_6': 'Chromatic Structure (PC7)',
            'color_pca_7': 'Color Noise Pattern (PC8)',
            
            # Direct Color Ratios
            'color_ratio_rg': 'Red-Green Color',
            'color_ratio_rb': 'Red-Blue Color',
            'color_ratio_gb': 'Green-Blue Color',
            'color_variation': 'Spatial Color Variation',
            
            # Environment
            'context_mean': 'Local Background',
            'context_std': 'Environmental Variability',
            'context_gradient': 'Density Gradient',
            'context_density': 'Local Galaxy Density',
            'neighbor_count': 'Nearby Galaxy Count',
            'local_density': 'Cluster Core Density',
            
            # Auxiliary features (2 features)
            'redshift_z': 'Photometric Redshift',
            'delta_m_star_z': 'Stellar Mass Indicator',
            
            # DESprior
            'delta_mstar': 'Stellar Mass Excess',
            'starflag': 'Star/Galaxy Classifier',
            'mag_auto_g': 'g-band Magnitude',
            'mag_auto_r': 'r-band Magnitude',
            'mag_auto_i': 'i-band Magnitude',
        }
    
    def map_feature_names(self, feature_names: List[str]) -> List[str]:
        """Map technical feature names to physical descriptions."""
        mapped_names = []
        for name in feature_names:
            if name in self.feature_mappings:
                mapped_names.append(self.feature_mappings[name])
            elif name.startswith('feature_'):
                # Handle generic features
                idx = name.split('_')[1]
                mapped_names.append(f'Extended Feature {idx}')
            elif name.startswith('color_pca_'):
                # Handle color PCA components with proper numbering
                try:
                    idx = int(name.split('_')[2])
                    # Map to predefined PCA component meanings
                    pca_meanings = [
                        'Red-Sequence Color (PC1)',
                        'Blue-Red Contrast (PC2)',
                        'Color Uniformity (PC3)',
                        'Spatial Color Gradient (PC4)',
                        'Multi-Band Color (PC5)',
                        'Color Asymmetry (PC6)',
                        'Chromatic Structure (PC7)',
                        'Color Noise Pattern (PC8)'
                    ]
                    if idx < len(pca_meanings):
                        mapped_names.append(pca_meanings[idx])
                    else:
                        mapped_names.append(f'Color Component {idx+1}')
                except (ValueError, IndexError):
                    mapped_names.append(f'Color Component (Unknown)')
            elif name.startswith('color_conv_'):
                # Handle color convolution features
                parts = name.split('_')
                if len(parts) >= 4:
                    channel = parts[2].upper()
                    kernel = parts[3].title()
                    mapped_names.append(f'{channel}-Channel {kernel} Response')
                else:
                    mapped_names.append('Color Convolution Feature')
            elif name.startswith('color_'):
                # Handle other color features with better naming
                clean_name = name.replace('color_', '').replace('_', ' ').title()
                mapped_names.append(f'Color: {clean_name}')
            else:
                # Keep original name if no mapping found
                mapped_names.append(name.replace('_', ' ').title())
        
        return mapped_names
    
    def compute_physical_importance(self, importance_results: Dict, feature_names: List[str]) -> Dict:
        """
        Compute importance scores for physical feature groups.
        
        Args:
            importance_results: Raw importance results from different methods
            feature_names: List of technical feature names
            
        Returns:
            Dictionary with physical group importance scores
        """
        # Validate feature names first
        validation_results = self.validate_feature_names(feature_names)
        if validation_results['warnings']:
            print("⚠️  Feature validation warnings:")
            for warning in validation_results['warnings']:
                print(f"    {warning}")
        
        physical_results = {
            'validation': validation_results
        }
        
        for method, results in importance_results.items():
            if method == 'shap':
                importance_scores = results['mean_abs_shap']
            else:
                importance_scores = results['importance']
            
            # Validate importance values for this method
            method_validation = self.validate_feature_names(feature_names, importance_scores)
            if method_validation['identical_values']:
                print(f"⚠️  {method.upper()} has identical values:")
                for group in method_validation['identical_values']:
                    print(f"    Value {group['value']:.6f}: {group['features']}")
            
            # Create feature importance mapping
            feature_importance_map = dict(zip(feature_names, importance_scores))
            
            # Compute group importance
            group_scores = {}
            group_details = {}
            
            for group_name, group_info in self.feature_groups.items():
                group_importance = 0.0
                group_count = 0
                contributing_features = []
                
                for tech_feature in group_info['technical_features']:
                    # Find matching features (handles partial matches and extensions)
                    matching_features = [f for f in feature_names if self._feature_matches(f, tech_feature)]
                    
                    for feature in matching_features:
                        if feature in feature_importance_map:
                            importance = abs(feature_importance_map[feature])
                            group_importance += importance
                            group_count += 1
                            contributing_features.append((feature, importance))
                
                if group_count > 0:
                    group_scores[group_name] = group_importance
                    group_details[group_name] = {
                        'total_importance': group_importance,
                        'average_importance': group_importance / group_count,
                        'feature_count': group_count,
                        'contributing_features': contributing_features,
                        'description': group_info['description'],
                        'color': group_info['color']
                    }
            
            physical_results[method] = {
                'group_scores': group_scores,
                'group_details': group_details
            }
        
        return physical_results
    
    def _feature_matches(self, actual_feature: str, template_feature: str) -> bool:
        """Check if an actual feature matches a template (handles prefixes and extensions)."""
        if actual_feature == template_feature:
            return True
        
        # Handle color_pca extensions - exact matching for PCA components
        if template_feature.startswith('color_pca_') and actual_feature.startswith('color_pca_'):
            # Only match if the component number is within the expected range
            try:
                actual_idx = int(actual_feature.split('_')[2])
                template_idx = int(template_feature.split('_')[2])
                return actual_idx == template_idx
            except (ValueError, IndexError):
                return actual_feature == template_feature
        
        # Handle color convolution features - exact match only
        if template_feature.startswith('color_conv_') and actual_feature.startswith('color_conv_'):
            return actual_feature == template_feature
        
        # Handle color gradient features - exact match only  
        if template_feature.startswith('color_grad_') and actual_feature.startswith('color_grad_'):
            return actual_feature == template_feature
        
        # Handle color spatial features - exact match only
        if template_feature.startswith('color_spatial_') and actual_feature.startswith('color_spatial_'):
            return actual_feature == template_feature
        
        # Handle context features - exact match only, no broad prefix matching
        if template_feature.startswith('context_') and actual_feature.startswith('context_'):
            # Make sure we don't match color context features to morphology
            if not actual_feature.startswith('color_'):
                # Exact match required for context features
                return actual_feature == template_feature
        
        # Handle moment extensions - exact match only
        if template_feature.startswith('moment_') and actual_feature.startswith('moment_'):
            return actual_feature == template_feature
        
        # REMOVED the overly broad general prefix matching rule that was causing issues
        
        return False
    
    def create_physical_importance_plot(self, physical_results: Dict, method: str, 
                                      save_path: Optional[Path] = None) -> plt.Figure:
        """Create importance plot with physical feature groups."""
        if method not in physical_results:
            raise ValueError(f"Method {method} not found in physical results")
        
        group_details = physical_results[method]['group_details']
        
        # Prepare data for plotting
        groups = list(group_details.keys())
        importances = [group_details[group]['total_importance'] for group in groups]
        colors = [group_details[group]['color'] for group in groups]
        
        # Sort by importance
        sorted_data = sorted(zip(groups, importances, colors), key=lambda x: x[1], reverse=True)
        groups, importances, colors = zip(*sorted_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(groups)), importances, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize plot
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([group.replace('_', ' ').title() for group in groups])
        ax.set_xlabel(f'{method.upper()} Importance Score', fontsize=12)
        ax.set_title(f'Physical Feature Group Importance ({method.upper()})', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        # Add descriptions as annotations
        for i, group in enumerate(groups):
            description = group_details[group]['description']
            feature_count = group_details[group]['feature_count']
            ax.text(0.02 * max(importances), i, f'({feature_count} features)', 
                   ha='left', va='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_detailed_breakdown_plot(self, physical_results: Dict, method: str,
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """Create detailed breakdown showing individual features within groups."""
        if method not in physical_results:
            raise ValueError(f"Method {method} not found in physical results")
        
        group_details = physical_results[method]['group_details']
        
        # Create subplot for each group with adaptive height
        n_groups = len(group_details)
        
        # Calculate adaptive height based on number of features in each group
        total_height = 0
        group_heights = []
        for group_name, details in group_details.items():
            n_features = len(details['contributing_features'])
            # Minimum 3 inches, then 0.4 inches per feature, with some padding
            group_height = max(3, n_features * 0.4 + 1.5)
            group_heights.append(group_height)
            total_height += group_height
        
        fig, axes = plt.subplots(n_groups, 1, figsize=(14, max(total_height, 6)))
        if n_groups == 1:
            axes = [axes]
        
        for idx, (group_name, details) in enumerate(group_details.items()):
            ax = axes[idx]
            
            # Get features and their importance
            features = [f[0] for f in details['contributing_features']]
            importances = [f[1] for f in details['contributing_features']]
            
            # Map to physical names
            physical_names = self.map_feature_names(features)
            
            # Sort by importance
            sorted_data = sorted(zip(physical_names, importances), key=lambda x: x[1], reverse=True)
            if sorted_data:
                physical_names, importances = zip(*sorted_data)
            
                # Create horizontal bar plot
                bars = ax.barh(range(len(physical_names)), importances, 
                              color=details['color'], alpha=0.7, edgecolor='black')
                
                # Adjust font size based on number of features
                n_features = len(physical_names)
                if n_features > 15:
                    label_fontsize = 8
                    title_fontsize = 10
                elif n_features > 10:
                    label_fontsize = 9
                    title_fontsize = 11
                else:
                    label_fontsize = 10
                    title_fontsize = 12
                
                ax.set_yticks(range(len(physical_names)))
                ax.set_yticklabels(physical_names, fontsize=label_fontsize)
                ax.set_xlabel('Importance Score', fontsize=label_fontsize)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=title_fontsize, fontweight='bold')
                
                # Add value labels with adaptive font size
                value_fontsize = max(6, label_fontsize - 2)
                for bar, importance in zip(bars, importances):
                    width = bar.get_width()
                    ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{importance:.3f}', ha='left', va='center', fontsize=value_fontsize)
            else:
                ax.text(0.5, 0.5, 'No features found for this group', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=12, fontweight='bold')
        
        # Use better spacing for plots with many subplots
        plt.tight_layout(pad=2.0, h_pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_table(self, physical_results: Dict) -> pd.DataFrame:
        """Create summary table of physical feature group importance across methods."""
        all_groups = set()
        for method_results in physical_results.values():
            all_groups.update(method_results['group_details'].keys())
        
        summary_data = []
        for group in all_groups:
            row = {'Physical_Group': group.replace('_', ' ').title()}
            
            for method, results in physical_results.items():
                if group in results['group_details']:
                    details = results['group_details'][group]
                    row[f'{method.upper()}_Importance'] = details['total_importance']
                    row[f'{method.upper()}_Features'] = details['feature_count']
                else:
                    row[f'{method.upper()}_Importance'] = 0.0
                    row[f'{method.upper()}_Features'] = 0
            
            # Add description
            if physical_results:
                first_method = list(physical_results.keys())[0]
                if group in physical_results[first_method]['group_details']:
                    row['Description'] = physical_results[first_method]['group_details'][group]['description']
                else:
                    row['Description'] = 'No description available'
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # Sort by average importance across methods
        importance_cols = [col for col in df.columns if col.endswith('_Importance')]
        if importance_cols:
            df['Average_Importance'] = df[importance_cols].mean(axis=1)
            df = df.sort_values('Average_Importance', ascending=False)
            df = df.drop('Average_Importance', axis=1)
        
        return df