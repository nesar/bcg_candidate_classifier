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

# Set style consistent with plot_physical_results.py
plt.rcParams.update({"text.usetex":False,"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":1.2})


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
            # 'color_pca_components': [],  # DEPRECATED: PCA was never actually used
            'color_features': [],  # Track actual color features instead
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
        
        # Check color features (all 54 raw color features, not PCA)
        color_features = [name for name in feature_names if name.startswith('color_')]
        validation_results['color_features'] = color_features

        # Check for unmapped features
        for name in feature_names:
            if name not in self.feature_mappings and not any([
                name.startswith('color_'),  # All color features handled together
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

        # NOTE: PCA validation commented out - model uses 54 raw color features, not 8 PCA components
        # if len(validation_results['color_pca_components']) != 8:
        #     validation_results['warnings'].append(f"Expected 8 color PCA components, found {len(validation_results['color_pca_components'])}")

        if len(validation_results['color_features']) != 54:
            validation_results['warnings'].append(f"Expected 54 raw color features, found {len(validation_results['color_features'])}")

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
            
            # NOTE: No longer using PCA - all 54 raw color features are used
            # if name.startswith('color_pca_'):
            #     color_features['pca_components'].append((name, importance))
            if 'ratio' in name and 'color' in name:
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
                'description': 'Red-sequence and RGB color properties (54 raw features: channel stats, ratios, spatial variation, gradients, convolutions)',
                'technical_features': [
                    # Basic channel statistics (9)
                    'color_mean_r', 'color_mean_g', 'color_mean_b',
                    'color_std_r', 'color_std_g', 'color_std_b',
                    'color_rel_r', 'color_rel_g', 'color_rel_b',
                    # Color ratios (7)
                    'color_rg_ratio', 'color_rg_diff', 'color_rb_ratio', 'color_rb_diff', 'color_gb_ratio',
                    'color_magnitude', 'color_red_sequence_score',
                    # Spatial variation (3)
                    'color_spatial_rg_std', 'color_spatial_rb_std', 'color_central_peripheral_rg_diff',
                    # Gradients (8)
                    'color_gradient_r_mean', 'color_gradient_r_std',
                    'color_gradient_g_mean', 'color_gradient_g_std',
                    'color_gradient_b_mean', 'color_gradient_b_std',
                    'color_gradient_rg_corr', 'color_gradient_rb_corr',
                    # Convolutions (27) - use prefix matching since there are many
                    # These will be matched by startswith('color_conv_')
                ],
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
            'patch_mean': 'Mean Intensity $\\mu_I$',
            'patch_std': 'Intensity Std. Dev. $\\sigma_I$',
            'patch_median': 'Median Intensity $I_\\text{median}$',
            'patch_max': 'Maximum Intensity $I_\\max$',
            'patch_min': 'Minimum Intensity $I_\\min$',
            'central_mean': 'Central Mean Intensity $\\mu_I^\\text{central}$',
            'peripheral_mean': 'Peripheral Mean Intensity $\\mu_I^\\text{peripheral}$',
            'concentration_ratio': 'Concentration Ratio $C_\\text{ratio}$',
            
            # Morphology (22 features)
            'gradient_mean': 'Mean Gradient Magnitude $\\langle|\\mathbf{G}|\\rangle$',
            'gradient_std': 'Gradient Std. Dev. $\\sigma_{|\\mathbf{G}|}$',
            'gradient_max': 'Max Gradient Magnitude $\\max(|\\mathbf{G}|)$',
            'x_relative': 'Normalized X Position $x_\\text{rel}$',
            'y_relative': 'Normalized Y Position $y_\\text{rel}$',
            'r_center': 'Radial Distance $r_\\text{center}$',
            'centroid_offset_x': 'Centroid X-Offset $\\bar{x}$',
            'centroid_offset_y': 'Centroid Y-Offset $\\bar{y}$',
            'eccentricity': 'Eccentricity $e$',
            'context_small_mean': 'Small-Scale Env. Mean $\\mu_\\text{env}^{(r_\\text{small})}$',
            'context_small_std': 'Small-Scale Env. Std. Dev. $\\sigma_\\text{env}^{(r_\\text{small})}$',
            'context_small_pixels': 'Small-Scale Env. Size $N_{r_\\text{small}}$',
            'context_medium_mean': 'Medium-Scale Env. Mean $\\mu_\\text{env}^{(r_\\text{medium})}$',
            'context_medium_std': 'Medium-Scale Env. Std. Dev. $\\sigma_\\text{env}^{(r_\\text{medium})}$',
            'context_medium_pixels': 'Medium-Scale Env. Size $N_{r_\\text{medium}}$',
            'context_large_mean': 'Large-Scale Env. Mean $\\mu_\\text{env}^{(r_\\text{large})}$',
            'context_large_std': 'Large-Scale Env. Std. Dev. $\\sigma_\\text{env}^{(r_\\text{large})}$',
            'context_large_pixels': 'Large-Scale Env. Size $N_{r_\\text{large}}$',
            'context_north_mean': 'Northern Direction Mean $\\mu_\\text{dir,N}$',
            'context_east_mean': 'Eastern Direction Mean $\\mu_\\text{dir,E}$',
            'context_south_mean': 'Southern Direction Mean $\\mu_\\text{dir,S}$',
            'context_west_mean': 'Western Direction Mean $\\mu_\\text{dir,W}$',
            
            # Raw RGB Color Features (54 total - no PCA)
            # Basic channel statistics (9 features)
            'color_mean_r': 'Red Mean $\\mu_R$',
            'color_mean_g': 'Green Mean $\\mu_G$',
            'color_mean_b': 'Blue Mean $\\mu_B$',
            'color_std_r': 'Red Std. Dev. $\\sigma_R$',
            'color_std_g': 'Green Std. Dev. $\\sigma_G$',
            'color_std_b': 'Blue Std. Dev. $\\sigma_B$',
            'color_rel_r': 'Red Fraction $f_R$',
            'color_rel_g': 'Green Fraction $f_G$',
            'color_rel_b': 'Blue Fraction $f_B$',

            # Color ratios (7 features)
            'color_rg_ratio': 'R/G Ratio $\\chi_{R/G}$',
            'color_rg_diff': 'Norm. R-G Diff. $\\chi^{\\text{norm}}_{R/G}$',
            'color_rb_ratio': 'R/B Ratio $\\chi_{R/B}$',
            'color_rb_diff': 'Norm. R-B Diff. $\\chi^{\\text{norm}}_{R/B}$',
            'color_gb_ratio': 'G/B Ratio $\\chi_{G/B}$',
            'color_magnitude': 'Chromatic Departure $\\chi_{\\text{col}}$',
            'color_red_sequence_score': 'Red Enhancement $S_{\\text{red}}$',

            # Spatial color variation (3 features)
            'color_spatial_rg_std': 'R/G Variability $\\sigma_{R/G}$',
            'color_spatial_rb_std': 'R/B Variability $\\sigma_{R/B}$',
            'color_central_peripheral_rg_diff': 'Central-Periph. Diff. $\\Delta_{c-p}$',

            # Color gradients (8 features)
            'color_gradient_r_mean': 'Red Gradient Mean $\\langle|\\mathbf{G}_R|\\rangle$',
            'color_gradient_g_mean': 'Green Gradient Mean $\\langle|\\mathbf{G}_G|\\rangle$',
            'color_gradient_b_mean': 'Blue Gradient Mean $\\langle|\\mathbf{G}_B|\\rangle$',
            'color_gradient_r_std': 'Red Gradient Std. $\\sigma_{|\\mathbf{G}_R|}$',
            'color_gradient_g_std': 'Green Gradient Std. $\\sigma_{|\\mathbf{G}_G|}$',
            'color_gradient_b_std': 'Blue Gradient Std. $\\sigma_{|\\mathbf{G}_B|}$',
            'color_gradient_rg_corr': 'R-G Gradient Corr. $\\rho_{RG}$',
            'color_gradient_rb_corr': 'R-B Gradient Corr. $\\rho_{RB}$',

            # Convolution features (27 features) - will be handled by pattern matching
            # Format: color_conv_{kernel}_{channel}_{stat} where kernel=edge/smooth/laplacian,
            # channel=r/g/b, stat=mean/std/max_abs
            # These represent $F_{c,k}$ in the notation
            
            # Environment
            'context_mean': 'Local Background',
            'context_std': 'Environmental Variability',
            'context_gradient': 'Density Gradient',
            'context_density': 'Local Galaxy Density',
            'neighbor_count': 'Nearby Galaxy Count',
            'local_density': 'Cluster Core Density',
            
            # Auxiliary features (2 features)
            'redshift_z': 'Photometric Redshift $z$',
            'delta_m_star_z': 'Luminosity measure $\\delta m_{zb}*$',
            
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
            # NOTE: PCA handling removed - model uses 54 raw color features
            # elif name.startswith('color_pca_'):
            #     ... PCA code commented out ...
            elif name.startswith('color_conv_'):
                # Handle color convolution features: color_conv_{kernel}_{channel}_{stat}
                # These represent $F_{c,k}$ in the LaTeX notation
                parts = name.split('_')
                if len(parts) >= 5:  # color_conv_kernel_channel_stat
                    kernel = parts[2]  # edge, smooth, laplacian
                    channel = parts[3].upper()  # R, G, B
                    stat = parts[4]  # mean, std, max_abs
                    kernel_map = {'edge': 'Edge', 'smooth': 'Smooth', 'laplacian': 'Laplacian'}
                    stat_map = {'mean': 'Mean', 'std': 'Std', 'max_abs': 'Max'}
                    mapped_names.append(f'{channel}-{kernel_map.get(kernel, kernel)} {stat_map.get(stat, stat)} $F_{{c,k}}$')
                else:
                    mapped_names.append('Color Conv. Feature $F_{c,k}$')
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
        
        # NOTE: PCA matching commented out - model uses raw color features
        # if template_feature.startswith('color_pca_') and actual_feature.startswith('color_pca_'):
        #     ... PCA matching code ...

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
        ax.set_xlabel(f'{method.upper()} Importance Score', fontsize=18)
        ax.set_title(f'Physical Feature Group Importance ({method.upper()})', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18)
        
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
                   ha='left', va='center', fontsize=18, style='italic')
        
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
                # Use consistent fontsize=18 for all cases
                label_fontsize = 18
                title_fontsize = 18
                
                ax.set_yticks(range(len(physical_names)))
                ax.set_yticklabels(physical_names, fontsize=label_fontsize)
                ax.set_xlabel('Importance Score', fontsize=label_fontsize)
                ax.tick_params(axis='both', labelsize=label_fontsize)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=title_fontsize, fontweight='bold')
                
                # Add value labels with consistent font size
                value_fontsize = 18
                for bar, importance in zip(bars, importances):
                    width = bar.get_width()
                    ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{importance:.3f}', ha='left', va='center', fontsize=value_fontsize)
            else:
                ax.text(0.5, 0.5, 'No features found for this group', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=18)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=18, fontweight='bold')
        
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