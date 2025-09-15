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
    
    def _create_feature_groups(self) -> Dict[str, Dict]:
        """Create physical feature groups from technical features."""
        return {
            'luminosity_profile': {
                'description': 'Surface brightness and luminosity distribution',
                'technical_features': ['patch_mean', 'patch_std', 'patch_median', 'patch_max', 'patch_min'],
                'combination_method': 'weighted_sum',
                'color': '#FF6B6B'
            },
            'morphology': {
                'description': 'Galaxy shape and structural parameters',
                'technical_features': ['concentration', 'eccentricity', 'moment_m00', 'moment_m10', 'moment_m01', 
                                     'moment_m20', 'moment_m11', 'moment_m02', 'asymmetry', 'smoothness'],
                'combination_method': 'weighted_sum',
                'color': '#4ECDC4'
            },
            'color_information': {
                'description': 'Red-sequence and color properties',
                'technical_features': ['color_pca_0', 'color_pca_1', 'color_pca_2', 'color_pca_3', 'color_pca_4',
                                     'color_pca_5', 'color_pca_6', 'color_pca_7', 'color_ratio_rg', 'color_ratio_rb',
                                     'color_ratio_gb', 'color_variation'],
                'combination_method': 'weighted_sum',
                'color': '#45B7D1'
            },
            'local_environment': {
                'description': 'Surrounding galaxy environment and context',
                'technical_features': ['context_mean', 'context_std', 'context_gradient', 'context_density',
                                     'neighbor_count', 'local_density'],
                'combination_method': 'weighted_sum',
                'color': '#96CEB4'
            },
            'physical_properties': {
                'description': 'Cosmological and physical parameters',
                'technical_features': ['cluster_z', 'delta_mstar_z', 'redshift', 'mass_proxy'],
                'combination_method': 'weighted_sum',
                'color': '#FECA57'
            },
            'candidate_properties': {
                'description': 'DESprior catalog properties',
                'technical_features': ['delta_mstar', 'starflag', 'mag_auto_g', 'mag_auto_r', 'mag_auto_i'],
                'combination_method': 'weighted_sum',
                'color': '#FF9FF3'
            }
        }
    
    def _create_feature_mappings(self) -> Dict[str, str]:
        """Create individual feature name mappings."""
        return {
            # Luminosity/brightness
            'patch_mean': 'Mean Surface Brightness',
            'patch_std': 'Brightness Variability',
            'patch_median': 'Median Luminosity',
            'patch_max': 'Peak Brightness',
            'patch_min': 'Background Level',
            
            # Morphology
            'concentration': 'Light Concentration',
            'eccentricity': 'Galaxy Ellipticity',
            'asymmetry': 'Structural Asymmetry',
            'smoothness': 'Surface Smoothness',
            'moment_m00': 'Total Flux',
            'moment_m10': 'X-centroid',
            'moment_m01': 'Y-centroid',
            'moment_m20': 'X-spread',
            'moment_m11': 'XY-correlation',
            'moment_m02': 'Y-spread',
            
            # Color
            'color_pca_0': 'Primary Color Component',
            'color_pca_1': 'Secondary Color Component',
            'color_pca_2': 'Tertiary Color Component',
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
            
            # Physical
            'cluster_z': 'Cluster Redshift',
            'delta_mstar_z': 'Stellar Mass Evolution',
            'redshift': 'Cosmological Redshift',
            'mass_proxy': 'Galaxy Mass Indicator',
            
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
                # Handle additional color PCA components
                idx = name.split('_')[2]
                mapped_names.append(f'Color Component {idx}')
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
        physical_results = {}
        
        for method, results in importance_results.items():
            if method == 'shap':
                importance_scores = results['mean_abs_shap']
            else:
                importance_scores = results['importance']
            
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
        
        # Handle color_pca extensions
        if template_feature.startswith('color_pca_') and actual_feature.startswith('color_pca_'):
            return True
        
        # Handle moment extensions
        if template_feature.startswith('moment_') and actual_feature.startswith('moment_'):
            return True
        
        # Handle context extensions
        if template_feature.startswith('context_') and actual_feature.startswith('context_'):
            return True
        
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
        
        # Create subplot for each group
        n_groups = len(group_details)
        fig, axes = plt.subplots(n_groups, 1, figsize=(14, 4 * n_groups))
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
                
                ax.set_yticks(range(len(physical_names)))
                ax.set_yticklabels(physical_names, fontsize=10)
                ax.set_xlabel('Importance Score', fontsize=10)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=12, fontweight='bold')
                
                # Add value labels
                for bar, importance in zip(bars, importances):
                    width = bar.get_width()
                    ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{importance:.3f}', ha='left', va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No features found for this group', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{group_name.replace('_', ' ').title()}: {details['description']}", 
                           fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
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