"""
Feature importance analysis for BCG candidate classification models.

This module provides comprehensive feature importance analysis including:
- SHAP (SHapley Additive exPlanations) analysis
- Gradient-based importance for neural networks
- Feature ranking and selection utilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class PyTorchModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper to make PyTorch models compatible with scikit-learn utilities.
    """
    
    def __init__(self, model, device='cpu', probabilistic=False):
        self.model = model
        self.device = device
        self.probabilistic = probabilistic
        self.model.eval()
    
    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility (model is already trained)."""
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(self.device)
            
            if self.probabilistic:
                # For probabilistic models, use mean prediction
                probs = []
                for _ in range(10):  # MC sampling
                    logits = self.model(X)
                    prob = torch.softmax(logits, dim=1)
                    probs.append(prob.cpu().numpy())
                return np.mean(probs, axis=0)
            else:
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1)
                return probs.cpu().numpy()
    
    def predict(self, X):
        """Predict classes."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        """Return accuracy score."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analyzer for BCG classification models.
    """
    
    def __init__(self, model, feature_names=None, device='cpu', probabilistic=False):
        """
        Initialize feature importance analyzer.
        
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names for interpretation
            device: PyTorch device
            probabilistic: Whether model is probabilistic (UQ model)
        """
        self.model = model
        self.device = device
        self.probabilistic = probabilistic
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self._get_feature_dim())]
        
        # Wrap model for sklearn compatibility
        self.wrapped_model = PyTorchModelWrapper(model, device, probabilistic)
        
        # Initialize analyzers
        self.shap_analyzer = None
        
        if SHAP_AVAILABLE:
            self.shap_analyzer = SHAPAnalyzer(model, feature_names, device, probabilistic)
    
    def _get_feature_dim(self):
        """Get input feature dimensionality from model."""
        # Try to get from first layer
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return None
    
    def analyze_feature_importance(self, X, y=None, methods=['shap', 'gradient'], 
                                 n_repeats=10, random_state=42):
        """
        Comprehensive feature importance analysis.
        
        Args:
            X: Input features (numpy array or torch tensor)
            y: True labels (optional)
            methods: List of methods to use ['shap', 'gradient']
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing importance scores from different methods
        """
        results = {}
        
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # SHAP analysis
        if 'shap' in methods and SHAP_AVAILABLE:
            print("Computing SHAP values...")
            shap_results = self.shap_analyzer.compute_shap_values(X_np)
            results['shap'] = shap_results
        
        # Gradient-based importance
        if 'gradient' in methods:
            print("Computing gradient-based importance...")
            grad_results = self._compute_gradient_importance(X)
            results['gradient'] = grad_results
        
        return results
    
    def _compute_gradient_importance(self, X):
        """
        Compute gradient-based feature importance.
        """
        self.model.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        X.requires_grad_(True)
        
        # Forward pass
        if self.probabilistic:
            # Average over multiple MC samples
            gradients = []
            for _ in range(10):
                logits = self.model(X)
                # Take gradient w.r.t. predicted class
                pred_class = torch.argmax(logits, dim=1)
                grad_outputs = torch.zeros_like(logits)
                grad_outputs[range(len(pred_class)), pred_class] = 1
                
                grad = torch.autograd.grad(
                    outputs=logits, inputs=X, 
                    grad_outputs=grad_outputs, 
                    create_graph=False, retain_graph=True
                )[0]
                gradients.append(grad.abs().mean(dim=0).cpu().numpy())
            
            importance = np.mean(gradients, axis=0)
        else:
            logits = self.model(X)
            pred_class = torch.argmax(logits, dim=1)
            grad_outputs = torch.zeros_like(logits)
            grad_outputs[range(len(pred_class)), pred_class] = 1
            
            grad = torch.autograd.grad(
                outputs=logits, inputs=X,
                grad_outputs=grad_outputs,
                create_graph=False
            )[0]
            
            importance = grad.abs().mean(dim=0).cpu().numpy()
        
        # Debug: Print top gradient features
        top_grad_indices = np.argsort(importance)[-10:][::-1]
        print(f"DEBUG GRADIENT: Top 10 features by importance:")
        for i, idx in enumerate(top_grad_indices):
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            print(f"  {i+1}. {feature_name}: {importance[idx]:.6f}")
        
        return {
            'importance': importance,
            'feature_names': self.feature_names,
            'ranking': np.argsort(importance)[::-1]
        }
    
    def get_feature_ranking(self, importance_results, method='shap'):
        """
        Get ranked list of features based on importance scores.
        
        Args:
            importance_results: Results from analyze_feature_importance
            method: Method to use for ranking
            
        Returns:
            DataFrame with ranked features
        """
        if method not in importance_results:
            raise ValueError(f"Method {method} not found in results")
        
        if method == 'shap':
            importance = importance_results[method]['mean_abs_shap']
        else:
            importance = importance_results[method]['importance']
        
        # Ensure feature names match importance array length
        if len(self.feature_names) != len(importance):
            print(f"Warning: Feature names ({len(self.feature_names)}) don't match importance array ({len(importance)})")
            if len(self.feature_names) < len(importance):
                # Extend feature names if too short
                extended_names = self.feature_names.copy()
                for i in range(len(self.feature_names), len(importance)):
                    extended_names.append(f"feature_{i}")
                feature_names = extended_names
            else:
                # Truncate feature names if too long
                feature_names = self.feature_names[:len(importance)]
        else:
            feature_names = self.feature_names
        
        ranking_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': importance,
            'rank': range(1, len(feature_names) + 1)
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def save_results(self, results, output_path):
        """Save importance analysis results."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_path / 'importance_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save feature rankings for each method
        for method in results:
            ranking_df = self.get_feature_ranking(results, method)
            ranking_df.to_csv(output_path / f'{method}_feature_ranking.csv', index=False)


class SHAPAnalyzer:
    """
    SHAP-based feature importance analysis.
    """
    
    def __init__(self, model, feature_names=None, device='cpu', probabilistic=False):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for SHAP analysis. Install with: pip install shap")
        
        self.model = model
        self.device = device
        self.probabilistic = probabilistic
        self.feature_names = feature_names
        self.explainer = None
    
    def _model_predict(self, X):
        """Model prediction function for SHAP."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(self.device)
            
            if self.probabilistic:
                # Average over multiple MC samples
                probs = []
                for _ in range(10):
                    logits = self.model(X)
                    prob = torch.softmax(logits, dim=1)
                    probs.append(prob.cpu().numpy())
                return np.mean(probs, axis=0)
            else:
                logits = self.model(X)
                return torch.softmax(logits, dim=1).cpu().numpy()
    
    def setup_explainer(self, background_data, explainer_type='deep'):
        """
        Setup SHAP explainer.
        
        Args:
            background_data: Background dataset for explainer
            explainer_type: Type of explainer ('deep', 'kernel', 'linear')
        """
        if explainer_type == 'deep':
            # DeepExplainer for neural networks
            if isinstance(background_data, np.ndarray):
                background_data = torch.FloatTensor(background_data).to(self.device)
            self.explainer = shap.DeepExplainer(self.model, background_data)
        
        elif explainer_type == 'kernel':
            # KernelExplainer - model agnostic but slower
            self.explainer = shap.KernelExplainer(self._model_predict, background_data)
        
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    def compute_shap_values(self, X, explainer_type='deep', background_sample_size=100):
        """
        Compute SHAP values for input data.
        
        Args:
            X: Input data
            explainer_type: Type of SHAP explainer to use
            background_sample_size: Size of background sample
            
        Returns:
            Dictionary containing SHAP values and derived metrics
        """
        # Setup background data
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Sample background data
        background_indices = np.random.choice(
            len(X_np), min(background_sample_size, len(X_np)), replace=False
        )
        background_data = X_np[background_indices]
        
        # Setup explainer if not already done
        if self.explainer is None:
            self.setup_explainer(background_data, explainer_type)
        
        # Compute SHAP values
        if explainer_type == 'deep':
            if isinstance(X_np, np.ndarray):
                X_tensor = torch.FloatTensor(X_np).to(self.device)
            else:
                X_tensor = X_np
            shap_values = self.explainer.shap_values(X_tensor)
        else:
            shap_values = self.explainer.shap_values(X_np)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # For binary classification, typically use class 1
            shap_values_class = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_values_class = shap_values
        
        # Compute summary statistics
        mean_abs_shap = np.mean(np.abs(shap_values_class), axis=0)
        std_abs_shap = np.std(np.abs(shap_values_class), axis=0)
        
        # Debug: Print top SHAP features
        top_shap_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        print(f"DEBUG SHAP: Top 10 features by importance:")
        for i, idx in enumerate(top_shap_indices):
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            print(f"  {i+1}. {feature_name}: {mean_abs_shap[idx]:.6f}")
        
        return {
            'shap_values': shap_values_class,
            'mean_abs_shap': mean_abs_shap,
            'std_abs_shap': std_abs_shap,
            'feature_names': self.feature_names,
            'ranking': np.argsort(mean_abs_shap)[::-1]
        }




class FeatureGroupAnalyzer:
    """
    Analyzer for grouped feature importance (e.g., morphological vs color features).
    """
    
    def __init__(self, feature_groups: Dict[str, List[str]]):
        """
        Initialize with feature group definitions.
        
        Args:
            feature_groups: Dictionary mapping group names to feature names
        """
        self.feature_groups = feature_groups
    
    def compute_group_importance(self, importance_scores, feature_names):
        """
        Aggregate importance scores by feature groups.
        
        Args:
            importance_scores: Individual feature importance scores
            feature_names: List of feature names
            
        Returns:
            Dictionary of group importance scores
        """
        group_scores = {}
        
        for group_name, features in self.feature_groups.items():
            # Find indices of features in this group
            feature_indices = [
                i for i, fname in enumerate(feature_names) 
                if fname in features
            ]
            
            if feature_indices:
                # Aggregate importance (sum or mean)
                group_scores[group_name] = {
                    'mean_importance': np.mean(importance_scores[feature_indices]),
                    'sum_importance': np.sum(importance_scores[feature_indices]),
                    'feature_count': len(feature_indices),
                    'features': [feature_names[i] for i in feature_indices]
                }
        
        return group_scores


def create_default_feature_groups():
    """
    Create default feature groups for BCG classification.
    Follows the sensitivity analysis classification scheme: Luminosity profile, Morphology, Color.
    """
    return {
        'luminosity_profile': [
            'patch_mean', 'patch_std', 'patch_max', 'patch_min', 'patch_median',
            'patch_skew'
        ],
        'morphology': [
            'concentration', 'eccentricity', 'gradient_mean', 'gradient_std', 'gradient_max',
            'x_rel', 'y_rel', 'r_center', 'brightness_rank', 'candidate_density',
            'background_level', 'north_intensity', 'east_intensity', 
            'south_intensity', 'west_intensity'
        ],
        'color_information': [
            'rg_ratio', 'rb_ratio', 'color_magnitude', 'red_sequence_score',
            'rg_ratio_std', 'rb_ratio_std', 'color_correlation_rg', 'color_correlation_rb'
        ] + [f'color_pca_{i}' for i in range(8)],  # PCA components
        'auxiliary': ['redshift', 'delta_m_star_z']
    }