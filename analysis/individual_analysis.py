"""
Individual sample analysis tools for BCG candidate classification.

This module provides detailed analysis and explanation capabilities for
individual samples, including:
- Local SHAP explanations
- Feature contribution analysis
- Sample similarity analysis
- Decision boundary visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Set style consistent with plot_physical_results.py
plt.rcParams.update({"text.usetex":False,"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":1.2})

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class IndividualSampleAnalyzer:
    """
    Analyzer for individual sample explanations and interpretability.
    """
    
    def __init__(self, model, feature_names=None, device='cpu', probabilistic=False):
        """
        Initialize individual sample analyzer.
        
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names
            device: PyTorch device
            probabilistic: Whether model is probabilistic
        """
        self.model = model
        self.device = device
        self.probabilistic = probabilistic
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self._get_feature_dim())]
        self.model.eval()
    
    def _get_feature_dim(self):
        """Get input feature dimensionality."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                return module.in_features
        return None
    
    def explain_sample(self, sample, shap_values=None, background_data=None):
        """
        Provide comprehensive explanation for a single sample.
        
        Args:
            sample: Single sample to explain (1D array)
            shap_values: Precomputed SHAP values (optional)
            background_data: Background dataset for SHAP
            
        Returns:
            Dictionary with explanation components
        """
        sample = np.array(sample).reshape(1, -1)
        
        # Get model prediction
        prediction_info = self.get_prediction_info(sample)
        
        # Compute feature contributions
        feature_contributions = self.compute_feature_contributions(sample, shap_values, background_data)
        
        # Analyze feature importance for this sample
        local_importance = self.analyze_local_importance(sample)
        
        # Get similar samples analysis if background data provided
        similarity_info = None
        if background_data is not None:
            similarity_info = self.analyze_sample_similarity(sample, background_data)
        
        return {
            'prediction': prediction_info,
            'feature_contributions': feature_contributions,
            'local_importance': local_importance,
            'similarity': similarity_info,
            'sample_features': dict(zip(self.feature_names, sample[0]))
        }
    
    def get_prediction_info(self, sample):
        """
        Get detailed prediction information for a sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Dictionary with prediction details
        """
        with torch.no_grad():
            if isinstance(sample, np.ndarray):
                sample_tensor = torch.FloatTensor(sample).to(self.device)
            else:
                sample_tensor = sample
            
            if self.probabilistic:
                # Multiple forward passes for uncertainty
                predictions = []
                logits_list = []
                for _ in range(50):  # More samples for better uncertainty estimate
                    logits = self.model(sample_tensor)
                    predictions.append(torch.softmax(logits, dim=1).cpu().numpy())
                    logits_list.append(logits.cpu().numpy())
                
                # Compute statistics
                pred_array = np.array(predictions)
                mean_prob = np.mean(pred_array, axis=0)[0]
                std_prob = np.std(pred_array, axis=0)[0]
                
                # Epistemic uncertainty (variance in predictions)
                epistemic_uncertainty = np.var(pred_array, axis=0)[0]
                
                # Aleatoric uncertainty approximation
                mean_logits = np.mean(logits_list, axis=0)[0]
                aleatoric_uncertainty = -np.sum(mean_prob * np.log(mean_prob + 1e-10))
                
                return {
                    'predicted_class': int(np.argmax(mean_prob)),
                    'class_probabilities': mean_prob.tolist(),
                    'probability_std': std_prob.tolist(),
                    'epistemic_uncertainty': epistemic_uncertainty.tolist(),
                    'aleatoric_uncertainty': float(aleatoric_uncertainty),
                    'prediction_confidence': float(np.max(mean_prob)),
                    'is_uncertain': float(np.max(std_prob)) > 0.1  # Threshold for uncertainty
                }
            
            else:
                logits = self.model(sample_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                return {
                    'predicted_class': int(np.argmax(probs)),
                    'class_probabilities': probs.tolist(),
                    'prediction_confidence': float(np.max(probs)),
                    'logits': logits.cpu().numpy()[0].tolist()
                }
    
    def compute_feature_contributions(self, sample, shap_values=None, background_data=None):
        """
        Compute feature contributions using SHAP or gradients.
        
        Args:
            sample: Input sample
            shap_values: Precomputed SHAP values
            background_data: Background data for SHAP
            
        Returns:
            Dictionary with feature contributions
        """
        contributions = {}
        
        # SHAP contributions
        if SHAP_AVAILABLE and (shap_values is not None or background_data is not None):
            if shap_values is None:
                # Compute SHAP values
                from .feature_importance import SHAPAnalyzer
                shap_analyzer = SHAPAnalyzer(self.model, self.feature_names, 
                                           self.device, self.probabilistic)
                if background_data is not None:
                    shap_results = shap_analyzer.compute_shap_values(
                        np.vstack([sample, background_data[:100]])  # Include sample + background
                    )
                    shap_values = shap_results['shap_values'][0:1]  # Take only sample's SHAP values
            
            if shap_values is not None:
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                contributions['shap'] = {
                    'values': shap_values[0] if shap_values.ndim > 1 else shap_values,
                    'feature_names': self.feature_names,
                    'positive_contributions': np.sum(shap_values[shap_values > 0]) if shap_values.ndim == 1 else np.sum(shap_values[0][shap_values[0] > 0]),
                    'negative_contributions': np.sum(shap_values[shap_values < 0]) if shap_values.ndim == 1 else np.sum(shap_values[0][shap_values[0] < 0])
                }
        
        # Gradient-based contributions
        grad_contributions = self.compute_gradient_contributions(sample)
        contributions['gradient'] = grad_contributions
        
        return contributions
    
    def compute_gradient_contributions(self, sample):
        """
        Compute gradient-based feature contributions.
        
        Args:
            sample: Input sample
            
        Returns:
            Dictionary with gradient contributions
        """
        if isinstance(sample, np.ndarray):
            sample_tensor = torch.FloatTensor(sample).to(self.device)
        else:
            sample_tensor = sample
            
        sample_tensor.requires_grad_(True)
        
        if self.probabilistic:
            # Average gradients over multiple forward passes
            gradients = []
            for _ in range(10):
                logits = self.model(sample_tensor)
                pred_class = torch.argmax(logits, dim=1)
                
                # Gradient w.r.t predicted class
                grad = torch.autograd.grad(
                    outputs=logits[0, pred_class[0]], 
                    inputs=sample_tensor,
                    create_graph=False, retain_graph=True
                )[0]
                gradients.append(grad.cpu().numpy())
            
            avg_gradient = np.mean(gradients, axis=0)[0]
        else:
            logits = self.model(sample_tensor)
            pred_class = torch.argmax(logits, dim=1)
            
            grad = torch.autograd.grad(
                outputs=logits[0, pred_class[0]],
                inputs=sample_tensor,
                create_graph=False
            )[0]
            avg_gradient = grad.cpu().numpy()[0]
        
        # Feature contributions = gradient * input
        feature_contributions = avg_gradient * sample[0]
        
        return {
            'gradients': avg_gradient.tolist(),
            'contributions': feature_contributions.tolist(),
            'input_values': sample[0].tolist(),
            'feature_names': self.feature_names
        }
    
    def analyze_local_importance(self, sample):
        """
        Analyze local feature importance for the sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Dictionary with local importance analysis
        """
        # Perturbation-based local importance
        baseline_pred = self.get_prediction_info(sample)
        baseline_prob = baseline_pred['class_probabilities'][baseline_pred['predicted_class']]
        
        feature_importance = []
        
        for i in range(len(self.feature_names)):
            # Create perturbed sample (set feature to 0)
            perturbed_sample = sample.copy()
            perturbed_sample[0, i] = 0
            
            perturbed_pred = self.get_prediction_info(perturbed_sample)
            perturbed_prob = perturbed_pred['class_probabilities'][baseline_pred['predicted_class']]
            
            # Importance = change in prediction probability
            importance = baseline_prob - perturbed_prob
            feature_importance.append(importance)
        
        feature_importance = np.array(feature_importance)
        
        return {
            'importance_scores': feature_importance.tolist(),
            'feature_names': self.feature_names,
            'ranking': np.argsort(np.abs(feature_importance))[::-1].tolist(),
            'top_positive': [(self.feature_names[i], feature_importance[i]) 
                           for i in np.argsort(feature_importance)[::-1][:5]],
            'top_negative': [(self.feature_names[i], feature_importance[i]) 
                           for i in np.argsort(feature_importance)[:5]]
        }
    
    def analyze_sample_similarity(self, sample, background_data, top_k=5):
        """
        Find and analyze similar samples.
        
        Args:
            sample: Target sample
            background_data: Background dataset
            top_k: Number of similar samples to return
            
        Returns:
            Dictionary with similarity analysis
        """
        # Compute similarities
        cosine_sim = cosine_similarity(sample, background_data)[0]
        euclidean_dist = euclidean_distances(sample, background_data)[0]
        
        # Get most similar samples
        most_similar_idx = np.argsort(cosine_sim)[::-1][:top_k]
        least_similar_idx = np.argsort(cosine_sim)[:top_k]
        closest_idx = np.argsort(euclidean_dist)[:top_k]
        
        # Analyze similar samples' predictions
        similar_predictions = []
        for idx in most_similar_idx:
            pred_info = self.get_prediction_info(background_data[idx:idx+1])
            similar_predictions.append({
                'sample_idx': int(idx),
                'cosine_similarity': float(cosine_sim[idx]),
                'euclidean_distance': float(euclidean_dist[idx]),
                'predicted_class': pred_info['predicted_class'],
                'confidence': pred_info['prediction_confidence']
            })
        
        return {
            'most_similar_samples': similar_predictions,
            'similarity_stats': {
                'mean_cosine_similarity': float(np.mean(cosine_sim)),
                'std_cosine_similarity': float(np.std(cosine_sim)),
                'mean_euclidean_distance': float(np.mean(euclidean_dist)),
                'std_euclidean_distance': float(np.std(euclidean_dist))
            },
            'most_similar_indices': most_similar_idx.tolist(),
            'least_similar_indices': least_similar_idx.tolist(),
            'closest_indices': closest_idx.tolist()
        }
    
    def plot_sample_explanation(self, explanation, save_path=None, figsize=(16, 12)):
        """
        Create comprehensive visualization of sample explanation.
        
        Args:
            explanation: Output from explain_sample()
            save_path: Path to save plot
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Prediction information
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_prediction_info(ax1, explanation['prediction'])
        
        # 2. Feature contributions (SHAP if available)
        ax2 = fig.add_subplot(gs[1, 0])
        if 'shap' in explanation['feature_contributions']:
            self._plot_shap_contributions(ax2, explanation['feature_contributions']['shap'])
        else:
            self._plot_gradient_contributions(ax2, explanation['feature_contributions']['gradient'])
        
        # 3. Local importance
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_local_importance(ax3, explanation['local_importance'])
        
        # 4. Feature values
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_feature_values(ax4, explanation['sample_features'])
        
        # 5. Similarity analysis (if available)
        if explanation['similarity'] is not None:
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_similarity_analysis(ax5, explanation['similarity'])
        
        plt.suptitle('Individual Sample Analysis', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_prediction_info(self, ax, prediction_info):
        """Plot prediction information."""
        if 'probability_std' in prediction_info:
            # Probabilistic model
            probs = prediction_info['class_probabilities']
            stds = prediction_info['probability_std']
            classes = [f'Class {i}' for i in range(len(probs))]
            
            bars = ax.bar(classes, probs, yerr=stds, capsize=5)
            ax.set_title('Prediction Probabilities (with uncertainty)')
            ax.set_ylabel('Probability')
            
            # Highlight predicted class
            predicted_class = prediction_info['predicted_class']
            bars[predicted_class].set_color('red')
            
            # Add uncertainty info
            ax.text(0.02, 0.98, 
                   f"Predicted: Class {predicted_class}\n"
                   f"Confidence: {prediction_info['prediction_confidence']:.3f}\n"
                   f"Uncertain: {'Yes' if prediction_info.get('is_uncertain', False) else 'No'}",
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Deterministic model
            probs = prediction_info['class_probabilities']
            classes = [f'Class {i}' for i in range(len(probs))]
            
            bars = ax.bar(classes, probs)
            ax.set_title('Prediction Probabilities')
            ax.set_ylabel('Probability')
            
            # Highlight predicted class
            predicted_class = prediction_info['predicted_class']
            bars[predicted_class].set_color('red')
            
            ax.text(0.02, 0.98,
                   f"Predicted: Class {predicted_class}\n"
                   f"Confidence: {prediction_info['prediction_confidence']:.3f}",
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_shap_contributions(self, ax, shap_info, top_n=10):
        """Plot SHAP feature contributions."""
        values = shap_info['values']
        feature_names = shap_info['feature_names']
        
        # Get top features by absolute SHAP value
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-top_n:]
        
        top_values = values[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Color by positive/negative
        colors = ['red' if v > 0 else 'blue' for v in top_values]
        
        bars = ax.barh(range(len(top_values)), top_values, color=colors)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Top {top_n} SHAP Feature Contributions')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_gradient_contributions(self, ax, grad_info, top_n=10):
        """Plot gradient-based contributions."""
        contributions = np.array(grad_info['contributions'])
        feature_names = grad_info['feature_names']
        
        # Get top features
        abs_contributions = np.abs(contributions)
        top_indices = np.argsort(abs_contributions)[-top_n:]
        
        top_values = contributions[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        colors = ['red' if v > 0 else 'blue' for v in top_values]
        
        bars = ax.barh(range(len(top_values)), top_values, color=colors)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Gradient × Input')
        ax.set_title(f'Top {top_n} Gradient Contributions')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_local_importance(self, ax, local_info, top_n=10):
        """Plot local feature importance."""
        importance = np.array(local_info['importance_scores'])
        feature_names = local_info['feature_names']
        
        # Get top features
        abs_importance = np.abs(importance)
        top_indices = np.argsort(abs_importance)[-top_n:]
        
        top_values = importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        colors = ['green' if v > 0 else 'orange' for v in top_values]
        
        bars = ax.barh(range(len(top_values)), top_values, color=colors)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Local Importance')
        ax.set_title(f'Top {top_n} Local Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_feature_values(self, ax, sample_features, top_n=15):
        """Plot sample feature values."""
        # Convert to sorted list
        feature_items = list(sample_features.items())
        values = [item[1] for item in feature_items]
        
        # Get features with largest absolute values
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-top_n:]
        
        top_items = [feature_items[i] for i in top_indices]
        top_names = [item[0] for item in top_items]
        top_values = [item[1] for item in top_items]
        
        bars = ax.barh(range(len(top_values)), top_values)
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Feature Value')
        ax.set_title(f'Top {top_n} Feature Values (by magnitude)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_similarity_analysis(self, ax, similarity_info):
        """Plot similarity analysis."""
        similar_samples = similarity_info['most_similar_samples']
        
        # Extract data
        similarities = [s['cosine_similarity'] for s in similar_samples]
        distances = [s['euclidean_distance'] for s in similar_samples]
        predictions = [s['predicted_class'] for s in similar_samples]
        confidences = [s['confidence'] for s in similar_samples]
        
        # Create scatter plot
        scatter = ax.scatter(distances, similarities, c=predictions, 
                           s=[c*100 for c in confidences], alpha=0.7, cmap='tab10')
        
        ax.set_xlabel('Euclidean Distance')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Most Similar Samples')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Class')
        
        # Add text info
        stats = similarity_info['similarity_stats']
        ax.text(0.02, 0.02,
               f"Mean cosine similarity: {stats['mean_cosine_similarity']:.3f}\n"
               f"Mean euclidean distance: {stats['mean_euclidean_distance']:.3f}",
               transform=ax.transAxes, va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def analyze_prediction_boundary(model, sample, feature_names, device='cpu', 
                              feature_idx=0, n_points=100, save_path=None):
    """
    Analyze how prediction changes when varying a specific feature.
    
    Args:
        model: Trained model
        sample: Base sample to analyze
        feature_names: List of feature names
        device: PyTorch device
        feature_idx: Index of feature to vary
        n_points: Number of points to sample
        save_path: Path to save plot
        
    Returns:
        matplotlib figure
    """
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        if isinstance(sample, np.ndarray):
            sample_tensor = torch.FloatTensor(sample.reshape(1, -1)).to(device)
        else:
            sample_tensor = sample
        
        original_logits = model(sample_tensor)
        original_probs = torch.softmax(original_logits, dim=1).cpu().numpy()[0]
    
    # Vary the selected feature
    original_value = sample[feature_idx] if sample.ndim == 1 else sample[0, feature_idx]
    feature_range = np.linspace(original_value - 2*np.std(sample), 
                               original_value + 2*np.std(sample), n_points)
    
    predictions = []
    for value in feature_range:
        # Create modified sample
        modified_sample = sample.copy()
        if sample.ndim == 1:
            modified_sample[feature_idx] = value
            input_tensor = torch.FloatTensor(modified_sample.reshape(1, -1)).to(device)
        else:
            modified_sample[0, feature_idx] = value
            input_tensor = torch.FloatTensor(modified_sample).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predictions.append(probs)
    
    predictions = np.array(predictions)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot probability curves for each class
    for class_idx in range(predictions.shape[1]):
        ax.plot(feature_range, predictions[:, class_idx], 
               label=f'Class {class_idx}', linewidth=2)
    
    # Mark original point
    ax.axvline(x=original_value, color='red', linestyle='--', alpha=0.7, 
              label='Original value')
    
    ax.set_xlabel(f'{feature_names[feature_idx]} Value')
    ax.set_ylabel('Prediction Probability')
    ax.set_title(f'Prediction Boundary Analysis - {feature_names[feature_idx]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig