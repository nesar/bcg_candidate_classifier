"""
Uncertainty Quantification BCG Classifier

This module provides probabilistic outputs and uncertainty quantification
for BCG candidate classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCGProbabilisticClassifier(nn.Module):
    """
    Probabilistic BCG classifier that outputs calibrated probabilities
    instead of raw scores for uncertainty quantification.
    """
    
    def __init__(self, feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2, 
                 use_temperature_scaling=True):
        """
        Initialize the probabilistic classifier.
        
        Args:
            feature_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
            use_temperature_scaling (bool): Whether to use temperature scaling for calibration
        """
        super(BCGProbabilisticClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_temperature_scaling = use_temperature_scaling
        
        layers = []
        prev_dim = feature_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer - logits for binary classification (BCG vs non-BCG)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Temperature parameter for calibration
        if use_temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = torch.ones(1)
    
    def forward(self, features):
        """
        Forward pass to get raw logits (no temperature scaling during training).
        
        Args:
            features (torch.Tensor): Features for candidates (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Raw logits for each candidate (batch_size, 1)
        """
        logits = self.network(features)
        return logits
    
    def forward_with_temperature(self, features):
        """
        Forward pass with temperature scaling for inference.
        
        Args:
            features (torch.Tensor): Features for candidates (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Temperature-scaled logits for each candidate (batch_size, 1)
        """
        logits = self.network(features)
        
        # Apply temperature scaling
        if self.use_temperature_scaling:
            logits = logits / self.temperature
        
        return logits
    
    def predict_probabilities(self, features):
        """
        Predict calibrated probabilities for being BCG.
        
        Args:
            features (torch.Tensor): Features for candidates
            
        Returns:
            torch.Tensor: Probabilities for each candidate being BCG
        """
        logits = self.forward_with_temperature(features)
        probabilities = torch.sigmoid(logits)
        return probabilities
    
    def predict_with_uncertainty(self, features, n_samples=10):
        """
        Predict with epistemic uncertainty using Monte Carlo Dropout.
        
        Args:
            features (torch.Tensor): Features for candidates
            n_samples (int): Number of MC samples for uncertainty estimation
            
        Returns:
            tuple: (mean_probabilities, uncertainty_estimates)
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward_with_temperature(features)
                # Each candidate gets independent probability (0-1), don't normalize
                probs = torch.sigmoid(logits)
                predictions.append(probs)
        
        self.eval()  # Return to eval mode
        
        predictions = torch.stack(predictions)  # (n_samples, n_candidates, 1)
        
        mean_probs = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)  # Epistemic uncertainty
        
        return mean_probs.squeeze(), uncertainty.squeeze()


class BCGEnsembleClassifier:
    """
    Ensemble of BCG classifiers for improved uncertainty quantification.
    """
    
    def __init__(self, n_models=5, feature_dim=30, **model_kwargs):
        """
        Initialize ensemble of classifiers.
        
        Args:
            n_models (int): Number of models in ensemble
            feature_dim (int): Feature dimension
            **model_kwargs: Arguments passed to individual models
        """
        self.n_models = n_models
        self.models = [
            BCGProbabilisticClassifier(feature_dim, **model_kwargs) 
            for _ in range(n_models)
        ]
        self.feature_scalers = []
    
    def train_ensemble(self, train_loader, val_loader, epochs=50, device='cpu'):
        """
        Train all models in the ensemble.
        """
        for i, model in enumerate(self.models):
            print(f"Training ensemble model {i+1}/{self.n_models}")
            # Train individual model (implementation would go here)
            # This is a placeholder for the training loop
            pass
    
    def predict_with_uncertainty(self, features):
        """
        Predict with ensemble uncertainty.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            tuple: (mean_probabilities, aleatoric_uncertainty, epistemic_uncertainty)
        """
        predictions = []
        uncertainties = []
        
        for i, model in enumerate(self.models):
            scaler = self.feature_scalers[i] if i < len(self.feature_scalers) else None
            
            if scaler:
                scaled_features = scaler.transform(features.numpy())
                scaled_features = torch.FloatTensor(scaled_features)
            else:
                scaled_features = features
            
            # Get prediction and MC uncertainty from individual model
            mean_prob, mc_uncertainty = model.predict_with_uncertainty(scaled_features)
            predictions.append(mean_prob)
            uncertainties.append(mc_uncertainty)
        
        predictions = torch.stack(predictions)  # (n_models, n_candidates)
        uncertainties = torch.stack(uncertainties)  # (n_models, n_candidates)
        
        # Ensemble statistics
        ensemble_mean = predictions.mean(dim=0)
        ensemble_epistemic = predictions.std(dim=0)  # Disagreement between models
        ensemble_aleatoric = uncertainties.mean(dim=0)  # Average MC uncertainty
        
        return ensemble_mean, ensemble_aleatoric, ensemble_epistemic


def predict_bcg_with_probabilities(image, model, feature_scaler=None, 
                                 detection_threshold=0.1, 
                                 return_all_candidates=False, **candidate_kwargs):
    """
    Predict BCG candidates with calibrated probabilities and uncertainty.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    model : BCGProbabilisticClassifier
        Trained probabilistic classifier
    feature_scaler : StandardScaler or None
        Feature scaler
    detection_threshold : float
        Probability threshold for considering a detection
    return_all_candidates : bool
        Whether to return all candidates or just detections
    **candidate_kwargs : dict
        Arguments for candidate finding
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - best_bcg: (x, y) coordinates of highest probability BCG
        - all_candidates: All candidate coordinates
        - probabilities: BCG probabilities for all candidates
        - uncertainties: Uncertainty estimates (if model supports it)
        - detections: Candidates above threshold
        - detection_probabilities: Probabilities for detected candidates
    """
    # Import here to avoid circular imports
    from utils.candidate_based_bcg import predict_bcg_from_candidates
    best_bcg, all_candidates, raw_scores = predict_bcg_from_candidates(
        image, model=None, feature_scaler=None, **candidate_kwargs
    )
    
    if len(all_candidates) == 0:
        return {
            'best_bcg': None,
            'all_candidates': np.array([]),
            'probabilities': np.array([]),
            'uncertainties': np.array([]),
            'detections': np.array([]),
            'detection_probabilities': np.array([])
        }
    
    # Extract features for probability prediction
    from utils.candidate_based_bcg import extract_candidate_features
    features, _ = extract_candidate_features(image, all_candidates)
    
    # Scale features
    if feature_scaler is not None:
        scaled_features = feature_scaler.transform(features)
        features_tensor = torch.FloatTensor(scaled_features)
    else:
        features_tensor = torch.FloatTensor(features)
    
    # Get probabilities and uncertainties
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict_with_uncertainty'):
            probabilities, uncertainties = model.predict_with_uncertainty(features_tensor)
            probabilities = probabilities.numpy()
            uncertainties = uncertainties.numpy()
        else:
            logits = model(features_tensor).squeeze()
            probabilities = torch.sigmoid(logits).numpy()
            uncertainties = np.zeros_like(probabilities)  # No uncertainty available
    
    # Find detections above threshold
    detection_mask = probabilities >= detection_threshold
    detections = all_candidates[detection_mask]
    detection_probabilities = probabilities[detection_mask]
    
    # Find best BCG (highest probability)
    if len(probabilities) > 0:
        best_idx = np.argmax(probabilities)
        best_bcg = tuple(all_candidates[best_idx])
    else:
        best_bcg = None
    
    results = {
        'best_bcg': best_bcg,
        'all_candidates': all_candidates,
        'probabilities': probabilities,
        'uncertainties': uncertainties,
        'detections': detections,
        'detection_probabilities': detection_probabilities
    }
    
    return results


def calibrate_temperature(model, val_loader, device='cpu'):
    """
    Calibrate the temperature parameter using validation data.
    
    Args:
        model: BCGProbabilisticClassifier with temperature scaling
        val_loader: Validation data loader
        device: Device to run calibration on
        
    Returns:
        Calibrated model
    """
    model.to(device)
    model.eval()
    
    # Collect validation predictions and targets
    logits_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            # Apply feature scaling if needed
            logits = model.network(features).squeeze()
            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())
    
    logits = torch.cat(logits_list)
    targets = torch.cat(targets_list)
    
    # Optimize temperature
    temperature_optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    
    def temperature_loss():
        temperature_optimizer.zero_grad()
        scaled_logits = logits / model.temperature
        loss = F.binary_cross_entropy_with_logits(scaled_logits, targets.float())
        loss.backward()
        return loss
    
    temperature_optimizer.step(temperature_loss)
    
    print(f"Optimal temperature: {model.temperature.item():.3f}")
    
    return model
