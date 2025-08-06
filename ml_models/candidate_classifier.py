"""
Candidate-Based BCG Classifier Model

This module contains the neural network model for ranking/classifying
BCG candidates based on their extracted features.
"""

import torch
import torch.nn as nn


class BCGCandidateClassifier(nn.Module):
    """
    Neural network to classify/rank BCG candidates based on their features.
    
    This is a multilayer perceptron (MLP) that takes candidate features
    and outputs a score for each candidate.
    """
    
    def __init__(self, feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        """
        Initialize the candidate classifier.
        
        Args:
            feature_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
        """
        super(BCGCandidateClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
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
        
        # Output layer - single score for each candidate
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Forward pass to score candidates.
        
        Args:
            features (torch.Tensor): Features for candidates (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Scores for each candidate (batch_size, 1)
        """
        return self.network(features)
    
    def predict_bcg(self, candidate_features):
        """
        Predict the best BCG candidate from a set of candidates.
        
        Args:
            candidate_features (torch.Tensor): Features for all candidates (n_candidates, feature_dim)
            
        Returns:
            int: Index of the best candidate
            torch.Tensor: Scores for all candidates
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(candidate_features).squeeze()
            best_idx = torch.argmax(scores).item()
            return best_idx, scores
    
    def get_model_info(self):
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'BCGCandidateClassifier',
            'feature_dim': self.feature_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }