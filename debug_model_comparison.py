#!/usr/bin/env python3
"""
Debug script to compare UQ and non-UQ model outputs directly
"""

import torch
import numpy as np
import joblib
from ml_models.candidate_classifier import BCGCandidateClassifier
from ml_models.uq_classifier import BCGProbabilisticClassifier

def compare_models(uq_model_path, nonuq_model_path, uq_scaler_path, nonuq_scaler_path, feature_dim):
    """Compare outputs of UQ and non-UQ models on the same synthetic data."""
    
    print("=" * 60)
    print("MODEL COMPARISON DIAGNOSTIC")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    uq_model = BCGProbabilisticClassifier(feature_dim)
    uq_model.load_state_dict(torch.load(uq_model_path, map_location='cpu'))
    uq_model.eval()
    
    nonuq_model = BCGCandidateClassifier(feature_dim)
    nonuq_model.load_state_dict(torch.load(nonuq_model_path, map_location='cpu'))
    nonuq_model.eval()
    
    # Load scalers
    uq_scaler = joblib.load(uq_scaler_path)
    nonuq_scaler = joblib.load(nonuq_scaler_path)
    
    print(f"Models loaded successfully")
    print(f"Feature dimension: {feature_dim}")
    
    # Create synthetic test data (5 candidates)
    np.random.seed(42)
    n_candidates = 5
    synthetic_features = np.random.randn(n_candidates, feature_dim)
    
    print(f"\nTesting with {n_candidates} synthetic candidates...")
    print(f"Synthetic features shape: {synthetic_features.shape}")
    print(f"Feature range: [{np.min(synthetic_features):.3f}, {np.max(synthetic_features):.3f}]")
    
    # Scale features with both scalers
    uq_scaled = uq_scaler.transform(synthetic_features)
    nonuq_scaled = nonuq_scaler.transform(synthetic_features)
    
    print(f"\nAfter UQ scaling: [{np.min(uq_scaled):.3f}, {np.max(uq_scaled):.3f}]")
    print(f"After non-UQ scaling: [{np.min(nonuq_scaled):.3f}, {np.max(nonuq_scaled):.3f}]")
    
    # Convert to tensors
    uq_tensor = torch.FloatTensor(uq_scaled)
    nonuq_tensor = torch.FloatTensor(nonuq_scaled)
    
    with torch.no_grad():
        # Non-UQ model outputs
        nonuq_scores = nonuq_model(nonuq_tensor).squeeze(-1)
        nonuq_probs = torch.sigmoid(nonuq_scores)  # Convert to probabilities for comparison
        nonuq_best_idx = torch.argmax(nonuq_scores).item()
        
        print(f"\n" + "="*30)
        print("NON-UQ MODEL RESULTS:")
        print("="*30)
        print(f"Raw scores: {nonuq_scores.numpy()}")
        print(f"Score range: [{torch.min(nonuq_scores):.4f}, {torch.max(nonuq_scores):.4f}]")
        print(f"Sigmoid probs: {nonuq_probs.numpy()}")
        print(f"Prob range: [{torch.min(nonuq_probs):.6f}, {torch.max(nonuq_probs):.6f}]")
        print(f"Best candidate: {nonuq_best_idx} (score: {nonuq_scores[nonuq_best_idx]:.4f})")
        
        # UQ model outputs
        print(f"\n" + "="*30)
        print("UQ MODEL RESULTS:")
        print("="*30)
        
        # Raw logits
        uq_raw_logits = uq_model(uq_tensor).squeeze(-1)
        print(f"Raw logits: {uq_raw_logits.numpy()}")
        print(f"Logits range: [{torch.min(uq_raw_logits):.4f}, {torch.max(uq_raw_logits):.4f}]")
        print(f"Temperature: {uq_model.temperature.item():.6f}")
        
        # Temperature-scaled logits
        temp_scaled_logits = uq_raw_logits / uq_model.temperature
        print(f"Temp-scaled logits: {temp_scaled_logits.numpy()}")
        print(f"Temp-scaled range: [{torch.min(temp_scaled_logits):.4f}, {torch.max(temp_scaled_logits):.4f}]")
        
        # Probabilities
        uq_raw_probs = torch.sigmoid(uq_raw_logits)
        uq_temp_probs = torch.sigmoid(temp_scaled_logits)
        
        print(f"Raw sigmoid probs: {uq_raw_probs.numpy()}")
        print(f"Raw prob range: [{torch.min(uq_raw_probs):.10f}, {torch.max(uq_raw_probs):.10f}]")
        
        print(f"Temp-scaled probs: {uq_temp_probs.numpy()}")
        print(f"Temp prob range: [{torch.min(uq_temp_probs):.10f}, {torch.max(uq_temp_probs):.10f}]")
        
        uq_raw_best_idx = torch.argmax(uq_raw_logits).item()
        uq_temp_best_idx = torch.argmax(temp_scaled_logits).item()
        
        print(f"Best candidate (raw): {uq_raw_best_idx} (logit: {uq_raw_logits[uq_raw_best_idx]:.4f})")
        print(f"Best candidate (temp): {uq_temp_best_idx} (logit: {temp_scaled_logits[uq_temp_best_idx]:.4f})")
        
        # MC Dropout uncertainty
        if hasattr(uq_model, 'predict_with_uncertainty'):
            print(f"\nMC Dropout Results:")
            mc_probs, mc_uncertainties = uq_model.predict_with_uncertainty(uq_tensor)
            print(f"MC probs: {mc_probs.numpy()}")
            print(f"MC prob range: [{torch.min(mc_probs):.10f}, {torch.max(mc_probs):.10f}]")
            print(f"MC uncertainties: {mc_uncertainties.numpy()}")
            mc_best_idx = torch.argmax(mc_probs).item()
            print(f"Best candidate (MC): {mc_best_idx} (prob: {mc_probs[mc_best_idx]:.10f})")
        
        print(f"\n" + "="*30)
        print("COMPARISON SUMMARY:")
        print("="*30)
        print(f"Non-UQ best candidate: {nonuq_best_idx}")
        print(f"UQ raw best candidate: {uq_raw_best_idx}")
        print(f"UQ temp best candidate: {uq_temp_best_idx}")
        if 'mc_best_idx' in locals():
            print(f"UQ MC best candidate: {mc_best_idx}")
        
        rankings_match = (nonuq_best_idx == uq_raw_best_idx)
        print(f"Rankings match (raw UQ vs non-UQ): {rankings_match}")
        
        if rankings_match:
            print("✅ GOOD: Rankings match! Problem is just probability calibration.")
            print("💡 SOLUTION: Use raw logits or retrain with proper temperature scaling.")
        else:
            print("❌ BAD: Rankings don't match. Models learned different features.")
            print("💡 SOLUTION: Must retrain UQ model with fixed training code.")
        
        # Scaling comparison
        scaling_similar = np.allclose(uq_scaled, nonuq_scaled, rtol=1e-3)
        print(f"Feature scaling similar: {scaling_similar}")
        if not scaling_similar:
            print("⚠️  WARNING: Different feature scaling between models!")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare UQ and non-UQ model outputs")
    parser.add_argument('--uq_model', required=True, help='Path to UQ model .pth file')
    parser.add_argument('--nonuq_model', required=True, help='Path to non-UQ model .pth file')
    parser.add_argument('--uq_scaler', required=True, help='Path to UQ scaler .pkl file')
    parser.add_argument('--nonuq_scaler', required=True, help='Path to non-UQ scaler .pkl file')
    parser.add_argument('--feature_dim', type=int, required=True, help='Feature dimension')
    
    args = parser.parse_args()
    
    compare_models(args.uq_model, args.nonuq_model, args.uq_scaler, args.nonuq_scaler, args.feature_dim)