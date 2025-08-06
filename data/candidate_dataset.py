"""
Candidate-Based Dataset for BCG Training

This module provides dataset classes that generate candidate features for training
instead of using full images for coordinate regression.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features


class BCGCandidateDataset(Dataset):
    """
    Dataset that converts BCG coordinate regression into candidate classification.
    
    For each image:
    1. Find bright spot candidates (local maxima)
    2. Extract features around each candidate
    3. Label which candidate is closest to true BCG
    4. Return (candidate_features, target_label)
    """
    
    def __init__(self, images, bcg_coords, candidate_params=None, min_candidates=3):
        """
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images or array of shape (N, H, W, C)
        bcg_coords : numpy.ndarray
            True BCG coordinates of shape (N, 2)
        candidate_params : dict
            Parameters for candidate finding
        min_candidates : int
            Minimum number of candidates required (skip images with fewer)
        """
        self.images = images
        self.bcg_coords = bcg_coords
        self.min_candidates = min_candidates
        
        # Default candidate finding parameters
        if candidate_params is None:
            candidate_params = {
                'min_distance': 15,
                'threshold_rel': 0.12,
                'exclude_border': 30,
                'max_candidates': 25
            }
        self.candidate_params = candidate_params
        
        # Process all images to create candidate-based samples
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """
        Process all images to generate candidate-based training samples.
        """
        print(f"Preparing candidate-based samples from {len(self.images)} images...")
        
        valid_samples = 0
        skipped_samples = 0
        total_candidates = 0
        
        for img_idx, (image, true_bcg) in enumerate(zip(self.images, self.bcg_coords)):
            # Convert image to numpy if needed
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif torch.is_tensor(image):
                image = image.numpy()
            
            # Find candidates in this image
            candidates, intensities = find_bcg_candidates(image, **self.candidate_params)
            
            if len(candidates) < self.min_candidates:
                skipped_samples += 1
                continue
            
            # Extract features for all candidates
            features, patches = extract_candidate_features(
                image, 
                candidates,
                patch_size=64,
                include_context=True
            )
            
            if len(features) == 0:
                skipped_samples += 1
                continue
            
            # Find which candidate is closest to true BCG
            distances = np.sqrt(np.sum((candidates - true_bcg)**2, axis=1))
            target_label = np.argmin(distances)
            
            # Store sample
            sample = {
                'features': features.astype(np.float32),
                'target': target_label,
                'candidates': candidates,
                'true_bcg': true_bcg,
                'image_idx': img_idx,
                'min_distance': distances[target_label]
            }
            
            self.samples.append(sample)
            valid_samples += 1
            total_candidates += len(candidates)
        
        print(f"Created {valid_samples} candidate-based samples")
        print(f"Skipped {skipped_samples} images (insufficient candidates)")
        print(f"Average candidates per image: {total_candidates/valid_samples:.1f}")
        
        # Report distance statistics
        min_distances = [s['min_distance'] for s in self.samples]
        print(f"True BCG distance to nearest candidate:")
        print(f"  Mean: {np.mean(min_distances):.1f} pixels")
        print(f"  Median: {np.median(min_distances):.1f} pixels") 
        print(f"  Max: {np.max(min_distances):.1f} pixels")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = torch.FloatTensor(sample['features'])
        target = torch.LongTensor([sample['target']])
        
        return {
            'features': features,
            'target': target,
            'candidates': sample['candidates'],
            'true_bcg': sample['true_bcg'],
            'image_idx': sample['image_idx']
        }


def collate_candidate_samples(batch):
    """
    Custom collate function for candidate-based samples.
    
    Since each image has a different number of candidates, we need to handle
    variable-length sequences.
    """
    batch_features = []
    batch_targets = []
    batch_sample_indices = []
    
    for sample_idx, sample in enumerate(batch):
        features = sample['features']  # Shape: (n_candidates, feature_dim)
        target = sample['target'].item()  # Scalar
        n_candidates = features.shape[0]
        
        # Add features for all candidates in this sample
        batch_features.append(features)
        
        # Create target for this sample (one-hot or index)
        sample_targets = torch.zeros(n_candidates, dtype=torch.long)
        sample_targets[target] = 1  # Mark the correct candidate
        batch_targets.append(sample_targets)
        
        # Track which sample each candidate belongs to
        batch_sample_indices.extend([sample_idx] * n_candidates)
    
    # Concatenate all candidates from all samples
    all_features = torch.cat(batch_features, dim=0)  # Shape: (total_candidates, feature_dim)
    all_targets = torch.cat(batch_targets, dim=0)    # Shape: (total_candidates,)
    sample_indices = torch.LongTensor(batch_sample_indices)  # Shape: (total_candidates,)
    
    return {
        'features': all_features,
        'targets': all_targets,
        'sample_indices': sample_indices,
        'batch_size': len(batch),
        'raw_batch': batch  # Keep original batch for additional info
    }


class CandidateBasedTrainer:
    """
    Trainer class for candidate-based BCG prediction.
    """
    
    def __init__(self, model, device='cpu', feature_scaler=None):
        self.model = model
        self.device = device
        self.feature_scaler = feature_scaler
        self.model.to(device)
        
    def train_step(self, batch, optimizer, criterion):
        """
        Training step for candidate-based model.
        """
        self.model.train()
        
        features = batch['features'].to(self.device)
        targets = batch['targets'].to(self.device)
        sample_indices = batch['sample_indices'].to(self.device)
        batch_size = batch['batch_size']
        
        # Apply feature scaling if available
        if self.feature_scaler is not None:
            features_np = features.cpu().numpy()
            features_scaled = self.feature_scaler.transform(features_np)
            features = torch.FloatTensor(features_scaled).to(self.device)
        
        # Forward pass: get scores for all candidates
        candidate_scores = self.model(features).squeeze()  # Shape: (total_candidates,)
        
        # For each sample, compute loss over its candidates
        total_loss = 0
        correct_predictions = 0
        
        for sample_idx in range(batch_size):
            # Get candidates for this sample
            sample_mask = sample_indices == sample_idx
            sample_scores = candidate_scores[sample_mask]
            sample_targets = targets[sample_mask]
            
            if len(sample_scores) == 0:
                continue
            
            # Find target candidate (the one with label 1)
            target_idx = torch.argmax(sample_targets).item()
            
            # Compute cross-entropy loss
            sample_loss = criterion(sample_scores.unsqueeze(0), torch.LongTensor([target_idx]).to(self.device))
            total_loss += sample_loss
            
            # Check if prediction is correct (highest score matches target)
            predicted_idx = torch.argmax(sample_scores).item()
            if predicted_idx == target_idx:
                correct_predictions += 1
        
        # Average loss over batch
        avg_loss = total_loss / batch_size if batch_size > 0 else 0
        accuracy = correct_predictions / batch_size if batch_size > 0 else 0
        
        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        return avg_loss.item(), accuracy
    
    def evaluate_step(self, batch, criterion):
        """
        Evaluation step for candidate-based model.
        """
        self.model.eval()
        
        with torch.no_grad():
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            sample_indices = batch['sample_indices'].to(self.device)
            batch_size = batch['batch_size']
            
            # Apply feature scaling if available
            if self.feature_scaler is not None:
                features_np = features.cpu().numpy()
                features_scaled = self.feature_scaler.transform(features_np)
                features = torch.FloatTensor(features_scaled).to(self.device)
            
            # Forward pass
            candidate_scores = self.model(features).squeeze()
            
            # Compute metrics
            total_loss = 0
            correct_predictions = 0
            
            for sample_idx in range(batch_size):
                sample_mask = sample_indices == sample_idx
                sample_scores = candidate_scores[sample_mask]
                sample_targets = targets[sample_mask]
                
                if len(sample_scores) == 0:
                    continue
                
                target_idx = torch.argmax(sample_targets).item()
                sample_loss = criterion(sample_scores.unsqueeze(0), torch.LongTensor([target_idx]).to(self.device))
                total_loss += sample_loss
                
                predicted_idx = torch.argmax(sample_scores).item()
                if predicted_idx == target_idx:
                    correct_predictions += 1
            
            avg_loss = total_loss / batch_size if batch_size > 0 else 0
            accuracy = correct_predictions / batch_size if batch_size > 0 else 0
            
            return avg_loss.item(), accuracy