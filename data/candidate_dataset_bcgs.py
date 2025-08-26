"""
Candidate-Based Dataset for BCG Training - BCG Specific Version

This module provides dataset classes that generate candidate features for training
BCG models using the new BCG data with additional features like redshift and delta_mstar_z.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features


class BCGCandidateDataset(Dataset):
    """
    Dataset that converts BCG coordinate regression into candidate classification for BCG data.
    
    For each image:
    1. Find bright spot candidates (local maxima)
    2. Extract features around each candidate
    3. Include additional features (redshift, delta_mstar_z) if available
    4. Label which candidate is closest to true BCG
    5. Return (candidate_features, target_label, additional_features)
    """
    
    def __init__(self, images, bcg_coords, additional_features=None, candidate_params=None, min_candidates=3):
        """
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images or array of shape (N, H, W, C)
        bcg_coords : numpy.ndarray
            True BCG coordinates of shape (N, 2)
        additional_features : numpy.ndarray or None
            Additional features like [redshift, delta_mstar_z] of shape (N, n_features)
        candidate_params : dict
            Parameters for candidate finding
        min_candidates : int
            Minimum number of candidates required (skip images with fewer)
        """
        self.images = images
        self.bcg_coords = bcg_coords
        self.additional_features = additional_features
        self.min_candidates = min_candidates
        
        # Default candidate finding parameters
        if candidate_params is None:
            candidate_params = {
                'min_distance': 15,
                'threshold_rel': 0.12,
                'exclude_border': 0,
                'max_candidates': 32
            }
        self.candidate_params = candidate_params
        
        # Process all images to create candidate-based samples
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """
        Process all images to generate candidate-based training samples.
        """
        print(f"Preparing BCG candidate-based samples from {len(self.images)} images...")
        
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
            
            # Get additional features for this image if available
            img_additional_features = None
            if self.additional_features is not None:
                img_additional_features = self.additional_features[img_idx]
            
            # Store sample
            sample = {
                'features': features.astype(np.float32),
                'target': target_label,
                'candidates': candidates,
                'true_bcg': true_bcg,
                'image_idx': img_idx,
                'min_distance': distances[target_label],
                'additional_features': img_additional_features
            }
            
            self.samples.append(sample)
            valid_samples += 1
            total_candidates += len(candidates)
        
        print(f"Created {valid_samples} BCG candidate-based samples")
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
        
        result = {
            'features': features,
            'target': target,
            'candidates': sample['candidates'],
            'true_bcg': sample['true_bcg'],
            'image_idx': sample['image_idx']
        }
        
        # Add additional features if available
        if sample['additional_features'] is not None:
            result['additional_features'] = torch.FloatTensor(sample['additional_features'])
        
        return result


# MultiScaleBCGCandidateDataset removed - use separate 2.2 or 3.8 arcmin datasets instead


def collate_bcg_candidate_samples(batch):
    """
    Custom collate function for BCG candidate-based samples.
    
    Handles variable-length candidate sequences and additional features.
    """
    batch_features = []
    batch_targets = []
    batch_sample_indices = []
    batch_additional_features = []
    
    for sample_idx, sample in enumerate(batch):
        features = sample['features']  # Shape: (n_candidates, feature_dim)
        target = sample['target'].item()  # Scalar
        n_candidates = features.shape[0]
        
        # Add features for all candidates in this sample
        batch_features.append(features)
        
        # Create target for this sample
        sample_targets = torch.zeros(n_candidates, dtype=torch.long)
        sample_targets[target] = 1  # Mark the correct candidate
        batch_targets.append(sample_targets)
        
        # Track which sample each candidate belongs to
        batch_sample_indices.extend([sample_idx] * n_candidates)
        
        # Handle additional features if present
        if 'additional_features' in sample:
            # Repeat additional features for each candidate
            additional_feats = sample['additional_features']
            batch_additional_features.extend([additional_feats] * n_candidates)
    
    # Concatenate all candidates from all samples
    all_features = torch.cat(batch_features, dim=0)  # Shape: (total_candidates, feature_dim)
    all_targets = torch.cat(batch_targets, dim=0)    # Shape: (total_candidates,)
    sample_indices = torch.LongTensor(batch_sample_indices)  # Shape: (total_candidates,)
    
    result = {
        'features': all_features,
        'targets': all_targets,
        'sample_indices': sample_indices,
        'batch_size': len(batch),
        'raw_batch': batch  # Keep original batch for additional info
    }
    
    # Add additional features if present
    if batch_additional_features:
        all_additional_features = torch.stack(batch_additional_features, dim=0)
        result['additional_features'] = all_additional_features
    
    return result


# collate_multiscale_bcg_candidate_samples removed - use separate 2.2 or 3.8 arcmin datasets instead


# Utility functions to create BCG candidate datasets from the data reading modules
def create_bcg_candidate_dataset_from_loader(dataset_loader, candidate_params=None):
    """
    Create a BCG candidate dataset from a regular BCG dataset loader.
    
    Args:
        dataset_loader: BCGDataset instance (2.2 or 3.8 arcmin)
        candidate_params: Parameters for candidate finding
        
    Returns:
        BCGCandidateDataset
    """
    # Extract data from the loader
    images = []
    bcg_coords = []
    additional_features = []
    
    print(f"Processing {len(dataset_loader)} samples to create candidate dataset...")
    
    for i in range(len(dataset_loader)):
        sample = dataset_loader[i]
        
        # Single scale dataset only
        images.append(sample['image'])
        bcg_coords.append(sample['BCG'])
        
        # Extract additional features if available
        if 'additional_features' in sample:
            additional_features.append(sample['additional_features'])
    
    bcg_coords = np.array(bcg_coords)
    additional_features = np.array(additional_features) if additional_features else None
    
    # Create candidate dataset
    candidate_dataset = BCGCandidateDataset(
        images=images,
        bcg_coords=bcg_coords,
        additional_features=additional_features,
        candidate_params=candidate_params
    )
    
    return candidate_dataset