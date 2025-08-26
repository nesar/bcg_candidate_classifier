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


class MultiScaleBCGCandidateDataset(Dataset):
    """
    Candidate-based dataset for multi-scale BCG data (both 2.2 and 3.8 arcmin).
    """
    
    def __init__(self, images_2p2, images_3p8, bcg_coords_2p2, bcg_coords_3p8, 
                 additional_features=None, candidate_params=None, min_candidates=3):
        """
        Parameters:
        -----------
        images_2p2 : list or numpy.ndarray
            2.2 arcmin images
        images_3p8 : list or numpy.ndarray
            3.8 arcmin images  
        bcg_coords_2p2 : numpy.ndarray
            BCG coordinates for 2.2 arcmin images
        bcg_coords_3p8 : numpy.ndarray
            BCG coordinates for 3.8 arcmin images
        additional_features : numpy.ndarray or None
            Additional features like [redshift, delta_mstar_z]
        candidate_params : dict
            Parameters for candidate finding
        min_candidates : int
            Minimum number of candidates required
        """
        self.images_2p2 = images_2p2
        self.images_3p8 = images_3p8
        self.bcg_coords_2p2 = bcg_coords_2p2
        self.bcg_coords_3p8 = bcg_coords_3p8
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
        Process all image pairs to generate multi-scale candidate-based training samples.
        """
        n_images = len(self.images_2p2)
        print(f"Preparing multi-scale BCG candidate samples from {n_images} image pairs...")
        
        valid_samples = 0
        skipped_samples = 0
        total_candidates_2p2 = 0
        total_candidates_3p8 = 0
        
        for img_idx in range(n_images):
            image_2p2 = self.images_2p2[img_idx]
            image_3p8 = self.images_3p8[img_idx]
            true_bcg_2p2 = self.bcg_coords_2p2[img_idx]
            true_bcg_3p8 = self.bcg_coords_3p8[img_idx]
            
            # Convert images to numpy if needed
            if hasattr(image_2p2, 'numpy'):
                image_2p2 = image_2p2.numpy()
            elif torch.is_tensor(image_2p2):
                image_2p2 = image_2p2.numpy()
                
            if hasattr(image_3p8, 'numpy'):
                image_3p8 = image_3p8.numpy()
            elif torch.is_tensor(image_3p8):
                image_3p8 = image_3p8.numpy()
            
            # Find candidates in both images
            candidates_2p2, intensities_2p2 = find_bcg_candidates(image_2p2, **self.candidate_params)
            candidates_3p8, intensities_3p8 = find_bcg_candidates(image_3p8, **self.candidate_params)
            
            if len(candidates_2p2) < self.min_candidates or len(candidates_3p8) < self.min_candidates:
                skipped_samples += 1
                continue
            
            # Extract features for candidates in both scales
            features_2p2, _ = extract_candidate_features(
                image_2p2, candidates_2p2, patch_size=64, include_context=True
            )
            features_3p8, _ = extract_candidate_features(
                image_3p8, candidates_3p8, patch_size=64, include_context=True
            )
            
            if len(features_2p2) == 0 or len(features_3p8) == 0:
                skipped_samples += 1
                continue
            
            # Find target candidates (closest to true BCG in each scale)
            distances_2p2 = np.sqrt(np.sum((candidates_2p2 - true_bcg_2p2)**2, axis=1))
            distances_3p8 = np.sqrt(np.sum((candidates_3p8 - true_bcg_3p8)**2, axis=1))
            target_label_2p2 = np.argmin(distances_2p2)
            target_label_3p8 = np.argmin(distances_3p8)
            
            # Get additional features for this image if available
            img_additional_features = None
            if self.additional_features is not None:
                img_additional_features = self.additional_features[img_idx]
            
            # Store sample
            sample = {
                'features_2p2': features_2p2.astype(np.float32),
                'features_3p8': features_3p8.astype(np.float32),
                'target_2p2': target_label_2p2,
                'target_3p8': target_label_3p8,
                'candidates_2p2': candidates_2p2,
                'candidates_3p8': candidates_3p8,
                'true_bcg_2p2': true_bcg_2p2,
                'true_bcg_3p8': true_bcg_3p8,
                'image_idx': img_idx,
                'min_distance_2p2': distances_2p2[target_label_2p2],
                'min_distance_3p8': distances_3p8[target_label_3p8],
                'additional_features': img_additional_features
            }
            
            self.samples.append(sample)
            valid_samples += 1
            total_candidates_2p2 += len(candidates_2p2)
            total_candidates_3p8 += len(candidates_3p8)
        
        print(f"Created {valid_samples} multi-scale BCG candidate samples")
        print(f"Skipped {skipped_samples} image pairs (insufficient candidates)")
        print(f"Average candidates per 2.2' image: {total_candidates_2p2/valid_samples:.1f}")
        print(f"Average candidates per 3.8' image: {total_candidates_3p8/valid_samples:.1f}")
        
        # Report distance statistics
        min_distances_2p2 = [s['min_distance_2p2'] for s in self.samples]
        min_distances_3p8 = [s['min_distance_3p8'] for s in self.samples]
        
        print(f"True BCG distance to nearest candidate (2.2'):")
        print(f"  Mean: {np.mean(min_distances_2p2):.1f} pixels")
        print(f"  Median: {np.median(min_distances_2p2):.1f} pixels")
        
        print(f"True BCG distance to nearest candidate (3.8'):")
        print(f"  Mean: {np.mean(min_distances_3p8):.1f} pixels")
        print(f"  Median: {np.median(min_distances_3p8):.1f} pixels")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features_2p2 = torch.FloatTensor(sample['features_2p2'])
        features_3p8 = torch.FloatTensor(sample['features_3p8'])
        target_2p2 = torch.LongTensor([sample['target_2p2']])
        target_3p8 = torch.LongTensor([sample['target_3p8']])
        
        result = {
            'features_2p2': features_2p2,
            'features_3p8': features_3p8,
            'target_2p2': target_2p2,
            'target_3p8': target_3p8,
            'candidates_2p2': sample['candidates_2p2'],
            'candidates_3p8': sample['candidates_3p8'],
            'true_bcg_2p2': sample['true_bcg_2p2'],
            'true_bcg_3p8': sample['true_bcg_3p8'],
            'image_idx': sample['image_idx']
        }
        
        # Add additional features if available
        if sample['additional_features'] is not None:
            result['additional_features'] = torch.FloatTensor(sample['additional_features'])
        
        return result


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


def collate_multiscale_bcg_candidate_samples(batch):
    """
    Custom collate function for multi-scale BCG candidate-based samples.
    """
    batch_features_2p2 = []
    batch_features_3p8 = []
    batch_targets_2p2 = []
    batch_targets_3p8 = []
    batch_sample_indices_2p2 = []
    batch_sample_indices_3p8 = []
    batch_additional_features = []
    
    for sample_idx, sample in enumerate(batch):
        features_2p2 = sample['features_2p2']
        features_3p8 = sample['features_3p8']
        target_2p2 = sample['target_2p2'].item()
        target_3p8 = sample['target_3p8'].item()
        n_candidates_2p2 = features_2p2.shape[0]
        n_candidates_3p8 = features_3p8.shape[0]
        
        # Add features for both scales
        batch_features_2p2.append(features_2p2)
        batch_features_3p8.append(features_3p8)
        
        # Create targets for both scales
        sample_targets_2p2 = torch.zeros(n_candidates_2p2, dtype=torch.long)
        sample_targets_2p2[target_2p2] = 1
        batch_targets_2p2.append(sample_targets_2p2)
        
        sample_targets_3p8 = torch.zeros(n_candidates_3p8, dtype=torch.long)
        sample_targets_3p8[target_3p8] = 1
        batch_targets_3p8.append(sample_targets_3p8)
        
        # Track sample indices
        batch_sample_indices_2p2.extend([sample_idx] * n_candidates_2p2)
        batch_sample_indices_3p8.extend([sample_idx] * n_candidates_3p8)
        
        # Handle additional features
        if 'additional_features' in sample:
            additional_feats = sample['additional_features']
            # Add for both scales
            batch_additional_features.extend([additional_feats] * (n_candidates_2p2 + n_candidates_3p8))
    
    # Concatenate all candidates
    all_features_2p2 = torch.cat(batch_features_2p2, dim=0)
    all_features_3p8 = torch.cat(batch_features_3p8, dim=0)
    all_targets_2p2 = torch.cat(batch_targets_2p2, dim=0)
    all_targets_3p8 = torch.cat(batch_targets_3p8, dim=0)
    sample_indices_2p2 = torch.LongTensor(batch_sample_indices_2p2)
    sample_indices_3p8 = torch.LongTensor(batch_sample_indices_3p8)
    
    result = {
        'features_2p2': all_features_2p2,
        'features_3p8': all_features_3p8,
        'targets_2p2': all_targets_2p2,
        'targets_3p8': all_targets_3p8,
        'sample_indices_2p2': sample_indices_2p2,
        'sample_indices_3p8': sample_indices_3p8,
        'batch_size': len(batch),
        'raw_batch': batch
    }
    
    # Add additional features if present
    if batch_additional_features:
        all_additional_features = torch.stack(batch_additional_features, dim=0)
        result['additional_features'] = all_additional_features
    
    return result


# Utility functions to create BCG candidate datasets from the data reading modules
def create_bcg_candidate_dataset_from_loader(dataset_loader, candidate_params=None):
    """
    Create a BCG candidate dataset from a regular BCG dataset loader.
    
    Args:
        dataset_loader: BCGDataset or MultiScaleBCGDataset instance
        candidate_params: Parameters for candidate finding
        
    Returns:
        BCGCandidateDataset or MultiScaleBCGCandidateDataset
    """
    # Extract data from the loader
    images = []
    bcg_coords = []
    additional_features = []
    
    print(f"Processing {len(dataset_loader)} samples to create candidate dataset...")
    
    for i in range(len(dataset_loader)):
        sample = dataset_loader[i]
        
        if 'image_2p2' in sample:
            # Multi-scale dataset
            # For now, we'll use the 2.2 arcmin images as primary
            images.append(sample['image_2p2'])
            bcg_coords.append(sample['BCG_2p2'])
        else:
            # Single scale dataset
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