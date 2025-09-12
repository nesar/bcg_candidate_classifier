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
    
    def __init__(self, images, bcg_coords, additional_features=None, candidate_params=None, min_candidates=3, patch_size=64):
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
        patch_size : int
            Size of square patches extracted around candidates (default: 64)
        """
        self.images = images
        self.bcg_coords = bcg_coords
        self.additional_features = additional_features
        self.min_candidates = min_candidates
        self.patch_size = patch_size
        
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
                patch_size=self.patch_size,
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


class DESpriorBCGCandidateDataset(Dataset):
    """
    Candidate-based dataset using DESprior candidates instead of automatic candidate finding.
    
    Uses pre-defined candidate locations from DESprior catalog for each cluster.
    """
    
    def __init__(self, images, bcg_coords, candidates_coords, candidate_features=None, 
                 additional_features=None, filter_inside_image=True, patch_size=64):
        """
        Parameters:
        -----------
        images : list or numpy.ndarray
            List of images or array of shape (N, H, W, C)
        bcg_coords : numpy.ndarray
            True BCG coordinates of shape (N, 2)
        candidates_coords : list of numpy.ndarray
            List of candidate coordinates for each image. Each element is array of shape (n_candidates_i, 2)
        candidate_features : list of numpy.ndarray or None
            List of additional features for each candidate (e.g., delta_mstar, starflag)
        additional_features : numpy.ndarray or None
            Additional features like [redshift, delta_mstar_z] of shape (N, n_features)
        filter_inside_image : bool
            Whether to filter out candidates outside image bounds
        patch_size : int
            Size of square patches extracted around candidates (default: 64)
        """
        self.images = images
        self.bcg_coords = bcg_coords
        self.candidates_coords = candidates_coords
        self.candidate_features = candidate_features
        self.additional_features = additional_features
        self.filter_inside_image = filter_inside_image
        self.patch_size = patch_size
        
        # Process all images to create candidate-based samples
        self.samples = []
        self._prepare_samples()
    
    def _prepare_samples(self):
        """
        Process all images to generate DESprior candidate-based training samples.
        """
        print(f"Preparing DESprior BCG candidate samples from {len(self.images)} images...")
        
        valid_samples = 0
        skipped_samples = 0
        total_candidates = 0
        candidates_outside = 0
        
        for img_idx, (image, true_bcg, candidates) in enumerate(zip(self.images, self.bcg_coords, self.candidates_coords)):
            # Convert image to numpy if needed
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif torch.is_tensor(image):
                image = image.numpy()
            
            # Filter candidates inside image if requested
            if self.filter_inside_image:
                inside_mask = ((candidates[:, 0] >= 0) & (candidates[:, 0] <= 512) & 
                              (candidates[:, 1] >= 0) & (candidates[:, 1] <= 512))
                candidates_filtered = candidates[inside_mask]
                candidates_outside += len(candidates) - len(candidates_filtered)
                
                # Filter candidate features if available
                if self.candidate_features is not None:
                    candidate_feats_filtered = self.candidate_features[img_idx][inside_mask]
                else:
                    candidate_feats_filtered = None
            else:
                candidates_filtered = candidates
                candidate_feats_filtered = self.candidate_features[img_idx] if self.candidate_features is not None else None
            
            if len(candidates_filtered) == 0:
                skipped_samples += 1
                continue
            
            # Extract features for all candidates using standard method
            features, patches = extract_candidate_features(
                image, 
                candidates_filtered,
                patch_size=self.patch_size,
                include_context=True
            )
            
            if len(features) == 0:
                skipped_samples += 1
                continue
            
            # Combine visual features with candidate-specific features if available
            if candidate_feats_filtered is not None:
                # Concatenate visual features with candidate features (delta_mstar, starflag, etc.)
                combined_features = np.hstack([features, candidate_feats_filtered])
            else:
                combined_features = features
            
            # Find which candidate is closest to true BCG
            distances = np.sqrt(np.sum((candidates_filtered - true_bcg)**2, axis=1))
            target_label = np.argmin(distances)
            
            # Get additional features for this image if available
            img_additional_features = None
            if self.additional_features is not None:
                img_additional_features = self.additional_features[img_idx]
            
            # Store sample
            sample = {
                'features': combined_features.astype(np.float32),
                'target': target_label,
                'candidates': candidates_filtered,
                'true_bcg': true_bcg,
                'image_idx': img_idx,
                'min_distance': distances[target_label],
                'additional_features': img_additional_features
            }
            
            self.samples.append(sample)
            valid_samples += 1
            total_candidates += len(candidates_filtered)
        
        print(f"Created {valid_samples} DESprior BCG candidate samples")
        print(f"Skipped {skipped_samples} images (no valid candidates)")
        print(f"Average candidates per image: {total_candidates/valid_samples:.1f}")
        if self.filter_inside_image:
            print(f"Filtered out {candidates_outside} candidates outside image bounds")
        
        # Report distance statistics
        min_distances = [s['min_distance'] for s in self.samples]
        print(f"True BCG distance to nearest DESprior candidate:")
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
def create_bcg_candidate_dataset_from_loader(dataset_loader, candidate_params=None, candidate_type='automatic'):
    """
    Create a BCG candidate dataset from a regular BCG dataset loader.
    
    Args:
        dataset_loader: BCGDataset instance (2.2 or 3.8 arcmin)
        candidate_params: Parameters for candidate finding (for automatic type)
        candidate_type: Either 'automatic' or 'DESprior'
        
    Returns:
        BCGCandidateDataset or DESpriorBCGCandidateDataset
    """
    # Extract data from the loader
    images = []
    bcg_coords = []
    additional_features = []
    
    print(f"Processing {len(dataset_loader)} samples to create {candidate_type} candidate dataset...")
    
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
    
    if candidate_type == 'automatic':
        # Create automatic candidate dataset
        candidate_dataset = BCGCandidateDataset(
            images=images,
            bcg_coords=bcg_coords,
            additional_features=additional_features,
            candidate_params=candidate_params
        )
    else:
        raise ValueError(f"Use create_desprior_candidate_dataset_from_files() for DESprior candidates")
    
    return candidate_dataset


def create_desprior_candidate_dataset_from_files(dataset_type='2p2arcmin', z_range=None, delta_mstar_z_range=None,
                                                filter_inside_image=True, candidate_delta_mstar_range=None,
                                                use_clean_data=True):
    """
    Create a DESprior candidate dataset from CSV files.
    
    Args:
        dataset_type: Either '2p2arcmin' or '3p8arcmin'
        z_range: Tuple (z_min, z_max) to filter BCG data by redshift
        delta_mstar_z_range: Tuple (delta_min, delta_max) to filter BCG data by delta_mstar_z
        filter_inside_image: Whether to filter out candidates outside image bounds
        candidate_delta_mstar_range: Tuple (delta_min, delta_max) to filter candidates by delta_mstar
        use_clean_data: Use clean matched datasets for pristine ML training (recommended: True)
        
    Returns:
        DESpriorBCGCandidateDataset
    """
    import pandas as pd
    from .data_read_bcgs import prepare_bcg_dataframe, BCGDataset
    
    # Load BCG data with filtering - use clean matched data by default for pristine ML training
    if dataset_type == '2p2arcmin':
        image_dir = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/2p2arcmin/'
        if use_clean_data:
            bcg_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_2p2arcmin_clean_matched.csv'
            candidates_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_2p2arcmin_clean_matched.csv'
        else:
            bcg_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_2p2arcmin_with_coordinates.csv'
            candidates_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_2p2arcmin_with_coordinates.csv'
    elif dataset_type == '3p8arcmin':
        image_dir = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin/'
        if use_clean_data:
            bcg_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_3p8arcmin_clean_matched.csv'
            candidates_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_3p8arcmin_clean_matched.csv'
        else:
            bcg_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_3p8arcmin_with_coordinates.csv'
            candidates_csv_path = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/desprior_candidates_3p8arcmin_with_coordinates.csv'
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use '2p2arcmin' or '3p8arcmin'")
    
    print(f"Loading DESprior data from: {'clean matched' if use_clean_data else 'all available'} dataset")
    
    # Load BCG dataset with filtering
    bcg_df = prepare_bcg_dataframe(bcg_csv_path, z_range=z_range, delta_mstar_z_range=delta_mstar_z_range)
    bcg_dataset = BCGDataset(image_dir, bcg_df)
    
    # Load DESprior candidates
    candidates_df = pd.read_csv(candidates_csv_path)
    
    # Filter candidates by delta_mstar if specified
    if candidate_delta_mstar_range is not None:
        delta_min, delta_max = candidate_delta_mstar_range
        before_filter = len(candidates_df)
        candidates_df = candidates_df[(candidates_df['delta_mstar'] >= delta_min) & (candidates_df['delta_mstar'] <= delta_max)]
        print(f"Applied candidate delta_mstar filter [{delta_min}, {delta_max}]: {len(candidates_df)} candidates remaining (filtered out {before_filter - len(candidates_df)})")
    
    # Extract data from BCG dataset
    images = []
    bcg_coords = []
    additional_features = []
    filenames = []
    
    for i in range(len(bcg_dataset)):
        sample = bcg_dataset[i]
        images.append(sample['image'])
        bcg_coords.append(sample['BCG'])
        filenames.append(sample['filename'])
        
        if 'additional_features' in sample:
            additional_features.append(sample['additional_features'])
    
    bcg_coords = np.array(bcg_coords)
    additional_features = np.array(additional_features) if additional_features else None
    
    # Organize candidates by filename
    candidates_coords = []
    candidate_features = []
    
    for filename in filenames:
        file_candidates = candidates_df[candidates_df['filename'] == filename]
        
        if len(file_candidates) == 0:
            print(f"Warning: No DESprior candidates found for {filename}")
            # Add dummy candidates to avoid skipping
            candidates_coords.append(np.array([[256, 256]]))  # Center pixel
            candidate_features.append(np.array([[0.0, 0]]))  # Dummy delta_mstar, starflag
        else:
            # Extract coordinates and features
            coords = file_candidates[['x', 'y']].values
            feats = file_candidates[['delta_mstar', 'starflag']].values
            
            candidates_coords.append(coords)
            candidate_features.append(feats)
    
    print(f"Loaded DESprior candidates for {len(filenames)} images")
    total_candidates = sum(len(cands) for cands in candidates_coords)
    print(f"Total DESprior candidates: {total_candidates}")
    print(f"Average candidates per image: {total_candidates / len(filenames):.1f}")
    
    # Create DESprior candidate dataset
    candidate_dataset = DESpriorBCGCandidateDataset(
        images=images,
        bcg_coords=bcg_coords,
        candidates_coords=candidates_coords,
        candidate_features=candidate_features,
        additional_features=additional_features,
        filter_inside_image=filter_inside_image
    )
    
    return candidate_dataset