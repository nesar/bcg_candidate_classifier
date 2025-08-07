#!/usr/bin/env python3
"""
Enhanced BCG Classifier Training Script

This script extends the original training with:
1. Multi-scale candidate detection for flexible object sizes
2. Uncertainty quantification with probabilistic outputs
"""

import os
# Fix NUMEXPR warning
os.environ['NUMEXPR_MAX_THREADS'] = '64'

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import maximum_filter, zoom

from data.data_read import prepare_dataframe, BCGDataset
from data.candidate_dataset import BCGCandidateDataset, collate_candidate_samples
from data.candidate_dataset import CandidateBasedTrainer
from ml_models.candidate_classifier import BCGCandidateClassifier
from utils.candidate_based_bcg import extract_patch_features, extract_context_features


# ============================================================================
# MULTI-SCALE CANDIDATE DETECTION
# ============================================================================

def find_multiscale_bcg_candidates(image, scales=[0.5, 1.0, 1.5], 
                                  base_min_distance=15, threshold_rel=0.12, 
                                  exclude_border=30, max_candidates_per_scale=10):
    """Find candidates at multiple scales to capture objects of different sizes."""
    # Convert to grayscale if RGB
    if len(image.shape) == 3:
        grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        grayscale = image.copy()
    
    all_candidates = []
    all_intensities = []
    all_scales = []
    
    for scale in scales:
        # Adjust parameters based on scale
        min_distance = max(int(base_min_distance * scale), 5)
        filter_size = max(int(3 * scale), 3)
        
        # Find local maxima with scale-adjusted filter
        local_max_mask = (grayscale == maximum_filter(grayscale, size=filter_size))
        
        # Apply threshold
        threshold_abs = threshold_rel * grayscale.max()
        local_max_mask &= (grayscale > threshold_abs)
        
        # Exclude border
        if exclude_border > 0:
            local_max_mask[:exclude_border, :] = False
            local_max_mask[-exclude_border:, :] = False
            local_max_mask[:, :exclude_border] = False
            local_max_mask[:, -exclude_border:] = False
        
        # Extract coordinates and intensities
        y_coords, x_coords = np.where(local_max_mask)
        if len(y_coords) == 0:
            continue
        
        candidates = np.column_stack((x_coords, y_coords))
        intensities = grayscale[y_coords, x_coords]
        
        # Sort by intensity (brightest first)
        sort_indices = np.argsort(intensities)[::-1]
        candidates = candidates[sort_indices]
        intensities = intensities[sort_indices]
        
        # Apply non-maximum suppression
        selected_candidates = []
        selected_intensities = []
        
        for candidate, intensity in zip(candidates, intensities):
            # Check distance to previously selected candidates
            too_close = False
            
            # Check against candidates from this scale
            for selected_candidate in selected_candidates:
                distance = np.sqrt(np.sum((candidate - selected_candidate)**2))
                if distance < min_distance:
                    too_close = True
                    break
            
            # Check against candidates from other scales
            if not too_close:
                for prev_candidate in all_candidates:
                    distance = np.sqrt(np.sum((candidate - prev_candidate[:2])**2))
                    min_scale_distance = min(min_distance, base_min_distance * prev_candidate[2])
                    if distance < min_scale_distance:
                        too_close = True
                        break
            
            if not too_close:
                selected_candidates.append(candidate)
                selected_intensities.append(intensity)
                
                if len(selected_candidates) >= max_candidates_per_scale:
                    break
        
        # Add scale information and store
        for candidate, intensity in zip(selected_candidates, selected_intensities):
            all_candidates.append(np.append(candidate, scale))
            all_intensities.append(intensity)
            all_scales.append(scale)
    
    if not all_candidates:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to arrays
    candidates_with_scale = np.array(all_candidates)
    intensities_array = np.array(all_intensities)
    scales_array = np.array(all_scales)
    
    # Compute adaptive patch sizes based on scale
    base_patch_size = 64
    patch_sizes = (base_patch_size * scales_array).astype(int)
    patch_sizes = np.clip(patch_sizes, 32, 128)
    
    return candidates_with_scale, intensities_array, patch_sizes


def extract_multiscale_candidate_features(image, candidate_coords_with_scale, patch_sizes, include_context=True):
    """Extract features for multi-scale candidates with adaptive patch sizes."""
    if len(candidate_coords_with_scale) == 0:
        return np.array([]), np.array([])
    
    # Ensure image is 3D
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    
    H, W = image.shape[:2]
    features_list = []
    patches_list = []
    
    for i, (candidate_info, patch_size) in enumerate(zip(candidate_coords_with_scale, patch_sizes)):
        x, y, scale = candidate_info[0], candidate_info[1], candidate_info[2]
        x, y = int(x), int(y)
        
        half_patch = patch_size // 2
        
        # Extract patch around candidate
        x_min = max(0, x - half_patch)
        x_max = min(W, x + half_patch)
        y_min = max(0, y - half_patch)
        y_max = min(H, y + half_patch)
        
        patch = image[y_min:y_max, x_min:x_max]
        
        # Resize to consistent size for feature extraction
        base_patch_size = 64
        if patch.shape[0] != base_patch_size or patch.shape[1] != base_patch_size:
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                scale_y = base_patch_size / patch.shape[0]
                scale_x = base_patch_size / patch.shape[1]
                patch = zoom(patch, (scale_y, scale_x, 1), order=1)
                patch = patch.astype(image.dtype)
            else:
                patch = np.zeros((base_patch_size, base_patch_size, image.shape[2]), dtype=image.dtype)
        
        patches_list.append(patch)
        
        # Extract features from patch
        patch_features = extract_patch_features(patch, x, y, image.shape[:2])
        
        # Add scale-specific features
        scale_features = np.array([
            scale,                    # Scale factor
            patch_size,              # Actual patch size used
            scale / np.mean([s[2] for s in candidate_coords_with_scale])  # Relative scale
        ])
        
        if include_context:
            context_features = extract_context_features(image, x, y, patch_size)
            patch_features = np.concatenate([patch_features, context_features, scale_features])
        else:
            patch_features = np.concatenate([patch_features, scale_features])
        
        features_list.append(patch_features)
    
    features = np.array(features_list) if features_list else np.array([])
    patches = np.array(patches_list) if patches_list else np.array([])
    
    return features, patches


# ============================================================================
# PROBABILISTIC CLASSIFIER FOR UQ
# ============================================================================

class BCGProbabilisticClassifier(nn.Module):
    """Probabilistic BCG classifier that outputs calibrated probabilities."""
    
    def __init__(self, feature_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(BCGProbabilisticClassifier, self).__init__()
        
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
        
        # Output layer - logits for binary classification (BCG vs non-BCG)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, features):
        """Forward pass to get logits."""
        logits = self.network(features)
        # Apply temperature scaling
        logits = logits / self.temperature
        return logits
    
    def predict_probabilities(self, features):
        """Predict calibrated probabilities for being BCG."""
        logits = self.forward(features)
        probabilities = torch.sigmoid(logits)
        return probabilities


# ============================================================================
# ENHANCED DATASET AND TRAINER
# ============================================================================

class EnhancedCandidateDataset(BCGCandidateDataset):
    """Enhanced dataset that supports multi-scale candidate detection."""
    
    def __init__(self, images, bcg_coords, candidate_params=None, min_candidates=3, 
                 use_multiscale=False, scales=[0.5, 1.0, 1.5]):
        self.use_multiscale = use_multiscale
        self.scales = scales
        super().__init__(images, bcg_coords, candidate_params, min_candidates)
    
    def _prepare_samples(self):
        """Process all images to generate candidate-based training samples."""
        scale_type = 'multi-scale' if self.use_multiscale else 'single-scale'
        print(f"Preparing {scale_type} candidate samples from {len(self.images)} images...")
        
        valid_samples = 0
        skipped_samples = 0
        total_candidates = 0
        
        for img_idx, (image, true_bcg) in enumerate(zip(self.images, self.bcg_coords)):
            # Convert image to numpy if needed
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif torch.is_tensor(image):
                image = image.numpy()
            
            # Find candidates with appropriate method
            if self.use_multiscale:
                # Map candidate_params to multiscale function parameters
                multiscale_params = {
                    'scales': self.scales,
                    'base_min_distance': self.candidate_params.get('min_distance', 15),
                    'threshold_rel': self.candidate_params.get('threshold_rel', 0.12),
                    'exclude_border': self.candidate_params.get('exclude_border', 30),
                    'max_candidates_per_scale': self.candidate_params.get('max_candidates_per_scale', 10)
                }
                candidates_with_scale, intensities, patch_sizes = find_multiscale_bcg_candidates(
                    image, **multiscale_params
                )
                if len(candidates_with_scale) < self.min_candidates:
                    skipped_samples += 1
                    continue
                
                candidates = candidates_with_scale[:, :2]  # Extract x, y coordinates
                features, patches = extract_multiscale_candidate_features(
                    image, candidates_with_scale, patch_sizes, include_context=True
                )
            else:
                from utils.candidate_based_bcg import find_bcg_candidates, extract_candidate_features
                candidates, intensities = find_bcg_candidates(image, **self.candidate_params)
                
                if len(candidates) < self.min_candidates:
                    skipped_samples += 1
                    continue
                
                features, patches = extract_candidate_features(
                    image, candidates, patch_size=64, include_context=True
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
        if valid_samples > 0:
            print(f"Average candidates per image: {total_candidates/valid_samples:.1f}")
            
            # Report distance statistics
            min_distances = [s['min_distance'] for s in self.samples]
            print(f"True BCG distance to nearest candidate:")
            print(f"  Mean: {np.mean(min_distances):.1f} pixels")
            print(f"  Median: {np.median(min_distances):.1f} pixels") 
            print(f"  Max: {np.max(min_distances):.1f} pixels")


class ProbabilisticTrainer(CandidateBasedTrainer):
    """Enhanced trainer for probabilistic models with UQ."""
    
    def __init__(self, model, device='cpu', feature_scaler=None, use_uq=False):
        super().__init__(model, device, feature_scaler)
        self.use_uq = use_uq
    
    def evaluate_step(self, batch, criterion):
        """Evaluation step for probabilistic model."""
        self.model.eval()
        
        features = batch['features'].to(self.device)
        targets = batch['targets'].to(self.device)
        sample_indices = batch['sample_indices'].to(self.device)
        batch_size = batch['batch_size']
        
        # Apply feature scaling if available
        if self.feature_scaler is not None:
            features_np = features.cpu().numpy()
            features_scaled = self.feature_scaler.transform(features_np)
            features = torch.FloatTensor(features_scaled).to(self.device)
        
        # Forward pass: get logits for all candidates
        candidate_logits = self.model(features).squeeze()
        
        # For probabilistic models, we need to create binary targets
        # Convert multi-class to binary (BCG vs non-BCG)
        binary_targets = torch.zeros_like(candidate_logits)
        
        start_idx = 0
        for sample_idx in range(batch_size):
            # Count candidates in this sample
            sample_mask = sample_indices == sample_idx
            n_candidates = sample_mask.sum().item()
            
            if n_candidates == 0:
                continue
            
            # Get the target index for this sample
            sample_targets = targets[sample_mask]
            if len(sample_targets) > 0:
                target_idx = sample_targets[0].item()  # BCG index within this sample
                # Set binary target: 1 for BCG, 0 for non-BCG
                if target_idx < n_candidates:
                    binary_targets[start_idx + target_idx] = 1.0
            
            start_idx += n_candidates
        
        # Compute binary cross-entropy loss
        total_loss = criterion(candidate_logits, binary_targets)
        
        # Compute accuracy (fraction of samples where highest prob is BCG)
        correct_predictions = 0
        start_idx = 0
        
        for sample_idx in range(batch_size):
            sample_mask = sample_indices == sample_idx
            n_candidates = sample_mask.sum().item()
            
            if n_candidates == 0:
                continue
            
            sample_logits = candidate_logits[start_idx:start_idx + n_candidates]
            sample_binary_targets = binary_targets[start_idx:start_idx + n_candidates]
            
            # Find predicted and true BCG
            predicted_idx = torch.argmax(torch.sigmoid(sample_logits)).item()
            true_idx = torch.argmax(sample_binary_targets).item() if torch.any(sample_binary_targets > 0) else -1
            
            if predicted_idx == true_idx and true_idx >= 0:
                correct_predictions += 1
                
            start_idx += n_candidates
        
        # Return loss and accuracy
        accuracy = correct_predictions / batch_size if batch_size > 0 else 0
        
        return total_loss.item(), accuracy
    
    def train_step(self, batch, optimizer, criterion):
        """Training step for probabilistic model."""
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
        
        # Forward pass: get logits for all candidates
        candidate_logits = self.model(features).squeeze()
        
        # For probabilistic models, we need to create binary targets
        # Convert multi-class to binary (BCG vs non-BCG)
        binary_targets = torch.zeros_like(candidate_logits)
        
        start_idx = 0
        for sample_idx in range(batch_size):
            # Count candidates in this sample
            sample_mask = sample_indices == sample_idx
            n_candidates = sample_mask.sum().item()
            
            if n_candidates == 0:
                continue
            
            # Get the target index for this sample
            sample_targets = targets[sample_mask]
            if len(sample_targets) > 0:
                target_idx = sample_targets[0].item()  # BCG index within this sample
                # Set binary target: 1 for BCG, 0 for non-BCG
                if target_idx < n_candidates:
                    binary_targets[start_idx + target_idx] = 1.0
            
            start_idx += n_candidates
        
        # Compute binary cross-entropy loss
        total_loss = criterion(candidate_logits, binary_targets)
        
        # Compute accuracy (fraction of samples where highest prob is BCG)
        correct_predictions = 0
        start_idx = 0
        
        for sample_idx in range(batch_size):
            sample_mask = sample_indices == sample_idx
            n_candidates = sample_mask.sum().item()
            
            if n_candidates == 0:
                continue
            
            sample_logits = candidate_logits[start_idx:start_idx + n_candidates]
            sample_binary_targets = binary_targets[start_idx:start_idx + n_candidates]
            
            # Find predicted and true BCG
            predicted_idx = torch.argmax(torch.sigmoid(sample_logits)).item()
            true_idx = torch.argmax(sample_binary_targets).item() if torch.any(sample_binary_targets > 0) else -1
            
            if predicted_idx == true_idx and true_idx >= 0:
                correct_predictions += 1
                
            start_idx += n_candidates
        
        # Return loss and accuracy
        accuracy = correct_predictions / batch_size if batch_size > 0 else 0
        
        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        return total_loss.item(), accuracy


# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, random_seed=42):
    """Split dataset into train/validation/test sets."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    n_samples = len(dataset)
    indices = torch.randperm(n_samples).tolist()
    
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def extract_images_and_coords(dataset):
    """Extract images and coordinates from dataset for candidate processing."""
    images = []
    coords = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        images.append(sample['image'])
        coords.append(sample['BCG'])
    
    return images, coords


def train_enhanced_classifier(train_dataset, val_dataset, args):
    """Train the enhanced BCG classifier with multi-scale and UQ options."""
    print("Setting up enhanced candidate-based training...")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_candidate_samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_candidate_samples
    )
    
    # Determine feature dimension from first batch
    sample_batch = next(iter(train_loader))
    feature_dim = sample_batch['features'].shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Fit feature scaler on training data
    print("Fitting feature scaler...")
    all_train_features = []
    for batch in train_loader:
        all_train_features.append(batch['features'].numpy())
    
    all_train_features = np.vstack(all_train_features)
    feature_scaler = StandardScaler()
    feature_scaler.fit(all_train_features)
    
    # Create model based on UQ setting
    if args.use_uq:
        print("Creating probabilistic classifier with uncertainty quantification...")
        model = BCGProbabilisticClassifier(
            feature_dim=feature_dim,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2
        )
        criterion = nn.BCEWithLogitsLoss()
    else:
        print("Creating standard classifier...")
        model = BCGCandidateClassifier(
            feature_dim=feature_dim,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2
        )
        criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    if args.use_uq:
        trainer = ProbabilisticTrainer(model, device, feature_scaler, use_uq=True)
    else:
        trainer = CandidateBasedTrainer(model, device, feature_scaler)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                loss, acc = trainer.train_step(batch, optimizer, criterion)
                
                epoch_train_loss += loss
                epoch_train_acc += acc
                num_train_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{loss:.4f}",
                    'Acc': f"{acc:.3f}"
                })
        
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_acc = epoch_train_acc / num_train_batches
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, acc = trainer.evaluate_step(batch, criterion)
                
                epoch_val_loss += loss
                epoch_val_acc += acc
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_acc = epoch_val_acc / num_val_batches if num_val_batches > 0 else 0
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.3f}")
        
        # Save best model
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            model_name = 'best_probabilistic_classifier' if args.use_uq else 'best_candidate_classifier'
            save_model(model, feature_scaler, args.output_dir, model_name)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.3f}")
    
    # Save final model
    final_model_name = 'final_probabilistic_classifier' if args.use_uq else 'final_candidate_classifier'
    save_model(model, feature_scaler, args.output_dir, final_model_name)
    
    # Plot training curves
    if args.plot:
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, args.output_dir)
    
    return model, feature_scaler, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }


def save_model(model, feature_scaler, output_dir, name):
    """Save model and scaler."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(output_dir, f"{name}.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save feature scaler
    scaler_path = os.path.join(output_dir, f"{name}_scaler.pkl")
    joblib.dump(feature_scaler, scaler_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if plt.get_backend() != 'Agg':  # Only show if display is available
        plt.show()
    
    print(f"Training curves saved to: {plot_path}")
    plt.close()


def main(args):
    """Main training function."""
    print("=" * 60)
    print("ENHANCED BCG CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Images: {args.image_dir}")
    print(f"Truth table: {args.truth_table}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {args.output_dir}")
    
    # Enhanced features
    if args.use_multiscale:
        print(f"Multi-scale: scales={args.scales}, max_per_scale={args.max_candidates_per_scale}")
    if args.use_uq:
        print(f"Uncertainty quantification: threshold={args.detection_threshold}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataframe = prepare_dataframe(args.image_dir, args.truth_table, args.dataset_type)
    print(f"Found {len(dataframe)} samples in dataset")
    
    # Create BCG dataset
    dataset = BCGDataset(args.image_dir, dataframe)
    
    # Split into train/val/test
    train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
    print(f"Dataset split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
    
    # Extract images and coordinates for candidate processing
    train_images, train_coords = extract_images_and_coords(train_subset)
    val_images, val_coords = extract_images_and_coords(val_subset)
    
    # Create enhanced candidate datasets
    print("\nCreating enhanced candidate-based datasets...")
    
    candidate_params = {
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'exclude_border': args.exclude_border,
        'max_candidates': args.max_candidates
    }
    
    # Add multiscale parameters if enabled
    if args.use_multiscale:
        candidate_params['max_candidates_per_scale'] = args.max_candidates_per_scale
    
    train_candidate_dataset = EnhancedCandidateDataset(
        train_images,
        train_coords,
        candidate_params,
        min_candidates=3,
        use_multiscale=args.use_multiscale,
        scales=args.scales if args.use_multiscale else [1.0]
    )
    
    val_candidate_dataset = EnhancedCandidateDataset(
        val_images, 
        val_coords,
        candidate_params,
        min_candidates=3,
        use_multiscale=args.use_multiscale,
        scales=args.scales if args.use_multiscale else [1.0]
    )
    
    if len(train_candidate_dataset) == 0 or len(val_candidate_dataset) == 0:
        print("Error: No valid candidate samples found. Try adjusting candidate parameters.")
        return
    
    # Train model
    print("\nStarting enhanced training...")
    model, scaler, history = train_enhanced_classifier(
        train_candidate_dataset, 
        val_candidate_dataset, 
        args
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.3f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced BCG Classifier")
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing .tif image files')
    parser.add_argument('--truth_table', type=str, required=True,
                       help='Path to CSV file with BCG coordinates')
    parser.add_argument('--dataset_type', type=str, default='SPT3G_1500d',
                       choices=['SPT3G_1500d', 'megadeep500'],
                       help='Type of dataset')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU if available')
    
    # Traditional candidate finding arguments
    parser.add_argument('--min_distance', type=int, default=15,
                       help='Minimum distance between candidates')
    parser.add_argument('--threshold_rel', type=float, default=0.12,
                       help='Relative threshold for candidate detection')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=25,
                       help='Maximum candidates per image (or per scale if multiscale)')
    
    # NEW: Multi-scale arguments
    parser.add_argument('--use_multiscale', action='store_true',
                       help='Enable multi-scale candidate detection')
    parser.add_argument('--scales', type=str, default='0.5,1.0,1.5',
                       help='Comma-separated scale factors for multiscale detection')
    parser.add_argument('--max_candidates_per_scale', type=int, default=10,
                       help='Maximum candidates per scale in multiscale mode')
    
    # NEW: Uncertainty quantification arguments
    parser.add_argument('--use_uq', action='store_true',
                       help='Enable uncertainty quantification with probabilistic outputs')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Probability threshold for BCG detection (0.0-1.0)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training curves')
    
    args = parser.parse_args()
    
    # Parse scales if multiscale is enabled
    if args.use_multiscale:
        args.scales = [float(s.strip()) for s in args.scales.split(',')]
    else:
        args.scales = [1.0]
    
    # Validate detection threshold
    if args.use_uq:
        args.detection_threshold = max(0.0, min(1.0, args.detection_threshold))
    
    main(args)