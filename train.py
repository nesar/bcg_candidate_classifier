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
from ml_models.uq_classifier import BCGProbabilisticClassifier
# NEW: BCG dataset support
from data.data_read_bcgs import create_bcg_datasets, BCGDataset as NewBCGDataset
from data.candidate_dataset_bcgs import (create_bcg_candidate_dataset_from_loader, 
                                        create_desprior_candidate_dataset_from_files,
                                        collate_bcg_candidate_samples)
from utils.candidate_based_bcg import extract_patch_features, extract_context_features


# ============================================================================
# MULTI-SCALE CANDIDATE DETECTION
# ============================================================================



# ============================================================================
# Use BCGProbabilisticClassifier from ml_models.uq_classifier
# ============================================================================


# ============================================================================
# ENHANCED DATASET AND TRAINER
# ============================================================================



class ProbabilisticTrainer(CandidateBasedTrainer):
    """Enhanced trainer for probabilistic models with UQ."""
    
    def __init__(self, model, device='cpu', feature_scaler=None, use_uq=False):
        super().__init__(model, device, feature_scaler)
        self.use_uq = use_uq
    
    def evaluate_step(self, batch, criterion):
        """Evaluation step for probabilistic model with ranking loss."""
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
            candidate_logits = self.model(features).squeeze(-1)
            
            # Use ranking loss for validation (same as training)
            total_loss = 0
            correct_predictions = 0
            loss_count = 0
            
            for sample_idx in range(batch_size):
                sample_mask = sample_indices == sample_idx
                sample_logits = candidate_logits[sample_mask]
                sample_targets = targets[sample_mask]
                
                if len(sample_logits) < 2:
                    continue
                
                target_idx = torch.argmax(sample_targets).item()
                target_logit = sample_logits[target_idx]
                
                # Compute validation loss using same ranking approach
                other_indices = [i for i in range(len(sample_logits)) if i != target_idx]
                if len(other_indices) > 0:
                    # Use first few negatives for validation loss
                    n_negatives = min(len(other_indices), 3)
                    for neg_idx in other_indices[:n_negatives]:
                        neg_logit = sample_logits[neg_idx]
                        ranking_target = torch.ones(1).to(self.device)
                        loss = criterion(target_logit.unsqueeze(0), neg_logit.unsqueeze(0), ranking_target)
                        total_loss += loss
                        loss_count += 1
                
                predicted_idx = torch.argmax(sample_logits).item()
                if predicted_idx == target_idx:
                    correct_predictions += 1
            
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            accuracy = correct_predictions / batch_size if batch_size > 0 else 0
            
            return avg_loss.item(), accuracy
    
    def train_step(self, batch, optimizer, criterion):
        """Training step for probabilistic model with ranking loss for better probability calibration."""
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
        
        # Forward pass: get raw logits for all candidates
        candidate_logits = self.model(features).squeeze(-1)  # Shape: (total_candidates,)
        
        # Use ranking loss for better probability calibration
        total_loss = 0
        correct_predictions = 0
        
        for sample_idx in range(batch_size):
            # Get candidates for this sample
            sample_mask = sample_indices == sample_idx
            sample_logits = candidate_logits[sample_mask]
            sample_targets = targets[sample_mask]
            
            if len(sample_logits) < 2:  # Need at least 2 candidates for ranking
                continue
            
            # Find target candidate (the one with label 1)
            target_idx = torch.argmax(sample_targets).item()
            target_logit = sample_logits[target_idx]
            
            # Create ranking pairs: target should be higher than all others
            other_indices = [i for i in range(len(sample_logits)) if i != target_idx]
            
            if len(other_indices) > 0:
                # Randomly sample a few negative examples to avoid too many pairs
                n_negatives = min(len(other_indices), 5)  # Limit to 5 negatives per sample
                negative_indices = torch.randperm(len(other_indices))[:n_negatives]
                selected_negatives = [other_indices[i] for i in negative_indices]
                
                for neg_idx in selected_negatives:
                    neg_logit = sample_logits[neg_idx]
                    # MarginRankingLoss: input1 should be ranked higher than input2
                    ranking_target = torch.ones(1).to(self.device)  # target_logit > neg_logit
                    loss = criterion(target_logit.unsqueeze(0), neg_logit.unsqueeze(0), ranking_target)
                    total_loss += loss
            
            # Check if prediction is correct (highest logit matches target)
            predicted_idx = torch.argmax(sample_logits).item()
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


def train_enhanced_classifier(train_dataset, val_dataset, args, collate_fn=None):
    """Train the enhanced BCG classifier with multi-scale and UQ options."""
    print("Setting up enhanced candidate-based training...")
    
    # Use provided collate function or default
    if collate_fn is None:
        collate_fn = collate_candidate_samples
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
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
        # Use ranking loss with smaller margin for better probability calibration
        criterion = nn.MarginRankingLoss(margin=0.5)
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
    if args.use_uq:
        print(f"Uncertainty quantification: threshold={args.detection_threshold}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    
    if args.use_bcg_data:
        # Use new BCG dataset
        print(f"Using BCG dataset: {args.bcg_arcmin_type}")
        if args.z_range:
            print(f"Redshift filter: {args.z_range}")
        if args.delta_mstar_z_range:
            print(f"Delta M* z filter: {args.delta_mstar_z_range}")
        
        # Create train and test datasets using the new BCG data reader
        # RedMapper probabilities are loaded in the dataset but NOT used as input features
        # They can be accessed for training supervision, evaluation, or loss weighting
        train_dataset, test_dataset = create_bcg_datasets(
            dataset_type=args.bcg_arcmin_type,
            split_ratio=0.8,  # 80% train, 20% test
            z_range=args.z_range,
            delta_mstar_z_range=args.delta_mstar_z_range,
            include_additional_features=args.use_additional_features,
            include_redmapper_probs=False  # Don't use as input features to match testing
        )
        
        # Split training set into train/val (70% train, 10% val, 20% test total)
        train_size = int(0.875 * len(train_dataset))  # 0.875 * 0.8 = 0.7 total
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        test_subset = test_dataset
        
        print(f"BCG Dataset split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
        
    else:
        # Use original dataset
        dataframe = prepare_dataframe(args.image_dir, args.truth_table, args.dataset_type)
        print(f"Found {len(dataframe)} samples in dataset")
        
        # Create BCG dataset
        dataset = BCGDataset(args.image_dir, dataframe)
        
        # Split into train/val/test
        train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
        print(f"Dataset split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
    
    # Create candidate datasets based on data type
    print("\nCreating candidate-based datasets...")
    
    if args.use_bcg_data and args.use_desprior_candidates:
        # Use DESprior candidates for BCG data
        print("Using DESprior candidates...")
        
        # Create full datasets with DESprior candidates 
        full_train_candidate_dataset = create_desprior_candidate_dataset_from_files(
            dataset_type=args.bcg_arcmin_type,
            z_range=args.z_range,
            delta_mstar_z_range=args.delta_mstar_z_range,
            candidate_delta_mstar_range=args.candidate_delta_mstar_range,
            filter_inside_image=args.filter_inside_image
        )
        
        # Split the DESprior dataset to match our train/val split ratios
        total_samples = len(full_train_candidate_dataset)
        train_size = int(0.875 * total_samples)  # Match the BCG dataset split
        val_size = total_samples - train_size
        
        train_candidate_dataset, val_candidate_dataset = torch.utils.data.random_split(
            full_train_candidate_dataset, [train_size, val_size]
        )
        
        # Use DESprior-specific collate function
        collate_fn = collate_bcg_candidate_samples
        
    elif args.use_bcg_data:
        # Use automatic candidates with BCG data
        print("Using automatic candidates with BCG data...")
        
        candidate_params = {
            'min_distance': args.min_distance,
            'threshold_rel': args.threshold_rel,
            'exclude_border': args.exclude_border,
            'max_candidates': args.max_candidates
        }
        
        train_candidate_dataset = create_bcg_candidate_dataset_from_loader(
            train_subset, candidate_params, candidate_type='automatic'
        )
        val_candidate_dataset = create_bcg_candidate_dataset_from_loader(
            val_subset, candidate_params, candidate_type='automatic'
        )
        
        # Use BCG-specific collate function
        collate_fn = collate_bcg_candidate_samples
        
    else:
        # Use original candidate system
        print("Using original candidate system...")
        
        # Extract images and coordinates for candidate processing
        train_images, train_coords = extract_images_and_coords(train_subset)
        val_images, val_coords = extract_images_and_coords(val_subset)
        
        candidate_params = {
            'min_distance': args.min_distance,
            'threshold_rel': args.threshold_rel,
            'exclude_border': args.exclude_border,
            'max_candidates': args.max_candidates
        }
        
        
        train_candidate_dataset = BCGCandidateDataset(
            train_images,
            train_coords,
            candidate_params,
            min_candidates=3
        )
        
        val_candidate_dataset = BCGCandidateDataset(
            val_images, 
            val_coords,
            candidate_params,
            min_candidates=3
        )
        
        # Use original collate function
        collate_fn = collate_candidate_samples
    
    if len(train_candidate_dataset) == 0 or len(val_candidate_dataset) == 0:
        print("Error: No valid candidate samples found. Try adjusting candidate parameters.")
        return
    
    print(f"Candidate datasets: Train={len(train_candidate_dataset)}, Val={len(val_candidate_dataset)}")
    
    # Train model
    print("\nStarting enhanced training...")
    model, scaler, history = train_enhanced_classifier(
        train_candidate_dataset, 
        val_candidate_dataset, 
        args,
        collate_fn=collate_fn
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
                       choices=['SPT3G_1500d', 'megadeep500', 'bcg_2p2arcmin', 'bcg_3p8arcmin'],
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
    parser.add_argument('--min_distance', type=int, default=8,
                       help='Minimum distance between candidates (reduced for higher precision)')
    parser.add_argument('--threshold_rel', type=float, default=0.1,
                       help='Relative threshold for candidate detection (lowered for more candidates)')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=50,
                       help='Maximum candidates per image (increased for better coverage)')
    
    
    # NEW: Uncertainty quantification arguments
    parser.add_argument('--use_uq', action='store_true',
                       help='Enable uncertainty quantification with probabilistic outputs')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Probability threshold for BCG detection (0.0-1.0)')
    
    # NEW: BCG dataset specific arguments
    parser.add_argument('--use_bcg_data', action='store_true',
                       help='Use new BCG dataset (2.2 or 3.8 arcmin)')
    parser.add_argument('--bcg_arcmin_type', type=str, default='2p2arcmin',
                       choices=['2p2arcmin', '3p8arcmin'],
                       help='BCG image scale (2.2 or 3.8 arcmin)')
    parser.add_argument('--z_range', type=str, default=None,
                       help='Redshift filter range as "z_min,z_max" (e.g. "0.3,0.7")')
    parser.add_argument('--delta_mstar_z_range', type=str, default=None,
                       help='Delta M* z filter range as "min,max" (e.g. "-2.0,-1.0")')
    parser.add_argument('--use_additional_features', action='store_true',
                       help='Include redshift and delta_mstar_z as additional features')
    parser.add_argument('--use_redmapper_probs', action='store_true',
                       help='Load RedMapper BCG probabilities for training supervision (not as input features)')
    
    # NEW: DESprior candidate arguments
    parser.add_argument('--use_desprior_candidates', action='store_true',
                       help='Use DESprior candidates instead of automatic detection')
    parser.add_argument('--candidate_delta_mstar_range', type=str, default=None,
                       help='Filter DESprior candidates by delta_mstar range as "min,max"')
    parser.add_argument('--filter_inside_image', action='store_true', default=True,
                       help='Filter out DESprior candidates outside image bounds')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training curves')
    
    args = parser.parse_args()
    
    
    # Validate detection threshold
    if args.use_uq:
        args.detection_threshold = max(0.0, min(1.0, args.detection_threshold))
    
    # Parse BCG filtering ranges
    if args.z_range:
        args.z_range = tuple(float(x.strip()) for x in args.z_range.split(','))
        if len(args.z_range) != 2:
            raise ValueError("z_range must be two values: 'min,max'")
    
    if args.delta_mstar_z_range:
        args.delta_mstar_z_range = tuple(float(x.strip()) for x in args.delta_mstar_z_range.split(','))
        if len(args.delta_mstar_z_range) != 2:
            raise ValueError("delta_mstar_z_range must be two values: 'min,max'")
    
    if args.candidate_delta_mstar_range:
        args.candidate_delta_mstar_range = tuple(float(x.strip()) for x in args.candidate_delta_mstar_range.split(','))
        if len(args.candidate_delta_mstar_range) != 2:
            raise ValueError("candidate_delta_mstar_range must be two values: 'min,max'")
    
    # Auto-set BCG data mode if BCG dataset types are selected
    if args.dataset_type in ['bcg_2p2arcmin', 'bcg_3p8arcmin']:
        args.use_bcg_data = True
        args.bcg_arcmin_type = args.dataset_type.replace('bcg_', '')
    
    main(args)