#!/usr/bin/env python3
"""
Candidate-Based BCG Classifier Training Script

This script trains a neural network to classify/rank BCG candidates
extracted from astronomical images.
"""

import argparse
import os
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

from data.data_read import prepare_dataframe, BCGDataset
from data.candidate_dataset import BCGCandidateDataset, collate_candidate_samples, CandidateBasedTrainer
from ml_models.candidate_classifier import BCGCandidateClassifier


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


def train_candidate_classifier(train_dataset, val_dataset, args):
    """Train the candidate-based BCG classifier."""
    print("Setting up candidate-based training...")
    
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
    
    # Create model
    model = BCGCandidateClassifier(
        feature_dim=feature_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = CandidateBasedTrainer(model, device, feature_scaler)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
            save_model(model, feature_scaler, args.output_dir, 'best_candidate_classifier')
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.3f}")
    
    # Save final model
    save_model(model, feature_scaler, args.output_dir, 'final_candidate_classifier')
    
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
    print("CANDIDATE-BASED BCG CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.dataset_type}")
    print(f"Images: {args.image_dir}")
    print(f"Truth table: {args.truth_table}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {args.output_dir}")
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
    
    # Create candidate datasets
    print("\nCreating candidate-based datasets...")
    
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
    
    if len(train_candidate_dataset) == 0 or len(val_candidate_dataset) == 0:
        print("Error: No valid candidate samples found. Try adjusting candidate parameters.")
        return
    
    # Train model
    print("\nStarting training...")
    model, scaler, history = train_candidate_classifier(
        train_candidate_dataset, 
        val_candidate_dataset, 
        args
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.3f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Candidate-Based BCG Classifier")
    
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
    
    # Candidate finding arguments
    parser.add_argument('--min_distance', type=int, default=15,
                       help='Minimum distance between candidates')
    parser.add_argument('--threshold_rel', type=float, default=0.12,
                       help='Relative threshold for candidate detection')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=25,
                       help='Maximum candidates per image')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training curves')
    
    args = parser.parse_args()
    main(args)