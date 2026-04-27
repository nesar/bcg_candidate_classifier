#!/usr/bin/env python3
"""
Test script to verify RedMapper probability integration works correctly.

This script tests:
1. Loading RedMapper probabilities from BCG dataset 
2. Passing them through to candidate dataset
3. Training with weighted loss
4. Ensuring feature dimensions remain consistent
"""

import numpy as np
import torch
from data.data_read_bcgs import create_bcg_datasets
from data.candidate_dataset_bcgs import create_bcg_candidate_dataset_from_loader, collate_bcg_candidate_samples
from train import ProbabilisticTrainer
from ml_models.uq_classifier import BCGProbabilisticClassifier
import torch.nn as nn
from torch.utils.data import DataLoader

def test_redmapper_integration():
    """Test that RedMapper integration works without breaking existing functionality."""
    print("Testing RedMapper probability integration...")
    
    # Test 1: Load dataset without RedMapper probabilities (existing functionality)
    print("\n1. Testing without RedMapper probabilities...")
    try:
        train_dataset, test_dataset = create_bcg_datasets(
            dataset_type='2p2arcmin',
            split_ratio=0.8,
            include_additional_features=True,
            include_redmapper_probs=False  # Standard approach
        )
        print(f"✓ Dataset loaded successfully: {len(train_dataset)} train, {len(test_dataset)} test")
        
        # Check a sample
        sample = train_dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        
        if 'additional_features' in sample:
            print(f"✓ Additional features shape: {sample['additional_features'].shape}")
        else:
            print("✓ No additional features (expected)")
            
        if 'bcg_probability' in sample:
            print(f"✓ BCG probability available: {sample['bcg_probability']}")
        else:
            print("✓ No BCG probability (expected)")
            
    except Exception as e:
        print(f"✗ Failed to load dataset without RedMapper probs: {e}")
        return False
    
    # Test 2: Load dataset with RedMapper probabilities  
    print("\n2. Testing with RedMapper probabilities...")
    try:
        train_dataset_rm, test_dataset_rm = create_bcg_datasets(
            dataset_type='2p2arcmin',
            split_ratio=0.8,
            include_additional_features=True,
            include_redmapper_probs=True  # New approach
        )
        print(f"✓ Dataset with RedMapper loaded: {len(train_dataset_rm)} train, {len(test_dataset_rm)} test")
        
        # Check a sample
        sample_rm = train_dataset_rm[0]
        print(f"✓ Sample keys: {list(sample_rm.keys())}")
        
        # Verify additional features are the same size
        if 'additional_features' in sample and 'additional_features' in sample_rm:
            feat_shape = sample['additional_features'].shape
            feat_rm_shape = sample_rm['additional_features'].shape
            if feat_shape == feat_rm_shape:
                print(f"✓ Additional features shape consistent: {feat_shape} == {feat_rm_shape}")
            else:
                print(f"✗ Additional features shape mismatch: {feat_shape} != {feat_rm_shape}")
                return False
        
        if 'bcg_probability' in sample_rm:
            print(f"✓ BCG probability available: {sample_rm['bcg_probability']}")
        else:
            print("✗ BCG probability missing when requested")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load dataset with RedMapper probs: {e}")
        return False
    
    # Test 3: Create candidate datasets and verify compatibility
    print("\n3. Testing candidate dataset creation...")
    try:
        # Test subset for speed
        train_subset = torch.utils.data.Subset(train_dataset_rm, range(min(10, len(train_dataset_rm))))
        
        candidate_dataset = create_bcg_candidate_dataset_from_loader(
            train_subset,
            candidate_params={'min_distance': 8, 'threshold_rel': 0.2, 'exclude_border': 0, 'max_candidates': 10}
        )
        print(f"✓ Candidate dataset created: {len(candidate_dataset)} samples")
        
        # Check if candidate dataset has RedMapper probs
        if len(candidate_dataset) > 0:
            cand_sample = candidate_dataset[0]
            print(f"✓ Candidate sample keys: {list(cand_sample.keys())}")
            
            if 'redmapper_prob' in cand_sample:
                print(f"✓ RedMapper probability in candidate: {cand_sample['redmapper_prob']}")
            else:
                print("✗ RedMapper probability missing from candidate")
                return False
                
    except Exception as e:
        print(f"✗ Failed to create candidate dataset: {e}")
        return False
    
    # Test 4: Test collate function
    print("\n4. Testing collate function...")
    try:
        data_loader = DataLoader(
            candidate_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_bcg_candidate_samples
        )
        
        for batch in data_loader:
            print(f"✓ Batch keys: {list(batch.keys())}")
            print(f"✓ Batch size: {batch['batch_size']}")
            print(f"✓ Features shape: {batch['features'].shape}")
            print(f"✓ Targets shape: {batch['targets'].shape}")
            
            if 'redmapper_probs' in batch:
                print(f"✓ RedMapper probs shape: {batch['redmapper_probs'].shape}")
                print(f"✓ RedMapper probs values: {batch['redmapper_probs']}")
            else:
                print("✗ RedMapper probs missing from batch")
                return False
            break  # Just test first batch
            
    except Exception as e:
        print(f"✗ Failed to test collate function: {e}")
        return False
    
    # Test 5: Test trainer with RedMapper weighting
    print("\n5. Testing trainer with RedMapper weighting...")
    try:
        # Create a simple model
        feature_dim = batch['features'].shape[1]
        model = BCGProbabilisticClassifier(
            feature_dim=feature_dim,
            hidden_dims=[32, 16],
            dropout_rate=0.1
        )
        
        # Create trainer with RedMapper weighting
        trainer = ProbabilisticTrainer(
            model=model,
            device='cpu',
            feature_scaler=None,
            use_uq=True,
            use_redmapper_weighting=True
        )
        
        # Test training step
        criterion = nn.MarginRankingLoss(margin=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        loss, acc = trainer.train_step(batch, optimizer, criterion)
        print(f"✓ Training step successful: loss={loss:.4f}, acc={acc:.3f}")
        
    except Exception as e:
        print(f"✗ Failed to test trainer: {e}")
        return False
    
    print("\n✅ All tests passed! RedMapper integration is working correctly.")
    print("\nKey features verified:")
    print("- RedMapper probabilities load correctly from dataset")
    print("- Feature dimensions remain consistent (no input feature contamination)")
    print("- Candidate dataset passes RedMapper probabilities correctly")  
    print("- Collate function handles RedMapper probabilities")
    print("- Trainer can use RedMapper probabilities for weighted loss")
    print("- Backward compatibility maintained")
    
    return True

if __name__ == "__main__":
    success = test_redmapper_integration()
    if not success:
        exit(1)
    print("\n🎉 RedMapper probability integration is ready to use!")