#!/usr/bin/env python3
"""
Test script for DESprior candidate system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.candidate_dataset_bcgs import create_desprior_candidate_dataset_from_files, create_bcg_candidate_dataset_from_loader
from data.data_read_bcgs import create_bcg_datasets


def test_basic_functionality():
    """Test basic DESprior candidate dataset creation"""
    print("=" * 60)
    print("Testing DESprior Candidate Dataset Creation")
    print("=" * 60)
    
    try:
        # Test DESprior candidate dataset creation with heavy filtering for speed
        print("\n1. Testing DESprior candidate dataset creation...")
        desprior_dataset = create_desprior_candidate_dataset_from_files(
            dataset_type='2p2arcmin',
            z_range=(0.5, 0.6),  # Narrow range for faster testing
            filter_inside_image=True
        )
        
        print(f"   ✓ Successfully created DESprior dataset with {len(desprior_dataset)} samples")
        
        # Test accessing a sample
        if len(desprior_dataset) > 0:
            print("\n2. Testing sample access...")
            sample = desprior_dataset[0]
            print(f"   ✓ Sample keys: {list(sample.keys())}")
            print(f"   ✓ Features shape: {sample['features'].shape}")
            print(f"   ✓ Target: {sample['target']}")
            print(f"   ✓ Number of candidates: {len(sample['candidates'])}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filtering():
    """Test filtering functionality"""
    print("\n" + "=" * 60)
    print("Testing Filtering Functionality")
    print("=" * 60)
    
    try:
        # Test with redshift filtering
        print("\n1. Testing with redshift filtering...")
        dataset_z_filter = create_desprior_candidate_dataset_from_files(
            dataset_type='2p2arcmin',
            z_range=(0.3, 0.5),  # Narrow range for speed
            filter_inside_image=True
        )
        print(f"   ✓ Z filtering [0.3, 0.5]: {len(dataset_z_filter)} samples")
        
        # Test with candidate delta_mstar filtering
        print("\n2. Testing with candidate delta_mstar filtering...")
        dataset_candidate_filter = create_desprior_candidate_dataset_from_files(
            dataset_type='2p2arcmin',
            z_range=(0.3, 0.7),  # Keep reasonable range
            candidate_delta_mstar_range=(-2.0, -0.5),
            filter_inside_image=True
        )
        print(f"   ✓ Candidate delta_mstar filtering [-2.0, -0.5]: {len(dataset_candidate_filter)} samples")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    import torch
    
    print("Testing DESprior Candidate System")
    print("=" * 80)
    
    tests = [
        test_basic_functionality,
        test_filtering,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)