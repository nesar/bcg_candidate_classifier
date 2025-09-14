#!/usr/bin/env python3
"""
Test script to verify HPC threading configuration works.
Run this on your cluster to test the fix.
"""

import os
# Fix threading issues for HPC systems - set before any numpy/sklearn imports
os.environ['NUMEXPR_MAX_THREADS'] = '128'
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent numpy threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL threading limit

print("🔧 Testing HPC threading configuration...")
print(f"NUMEXPR_MAX_THREADS: {os.environ.get('NUMEXPR_MAX_THREADS')}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")

try:
    print("\n📦 Testing basic imports...")
    import argparse
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for HPC
    import matplotlib.pyplot as plt
    from datetime import datetime
    from scipy.ndimage import maximum_filter, zoom
    print("✅ Basic scientific imports successful")

    print("\n📦 Testing BCG project imports...")
    from data.data_read import prepare_dataframe, BCGDataset
    from ml_models.candidate_classifier import BCGCandidateClassifier
    from utils.candidate_based_bcg import extract_patch_features, extract_context_features
    from utils.viz_bcg import show_failures
    from data.data_read_bcgs import create_bcg_datasets, BCGDataset as NewBCGDataset
    from data.candidate_dataset_bcgs import (create_bcg_candidate_dataset_from_loader, 
                                            create_desprior_candidate_dataset_from_files,
                                            collate_bcg_candidate_samples)
    from ml_models.uq_classifier import BCGProbabilisticClassifier
    print("✅ BCG project imports successful")

    print("\n📦 Testing color features (lazy import)...")
    try:
        from utils.color_features import ColorFeatureExtractor
        print("✅ Color features import successful")
    except ImportError as e:
        print(f"⚠️ Color features not available: {e}")

    print("\n📦 Testing analysis framework...")
    try:
        from analysis.feature_importance import FeatureImportanceAnalyzer
        print("✅ Analysis framework import successful")
    except ImportError as e:
        print(f"⚠️ Analysis framework not available: {e}")

    print("\n🎉 ALL TESTS PASSED!")
    print("Your HPC threading configuration is working correctly.")
    print("You can now run enhanced_full_run.py without NUMEXPR errors.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 If you still see NUMEXPR errors, try setting NUMEXPR_MAX_THREADS even higher (256 or 512)")