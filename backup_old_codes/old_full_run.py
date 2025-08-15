#!/usr/bin/env python3
"""
BCG Candidate Classifier - Full Run Demo

This script demonstrates the complete workflow for training and testing 
candidate-based BCG classifiers on astronomical images.

Features:
- Candidate-based classification approach (no coordinate regression)
- Feature extraction from bright spots (local maxima)
- Neural network ranking/classification of candidates
- Comprehensive evaluation and visualization
"""

import os
import subprocess
import sys
from datetime import datetime


def run_command(command, description):
    """Run a subprocess command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    """Main function demonstrating the complete candidate-based BCG workflow."""
    print("="*80)
    print("BCG CANDIDATE CLASSIFIER - FULL WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Dataset selection
    print("\nDataset options:")
    print("1. SPT3G_1500d (default)")
    print("2. megadeep500")
    
    dataset_choice = input("\nSelect dataset (1 or 2): ").strip()
    
    if dataset_choice == "1":
        DATASET_TYPE = "SPT3G_1500d"
        IMAGE_DIR = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/SPT3G_1500d_data/1-5-7mix/'
        TRUTH_TABLE = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/ML/1-5-7mix_cart.csv'
        print(f"\nSelected: {DATASET_TYPE} dataset")
    else:
        DATASET_TYPE = "megadeep500"
        IMAGE_DIR = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/pruned_megadeep500/'
        TRUTH_TABLE = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/ML/truth_table.csv'
        print(f"\nSelected: {DATASET_TYPE} dataset")
    
    # Check if real data exists
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(TRUTH_TABLE):
        print("\nWarning: Default data paths do not exist.")
        print(f"Please update IMAGE_DIR and TRUTH_TABLE in {__file__}")
        print("\nExpected paths:")
        print(f"  IMAGE_DIR: {IMAGE_DIR}")
        print(f"  TRUTH_TABLE: {TRUTH_TABLE}")
        print("\nFor this demo, you can:")
        print("1. Update the paths above to your data location")
        print("2. Use demo mode with synthetic data (if implemented)")
        
        use_real_data = input("\nContinue with these paths anyway? (y/N): ").strip().lower()
        if use_real_data not in ['y', 'yes']:
            print("Please update the data paths and run again.")
            return
    
    # Candidate detection parameters
    print("\nCandidate Detection Parameters:")
    print("These parameters control how bright spots (candidates) are found in images")
    print("Current defaults work well for most astronomical images")
    
    modify_params = input("Modify candidate detection parameters? (y/N): ").strip().lower()
    
    if modify_params in ['y', 'yes']:
        print("\nCandidate detection parameters:")
        min_distance = int(input("Minimum distance between candidates (default 15): ") or "15")
        threshold_rel = float(input("Relative brightness threshold (default 0.12): ") or "0.12")
        exclude_border = int(input("Exclude border pixels (default 30): ") or "30")
        max_candidates = int(input("Maximum candidates per image (default 25): ") or "25")
    else:
        min_distance = 15
        threshold_rel = 0.12
        exclude_border = 30
        max_candidates = 25
        print(f"Using default parameters: min_distance={min_distance}, threshold_rel={threshold_rel}")
    
    # Training parameters
    print("\nTraining Parameters:")
    modify_training = input("Modify training parameters? (y/N): ").strip().lower()
    
    if modify_training in ['y', 'yes']:
        epochs = int(input("Number of epochs (default 50): ") or "50")
        batch_size = int(input("Batch size (default 16): ") or "16")
        lr = float(input("Learning rate (default 0.001): ") or "0.001")
    else:
        epochs = 50
        batch_size = 16
        lr = 0.001
        print(f"Using default training parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./trained_models/candidate_classifier_run_{timestamp}"
    
    print(f"\nExperiment output directory: {output_dir}")
    
    # Build training command
    train_command = f"""python old_train.py \\
        --image_dir "{IMAGE_DIR}" \\
        --truth_table "{TRUTH_TABLE}" \\
        --dataset_type {DATASET_TYPE} \\
        --epochs {epochs} \\
        --batch_size {batch_size} \\
        --lr {lr} \\
        --min_distance {min_distance} \\
        --threshold_rel {threshold_rel} \\
        --exclude_border {exclude_border} \\
        --max_candidates {max_candidates} \\
        --output_dir "{output_dir}" \\
        --plot"""
    
    # Check for GPU
    gpu_available = input("\nUse GPU if available? (Y/n): ").strip().lower()
    if gpu_available not in ['n', 'no']:
        train_command += " --use_gpu"
    
    # Step 1: Training
    print("\n" + "="*80)
    print("STEP 1: TRAINING CANDIDATE-BASED BCG CLASSIFIER")
    print("="*80)
    
    if not run_command(train_command, "Training candidate-based BCG classifier"):
        print("Training failed. Stopping execution.")
        return
    
    # Step 2: Find best model
    model_path = os.path.join(output_dir, "best_candidate_classifier.pth")
    scaler_path = os.path.join(output_dir, "best_candidate_classifier_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Expected model files not found:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print("Using final model instead...")
        model_path = os.path.join(output_dir, "final_candidate_classifier.pth")
        scaler_path = os.path.join(output_dir, "final_candidate_classifier_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("No trained models found. Training may have failed.")
        return
    
    # Step 3: Testing
    print("\n" + "="*80)
    print("STEP 2: TESTING CANDIDATE-BASED BCG CLASSIFIER")
    print("="*80)
    
    test_output_dir = os.path.join(output_dir, "evaluation_results")
    
    test_command = f"""python old_test.py \\
        --model_path "{model_path}" \\
        --scaler_path "{scaler_path}" \\
        --image_dir "{IMAGE_DIR}" \\
        --truth_table "{TRUTH_TABLE}" \\
        --dataset_type {DATASET_TYPE} \\
        --min_distance {min_distance} \\
        --threshold_rel {threshold_rel} \\
        --exclude_border {exclude_border} \\
        --max_candidates {max_candidates} \\
        --show_samples 5 \\
        --show_failures 3 \\
        --output_dir "{test_output_dir}" \\
        --save_results"""
    
    if not run_command(test_command, "Testing candidate-based BCG classifier"):
        print("Testing failed.")
        return
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nExperiment results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  Training curves: {output_dir}/training_curves.png")
    print(f"  Best model: {model_path}")
    print(f"  Feature scaler: {scaler_path}")
    print(f"  Evaluation results: {test_output_dir}/")
    print(f"  Sample predictions: {test_output_dir}/CandidateBasedTesting_prediction_sample_*.png")
    print(f"  Failure cases: {test_output_dir}/CandidateBasedTesting_failure_sample_*.png")
    print(f"  Detailed results: {test_output_dir}/evaluation_results.csv")
    
    print(f"\nApproach: Candidate-based BCG classification")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Candidate detection: min_distance={min_distance}, threshold_rel={threshold_rel}")
    
    print("\nCandidate-based approach:")
    print("1. Finds bright spots (local maxima) in each image")
    print("2. Extracts 30+ dimensional features around each bright spot")
    print("3. Trains neural network to rank/classify candidates")
    print("4. Selects highest-scoring candidate as BCG prediction")
    
    print(f"\nTo re-run evaluation with different parameters:")
    print(f"python test.py --model_path '{model_path}' --scaler_path '{scaler_path}' [other args]")
    
    print("\nWorkflow demonstration completed successfully!")


if __name__ == "__main__":
    main()