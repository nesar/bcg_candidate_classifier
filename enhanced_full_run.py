#!/usr/bin/env python3
"""
Enhanced BCG Candidate Classifier - Full Run Demo

This script extends the working old_full_run.py with two key enhancements:
1. Multi-scale inference for flexible candidate square sizes
2. Uncertainty quantification (UQ) with probability thresholds for detections

The script maintains all existing functionality while adding options for these improvements.
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
    """Main function demonstrating the enhanced BCG workflow."""
    print("="*80)
    print("ENHANCED BCG CANDIDATE CLASSIFIER - FULL WORKFLOW")
    print("="*80)
    print("This script extends the working implementation with:")
    print("1. Multi-scale inference for flexible candidate square sizes")
    print("2. Uncertainty quantification with probability thresholds")
    print("="*80)
    
    # Dataset selection
    print("\nDataset options:")
    print("1. SPT3G_1500d (default)")
    print("2. megadeep500")
    
    dataset_choice = input("\nSelect dataset (1 or 2): ").strip()
    
    if dataset_choice == "2":
        DATASET_TYPE = "megadeep500"
        IMAGE_DIR = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/pruned_megadeep500/'
        TRUTH_TABLE = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/ML/truth_table.csv'
        print(f"\nSelected: {DATASET_TYPE} dataset")
    else:
        DATASET_TYPE = "SPT3G_1500d"
        IMAGE_DIR = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/SPT3G_1500d_data/1-5-7mix/'
        TRUTH_TABLE = '/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/ryanwalker/ML/1-5-7mix_cart.csv'
        print(f"\nSelected: {DATASET_TYPE} dataset")
    
    # Check if real data exists
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(TRUTH_TABLE):
        print("\nWarning: Default data paths do not exist.")
        print(f"Please update IMAGE_DIR and TRUTH_TABLE paths")
        print("\nExpected paths:")
        print(f"  IMAGE_DIR: {IMAGE_DIR}")
        print(f"  TRUTH_TABLE: {TRUTH_TABLE}")
        
        use_real_data = input("\nContinue with these paths anyway? (y/N): ").strip().lower()
        if use_real_data not in ['y', 'yes']:
            print("Please update the data paths and run again.")
            return
    
    # ENHANCEMENT 1: Multi-scale inference option
    print("\n" + "="*60)
    print("ENHANCEMENT 1: MULTI-SCALE CANDIDATE DETECTION")
    print("="*60)
    print("Multi-scale detection finds candidates at different sizes to handle")
    print("large bright objects that span multiple traditional candidate regions.")
    print("This addresses the issue where a single large object appears as multiple candidates.")
    
    use_multiscale = input("\nEnable multi-scale inference? (Y/n): ").strip().lower()
    use_multiscale = use_multiscale not in ['n', 'no']
    
    # Multi-scale parameters
    if use_multiscale:
        print("\nMulti-scale parameters:")
        modify_multiscale = input("Modify multi-scale parameters? (y/N): ").strip().lower()
        
        if modify_multiscale in ['y', 'yes']:
            scales_input = input("Scale factors [0.5,1.0,1.5] (comma-separated): ").strip()
            if scales_input:
                scales = [float(s.strip()) for s in scales_input.split(',')]
            else:
                scales = [0.5, 1.0, 1.5]
            
            max_per_scale = int(input("Max candidates per scale (default 10): ") or "10")
        else:
            scales = [0.5, 1.0, 1.5]
            max_per_scale = 10
        
        print(f"Using scales: {scales}, max per scale: {max_per_scale}")
    else:
        scales = [1.0]  # Single scale (traditional approach)
        max_per_scale = 25
        print("Using traditional single-scale detection")
    
    # ENHANCEMENT 2: Uncertainty Quantification option
    print("\n" + "="*60)
    print("ENHANCEMENT 2: UNCERTAINTY QUANTIFICATION")
    print("="*60)
    print("Uncertainty quantification provides calibrated probabilities for each")
    print("candidate being a BCG. Over a threshold = 'detection'.")
    print("This addresses the need for probabilistic outputs and confidence estimates.")
    
    use_uq = input("\nEnable uncertainty quantification? (Y/n): ").strip().lower()
    use_uq = use_uq not in ['n', 'no']
    
    # UQ parameters
    detection_threshold = 0.5
    if use_uq:
        print("\nUncertainty quantification parameters:")
        modify_uq = input("Modify UQ parameters? (y/N): ").strip().lower()
        
        if modify_uq in ['y', 'yes']:
            detection_threshold = float(input("Detection threshold (0.0-1.0, default 0.5): ") or "0.5")
            detection_threshold = max(0.0, min(1.0, detection_threshold))
        
        print(f"Using detection threshold: {detection_threshold}")
        print("Above threshold = BCG detection, below = insufficient confidence")
    else:
        print("Using traditional deterministic classification")
    
    # Traditional candidate detection parameters
    print("\n" + "="*60)
    print("TRADITIONAL CANDIDATE DETECTION PARAMETERS")
    print("="*60)
    print("These control the base candidate finding algorithm")
    
    modify_params = input("Modify candidate detection parameters? (y/N): ").strip().lower()
    
    if modify_params in ['y', 'yes']:
        print("\nCandidate detection parameters:")
        min_distance = int(input("Minimum distance between candidates (default 15): ") or "15")
        threshold_rel = float(input("Relative brightness threshold (default 0.12): ") or "0.12")
        exclude_border = int(input("Exclude border pixels (default 30): ") or "30")
        if not use_multiscale:
            max_candidates = int(input("Maximum candidates per image (default 25): ") or "25")
        else:
            max_candidates = max_per_scale  # Use multiscale parameter
    else:
        min_distance = 15
        threshold_rel = 0.12
        exclude_border = 30
        max_candidates = max_per_scale if use_multiscale else 25
        print(f"Using defaults: min_distance={min_distance}, threshold_rel={threshold_rel}")
    
    # Training parameters
    print("\n" + "="*60)
    print("TRAINING PARAMETERS")
    print("="*60)
    modify_training = input("Modify training parameters? (y/N): ").strip().lower()
    
    if modify_training in ['y', 'yes']:
        epochs = int(input("Number of epochs (default 50): ") or "50")
        batch_size = int(input("Batch size (default 16): ") or "16")
        lr = float(input("Learning rate (default 0.001): ") or "0.001")
    else:
        epochs = 50
        batch_size = 16
        lr = 0.001
        print(f"Using defaults: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Choose which implementation to use
    print("\n" + "="*60)
    print("IMPLEMENTATION CHOICE")
    print("="*60)
    
    if use_multiscale or use_uq:
        print("Enhanced features requested - using enhanced scripts")
        use_enhanced = True
        train_script = "train.py"
        test_script = "test.py"
    else:
        print("No enhanced features requested - using proven working scripts")
        choice = input("Use enhanced scripts anyway for consistency? (y/N): ").strip().lower()
        use_enhanced = choice in ['y', 'yes']
        if use_enhanced:
            train_script = "train.py"
            test_script = "test.py"
        else:
            train_script = "old_train.py"
            test_script = "old_test.py"
    
    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "candidate_classifier"
    if use_multiscale:
        experiment_name += "_multiscale"
    if use_uq:
        experiment_name += "_uq"
    
    output_dir = f"./trained_models/{experiment_name}_run_{timestamp}"
    print(f"\nExperiment output directory: {output_dir}")
    
    # Build training command
    train_command = f"""python {train_script} \\
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
    
    # Add enhanced feature flags if using enhanced scripts
    if use_enhanced:
        if use_multiscale:
            scales_str = ",".join(map(str, scales))
            train_command += f" --use_multiscale --scales {scales_str} --max_candidates_per_scale {max_per_scale}"
        
        if use_uq:
            train_command += f" --use_uq --detection_threshold {detection_threshold}"
    
    # Check for GPU
    gpu_available = input("\nUse GPU if available? (Y/n): ").strip().lower()
    if gpu_available not in ['n', 'no']:
        train_command += " --use_gpu"
    
    # Step 1: Training
    print("\n" + "="*80)
    print("STEP 1: TRAINING BCG CLASSIFIER")
    print("="*80)
    
    feature_summary = []
    if use_multiscale:
        feature_summary.append("Multi-scale candidate detection")
    if use_uq:
        feature_summary.append("Uncertainty quantification")
    
    if feature_summary:
        print("Enhanced features enabled:")
        for feature in feature_summary:
            print(f"  ✓ {feature}")
        print()
    else:
        print("Using traditional approach (proven working implementation)")
        print()
    
    if not run_command(train_command, f"Training BCG classifier with {train_script}"):
        print("Training failed. Stopping execution.")
        return
    
    # Step 2: Find best model
    if use_uq:
        model_name = "best_probabilistic_classifier"
    else:
        model_name = "best_candidate_classifier"
    
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    scaler_path = os.path.join(output_dir, f"{model_name}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Expected model files not found:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print("Checking for final model...")
        final_model_name = model_name.replace("best_", "final_")
        model_path = os.path.join(output_dir, f"{final_model_name}.pth")
        scaler_path = os.path.join(output_dir, f"{final_model_name}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("No trained models found. Training may have failed.")
        return
    
    # Step 3: Testing
    print("\n" + "="*80)
    print("STEP 2: TESTING BCG CLASSIFIER")
    print("="*80)
    
    test_output_dir = os.path.join(output_dir, "evaluation_results")
    
    test_command = f"""python {test_script} \\
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
    
    # Add enhanced feature flags to test command if using enhanced scripts
    if use_enhanced:
        if use_multiscale:
            scales_str = ",".join(map(str, scales))
            test_command += f" --use_multiscale --scales {scales_str} --max_candidates_per_scale {max_per_scale}"
        
        if use_uq:
            test_command += f" --use_uq --detection_threshold {detection_threshold}"
    
    if not run_command(test_command, f"Testing BCG classifier with {test_script}"):
        print("Testing failed.")
        return
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("ENHANCED WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nExperiment results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  Training curves: {output_dir}/training_curves.png")
    print(f"  Best model: {model_path}")
    print(f"  Feature scaler: {scaler_path}")
    print(f"  Evaluation results: {test_output_dir}/")
    print(f"  Sample predictions: {test_output_dir}/*_prediction_sample_*.png")
    print(f"  Failure cases: {test_output_dir}/*_failure_sample_*.png")
    print(f"  Detailed results: {test_output_dir}/evaluation_results.csv")
    
    if use_uq:
        print(f"  Probability analysis: {test_output_dir}/probability_analysis.csv")
        print(f"  Uncertainty plots: {test_output_dir}/probability_analysis.png")
    
    print(f"\nApproach: Enhanced candidate-based BCG classification")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Scripts used: {train_script}, {test_script}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Candidate detection: min_distance={min_distance}, threshold_rel={threshold_rel}")
    
    if use_multiscale:
        print(f"Multi-scale: scales={scales}, max_per_scale={max_per_scale}")
    
    if use_uq:
        print(f"Uncertainty quantification: threshold={detection_threshold}")
    
    print(f"\nEnhancements implemented:")
    if use_multiscale:
        print("1. ✓ Multi-scale candidate detection:")
        print("   - Flexible candidate square sizes")
        print("   - Handles large objects spanning multiple regions")
        print("   - Adaptive patch sizes for feature extraction")
    else:
        print("1. ✗ Multi-scale detection (using traditional single-scale)")
    
    if use_uq:
        print("2. ✓ Uncertainty quantification:")
        print("   - Probabilistic outputs (0-1 scale)")
        print("   - Detection threshold for confident predictions") 
        print("   - Uncertainty estimates for risk assessment")
    else:
        print("2. ✗ Uncertainty quantification (using deterministic scores)")
    
    print(f"\nTo re-run evaluation with different parameters:")
    test_cmd_simple = f"python {test_script} --model_path '{model_path}' --scaler_path '{scaler_path}'"
    if use_enhanced and use_multiscale:
        test_cmd_simple += " --use_multiscale"
    if use_enhanced and use_uq:
        test_cmd_simple += " --use_uq"
    print(f"{test_cmd_simple} [other args]")
    
    print("\nWorkflow demonstration completed successfully!")
    print("Both requested enhancements have been implemented while preserving")
    print("all existing functionality and maintaining backward compatibility.")


if __name__ == "__main__":
    main()