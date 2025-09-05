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


def get_bcg_data_ranges(bcg_arcmin_type):
    """Get the actual min/max ranges from BCG data files."""
    try:
        import pandas as pd
        
        if bcg_arcmin_type == "2p2arcmin":
            csv_path = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_2p2arcmin_clean_matched.csv"
        else:  # 3p8arcmin
            csv_path = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_3p8arcmin_clean_matched.csv"
        
        if not os.path.exists(csv_path):
            # Fall back to non-clean version
            csv_path = csv_path.replace("_clean_matched.csv", "_with_coordinates.csv")
            
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            z_min, z_max = df['Cluster z'].min(), df['Cluster z'].max()
            delta_min, delta_max = df['delta_mstar_z'].min(), df['delta_mstar_z'].max()
            return f"{z_min:.2f},{z_max:.2f}", f"{delta_min:.2f},{delta_max:.2f}"
    except:
        pass
    
    # Default fallback ranges
    return "0.1,1.2", "-4.0,2.0"


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
    print("3. BCG 2.2 arcmin (new dataset)")
    print("4. BCG 3.8 arcmin (new dataset)")
    
    dataset_choice = input("\nSelect dataset (1-4): ").strip()
    
    # Initialize BCG-specific variables
    use_bcg_data = False
    bcg_arcmin_type = None
    z_range = None
    delta_mstar_z_range = None
    use_additional_features = False
    use_redmapper_probs = False
    use_desprior_candidates = False
    candidate_delta_mstar_range = None
    
    if dataset_choice == "3":
        DATASET_TYPE = "bcg_2p2arcmin"
        IMAGE_DIR = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/2p2arcmin/"
        TRUTH_TABLE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_2p2arcmin_with_coordinates.csv"
        use_bcg_data = True
        bcg_arcmin_type = "2p2arcmin"
        print(f"\nSelected: BCG 2.2 arcmin dataset")
    elif dataset_choice == "4":
        DATASET_TYPE = "bcg_3p8arcmin" 
        IMAGE_DIR = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin/"
        TRUTH_TABLE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_3p8arcmin_with_coordinates.csv"
        use_bcg_data = True
        bcg_arcmin_type = "3p8arcmin"
        print(f"\nSelected: BCG 3.8 arcmin dataset")
    elif dataset_choice == "2":
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
    
    # BCG dataset specific options
    if use_bcg_data:
        print("\n" + "="*60)
        print("BCG DATASET CONFIGURATION")
        print("="*60)
        print("Configure options specific to the new BCG dataset.")
        
        # Additional features option
        use_additional_features_input = input("\nInclude redshift and delta_mstar_z as additional features? (Y/n): ").strip().lower()
        use_additional_features = use_additional_features_input not in ['n', 'no']
        
        # RedMapper probabilities option  
        print("\nRedMapper BCG Probabilities:")
        print("These can be used for training supervision (loss weighting, evaluation) but")
        print("will NOT be used as input features during testing (to avoid cheating).")
        use_redmapper_probs_input = input("Include RedMapper BCG probabilities for training supervision? (y/N): ").strip().lower()
        use_redmapper_probs = use_redmapper_probs_input in ['y', 'yes']
        
        # Filtering options
        print("\nFiltering options:")
        apply_filters = input("Apply redshift or delta_mstar_z filters? (y/N): ").strip().lower()
        
        if apply_filters in ['y', 'yes']:
            # Get actual data ranges
            z_data_range, delta_data_range = get_bcg_data_ranges(bcg_arcmin_type)
            
            # Redshift filtering
            z_input = input(f"Redshift range (format: min,max, data range: {z_data_range}, or press Enter to skip): ").strip()
            if z_input:
                try:
                    z_min, z_max = map(lambda x: float(x.strip()), z_input.split(','))
                    z_range = f"{z_min},{z_max}"
                    print(f"Applied redshift filter: [{z_min}, {z_max}]")
                except:
                    print("Invalid format, skipping redshift filter")
            
            # Delta M* z filtering
            delta_input = input(f"Delta M* z range (format: min,max, data range: {delta_data_range}, or press Enter to skip): ").strip()
            if delta_input:
                try:
                    delta_min, delta_max = map(lambda x: float(x.strip()), delta_input.split(','))
                    delta_mstar_z_range = f"{delta_min},{delta_max}"
                    print(f"Applied delta M* z filter: [{delta_min}, {delta_max}]")
                except:
                    print("Invalid format, skipping delta M* z filter")
        
        # DESprior candidates option
        print("\nCandidate selection:")
        use_desprior_input = input("Use DESprior candidates instead of automatic detection? (y/N): ").strip().lower()
        use_desprior_candidates = use_desprior_input in ['y', 'yes']
        
        if use_desprior_candidates:
            # DESprior candidate filtering
            candidate_delta_input = input("Filter DESprior candidates by delta_mstar range (format: min,max, or press Enter to skip): ").strip()
            if candidate_delta_input:
                try:
                    candidate_delta_min, candidate_delta_max = map(lambda x: float(x.strip()), candidate_delta_input.split(','))
                    candidate_delta_mstar_range = f"{candidate_delta_min},{candidate_delta_max}"
                    print(f"Applied candidate delta_mstar filter: [{candidate_delta_min}, {candidate_delta_max}]")
                except:
                    print("Invalid format, skipping candidate delta_mstar filter")
            
            print("\nUsing DESprior catalog candidates")
        else:
            print("Using automatic candidate detection")
        
        print(f"\nBCG Configuration Summary:")
        print(f"  Dataset: {bcg_arcmin_type}")
        print(f"  Additional features: {use_additional_features}")
        print(f"  RedMapper probabilities: {use_redmapper_probs}")
        print(f"  Redshift filter: {z_range if z_range else 'None'}")
        print(f"  Delta M* z filter: {delta_mstar_z_range if delta_mstar_z_range else 'None'}")
        print(f"  DESprior candidates: {use_desprior_candidates}")
        if candidate_delta_mstar_range:
            print(f"  Candidate delta_mstar filter: {candidate_delta_mstar_range}")
    
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
        print(f"Detection threshold: {detection_threshold} (candidates above this probability are considered 'detections')")
        modify_uq = input("Modify detection threshold? (y/N): ").strip().lower()
        
        if modify_uq in ['y', 'yes']:
            detection_threshold = float(input("Detection threshold (0.0-1.0, default 0.5): ") or "0.5")
            detection_threshold = max(0.0, min(1.0, detection_threshold))
            print(f"Using detection threshold: {detection_threshold}")
        
        print("Above threshold = BCG detection, below = insufficient confidence")
    else:
        print("Using traditional deterministic classification")
    
    # Traditional candidate detection parameters (only for automatic candidate detection)
    if not use_desprior_candidates:
        print("\n" + "="*60)
        print("TRADITIONAL CANDIDATE DETECTION PARAMETERS")
        print("="*60)
        print("These control the base candidate finding algorithm")
        
        modify_params = input("Modify candidate detection parameters? (y/N): ").strip().lower()
        
        if modify_params in ['y', 'yes']:
            print("\nCandidate detection parameters:")
            min_distance = int(input("Minimum distance between candidates (default 20): ") or "20")
            threshold_rel = float(input("Relative brightness threshold (default 0.12): ") or "0.12")
            exclude_border = int(input("Exclude border pixels (default 0): ") or "0")
            if not use_multiscale:
                max_candidates = int(input("Maximum candidates per image (default 30): ") or "30")
            else:
                max_candidates = max_per_scale  # Use multiscale parameter
        else:
            min_distance = 20
            threshold_rel = 0.12
            exclude_border = 0
            max_candidates = max_per_scale if use_multiscale else 30
            print(f"Using defaults: min_distance={min_distance}, threshold_rel={threshold_rel}")
    else:
        # Use default values for DESprior (these parameters won't be used in DESprior mode)
        min_distance = 20
        threshold_rel = 0.12
        exclude_border = 0
        max_candidates = max_per_scale if use_multiscale else 30
        print("\n" + "="*60)
        print("CANDIDATE DETECTION PARAMETERS")
        print("="*60)
        print("Using DESprior catalog candidates - automatic detection parameters not needed")
    
    # Training parameters
    print("\n" + "="*60)
    print("TRAINING PARAMETERS")
    print("="*60)
    modify_training = input("Modify training parameters? (y/N): ").strip().lower()
    
    if modify_training in ['y', 'yes']:
        epochs = int(input("Number of epochs (default 100): ") or "100")
        batch_size = int(input("Batch size (default 16): ") or "16")
        lr = float(input("Learning rate (default 0.0001): ") or "0.0001")
    else:
        epochs = 100
        batch_size = 16
        lr = 0.0001
        print(f"Using defaults: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Choose which implementation to use
    print("\n" + "="*60)
    print("IMPLEMENTATION CHOICE")
    print("="*60)
    
    if use_multiscale or use_uq or use_bcg_data:
        reasons = []
        if use_multiscale:
            reasons.append("Multi-scale inference")
        if use_uq:
            reasons.append("Uncertainty quantification") 
        if use_bcg_data:
            reasons.append("BCG dataset support")
        
        print(f"Enhanced features requested - using enhanced scripts")
        print(f"Reasons: {', '.join(reasons)}")
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
    
    # Add BCG dataset flags
    if use_bcg_data:
        train_command += f" --use_bcg_data --bcg_arcmin_type {bcg_arcmin_type}"
        
        if use_additional_features:
            train_command += " --use_additional_features"
            
        if use_redmapper_probs:
            train_command += " --use_redmapper_probs"
        
        if z_range:
            train_command += f" --z_range \"{z_range}\""
        
        if delta_mstar_z_range:
            train_command += f" --delta_mstar_z_range \"{delta_mstar_z_range}\""
        
        if use_desprior_candidates:
            train_command += " --use_desprior_candidates"
            
            if candidate_delta_mstar_range:
                train_command += f" --candidate_delta_mstar_range \"{candidate_delta_mstar_range}\""
    
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
        --show_failures 20 \\
        --output_dir "{test_output_dir}" \\
        --save_results"""
    
    # Add enhanced feature flags to test command if using enhanced scripts
    if use_enhanced:
        if use_multiscale:
            scales_str = ",".join(map(str, scales))
            test_command += f" --use_multiscale --scales {scales_str} --max_candidates_per_scale {max_per_scale}"
        
        if use_uq:
            test_command += f" --use_uq --detection_threshold {detection_threshold}"
    
    # Add BCG dataset flags to test command
    if use_bcg_data:
        test_command += f" --use_bcg_data --bcg_arcmin_type {bcg_arcmin_type}"
        
        if use_additional_features:
            test_command += " --use_additional_features"
            
        if use_redmapper_probs:
            test_command += " --use_redmapper_probs"
        
        if z_range:
            test_command += f" --z_range \"{z_range}\""
        
        if delta_mstar_z_range:
            test_command += f" --delta_mstar_z_range \"{delta_mstar_z_range}\""
        
        if use_desprior_candidates:
            test_command += " --use_desprior_candidates"
            
            if candidate_delta_mstar_range:
                test_command += f" --candidate_delta_mstar_range \"{candidate_delta_mstar_range}\""
    
    if not run_command(test_command, f"Testing BCG classifier with {test_script}"):
        print("Testing failed.")
        return
    
    # Step 3: Generate Diagnostic Plots
    print("\n" + "="*80)
    print("STEP 3: GENERATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    # Generate diagnostic plots from evaluation results
    evaluation_csv = os.path.join(test_output_dir, "evaluation_results.csv")
    if os.path.exists(evaluation_csv):
        diagnostic_command = f"python -c \"from utils.diagnostic_plots import create_diagnostic_plots; create_diagnostic_plots('{evaluation_csv}', '{output_dir}')\""
        
        if not run_command(diagnostic_command, "Generating diagnostic plots"):
            print("Diagnostic plotting failed, but continuing...")
        else:
            print(f"Diagnostic plots saved to: {output_dir}/diagnostic_plots.png")
    else:
        print(f"Warning: Evaluation results file not found: {evaluation_csv}")
        print("Skipping diagnostic plots generation.")
    
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
    print(f"  Diagnostic plots: {output_dir}/diagnostic_plots.png")
    print(f"  Diagnostic plots (PDF): {output_dir}/diagnostic_plots.pdf")
    
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
    
    # Add BCG dataset info to summary
    if use_bcg_data:
        print(f"BCG Dataset: {bcg_arcmin_type}")
        print(f"Additional features: {use_additional_features}")
        print(f"RedMapper probabilities: {use_redmapper_probs}")
        if z_range:
            print(f"Redshift filtering: {z_range}")
        if delta_mstar_z_range:
            print(f"Delta M* z filtering: {delta_mstar_z_range}")
        if use_desprior_candidates:
            print(f"DESprior candidates: enabled")
            if candidate_delta_mstar_range:
                print(f"Candidate delta_mstar filtering: {candidate_delta_mstar_range}")
        else:
            print(f"Candidate detection: automatic")
    
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
    
    if use_bcg_data:
        print("3. ✓ BCG Dataset Integration:")
        print(f"   - New astronomical data ({bcg_arcmin_type} scale)")
        if use_additional_features:
            print("   - Additional features: redshift, delta_mstar_z")
        if use_redmapper_probs:
            print("   - RedMapper BCG probabilities for training supervision (not input features)")
        if use_desprior_candidates:
            print("   - DESprior catalog candidates")
            print("   - Advanced candidate filtering")
        else:
            print("   - Automatic candidate detection")
        if z_range or delta_mstar_z_range:
            print("   - Data filtering by physical properties")
    else:
        print("3. ✗ BCG Dataset Integration (using original dataset)")
    
    print(f"\nTo re-run evaluation with different parameters:")
    test_cmd_simple = f"python {test_script} --model_path '{model_path}' --scaler_path '{scaler_path}'"
    if use_enhanced and use_multiscale:
        test_cmd_simple += " --use_multiscale"
    if use_enhanced and use_uq:
        test_cmd_simple += " --use_uq"
    if use_bcg_data:
        test_cmd_simple += f" --use_bcg_data --bcg_arcmin_type {bcg_arcmin_type}"
    print(f"{test_cmd_simple} [other args]")
    
    print("\nWorkflow demonstration completed successfully!")
    if use_bcg_data:
        print("All enhancements have been implemented with new BCG dataset support:")
        print("- Multi-scale and uncertainty quantification features")
        print("- BCG dataset integration with additional features")
        print("- DESprior candidate system with filtering capabilities")
    else:
        print("Both requested enhancements have been implemented while preserving")
    print("all existing functionality and maintaining backward compatibility.")


if __name__ == "__main__":
    main()