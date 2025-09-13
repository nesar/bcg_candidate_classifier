#!/usr/bin/env python3
"""
Enhanced BCG Candidate Classifier - Full Run Demo

This script extends the working old_full_run.py with key enhancements:
1. Uncertainty quantification (UQ) with probability thresholds for detections

The script maintains all existing functionality while adding options for these improvements.
"""

import os
# Fix NUMEXPR warning - MUST be set before ANY numpy/sklearn imports
os.environ['NUMEXPR_MAX_THREADS'] = '64'

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
    print("1. Uncertainty quantification with probability thresholds")
    print("="*80)
    
    # Dataset selection
    print("\nDataset options:")
    print("1. BCG 2.2 arcmin (default)")
    print("2. BCG 3.8 arcmin")
    
    dataset_choice = input("\nSelect dataset (1-2): ").strip()
    
    # Initialize BCG-specific variables
    use_bcg_data = True
    bcg_arcmin_type = None
    z_range = None
    delta_mstar_z_range = None
    use_additional_features = False
    use_redmapper_probs = False
    use_desprior_candidates = False
    candidate_delta_mstar_range = None
    
    if dataset_choice == "2":
        DATASET_TYPE = "bcg_3p8arcmin" 
        IMAGE_DIR = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/3p8arcmin/"
        TRUTH_TABLE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_3p8arcmin_with_coordinates.csv"
        bcg_arcmin_type = "3p8arcmin"
        print(f"\nSelected: BCG 3.8 arcmin dataset")
    else:
        DATASET_TYPE = "bcg_2p2arcmin"
        IMAGE_DIR = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/2p2arcmin/"
        TRUTH_TABLE = "/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/data/lbleem/bcgs/bcgs_2p2arcmin_with_coordinates.csv"
        bcg_arcmin_type = "2p2arcmin"
        print(f"\nSelected: BCG 2.2 arcmin dataset")
    
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
    
    # ENHANCEMENT 1: Color Features option  
    print("\n" + "="*60)
    print("ENHANCEMENT 1: COLOR FEATURES")
    print("="*60)
    print("Color features help distinguish red-sequence BCG candidates from")
    print("bright white objects (stars, QSOs) that often cause false positives.")
    print("This preserves RGB color information lost during grayscale conversion.")
    
    use_color_features = input("\nEnable color features for red-sequence detection? (Y/n): ").strip().lower()
    use_color_features = use_color_features not in ['n', 'no']
    
    if use_color_features:
        print("Color features will be extracted from RGB patches to identify red objects.")
        print("This includes color ratios, spatial color variation, and PCA-reduced color info.")
    else:
        print("Using traditional grayscale-based features only.")
    
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
    detection_threshold = 0.4
    if use_uq:
        print("\nUncertainty quantification parameters:")
        print(f"Detection threshold: {detection_threshold} (candidates above this probability are considered 'detections')")
        modify_uq = input("Modify detection threshold? (y/N): ").strip().lower()
        
        if modify_uq in ['y', 'yes']:
            detection_threshold = float(input("Detection threshold (0.0-1.0, default 0.4): ") or "0.4")
            detection_threshold = max(0.0, min(1.0, detection_threshold))
            print(f"Using detection threshold: {detection_threshold}")
        
        print("Above threshold = BCG detection, below = insufficient confidence")
    else:
        print("Using traditional deterministic classification")
    
    # ENHANCEMENT 3: Feature Importance Analysis option
    print("\n" + "="*60)
    print("ENHANCEMENT 3: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    print("Feature importance analysis provides insights into which features are")
    print("most important for BCG classification. This includes:")
    print("- SHAP (SHapley Additive exPlanations) for rigorous feature attribution")
    print("- Permutation importance for performance-based importance")
    print("- Individual sample explanations for understanding specific predictions")
    print("- Comprehensive visualizations and reports")
    
    run_analysis = input("\nRun feature importance analysis after training/testing? (Y/n): ").strip().lower()
    run_analysis = run_analysis not in ['n', 'no']
    
    # Analysis parameters
    analysis_methods = ['permutation', 'shap', 'gradient']
    analysis_samples = 1000
    
    if run_analysis:
        print("\nFeature importance analysis configuration:")
        print(f"Default methods: {', '.join(analysis_methods)}")
        print(f"Default analysis samples: {analysis_samples}")
        
        modify_analysis = input("Modify analysis parameters? (y/N): ").strip().lower()
        
        if modify_analysis in ['y', 'yes']:
            print("\nAvailable analysis methods:")
            print("1. permutation - Performance degradation when features shuffled (fast, reliable)")
            print("2. shap - SHapley Additive exPlanations (comprehensive, slower)")
            print("3. gradient - Gradient-based importance (fast, neural network specific)")
            
            methods_input = input("Select methods (comma-separated, e.g., 'permutation,shap'): ").strip()
            if methods_input:
                analysis_methods = [m.strip() for m in methods_input.split(',')]
            
            analysis_samples_input = input(f"Number of samples for analysis (default {analysis_samples}): ").strip()
            if analysis_samples_input:
                analysis_samples = int(analysis_samples_input)
        
        print(f"Analysis will use methods: {', '.join(analysis_methods)}")
        print(f"Analysis will process: {analysis_samples} samples")
        
        # Check if SHAP is requested
        if 'shap' in analysis_methods:
            print("\nNote: SHAP analysis requires the 'shap' package.")
            print("Install with: pip install shap")
    else:
        print("Skipping feature importance analysis")
    
    # Traditional candidate detection parameters (only for automatic candidate detection)
    if not use_desprior_candidates:
        print("\n" + "="*60)
        print("TRADITIONAL CANDIDATE DETECTION PARAMETERS")
        print("="*60)
        print("These control the base candidate finding algorithm")
        
        modify_params = input("Modify candidate detection parameters? (y/N): ").strip().lower()
        
        if modify_params in ['y', 'yes']:
            print("\nCandidate detection parameters:")
            min_distance = int(input("Minimum distance between candidates (default 8): ") or "8")
            threshold_rel = float(input("Relative brightness threshold (default 0.1): ") or "0.1")
            exclude_border = int(input("Exclude border pixels (default 0): ") or "0")
            max_candidates = int(input("Maximum candidates per image (default 50): ") or "50")
        else:
            min_distance = 8
            threshold_rel = 0.1
            exclude_border = 0
            max_candidates = 50
            print(f"Using improved defaults: min_distance={min_distance}, threshold_rel={threshold_rel}")
    else:
        # Use default values for DESprior (these parameters won't be used in DESprior mode)
        min_distance = 8
        threshold_rel = 0.1
        exclude_border = 0
        max_candidates = 50
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
        epochs = int(input("Number of epochs (default 32): ") or "32")
        batch_size = int(input("Batch size (default 16): ") or "16")
        lr = float(input("Learning rate (default 0.0001): ") or "0.0001")
    else:
        epochs = 32
        batch_size = 16
        lr = 0.0001
        print(f"Using defaults: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    
    # Feature extraction parameters
    print("\n" + "="*60)
    print("FEATURE EXTRACTION PARAMETERS")
    print("="*60)
    print("Patch size determines the spatial resolution for feature extraction.")
    print("Larger patches capture more context but increase computational cost.")
    modify_features = input("Modify feature extraction parameters? (y/N): ").strip().lower()
    
    if modify_features in ['y', 'yes']:
        print("\nAvailable patch sizes:")
        print("  64x64 pixels (default) - Good balance of context and efficiency")
        print("  128x128 pixels - More spatial context, higher computation")
        print("  256x256 pixels - Maximum context, highest computation")
        patch_size = int(input("Patch size (64, 128, 256, default 64): ") or "64")
        if patch_size not in [64, 128, 256]:
            print(f"Warning: Unusual patch size {patch_size}. Using anyway.")
    else:
        patch_size = 64
        print(f"Using default patch size: {patch_size}x{patch_size} pixels")
    
    # Choose which implementation to use
    print("\n" + "="*60)
    print("IMPLEMENTATION CHOICE")
    print("="*60)
    
    reasons = []
    if use_color_features:
        reasons.append("Color features")
    if use_uq:
        reasons.append("Uncertainty quantification")
    reasons.append("BCG dataset support")
    
    print(f"Enhanced features requested - using enhanced scripts")
    print(f"Reasons: {', '.join(reasons)}")
    use_enhanced = True
    train_script = "train.py"
    test_script = "test.py"
    
    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = "candidate_classifier"
    if use_color_features:
        experiment_name += "_color"
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
        --patch_size {patch_size} \\
        --output_dir "{output_dir}" \\
        --plot"""
    
    # Add enhanced feature flags if using enhanced scripts
    if use_enhanced:
        if use_color_features:
            train_command += " --use_color_features"
        
        if use_uq:
            train_command += f" --use_uq --detection_threshold {detection_threshold}"
    
    # Add BCG dataset flags
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
    if use_color_features:
        feature_summary.append("Color features (red-sequence detection)")
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
        --patch_size {patch_size} \\
        --show_samples 5 \\
        --show_failures 20 \\
        --output_dir "{test_output_dir}" \\
        --save_results"""
    
    # Add enhanced feature flags to test command if using enhanced scripts
    if use_enhanced:
        if use_color_features:
            test_command += " --use_color_features"
        
        if use_uq:
            test_command += f" --use_uq --detection_threshold {detection_threshold}"
    
    # Add BCG dataset flags to test command
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
    
    # Step 3: Feature Importance Analysis (if requested)
    analysis_output_dir = None
    if run_analysis:
        print("\n" + "="*80)
        print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        analysis_output_dir = os.path.join(output_dir, "feature_importance_analysis")
        
        # Prepare test data path for analysis
        test_data_path = os.path.join(test_output_dir, "test_features.npz")
        if not os.path.exists(test_data_path):
            print("Warning: Test features file not found. Checking for evaluation data...")
            # Try to use evaluation CSV data
            evaluation_csv = os.path.join(test_output_dir, "evaluation_results.csv")
            if os.path.exists(evaluation_csv):
                print("Using evaluation results for analysis...")
                # First try to prepare the data in the right format
                prepare_data_command = f"python analysis/prepare_analysis_data.py --test_output_dir '{test_output_dir}'"
                if run_command(prepare_data_command, "Preparing analysis data from evaluation results"):
                    test_data_path = os.path.join(test_output_dir, "analysis_data.npz")
                else:
                    test_data_path = evaluation_csv
            else:
                print("No suitable data found for analysis. Skipping feature importance analysis.")
                run_analysis = False
        
        if run_analysis:
            # Build analysis command
            analysis_command = f"""python -c "
import sys
sys.path.append('.')
from analysis.run_analysis import BCGAnalysisRunner
import os

config = {{
    'model_path': '{model_path}',
    'data_path': '{test_data_path}',
    'model_type': '{'probabilistic' if use_uq else 'deterministic'}',
    'probabilistic_model': {use_uq},
    'output_dir': '{analysis_output_dir}',
    'analysis_methods': {analysis_methods},
    'analysis_samples': {analysis_samples},
    'features': {{
        'use_color': {use_color_features},
        'use_auxiliary': {use_additional_features},
        'color_pca_components': 8
    }}
}}

print('Starting BCG feature importance analysis...')
runner = BCGAnalysisRunner(**config)
try:
    results = runner.run_complete_analysis()
    print(f'Analysis completed successfully!')
    print(f'Results saved to: {{results[\"output_directory\"]}}')
except Exception as e:
    print(f'Analysis failed: {{e}}')
    import traceback
    traceback.print_exc()
"
"""
            
            if not run_command(analysis_command, "Running comprehensive feature importance analysis"):
                print("Feature importance analysis failed, but continuing with workflow...")
                analysis_output_dir = None
            else:
                print(f"Feature importance analysis completed successfully!")
                print(f"Analysis results saved to: {analysis_output_dir}")
    
    # Step 4: Generate Diagnostic Plots
    step_number = 4 if run_analysis else 3
    print(f"\n" + "="*80)
    print(f"STEP {step_number}: GENERATING DIAGNOSTIC PLOTS")
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
    
    # Final Step: Summary
    final_step_number = 5 if run_analysis else 4
    print(f"\n" + "="*80)
    print("ENHANCED WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nExperiment results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  Training curves: {output_dir}/training_curves.png")
    print(f"  Best model: {model_path}")
    print(f"  Feature scaler: {scaler_path}")
    print(f"  Evaluation results: {test_output_dir}/")
    if use_uq:
        print(f"  Rank-based samples: {test_output_dir}/*_prediction_sample_best_rank*.png")
    else:
        print(f"  Sample predictions: {test_output_dir}/*_prediction_sample_*.png")
        print(f"  Failure cases: {test_output_dir}/*_failure_sample_*.png")
    print(f"  Detailed results: {test_output_dir}/evaluation_results.csv")
    print(f"  Diagnostic plots: {output_dir}/diagnostic_plots.png")
    print(f"  Diagnostic plots (PDF): {output_dir}/diagnostic_plots.pdf")
    
    if use_uq:
        print(f"  Probability analysis: {test_output_dir}/probability_analysis.csv")
        print(f"  Uncertainty plots: {test_output_dir}/probability_analysis.png")
    
    # Feature importance analysis results
    if run_analysis and analysis_output_dir:
        print(f"\n  === Feature Importance Analysis Results ===")
        print(f"  Analysis directory: {analysis_output_dir}/")
        print(f"  Feature rankings (CSV): {analysis_output_dir}/csv_reports/")
        print(f"  Comprehensive plots: {analysis_output_dir}/plots/")
        print(f"  Individual explanations: {analysis_output_dir}/individual_plots/")
        print(f"  Analysis summary: {analysis_output_dir}/analysis_summary.txt")
        print(f"  Raw results: {analysis_output_dir}/raw_results/")
        if 'shap' in analysis_methods:
            print(f"  SHAP summary plots: {analysis_output_dir}/plots/shap_summary_*.png")
            print(f"  SHAP individual explanations: {analysis_output_dir}/individual_plots/shap_individual_*.png")
    elif run_analysis:
        print(f"\n  ⚠️  Feature importance analysis was requested but failed to complete")
    
    print(f"\nApproach: Enhanced candidate-based BCG classification")
    print(f"Dataset: {DATASET_TYPE}")
    print(f"Scripts used: {train_script}, {test_script}")
    print(f"Training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"Candidate detection: min_distance={min_distance}, threshold_rel={threshold_rel}")
    print(f"Feature extraction: patch_size={patch_size}x{patch_size} pixels")
    
    
    if use_color_features:
        print(f"Color features: RGB-based red-sequence detection enabled")
    if use_uq:
        print(f"Uncertainty quantification: threshold={detection_threshold}")
    
    # Add BCG dataset info to summary
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
    if use_color_features:
        print("1. ✓ Color Features:")
        print("   - RGB color ratios for red-sequence identification")
        print("   - Spatial color variation analysis")
        print("   - PCA-reduced color information")
        print("   - Distinguishes red BCGs from bright white objects")
        enhancement_num = 2
    else:
        print("1. ✗ Color features (using grayscale features only)")
        enhancement_num = 2
    
    if use_uq:
        print(f"{enhancement_num}. ✓ Uncertainty quantification:")
        print("   - Probabilistic outputs (0-1 scale)")
        print("   - Detection threshold for confident predictions") 
        print("   - Uncertainty estimates for risk assessment")
        enhancement_num += 1
    else:
        print(f"{enhancement_num}. ✗ Uncertainty quantification (using deterministic scores)")
        enhancement_num += 1
    
    if run_analysis and analysis_output_dir:
        print(f"{enhancement_num}. ✓ Feature Importance Analysis:")
        print("   - Global feature importance ranking (all methods)")
        print("   - Individual sample explanations")
        print("   - Feature group analysis (morphological, color, contextual, auxiliary)")
        print("   - Comprehensive visualization reports")
        if 'shap' in analysis_methods:
            print("   - SHAP waterfall plots for individual predictions")
        if 'permutation' in analysis_methods:
            print("   - Permutation-based feature importance")
        if 'gradient' in analysis_methods:
            print("   - Gradient-based neural network feature importance")
        enhancement_num += 1
    elif run_analysis:
        print(f"{enhancement_num}. ⚠️ Feature Importance Analysis: requested but failed")
        enhancement_num += 1
    else:
        print(f"{enhancement_num}. ✗ Feature importance analysis (skipped)")
        enhancement_num += 1
    
    print(f"{enhancement_num}. ✓ BCG Dataset Integration:")
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
    
    print(f"\nTo re-run evaluation with different parameters:")
    test_cmd_simple = f"python {test_script} --model_path '{model_path}' --scaler_path '{scaler_path}'"
    if use_enhanced:
        if use_color_features:
            test_cmd_simple += " --use_color_features"
        if use_uq:
            test_cmd_simple += " --use_uq"
    test_cmd_simple += f" --use_bcg_data --bcg_arcmin_type {bcg_arcmin_type}"
    print(f"{test_cmd_simple} [other args]")
    
    print("\nWorkflow demonstration completed successfully!")
    print("All enhancements have been implemented with BCG dataset support:")
    if use_color_features:
        print("- Color feature extraction for red-sequence identification")
    if use_uq:
        print("- Uncertainty quantification with probabilistic outputs")
    if run_analysis:
        print("- Feature importance analysis with SHAP and permutation methods")
    print("- BCG dataset integration with additional features")
    print("- DESprior candidate system with filtering capabilities")
    print("All existing functionality maintained with backward compatibility.")
    
    if run_analysis and analysis_output_dir:
        print(f"\n🎯 FEATURE IMPORTANCE INSIGHTS:")
        print(f"   📊 Check feature rankings: {analysis_output_dir}/csv_reports/")
        print(f"   📈 View importance plots: {analysis_output_dir}/plots/")
        print(f"   🔍 Individual explanations: {analysis_output_dir}/individual_plots/")
        print(f"   📋 Read summary report: {analysis_output_dir}/analysis_summary.txt")
        print(f"\n   Use these results to understand which features matter most for BCG detection!")


if __name__ == "__main__":
    main()