#!/usr/bin/env python3
"""
Enhanced BCG Classifier Testing Script

This script evaluates trained BCG classifiers with:
1. Multi-scale candidate detection
2. Uncertainty quantification and probabilistic outputs
3. Detection threshold analysis
4. Enhanced visualizations with probability information
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from data.data_read import prepare_dataframe, BCGDataset
from ml_models.candidate_classifier import BCGCandidateClassifier
from utils.uq_classifier import BCGProbabilisticClassifier, predict_bcg_with_probabilities
from utils.multiscale_candidates import predict_bcg_from_multiscale_candidates
from utils.candidate_based_bcg import predict_bcg_from_candidates
from utils.viz_bcg import show_predictions_with_candidates, show_failures


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, random_seed=42):
    """Split dataset into train/validation/test sets (same as training)."""
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


def load_trained_model(model_path, scaler_path, feature_dim, use_uq=False):
    """Load trained model and feature scaler."""
    # Load appropriate model type
    if use_uq:
        model = BCGProbabilisticClassifier(feature_dim)
    else:
        model = BCGCandidateClassifier(feature_dim)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load scaler
    feature_scaler = joblib.load(scaler_path)
    
    return model, feature_scaler


def evaluate_enhanced_model(model, scaler, test_dataset, candidate_params, 
                          original_dataframe=None, dataset_type='SPT3G_1500d',
                          use_multiscale=False, use_uq=False, detection_threshold=0.5):
    """Evaluate enhanced model with multiscale and UQ capabilities."""
    print(f"Evaluating {'probabilistic' if use_uq else 'deterministic'} model on {len(test_dataset)} test images...")
    if use_multiscale:
        print("Using multi-scale candidate detection")
    
    predictions = []
    targets = []
    distances = []
    candidate_counts = []
    failed_predictions = []
    all_candidates_list = []
    all_scores_list = []
    test_images = []
    sample_metadata = []
    
    # UQ-specific tracking
    all_probabilities_list = []
    all_uncertainties_list = []
    detection_counts = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        image = sample['image']
        true_bcg = sample['BCG']
        filename = sample.get('filename', f'sample_{i}')
        
        # Store image for visualization
        test_images.append(image)
        
        # Extract metadata from original dataframe if available
        metadata = {'filename': filename}
        if original_dataframe is not None:
            cluster_name = filename.replace('.tif', '').split('_')[0]
            metadata['cluster_name'] = cluster_name
            
            cluster_col = 'Cluster name' if 'Cluster name' in original_dataframe.columns else 'cluster_name'
            if cluster_col in original_dataframe.columns:
                matching_rows = original_dataframe[original_dataframe[cluster_col] == cluster_name]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    if 'z' in row:
                        metadata['z'] = row['z']
                    prob_cols = [col for col in row.index if 'prob' in col.lower()]
                    if prob_cols:
                        metadata['bcg_prob'] = row[prob_cols[0]]
        
        # Make prediction with appropriate method
        if use_uq:
            results = predict_bcg_with_probabilities(
                image, model, scaler, 
                detection_threshold=detection_threshold,
                use_multiscale=use_multiscale,
                **candidate_params
            )
            
            predicted_bcg = results['best_bcg']
            all_candidates = results['all_candidates']
            scores = results['probabilities']  # These are probabilities, not raw scores
            probabilities = results['probabilities']
            uncertainties = results['uncertainties']
            detections = results['detections']
            
            # Track UQ metrics
            all_probabilities_list.append(probabilities)
            all_uncertainties_list.append(uncertainties)
            detection_counts.append(len(detections))
            
        else:
            # Use traditional method
            if use_multiscale:
                predicted_bcg, all_candidates, scores = predict_bcg_from_multiscale_candidates(
                    image, model, scaler, use_multiscale=True, **candidate_params
                )
            else:
                predicted_bcg, all_candidates, scores = predict_bcg_from_candidates(
                    image, model, scaler, **candidate_params
                )
            
            # No UQ information available
            probabilities = np.array([])
            uncertainties = np.array([])
            all_probabilities_list.append(probabilities)
            all_uncertainties_list.append(uncertainties)
            detection_counts.append(len(all_candidates) if len(all_candidates) > 0 else 0)
        
        if predicted_bcg is None:
            # No candidates found
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': 'no_candidates',
                'true_bcg': true_bcg
            })
            # Add empty entries to maintain list consistency
            all_candidates_list.append(np.array([]).reshape(0, 2))
            all_scores_list.append(np.array([]))
            sample_metadata.append(metadata)
            continue
        
        # Compute distance error
        distance = np.sqrt(np.sum((np.array(predicted_bcg) - true_bcg)**2))
        distances.append(distance)
        candidate_counts.append(len(all_candidates))
        
        # Store results
        predictions.append(predicted_bcg)
        targets.append(true_bcg)
        all_candidates_list.append(all_candidates)
        all_scores_list.append(scores)
        sample_metadata.append(metadata)
        
        # Check for potential failure cases
        if distance > 50:  # Large error threshold
            failed_predictions.append({
                'index': i,
                'filename': filename,
                'reason': 'large_error',
                'predicted': predicted_bcg,
                'true_bcg': true_bcg,
                'distance': distance,
                'candidates': all_candidates,
                'scores': scores
            })
    
    # Compute metrics
    distances = np.array(distances)
    success_rates = {}
    
    for threshold in [10, 20, 30, 50]:
        success_rate = np.mean(distances <= threshold) if len(distances) > 0 else 0
        success_rates[f'success_rate_{threshold}px'] = success_rate
    
    metrics = {
        'n_predictions': len(predictions),
        'n_failed': len(failed_predictions),
        'mean_distance': np.mean(distances) if len(distances) > 0 else float('inf'),
        'median_distance': np.median(distances) if len(distances) > 0 else float('inf'),
        'std_distance': np.std(distances) if len(distances) > 0 else 0,
        'min_distance': np.min(distances) if len(distances) > 0 else float('inf'),
        'max_distance': np.max(distances) if len(distances) > 0 else 0,
        'mean_candidates': np.mean(candidate_counts) if len(candidate_counts) > 0 else 0,
        **success_rates
    }
    
    # Add UQ-specific metrics
    if use_uq:
        metrics.update({
            'mean_detections': np.mean(detection_counts) if len(detection_counts) > 0 else 0,
            'detection_threshold': detection_threshold,
            'mean_probability': np.mean([np.mean(p) for p in all_probabilities_list if len(p) > 0]),
            'mean_uncertainty': np.mean([np.mean(u) for u in all_uncertainties_list if len(u) > 0])
        })
    
    return (predictions, targets, distances, failed_predictions, metrics,
            all_candidates_list, all_scores_list, test_images, sample_metadata,
            all_probabilities_list, all_uncertainties_list)


def print_enhanced_evaluation_report(metrics, failed_predictions, use_uq=False):
    """Print detailed evaluation report with UQ information."""
    print("\n" + "="*60)
    print("ENHANCED EVALUATION RESULTS")
    print("="*60)
    
    print(f"Total predictions: {metrics['n_predictions']}")
    print(f"Failed predictions: {metrics['n_failed']}")
    print(f"Average candidates per image: {metrics['mean_candidates']:.1f}")
    
    if use_uq:
        print(f"Average detections per image: {metrics['mean_detections']:.1f}")
        print(f"Detection threshold: {metrics['detection_threshold']:.3f}")
        print(f"Average probability: {metrics.get('mean_probability', 0):.3f}")
        print(f"Average uncertainty: {metrics.get('mean_uncertainty', 0):.3f}")
    print()
    
    if metrics['n_predictions'] > 0:
        print("Distance Metrics:")
        print(f"  Mean error: {metrics['mean_distance']:.2f} pixels")
        print(f"  Median error: {metrics['median_distance']:.2f} pixels")
        print(f"  Std deviation: {metrics['std_distance']:.2f} pixels")
        print(f"  Min error: {metrics['min_distance']:.2f} pixels")
        print(f"  Max error: {metrics['max_distance']:.2f} pixels")
        print()
        
        print("Success Rates:")
        for key, value in metrics.items():
            if 'success_rate' in key:
                threshold = key.split('_')[-1]
                print(f"  Within {threshold}: {value*100:.1f}%")
        print()
    
    if failed_predictions:
        print("Failed Prediction Analysis:")
        failure_reasons = {}
        for failure in failed_predictions:
            reason = failure['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} cases")


def plot_probability_analysis(all_probabilities_list, all_uncertainties_list, 
                            distances, save_dir=None):
    """Plot probability and uncertainty analysis."""
    if not all_probabilities_list or not any(len(p) > 0 for p in all_probabilities_list):
        return
    
    # Collect all probabilities and uncertainties
    all_probs = []
    all_uncs = []
    best_probs = []  # Probability of best candidate
    best_uncs = []   # Uncertainty of best candidate
    
    for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
        if len(probs) > 0:
            all_probs.extend(probs)
            best_probs.append(np.max(probs))
            
            if len(uncs) > 0:
                all_uncs.extend(uncs)
                best_idx = np.argmax(probs)
                best_uncs.append(uncs[best_idx])
    
    if not all_probs:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Probability distribution
    axes[0, 0].hist(all_probs, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('BCG Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of All Candidate Probabilities')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Best candidate probabilities
    axes[0, 1].hist(best_probs, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Best Candidate Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Best Candidate Probabilities')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty analysis
    if all_uncs:
        axes[1, 0].hist(all_uncs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Uncertainty')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of All Candidate Uncertainties')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Probability vs Uncertainty scatter
        if len(best_probs) == len(best_uncs):
            scatter = axes[1, 1].scatter(best_probs, best_uncs, c=distances[:len(best_probs)], 
                                       cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Best Candidate Probability')
            axes[1, 1].set_ylabel('Best Candidate Uncertainty')
            axes[1, 1].set_title('Probability vs Uncertainty (colored by distance error)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Distance Error (pixels)')
    else:
        # If no uncertainties, just show probability vs distance
        if len(best_probs) <= len(distances):
            axes[1, 0].scatter(best_probs, distances[:len(best_probs)], alpha=0.6)
            axes[1, 0].set_xlabel('Best Candidate Probability')
            axes[1, 0].set_ylabel('Distance Error (pixels)')
            axes[1, 0].set_title('Probability vs Distance Error')
            axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].text(0.5, 0.5, 'No uncertainty\ninformation available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Uncertainty Analysis')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'probability_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability analysis saved to: {save_path}")
    
    plt.show()
    plt.close()


def show_enhanced_predictions(images, targets, predictions, all_candidates_list, 
                            all_scores_list, all_probabilities_list=None,
                            indices=None, save_dir=None, phase=None, use_uq=False):
    """Enhanced visualization with probability information."""
    if indices is None:
        indices = range(min(5, len(images)))
    
    for i, idx in enumerate(indices):
        if idx >= len(images):
            continue
            
        image = images[idx]
        target = targets[idx]
        prediction = predictions[idx]
        candidates = all_candidates_list[idx] if idx < len(all_candidates_list) else []
        scores = all_scores_list[idx] if idx < len(all_scores_list) else np.array([])
        
        # Get probabilities if available
        if use_uq and all_probabilities_list and idx < len(all_probabilities_list):
            probabilities = all_probabilities_list[idx]
        else:
            probabilities = scores  # Use scores as proxy
        
        # Calculate distance between target and prediction
        distance = np.sqrt(np.sum((target - prediction)**2))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = np.clip(image.astype(np.uint8), 0, 255)
        else:
            display_image = image
        
        plt.imshow(display_image)
        
        # Plot all candidates with probability-based coloring
        if len(candidates) > 0:
            candidates_array = np.array(candidates)
            
            if len(probabilities) > 0 and use_uq:
                # Color by probability
                scatter = plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                                    c=probabilities, cmap='coolwarm', 
                                    marker='s', s=200, alpha=0.7, 
                                    vmin=0, vmax=1, edgecolors='black', linewidths=1)
                cbar = plt.colorbar(scatter, ax=plt.gca(), shrink=0.8)
                cbar.set_label('BCG Probability', rotation=270, labelpad=20)
                candidate_label = f'Candidates ({len(candidates)}) - colored by probability'
            else:
                # Traditional visualization
                plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                          marker='s', s=200, facecolors='none', edgecolors='cyan', 
                          linewidths=1, alpha=0.5, label=f'Candidates ({len(candidates)})')
                candidate_label = f'Candidates ({len(candidates)})'
        
        # Plot selected BCG (prediction) as red circle
        plt.scatter(prediction[0], prediction[1], marker='o', s=400, 
                   facecolors='none', edgecolors='red', linewidths=3, alpha=0.9,
                   label='Predicted BCG')
        
        # Plot true BCG location as yellow circle
        plt.scatter(target[0], target[1], marker='o', s=250, 
                   facecolors='none', edgecolors='yellow', linewidths=3, alpha=0.9,
                   label='True BCG')
        
        # Enhanced title with UQ information
        title = f'Enhanced BCG Prediction - Sample {idx+1}'
        if phase:
            title = f'{phase} - Sample {idx+1}'
        
        subtitle = f'Distance: {distance:.1f} px | Candidates: {len(candidates)}'
        
        if len(scores) > 0:
            if use_uq and len(probabilities) > 0:
                max_prob = np.max(probabilities)
                avg_prob = np.mean(probabilities)
                subtitle += f' | Max Prob: {max_prob:.3f} | Avg Prob: {avg_prob:.3f}'
            else:
                max_score = np.max(scores)
                avg_score = np.mean(scores)
                subtitle += f' | Max Score: {max_score:.3f} | Avg Score: {avg_score:.3f}'
        
        plt.title(f'{title}\n{subtitle}', fontsize=12)
        
        # Adjust legend
        if not (len(probabilities) > 0 and use_uq):
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.axis('off')
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            phase_str = f"{phase}_" if phase else ""
            uq_str = "Probabilistic_" if use_uq else ""
            filename = f'{phase_str}{uq_str}prediction_sample_{idx+1}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced prediction plot saved: {save_path}")
        
        plt.show()
        plt.close()


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("ENHANCED BCG CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Scaler: {args.scaler_path}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Images: {args.image_dir}")
    
    if args.use_multiscale:
        print(f"Multi-scale: scales={args.scales}")
    if args.use_uq:
        print(f"Uncertainty quantification: threshold={args.detection_threshold}")
    print()
    
    # Load original truth table for metadata
    print("Loading original truth table...")
    original_df = pd.read_csv(args.truth_table)
    
    # Load processed dataset
    print("Loading processed dataset...")
    dataframe = prepare_dataframe(args.image_dir, args.truth_table, args.dataset_type)
    print(f"Found {len(dataframe)} samples in dataset")
    
    # Create BCG dataset
    dataset = BCGDataset(args.image_dir, dataframe)
    
    # Split dataset (use same random seed as training)
    train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.2)
    print(f"Using test split: {len(test_subset)} samples")
    
    # Feature dimension (should match training)
    base_feature_dim = 30
    if args.use_multiscale:
        base_feature_dim += 3  # Additional scale features
    
    # Load trained model
    print("Loading trained model...")
    model, scaler = load_trained_model(args.model_path, args.scaler_path, 
                                     base_feature_dim, use_uq=args.use_uq)
    
    # Set up candidate parameters
    candidate_params = {
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'exclude_border': args.exclude_border,
        'max_candidates': args.max_candidates
    }
    
    if args.use_multiscale:
        candidate_params.update({
            'scales': args.scales,
            'max_candidates_per_scale': args.max_candidates_per_scale
        })
    
    # Evaluate model
    results = evaluate_enhanced_model(
        model, scaler, test_subset, candidate_params, original_df, args.dataset_type,
        use_multiscale=args.use_multiscale, use_uq=args.use_uq, 
        detection_threshold=args.detection_threshold
    )
    
    (predictions, targets, distances, failures, metrics, 
     all_candidates_list, all_scores_list, test_images, sample_metadata,
     all_probabilities_list, all_uncertainties_list) = results
    
    # Print results
    print_enhanced_evaluation_report(metrics, failures, use_uq=args.use_uq)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # UQ-specific analysis
    if args.use_uq and len(predictions) > 0:
        print("\nGenerating probability and uncertainty analysis...")
        plot_probability_analysis(all_probabilities_list, all_uncertainties_list, 
                                distances, save_dir=args.output_dir)
    
    # Show sample predictions with enhanced visualization
    if args.show_samples > 0 and len(predictions) > 0:
        print(f"\nShowing {min(args.show_samples, len(predictions))} sample predictions...")
        
        # Sort by distance for best examples
        sorted_indices = np.argsort(distances)[:args.show_samples]
        
        # Prepare data for visualization
        sample_images = [test_images[i] for i in sorted_indices]
        sample_targets = [targets[i] for i in sorted_indices]
        sample_predictions = [predictions[i] for i in sorted_indices]
        sample_candidates = [all_candidates_list[i] for i in sorted_indices]
        sample_scores = [all_scores_list[i] for i in sorted_indices]
        
        # Get probabilities if available
        if args.use_uq:
            sample_probabilities = [all_probabilities_list[i] for i in sorted_indices]
        else:
            sample_probabilities = None
        
        phase_name = "EnhancedTesting"
        if args.use_multiscale:
            phase_name = "MultiscaleTesting"
        if args.use_uq:
            phase_name = "ProbabilisticTesting"
        if args.use_multiscale and args.use_uq:
            phase_name = "MultiscaleProbabilisticTesting"
        
        show_enhanced_predictions(
            sample_images, sample_targets, sample_predictions,
            sample_candidates, sample_scores, sample_probabilities,
            indices=range(len(sample_images)),
            save_dir=args.output_dir,
            phase=phase_name,
            use_uq=args.use_uq
        )
    
    # Show failure cases
    if args.show_failures and len(distances) > 0:
        print(f"\nShowing worst prediction failures...")
        
        # Get worst cases
        worst_indices = np.argsort(distances)[-args.show_failures:]
        
        failure_images = [test_images[i] for i in worst_indices]
        failure_targets = [targets[i] for i in worst_indices]
        failure_predictions = [predictions[i] for i in worst_indices]
        
        phase_name = "EnhancedTesting"
        if args.use_multiscale:
            phase_name = "MultiscaleTesting"
        if args.use_uq:
            phase_name = "ProbabilisticTesting"
        if args.use_multiscale and args.use_uq:
            phase_name = "MultiscaleProbabilisticTesting"
        
        show_failures(
            failure_images, failure_targets, failure_predictions,
            threshold=20, max_failures=args.show_failures,
            save_dir=args.output_dir,
            phase=phase_name
        )
    
    # Save detailed results
    if args.save_results and len(predictions) > 0:
        results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
        
        # Create enhanced results dictionary
        results_data = {
            'pred_x': [pred[0] for pred in predictions],
            'pred_y': [pred[1] for pred in predictions], 
            'true_x': [target[0] for target in targets],
            'true_y': [target[1] for target in targets],
            'distance_error': distances,
            'n_candidates': [len(cand) for cand in all_candidates_list]
        }
        
        # Add UQ-specific columns
        if args.use_uq:
            max_probabilities = []
            avg_probabilities = []
            max_uncertainties = []
            avg_uncertainties = []
            n_detections = []
            
            for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
                if len(probs) > 0:
                    max_probabilities.append(np.max(probs))
                    avg_probabilities.append(np.mean(probs))
                    n_detections.append(np.sum(probs >= args.detection_threshold))
                else:
                    max_probabilities.append(np.nan)
                    avg_probabilities.append(np.nan)
                    n_detections.append(0)
                
                if len(uncs) > 0:
                    max_uncertainties.append(np.max(uncs))
                    avg_uncertainties.append(np.mean(uncs))
                else:
                    max_uncertainties.append(np.nan)
                    avg_uncertainties.append(np.nan)
            
            results_data.update({
                'max_probability': max_probabilities,
                'avg_probability': avg_probabilities,
                'max_uncertainty': max_uncertainties,
                'avg_uncertainty': avg_uncertainties,
                'n_detections': n_detections,
                'detection_threshold': [args.detection_threshold] * len(predictions)
            })
        
        # Add metadata columns
        if sample_metadata:
            cluster_names = [meta.get('cluster_name', 'unknown') for meta in sample_metadata]
            results_data['cluster_name'] = cluster_names
            
            if any('z' in meta for meta in sample_metadata):
                redshifts = [meta.get('z', np.nan) for meta in sample_metadata]
                results_data['z'] = redshifts
            
            if any('bcg_prob' in meta for meta in sample_metadata):
                bcg_probs = [meta.get('bcg_prob', np.nan) for meta in sample_metadata]
                results_data['bcg_prob'] = bcg_probs
        
        results_df = pd.DataFrame(results_data)
        
        # Reorder columns to put metadata first
        cols = ['cluster_name'] if 'cluster_name' in results_df.columns else []
        if 'z' in results_df.columns:
            cols.append('z')
        if 'bcg_prob' in results_df.columns:
            cols.append('bcg_prob')
        
        # Add coordinate and error columns
        cols.extend(['pred_x', 'pred_y', 'true_x', 'true_y', 'distance_error'])
        
        # Add UQ columns
        if args.use_uq:
            cols.extend(['max_probability', 'avg_probability', 'n_detections', 
                        'detection_threshold', 'max_uncertainty', 'avg_uncertainty'])
        
        # Add remaining columns
        cols.extend(['n_candidates'])
        
        # Only include columns that exist
        cols = [col for col in cols if col in results_df.columns]
        results_df = results_df[cols]
        
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Results include {len(results_df.columns)} columns: {', '.join(results_df.columns)}")
        
        # Save UQ-specific analysis if enabled
        if args.use_uq:
            uq_analysis_file = os.path.join(args.output_dir, 'probability_analysis.csv')
            
            # Compile probability analysis
            prob_analysis_data = []
            for i, (probs, uncs) in enumerate(zip(all_probabilities_list, all_uncertainties_list)):
                sample_name = sample_metadata[i].get('cluster_name', f'sample_{i}') if i < len(sample_metadata) else f'sample_{i}'
                
                for j, prob in enumerate(probs):
                    unc = uncs[j] if j < len(uncs) else np.nan
                    is_detection = prob >= args.detection_threshold
                    is_best = j == np.argmax(probs) if len(probs) > 0 else False
                    
                    prob_analysis_data.append({
                        'sample_name': sample_name,
                        'candidate_idx': j,
                        'probability': prob,
                        'uncertainty': unc,
                        'is_detection': is_detection,
                        'is_best_candidate': is_best,
                        'distance_error': distances[i] if i < len(distances) else np.nan
                    })
            
            if prob_analysis_data:
                prob_df = pd.DataFrame(prob_analysis_data)
                prob_df.to_csv(uq_analysis_file, index=False)
                print(f"Probability analysis saved to: {uq_analysis_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Enhanced BCG Classifier")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--scaler_path', type=str, required=True,
                       help='Path to feature scaler (.pkl file)')
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing .tif image files')
    parser.add_argument('--truth_table', type=str, required=True,
                       help='Path to CSV file with BCG coordinates')
    parser.add_argument('--dataset_type', type=str, default='SPT3G_1500d',
                       choices=['SPT3G_1500d', 'megadeep500'],
                       help='Type of dataset')
    
    # Candidate finding arguments (should match training)
    parser.add_argument('--min_distance', type=int, default=15,
                       help='Minimum distance between candidates')
    parser.add_argument('--threshold_rel', type=float, default=0.12,
                       help='Relative threshold for candidate detection')
    parser.add_argument('--exclude_border', type=int, default=30,
                       help='Exclude candidates near borders')
    parser.add_argument('--max_candidates', type=int, default=25,
                       help='Maximum candidates per image')
    
    # Enhanced feature arguments
    parser.add_argument('--use_multiscale', action='store_true',
                       help='Enable multi-scale candidate detection')
    parser.add_argument('--scales', type=str, default='0.5,1.0,1.5',
                       help='Comma-separated scale factors for multiscale detection')
    parser.add_argument('--max_candidates_per_scale', type=int, default=10,
                       help='Maximum candidates per scale in multiscale mode')
    
    parser.add_argument('--use_uq', action='store_true',
                       help='Enable uncertainty quantification with probabilistic outputs')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Probability threshold for BCG detection (0.0-1.0)')
    
    # Visualization arguments
    parser.add_argument('--show_samples', type=int, default=5,
                       help='Number of sample predictions to visualize')
    parser.add_argument('--show_failures', type=int, default=3,
                       help='Number of failure cases to visualize')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results to CSV')
    
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