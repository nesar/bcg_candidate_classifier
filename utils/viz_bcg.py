import os
import numpy as np
import matplotlib.pyplot as plt


def show_BCG(image, BCG, sample_idx=None, save_path=None):
    """
    Display image with BCG location marked.
    
    Args:
        image: Image array
        BCG: BCG coordinates [x, y]
        sample_idx: Optional sample index for title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    if sample_idx is not None:
        plt.title(f'Sample #{sample_idx}')
    
    plt.axis('off')
    plt.imshow(image.astype(np.uint8))
    plt.scatter(BCG[0], BCG[1], marker='o', s=300, c='r', 
                facecolors='none', linewidth=2, label='BCG')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"BCG plot saved to {save_path}")
    
    plt.show()


def show_predictions_with_candidates(images, targets, predictions, all_candidates_list, candidate_scores_list=None, indices=None, save_dir=None, phase=None, 
                                   probabilities_list=None, detection_threshold=0.5, use_uq=False, metadata_list=None):
    """
    Show images with candidate local maxima (squares) and selected BCG (circle).
    Enhanced with UQ support: probability labels and adaptive candidate display.
    
    Args:
        images: Array of images
        targets: Array of target BCG coordinates  
        predictions: Array of predicted BCG coordinates
        all_candidates_list: List of all candidate coordinates for each image
        candidate_scores_list: List of candidate scores for each image (optional)
        indices: List of indices to display (default: first 5)
        save_dir: Optional directory to save plots
        phase: Optional phase indicator for title (e.g., 'CandidateBasedTesting')
        probabilities_list: List of probability arrays for each image (for UQ mode)
        detection_threshold: Probability threshold for detections (for UQ mode)
        use_uq: Whether to use UQ-specific visualization features
        metadata_list: List of metadata dictionaries for each image (for cluster names)
    """
    if indices is None:
        indices = range(min(5, len(images)))
    
    for i, idx in enumerate(indices):
        if idx >= len(images):
            continue
            
        image = images[idx]
        target = targets[idx]
        prediction = predictions[idx]
        candidates = all_candidates_list[idx] if idx < len(all_candidates_list) else []
        scores = candidate_scores_list[idx] if candidate_scores_list and idx < len(candidate_scores_list) else np.array([])
        probabilities = probabilities_list[idx] if probabilities_list and idx < len(probabilities_list) else np.array([])
        
        # Calculate distance between target and prediction
        distance = np.sqrt(np.sum((target - prediction)**2))
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Display image (ensure proper format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            display_image = np.clip(image.astype(np.uint8), 0, 255)
        else:
            # Grayscale or other format
            display_image = image
        
        plt.imshow(display_image)
        
        # Enhanced UQ visualization
        if use_uq and len(probabilities) > 0 and len(candidates) > 0:
            # For rank-based visualization, show enough candidates to see the relevant ranks
            # Determine number of candidates based on filename/phase to ensure rank visibility
            if phase and 'rank3' in phase:
                n_candidates_to_show = max(3, min(5, len(candidates)))  # Show at least 3 for rank3
            elif phase and ('rank2' in phase or 'rank1' in phase):
                n_candidates_to_show = max(2, min(3, len(candidates)))  # Show at least 2 for rank2/rank1
            else:
                # Default logic for non-rank-specific phases
                max_prob = np.max(probabilities)
                if max_prob >= 0.85:
                    n_candidates_to_show = min(2, len(candidates))  # Always show at least 2
                elif max_prob >= 0.6:
                    n_candidates_to_show = min(3, len(candidates))
                else:
                    n_candidates_to_show = min(5, len(candidates))  # Show more when uncertain
            
            # Get top candidates by probability
            top_indices = np.argsort(probabilities)[-n_candidates_to_show:][::-1]
            colors = ["#FF0000", "#ff7b00", "#ff9900", "#ffe100", "#b7ff00"]  # More colors for up to 5 ranks
            
            # Plot all candidates as light gray squares first
            candidates_array = np.array(candidates)
            plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                      marker='s', s=250, facecolors='none', edgecolors='#E0E0E0', 
                      linewidths=2, alpha=0.5)
            
            # Plot top candidates with different colors and probability labels
            for rank, cand_idx in enumerate(top_indices):
                if cand_idx >= len(candidates):
                    continue
                    
                candidate = candidates[cand_idx]
                prob = probabilities[cand_idx]
                color = colors[rank] if rank < len(colors) else 'red'
                # size = 800 - rank * 50  # Decreasing size for lower ranks
                
                # Plot candidate circle
                plt.scatter(candidate[0], candidate[1], marker='o', s=800, 
                           facecolors='none', edgecolors=color, linewidths=2, alpha=0.9,
                           label=f'Candidate {rank+1}' if rank < 3 else None)
                
                # Add probability label
                plt.text(candidate[0] + 8, candidate[1] - 8, f'{prob:.2f}', 
                        fontsize=10, color='k', weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.5))
                
            # Update label for best candidate
            if len(top_indices) > 0:
                best_idx = top_indices[0]
                best_prob = probabilities[best_idx]
                # plt.scatter([], [], marker='o', s=400, 
                #            facecolors='none', edgecolors="red", linewidths=2, alpha=0.9,
                #            label=f'Best BCG (p={best_prob:.2f})')
        else:
            # Traditional visualization
            # Plot all candidates as grey squares (transparent with edges only)
            if len(candidates) > 0:
                candidates_array = np.array(candidates)
                plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                          marker='s', s=250, facecolors='none', edgecolors='#E0E0E0', 
                          linewidths=2, alpha=0.5,
                          label=f'Candidates ({len(candidates)})')
            
            # Plot selected BCG (prediction) as red circle (transparent with edges only)
            plt.scatter(prediction[0], prediction[1], marker='o', s=400, 
                       facecolors='none', edgecolors='red', linewidths=3, alpha=0.9,
                       label='Predicted BCG')
        
        # Always plot true BCG location as yellow circle (transparent with edges only)
        plt.scatter(target[0], target[1], marker='o', s=950, 
                   facecolors='none', edgecolors="#59F5ED", linewidths=3, alpha=1.0, ls='dashed',
                   label='True BCG')
        
        # Add title with information including cluster name
        cluster_name = 'Unknown'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            cluster_name = metadata_list[idx].get('cluster_name', 'Unknown')
        
        title = f'Candidate-Based BCG Prediction - Sample {idx+1}'
        if phase:
            title = f'{phase} - Sample {idx+1}'
        
        # Add cluster name as second line
        cluster_line = f'{cluster_name}'
        subtitle = f'Distance: {distance:.1f} px | Candidates: {len(candidates)}'
        if use_uq and len(probabilities) > 0:
            max_prob = np.max(probabilities)
            n_detections = np.sum(probabilities >= detection_threshold)
            subtitle += f' | Max Prob: {max_prob:.3f} | Detections: {n_detections} (≥{detection_threshold:.2f})'
        elif len(scores) > 0:
            max_score = np.max(scores)
            avg_score = np.mean(scores)
            subtitle += f' | Selected Score: {max_score:.3f} | Avg Score: {avg_score:.3f}'
        
        plt.title(f'{title}\n{cluster_line}\n{subtitle}', fontsize=12)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.axis('off')
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            phase_str = f"{phase}_" if phase else ""
            filename = f'{phase_str}prediction_sample_{idx+1}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Candidate-based prediction plot saved: {save_path}")
        
        plt.show()
        plt.close()


def show_predictions(images, targets, predictions, indices=None, save_dir=None, phase=None):
    """
    Show images with both target and predicted BCG locations.
    
    Args:
        images: Array of images
        targets: Array of target BCG coordinates
        predictions: Array of predicted BCG coordinates
        indices: List of indices to display (default: first 5)
        save_dir: Optional directory to save plots
        phase: Optional phase indicator for title
    """
    if indices is None:
        indices = range(min(5, len(images)))
    
    for i, idx in enumerate(indices):
        if idx >= len(images):
            continue
            
        image = images[idx]
        target = targets[idx]
        prediction = predictions[idx]
        
        # Calculate distance between target and prediction
        distance = np.sqrt(np.sum((target - prediction)**2))
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Display image
        display_image = np.clip(image.astype(np.uint8), 0, 255)
        plt.imshow(display_image)
        
        # Plot predicted BCG as red circle
        plt.scatter(prediction[0], prediction[1], marker='o', s=300, 
                   facecolors='none', edgecolors='red', linewidths=3,
                   label='Predicted BCG')
        
        # Plot true BCG location as yellow circle
        plt.scatter(target[0], target[1], marker='o', s=200, 
                   facecolors='none', edgecolors='yellow', linewidths=3,
                   label='True BCG')
        
        # Add title with information
        title = f'BCG Prediction - Sample {idx+1}'
        if phase:
            title = f'{phase} - Sample {idx+1}'
        
        plt.title(f'{title}\nDistance: {distance:.1f} px', fontsize=12)
        plt.legend()
        plt.axis('off')
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            phase_str = f"{phase}_" if phase else ""
            filename = f'{phase_str}prediction_sample_{idx+1}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prediction plot saved: {save_path}")
        
        plt.show()
        plt.close()


def find_failed(targets, predictions, threshold=50):
    """
    Find indices of failed predictions based on distance threshold.
    
    Args:
        targets: Array of target coordinates
        predictions: Array of predicted coordinates  
        threshold: Distance threshold for failure (pixels)
        
    Returns:
        List of indices where distance > threshold
    """
    distances = [np.sqrt(np.sum((target - pred)**2)) for target, pred in zip(targets, predictions)]
    failed_indices = [i for i, dist in enumerate(distances) if dist > threshold]
    return failed_indices


def show_failures(images, targets, predictions, threshold=50, max_failures=5, save_dir=None, phase=None, metadata_list=None,
                 all_candidates_list=None, candidate_scores_list=None, probabilities_list=None, detection_threshold=0.5, use_uq=False):
    """
    Show worst prediction failures with enhanced candidate visualization.
    
    Args:
        images: Array of images
        targets: Array of target coordinates
        predictions: Array of predicted coordinates
        threshold: Distance threshold for considering a failure
        max_failures: Maximum number of failures to show
        save_dir: Optional directory to save plots
        phase: Optional phase indicator for title
        metadata_list: Optional metadata for each sample
        all_candidates_list: List of all candidate coordinates for each image
        candidate_scores_list: List of candidate scores for each image (optional)
        probabilities_list: List of probability arrays for each image (for UQ mode)
        detection_threshold: Probability threshold for detections (for UQ mode)
        use_uq: Whether to use UQ-specific visualization features
    """
    # Calculate distances and find worst cases
    distances = []
    for target, pred in zip(targets, predictions):
        if np.any(np.isnan(pred)):
            # Complete failure case (no candidates found)
            distances.append(float('inf'))
        else:
            distances.append(np.sqrt(np.sum((target - pred)**2)))
    
    # Separate infinite and finite distance cases
    infinite_indices = [i for i, d in enumerate(distances) if np.isinf(d)]
    finite_indices = [i for i, d in enumerate(distances) if np.isfinite(d) and d > threshold]
    
    # Sort finite cases by distance (worst first)
    finite_indices = sorted(finite_indices, key=lambda i: distances[i], reverse=True)
    
    # Combine: infinite distance failures first, then worst finite failures
    failure_indices = infinite_indices + finite_indices
    failure_indices = failure_indices[:max_failures]
    
    if not failure_indices:
        print(f"No failures found with threshold {threshold} pixels")
        return
    
    print(f"Showing {len(failure_indices)} worst failures:")
    
    for idx in failure_indices:
        image = images[idx]
        target = targets[idx]
        prediction = predictions[idx]
        distance = distances[idx]
        
        plt.figure(figsize=(8, 6))
        
        # Display image
        display_image = np.clip(image.astype(np.uint8), 0, 255)
        plt.imshow(display_image)
        
        # Get candidate information for this sample
        candidates = all_candidates_list[idx] if all_candidates_list and idx < len(all_candidates_list) else []
        scores = candidate_scores_list[idx] if candidate_scores_list and idx < len(candidate_scores_list) else np.array([])
        probabilities = probabilities_list[idx] if probabilities_list and idx < len(probabilities_list) else np.array([])
        
        # Enhanced UQ visualization (same as successful cases)
        if use_uq and len(probabilities) > 0 and len(candidates) > 0:
            # Determine number of candidates to show based on max probability
            max_prob = np.max(probabilities)
            if max_prob >= 0.85:
                n_candidates_to_show = 1
            elif max_prob >= 0.6:
                n_candidates_to_show = 2
            else:
                n_candidates_to_show = min(3, len(candidates))
            
            # Get top candidates by probability
            top_indices = np.argsort(probabilities)[-n_candidates_to_show:][::-1]
            colors = ["#FF0000", "#ff7b00", "#ff9900", "#ffe100", "#b7ff00"]  # More colors for up to 5 ranks
            
            # Plot all candidates as light gray squares first
            candidates_array = np.array(candidates)
            plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                      marker='s', s=250, facecolors='none', edgecolors='#E0E0E0', 
                      linewidths=2, alpha=0.5)
            
            # Plot top candidates with different colors and probability labels
            for rank, cand_idx in enumerate(top_indices):
                if cand_idx >= len(candidates):
                    continue
                    
                candidate = candidates[cand_idx]
                prob = probabilities[cand_idx]
                color = colors[rank] if rank < len(colors) else 'red'
                
                # Plot candidate circle
                plt.scatter(candidate[0], candidate[1], marker='o', s=800, 
                           facecolors='none', edgecolors=color, linewidths=2, alpha=0.9,
                           label=f'Candidate {rank+1}' if rank < 3 else None)
                
                # Add probability label
                plt.text(candidate[0] + 8, candidate[1] - 8, f'{prob:.2f}', 
                        fontsize=10, color='k', weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.5))
                
            # Update label for best candidate
            if len(top_indices) > 0:
                best_idx = top_indices[0]
                best_prob = probabilities[best_idx]
                # plt.scatter([], [], marker='o', s=400, 
                #            facecolors='none', edgecolors="red", linewidths=2, alpha=0.9,
                #            label=f'Best BCG (p={best_prob:.2f})')
        else:
            # Traditional visualization or fallback
            # Plot all candidates as gray squares (if available)
            if len(candidates) > 0:
                candidates_array = np.array(candidates)
                plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                          marker='s', s=250, facecolors='none', edgecolors='#E0E0E0', 
                          linewidths=2, alpha=0.5,
                          label=f'Candidates ({len(candidates)})')
            
            # Plot predicted BCG (if valid coordinates)
            if not np.any(np.isnan(prediction)):
                plt.scatter(prediction[0], prediction[1], marker='o', s=800, 
                           facecolors='none', edgecolors='red', linewidths=2, alpha=0.9,
                           label='Predicted BCG')
            else:
                # For complete failures, add text indicating no prediction
                plt.text(0.05, 0.95, 'NO PREDICTION\n(No candidates found)', 
                        transform=plt.gca().transAxes, fontsize=12, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Always plot true BCG location as blue dashed circle (same as successful cases)
        plt.scatter(target[0], target[1], marker='o', s=950, 
                   facecolors='none', edgecolors="#59F5ED", linewidths=3, alpha=1.0, ls='dashed',
                   label='True BCG')
        
        # Add title with failure information and cluster name
        cluster_name = 'Unknown'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            cluster_name = metadata_list[idx].get('cluster_name', 'Unknown')
        
        title = f'Prediction Failure - Sample {idx+1} ({cluster_name})'
        if phase:
            title = f'{phase} Failure - Sample {idx+1} ({cluster_name})'
        
        # Create enhanced subtitle with candidate and probability information (same as successful cases)
        subtitle = f'Distance: {distance:.1f} px | Candidates: {len(candidates)}'
        if np.isinf(distance):
            subtitle = f'Complete Failure (No candidates) | Candidates: {len(candidates)}'
            
        if use_uq and len(probabilities) > 0:
            max_prob = np.max(probabilities)
            n_detections = np.sum(probabilities >= detection_threshold)
            subtitle += f' | Max Prob: {max_prob:.3f} | Detections: {n_detections} (≥{detection_threshold:.2f})'
        elif len(scores) > 0:
            max_score = np.max(scores)
            avg_score = np.mean(scores)
            subtitle += f' | Selected Score: {max_score:.3f} | Avg Score: {avg_score:.3f}'
            
        plt.title(f'{title}\n{subtitle}', fontsize=12, color='red')
        plt.legend()
        plt.axis('off')
        
        # Save plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            phase_str = f"{phase}_" if phase else ""
            filename = f'{phase_str}failure_sample_{idx+1}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Failure plot saved: {save_path}")
        
        plt.show()
        plt.close()