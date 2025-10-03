import os
import numpy as np
import matplotlib.pyplot as plt

# Set style consistent with plot_physical_results.py
plt.rcParams.update({"text.usetex":False,"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":1.2})


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
    plt.legend(fontsize=14)
    
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
        indices = range(min(10, len(images)))
    
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
                        fontsize=14, color='k', #weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6))
                
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
        # Get BCG probability from metadata (ground truth) if available
        true_bcg_label = 'True BCG'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            bcg_prob = metadata_list[idx].get('bcg_prob')
            if bcg_prob is not None and not np.isnan(bcg_prob):
                true_bcg_label = f'True BCG (p={bcg_prob:.3f})'
        
        plt.scatter(target[0], target[1], marker='o', s=950, 
                   facecolors='none', edgecolors="#59F5ED", linewidths=3, alpha=1.0, ls='dashed',
                   label=true_bcg_label)
        
        # Add title with information including cluster name
        cluster_name = 'Unknown'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            cluster_name = metadata_list[idx].get('cluster_name', 'Unknown')
        
        # Show only cluster name in title
        title = f'{cluster_name}'
        
        plt.title(title, fontsize=18)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=14)
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
        indices = range(min(10, len(images)))
    
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
        
        plt.title(f'{title}\nDistance: {distance:.1f} px', fontsize=18)
        plt.legend(fontsize=14)
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


def show_failures(images, targets, predictions, threshold=50, max_failures=10, save_dir=None, phase=None, metadata_list=None,
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
                        fontsize=16, color='k', #weight='bold', 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6))
                
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
                        transform=plt.gca().transAxes, fontsize=18, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Always plot true BCG location as blue dashed circle (same as successful cases)
        # Get BCG probability from metadata (ground truth) if available
        true_bcg_label = 'True BCG'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            bcg_prob = metadata_list[idx].get('bcg_prob')
            if bcg_prob is not None and not np.isnan(bcg_prob):
                true_bcg_label = f'True BCG (p={bcg_prob:.3f})'
        
        plt.scatter(target[0], target[1], marker='o', s=950, 
                   facecolors='none', edgecolors="#59F5ED", linewidths=3, alpha=1.0, ls='dashed',
                   label=true_bcg_label)
        
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
            
        plt.title(f'{title}\n{subtitle}', fontsize=16, color='black')
        plt.legend(fontsize=14)
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


def show_predictions_with_candidates_enhanced(images, targets, predictions, all_candidates_list, candidate_scores_list=None, 
                                            indices=None, save_dir=None, phase=None, probabilities_list=None, 
                                            detection_threshold=0.5, use_uq=False, metadata_list=None, dataset_type="bcg_2p2arcmin"):
    """
    Enhanced version with physical coordinates, improved layout, and separate save directory.
    
    Args:
        dataset_type: Either "bcg_2p2arcmin" or "bcg_3p8arcmin" for proper coordinate scaling
    """
    if indices is None:
        indices = range(min(10, len(images)))
    
    # Calculate pixel to arcmin conversion
    if "2p2arcmin" in dataset_type:
        arcmin_per_pixel = 2.2 / 512  # 2.2 arcmin across 512 pixels
        total_arcmin = 2.2
    elif "3p8arcmin" in dataset_type:
        arcmin_per_pixel = 3.8 / 512  # 3.8 arcmin across 512 pixels  
        total_arcmin = 3.8
    else:
        # Default to 2.2 arcmin
        arcmin_per_pixel = 2.2 / 512
        total_arcmin = 2.2
    
    # Create physical_images subdirectory
    if save_dir:
        physical_save_dir = os.path.join(save_dir, "physical_images")
        os.makedirs(physical_save_dir, exist_ok=True)
    else:
        physical_save_dir = None
    
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
        
        # Create figure with enhanced layout
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display image (ensure proper format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            display_image = np.clip(image.astype(np.uint8), 0, 255)
        else:
            # Grayscale or other format
            display_image = image
        
        ax.imshow(display_image)
        
        # Enhanced UQ visualization with detection threshold filtering
        if use_uq and len(probabilities) > 0 and len(candidates) > 0:
            # First filter candidates by detection threshold
            above_threshold_mask = probabilities >= detection_threshold
            above_threshold_indices = np.where(above_threshold_mask)[0]
            
            if len(above_threshold_indices) > 0:
                # Only show candidates above threshold, up to 10 max
                max_candidates = 10
                n_candidates_to_show = min(max_candidates, len(above_threshold_indices))
                
                # Get top candidates by probability from those above threshold
                above_threshold_probs = probabilities[above_threshold_indices]
                sorted_indices = np.argsort(above_threshold_probs)[-n_candidates_to_show:][::-1]
                top_indices = above_threshold_indices[sorted_indices]
            else:
                # If no candidates above threshold, show top 3 regardless
                max_candidates = 3
                n_candidates_to_show = min(max_candidates, len(candidates))
                top_indices = np.argsort(probabilities)[-n_candidates_to_show:][::-1]
            colors = ["#FF0000", "#ff7b00", "#ff9900", "#ffe100", "#b7ff00", 
                     "#90ee90", "#87ceeb", "#dda0dd", "#f0e68c", "#ffa07a"]  # More colors for 10 ranks
            
            # Plot all candidates as light gray squares first
            candidates_array = np.array(candidates)
            ax.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                      marker='s', s=250, facecolors='none', edgecolors='#E0E0E0', 
                      linewidths=2, alpha=0.5)
            
            # Plot top candidates with different colors and probability labels
            legend_elements = []
            for rank, cand_idx in enumerate(top_indices):
                if cand_idx >= len(candidates):
                    continue
                    
                candidate = candidates[cand_idx]
                prob = probabilities[cand_idx]
                color = colors[rank] if rank < len(colors) else 'red'
                
                # Plot candidate circle
                ax.scatter(candidate[0], candidate[1], marker='o', s=800, 
                          facecolors='none', edgecolors=color, linewidths=2, alpha=0.9)
                
                # Add probability label
                ax.text(candidate[0] + 8, candidate[1] - 8, f'{prob:.2f}', 
                       fontsize=14, color='red', 
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.2))
                
                # Create legend entry
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markeredgecolor=color, markersize=12, 
                                                markeredgewidth=2, linestyle='None',
                                                label=f'Rank {rank+1} ({prob:.2f})'))
        else:
            # Standard visualization for non-UQ mode
            legend_elements = []
            if len(candidates) > 0:
                candidates_array = np.array(candidates)
                ax.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                          marker='s', s=300, facecolors='none', edgecolors='orange', 
                          linewidths=2, alpha=0.7)
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                                markeredgecolor='orange', markersize=12, 
                                                markeredgewidth=2, linestyle='None', label='Candidates'))
            
            # Plot selected BCG (prediction) as red circle
            ax.scatter(prediction[0], prediction[1], marker='o', s=400, 
                      facecolors='none', edgecolors='red', linewidths=3, alpha=0.9)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markeredgecolor='red', markersize=12, 
                                            markeredgewidth=3, linestyle='None', label='Predicted BCG'))
        
        # Always plot true BCG location
        true_bcg_label = 'True BCG'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            bcg_prob = metadata_list[idx].get('bcg_prob')
            if bcg_prob is not None and not np.isnan(bcg_prob):
                true_bcg_label = f'True BCG (p={bcg_prob:.3f})'
        
        ax.scatter(target[0], target[1], marker='o', s=950, 
                  facecolors='none', edgecolors="#59F5ED", linewidths=3, alpha=1.0, 
                  linestyle='dashed')
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markeredgecolor="#59F5ED", markersize=15, 
                                        markeredgewidth=3, linestyle='None', label=true_bcg_label))
        
        # Get cluster name and redshift for display
        cluster_name = 'Unknown'
        redshift = None
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            metadata = metadata_list[idx]
            cluster_name = metadata.get('cluster_name', 'Unknown')
            redshift = metadata.get('z')
            
            # Debug: print available metadata to help troubleshoot
            print(f"Debug - Cluster: {cluster_name}, Metadata keys: {list(metadata.keys())}")
            print(f"Debug - Redshift value: {redshift}")
        
        # Create display text with cluster name and redshift
        display_text = cluster_name
        if redshift is not None:
            display_text = f"{cluster_name}, z={redshift:.2f}"
        
        # Add cluster name and redshift as text in top-left corner
        ax.text(0.02, 0.98, display_text, transform=ax.transAxes, fontsize=18, 
               weight='bold', verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Check if we have RA/Dec coordinates in metadata for proper physical coordinate display
        use_radec = False
        bcg_ra, bcg_dec = None, None
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            bcg_ra = metadata_list[idx].get('bcg_ra')
            bcg_dec = metadata_list[idx].get('bcg_dec')
            if bcg_ra is not None and bcg_dec is not None:
                use_radec = True
                print(f"Using RA/Dec coordinates: RA={bcg_ra:.6f}, Dec={bcg_dec:.6f}")
        
        if use_radec:
            ax.set_xlabel("RA", fontsize=14)
            ax.set_ylabel("Dec", fontsize=14)
        else:
            # Fall back to relative coordinates
            ax.set_xlabel("Relative Position", fontsize=18)
            ax.set_ylabel("Relative Position", fontsize=18)
        
        # Create coordinate tick labels
        # Center coordinates at image center
        center_pixel = 256  # 512/2
        
        # Create tick positions in pixels - use 5 ticks instead of 6
        # pixel_ticks = np.linspace(0, 512, 5)  # 5 ticks from 0 to 512
        pixel_ticks = np.linspace(128, 384, 3)
        # Convert to arcmin coordinates (centered)
        arcmin_offsets = (pixel_ticks - center_pixel) * arcmin_per_pixel
        
        def format_relative_coordinate(offset_arcmin):
            """Format coordinate as relative arcmin'arcsec" offset from image center"""
            abs_arcmin = abs(offset_arcmin)
            sign = '-' if offset_arcmin < 0 else ''
            
            # Get integer arcmin and fractional part
            arcmin_int = int(abs_arcmin)
            arcmin_frac = abs_arcmin - arcmin_int
            
            # Convert fractional arcmin to arcsec (1 arcmin = 60 arcsec)
            arcsec = arcmin_frac * 60
            
            # Format based on magnitude
            if abs_arcmin < 0.1:  # Less than 6 arcsec, show only arcsec
                return f'{sign}{arcsec:.0f}"'
            elif arcmin_int == 0:  # Less than 1 arcmin, show only arcsec
                return f'{sign}{arcsec:.0f}"'
            elif arcsec < 1:  # Exactly on arcmin boundary
                return f'{sign}{arcmin_int}\''
            else:  # Show both arcmin and arcsec
                return f'{sign}{arcmin_int}\'{arcsec:.0f}"'
        
        ax.set_xticks(pixel_ticks)
        ax.set_yticks(pixel_ticks)
        
        if use_radec and bcg_ra is not None and bcg_dec is not None:
            # Convert pixel positions to RA/Dec coordinates
            # Assume BCG is at center of image for coordinate transformation
            pixel_offsets = pixel_ticks - center_pixel
            arcmin_offsets_radec = pixel_offsets * arcmin_per_pixel
            
            # Convert arcmin offsets to degree offsets (1 arcmin = 1/60 degrees)
            degree_offsets = arcmin_offsets_radec / 60.0
            
            # Calculate RA/Dec values at tick positions
            # Note: RA decreases with increasing x-pixel (East is left)
            ra_values = bcg_ra - degree_offsets  # RA decreases towards East
            dec_values = bcg_dec + degree_offsets  # Dec increases towards North
            
            def format_radec_coordinate(coord_deg, coord_type='ra'):
                """Format RA/Dec coordinate in proper astronomical notation"""
                if coord_type == 'ra':
                    # Convert RA degrees to hours:minutes:seconds
                    hours = coord_deg / 15.0  # 1 hour = 15 degrees
                    h = int(hours)
                    m = int((hours - h) * 60)
                    s = ((hours - h) * 60 - m) * 60
                    return f"{h:02d}h{m:02d}m{s:04.1f}s"
                else:  # dec
                    # Convert Dec degrees to degrees:arcminutes:arcseconds
                    sign = '+' if coord_deg >= 0 else '-'
                    abs_deg = abs(coord_deg)
                    deg = int(abs_deg)
                    arcmin = int((abs_deg - deg) * 60)
                    arcsec = ((abs_deg - deg) * 60 - arcmin) * 60
                    return f"{sign}{deg:02d}°{arcmin:02d}'{arcsec:04.1f}\""
            
            x_labels = [format_radec_coordinate(ra, 'ra') for ra in ra_values]
            y_labels = [format_radec_coordinate(dec, 'dec') for dec in dec_values]
        else:
            # Format tick labels as relative coordinates
            x_labels = [format_relative_coordinate(offset) for offset in arcmin_offsets]
            y_labels = [format_relative_coordinate(offset) for offset in arcmin_offsets]
            
        ax.set_xticklabels(x_labels, fontsize=18)
        ax.set_yticklabels(y_labels, fontsize=18)
        
        # Move legend to bottom-left and make it column-wise with smaller font
        ncol = min(3, len(legend_elements))  # Adaptive column count
        ax.legend(handles=legend_elements, loc='lower left', 
                 bbox_to_anchor=(0.02, 0.02), ncol=ncol, fontsize=12,
                 frameon=True, fancybox=True, shadow=False, framealpha=0.5,
                 columnspacing=0.5, handletextpad=0.3)
        
        # Remove title (cluster name now in corner)
        # Keep axis visible for coordinate reference
        
        # Save enhanced plot
        if physical_save_dir:
            phase_str = f"{phase}_" if phase else ""
            filename = f'{phase_str}prediction_sample_{idx+1}_enhanced.png'
            save_path = os.path.join(physical_save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced prediction plot saved: {save_path}")
        
        plt.show()
        plt.close()
