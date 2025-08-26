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


def show_predictions_with_candidates(images, targets, predictions, all_candidates_list, candidate_scores_list=None, indices=None, save_dir=None, phase=None):
    """
    Show images with candidate local maxima (squares) and selected BCG (circle).
    
    Args:
        images: Array of images
        targets: Array of target BCG coordinates  
        predictions: Array of predicted BCG coordinates
        all_candidates_list: List of all candidate coordinates for each image
        candidate_scores_list: List of candidate scores for each image (optional)
        indices: List of indices to display (default: first 5)
        save_dir: Optional directory to save plots
        phase: Optional phase indicator for title (e.g., 'CandidateBasedTesting')
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
        
        # Plot all candidates as cyan squares (transparent with edges only)
        if len(candidates) > 0:
            candidates_array = np.array(candidates)
            plt.scatter(candidates_array[:, 0], candidates_array[:, 1], 
                      marker='s', s=200, facecolors='none', edgecolors='cyan', 
                      linewidths=1, alpha=0.5,
                      label=f'Candidates ({len(candidates)})')
        
        # Plot selected BCG (prediction) as red circle (transparent with edges only)
        plt.scatter(prediction[0], prediction[1], marker='o', s=400, 
                   facecolors='none', edgecolors='red', linewidths=3, alpha=0.9,
                   label='Predicted BCG')
        
        # Plot true BCG location as yellow circle (transparent with edges only)
        plt.scatter(target[0], target[1], marker='o', s=250, 
                   facecolors='none', edgecolors='yellow', linewidths=3, alpha=0.9,
                   label='True BCG')
        
        # Add title with information
        title = f'Candidate-Based BCG Prediction - Sample {idx+1}'
        if phase:
            title = f'{phase} - Sample {idx+1}'
        
        subtitle = f'Distance: {distance:.1f} px | Candidates: {len(candidates)}'
        if len(scores) > 0:
            max_score = np.max(scores)
            avg_score = np.mean(scores)
            subtitle += f' | Selected Score: {max_score:.3f} | Avg Score: {avg_score:.3f}'
        
        plt.title(f'{title}\n{subtitle}', fontsize=12)
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


def show_failures(images, targets, predictions, threshold=50, max_failures=5, save_dir=None, phase=None, metadata_list=None):
    """
    Show worst prediction failures.
    
    Args:
        images: Array of images
        targets: Array of target coordinates
        predictions: Array of predicted coordinates
        threshold: Distance threshold for considering a failure
        max_failures: Maximum number of failures to show
        save_dir: Optional directory to save plots
        phase: Optional phase indicator for title
    """
    # Calculate distances and find worst cases
    distances = [np.sqrt(np.sum((target - pred)**2)) for target, pred in zip(targets, predictions)]
    
    # Get indices sorted by distance (worst first)
    worst_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)
    
    # Filter for actual failures and limit number
    failure_indices = [i for i in worst_indices if distances[i] > threshold][:max_failures]
    
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
        
        # Plot predicted BCG as red X
        plt.scatter(prediction[0], prediction[1], marker='x', s=400, 
                   c='red', linewidths=4, label='Predicted BCG')
        
        # Plot true BCG location as yellow circle
        plt.scatter(target[0], target[1], marker='o', s=200, 
                   facecolors='none', edgecolors='yellow', linewidths=3,
                   label='True BCG')
        
        # Add title with failure information and cluster name
        cluster_name = 'Unknown'
        if metadata_list and idx < len(metadata_list) and metadata_list[idx]:
            cluster_name = metadata_list[idx].get('cluster_name', 'Unknown')
        
        title = f'Prediction Failure - Sample {idx+1} ({cluster_name})'
        if phase:
            title = f'{phase} Failure - Sample {idx+1} ({cluster_name})'
        
        plt.title(f'{title}\nError: {distance:.1f} px', fontsize=12, color='red')
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