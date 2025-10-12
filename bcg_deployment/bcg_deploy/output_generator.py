"""
Output Generation Utilities

Handles creation of results files and annotated images.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Circle
import matplotlib


class OutputGenerator:
    """Generates output files and visualizations from inference results."""

    def __init__(self, output_dir):
        """
        Initialize output generator.

        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.images_dir = self.output_dir / "annotated_images"
        self.images_dir.mkdir(exist_ok=True)

    def save_evaluation_results(self, results):
        """
        Save evaluation results to CSV.

        Parameters:
        -----------
        results : list of dict
            List of result dictionaries, each containing:
            - 'filename': Image filename
            - 'pred_x', 'pred_y': Predicted BCG coordinates
            - 'rank': Rank of best candidate
            - 'probability': BCG probability
            - 'uncertainty': Uncertainty estimate
            - 'n_candidates': Number of candidates
            - 'all_candidates': List of all candidate coordinates
            - 'all_probabilities': List of all candidate probabilities
        """
        results_file = self.output_dir / "evaluation_results.csv"

        # Prepare data
        data = []
        for result in results:
            data.append({
                'filename': result['filename'],
                'pred_x': result.get('pred_x', np.nan),
                'pred_y': result.get('pred_y', np.nan),
                'rank': result.get('rank', np.nan),
                'max_probability': result.get('probability', np.nan),
                'max_uncertainty': result.get('uncertainty', np.nan),
                'n_candidates': result.get('n_candidates', 0),
                'n_detections': result.get('n_detections', 0)
            })

        df = pd.DataFrame(data)
        df.to_csv(results_file, index=False)
        print(f"Evaluation results saved to: {results_file}")

        return results_file

    def save_probability_analysis(self, results):
        """
        Save detailed probability analysis to CSV.

        Parameters:
        -----------
        results : list of dict
            List of result dictionaries with candidate-level details
        """
        analysis_file = self.output_dir / "probability_analysis.csv"

        # Prepare detailed per-candidate data
        data = []
        for result in results:
            filename = result['filename']
            all_candidates = result.get('all_candidates', [])
            all_probs = result.get('all_probabilities', [])
            all_uncs = result.get('all_uncertainties', [])

            for idx, (cand, prob) in enumerate(zip(all_candidates, all_probs)):
                unc = all_uncs[idx] if idx < len(all_uncs) else np.nan

                data.append({
                    'filename': filename,
                    'candidate_idx': idx,
                    'x': cand[0],
                    'y': cand[1],
                    'probability': prob,
                    'uncertainty': unc,
                    'is_best_candidate': idx == np.argmax(all_probs) if len(all_probs) > 0 else False
                })

        if data:
            df = pd.DataFrame(data)
            df.to_csv(analysis_file, index=False)
            print(f"Probability analysis saved to: {analysis_file}")
            return analysis_file

        return None

    def create_annotated_image(self, image, result, detection_threshold=0.5):
        """
        Create annotated image with ranked candidates.

        Parameters:
        -----------
        image : numpy.ndarray
            Original image array
        result : dict
            Inference result dictionary
        detection_threshold : float
            Probability threshold for marking detections

        Returns:
        --------
        str
            Path to saved annotated image
        """
        filename = result['filename']
        all_candidates = result.get('all_candidates', [])
        all_probs = result.get('all_probabilities', [])
        all_uncs = result.get('all_uncertainties', [])

        if len(all_candidates) == 0:
            print(f"No candidates to visualize for {filename}")
            return None

        # Create figure
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2
        })

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Show image
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')

        # Sort candidates by probability (descending)
        if len(all_probs) > 0:
            sorted_indices = np.argsort(all_probs)[::-1]
        else:
            sorted_indices = range(len(all_candidates))

        # Annotate candidates
        for rank, idx in enumerate(sorted_indices[:10], start=1):  # Show top 10
            cand = all_candidates[idx]
            prob = all_probs[idx] if idx < len(all_probs) else 0
            unc = all_uncs[idx] if idx < len(all_uncs) else 0

            # Color based on rank
            if rank == 1:
                color = 'red'
                marker_size = 150
            elif rank <= 3:
                color = 'orange'
                marker_size = 100
            else:
                color = 'yellow'
                marker_size = 80

            # Mark candidate
            circle = Circle((cand[0], cand[1]), radius=15,
                          fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(circle)

            # Add label with rank and probability
            label = f"#{rank}: {prob:.3f}"
            if unc > 0:
                label += f"±{unc:.3f}"

            ax.text(cand[0], cand[1] - 20, label,
                   color=color, fontsize=12, fontweight='bold',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_title(f"{filename}\n{len(all_candidates)} candidates, "
                    f"Best prob: {max(all_probs):.3f}" if len(all_probs) > 0 else filename,
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        # Save
        output_path = self.images_dir / f"{Path(filename).stem}_annotated.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_all_outputs(self, results, images, detection_threshold=0.5):
        """
        Generate all output files and visualizations.

        Parameters:
        -----------
        results : list of dict
            Inference results
        images : dict
            Dictionary mapping filenames to image arrays
        detection_threshold : float
            Probability threshold for detections

        Returns:
        --------
        dict
            Paths to generated output files
        """
        print("Generating output files...")

        # Save CSV files
        eval_csv = self.save_evaluation_results(results)
        prob_csv = self.save_probability_analysis(results)

        # Generate annotated images
        print("Creating annotated images...")
        annotated_images = []
        for result in results:
            filename = result['filename']
            if filename in images:
                image = images[filename]
                img_path = self.create_annotated_image(image, result, detection_threshold)
                if img_path:
                    annotated_images.append(img_path)

        print(f"Generated {len(annotated_images)} annotated images")

        return {
            'evaluation_csv': str(eval_csv),
            'probability_csv': str(prob_csv) if prob_csv else None,
            'annotated_images': annotated_images,
            'output_dir': str(self.output_dir)
        }

    def print_summary(self, results):
        """
        Print summary statistics of results.

        Parameters:
        -----------
        results : list of dict
            Inference results
        """
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)

        n_images = len(results)
        n_with_candidates = sum(1 for r in results if r.get('n_candidates', 0) > 0)

        print(f"Processed images: {n_images}")
        print(f"Images with candidates: {n_with_candidates}")

        if n_with_candidates > 0:
            avg_candidates = np.mean([r.get('n_candidates', 0) for r in results])
            avg_prob = np.mean([r.get('probability', 0) for r in results if r.get('probability') is not None])
            avg_unc = np.mean([r.get('uncertainty', 0) for r in results if r.get('uncertainty') is not None])

            print(f"Average candidates per image: {avg_candidates:.1f}")
            print(f"Average best probability: {avg_prob:.3f}")
            print(f"Average uncertainty: {avg_unc:.3f}")

        print("=" * 60)
