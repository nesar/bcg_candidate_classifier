"""
BCG Inference Engine

Main inference class that orchestrates model loading, image processing,
and prediction generation.
"""

import numpy as np
import torch
import sys
from pathlib import Path

from .model_loader import ModelLoader
from .image_processor import ImageProcessor
from .output_generator import OutputGenerator


class BCGInference:
    """Main inference engine for BCG classification."""

    def __init__(self, model_dir, image_dir, candidates_csv, output_dir, use_gpu=True):
        """
        Initialize BCG inference engine.

        Parameters:
        -----------
        model_dir : str
            Directory containing trained model files
        image_dir : str
            Directory containing cluster images
        candidates_csv : str
            Path to BCG candidates CSV file
        output_dir : str
            Directory to save results
        use_gpu : bool
            Whether to use GPU if available
        """
        self.model_dir = Path(model_dir)
        self.image_dir = Path(image_dir)
        self.candidates_csv = candidates_csv
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu

        # Initialize components
        self.model_loader = ModelLoader(model_dir)
        self.image_processor = ImageProcessor(image_dir, candidates_csv)
        self.output_generator = OutputGenerator(output_dir)

        # Model components (loaded later)
        self.model = None
        self.scaler = None
        self.color_extractor = None
        self.device = None

        # Add parent module to path for imports
        self._setup_module_paths()

    def _setup_module_paths(self):
        """Add parent module to Python path."""
        parent_path = str(self.model_dir.parent.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)

    def load_models(self, model_name='best_probabilistic_classifier'):
        """
        Load all model components.

        Parameters:
        -----------
        model_name : str
            Name of model files (without extension)
        """
        print("Loading model components...")
        components = self.model_loader.load_all(model_name, self.use_gpu)

        self.model = components['model']
        self.scaler = components['scaler']
        self.color_extractor = components['color_extractor']
        self.device = components['device']

        print("All model components loaded successfully")

    def _extract_features_for_candidates(self, image, candidates_df):
        """
        Extract features for DESprior candidates.

        Parameters:
        -----------
        image : numpy.ndarray
            Image array
        candidates_df : pandas.DataFrame
            Candidates for this image

        Returns:
        --------
        numpy.ndarray
            Feature array for all candidates
        """
        # Import feature extraction utilities
        from utils.candidate_based_bcg import extract_candidate_features

        # Get candidate coordinates
        candidates = candidates_df[['x', 'y']].values

        # Get candidate-specific features
        candidate_features = candidates_df[['delta_mstar', 'starflag']].values

        # Extract visual features
        use_color = self.color_extractor is not None
        visual_features, _ = extract_candidate_features(
            image, candidates,
            patch_size=64,
            include_context=True,
            include_color=use_color,
            color_extractor=self.color_extractor
        )

        # Combine visual and candidate-specific features
        features = np.hstack([visual_features, candidate_features])

        return features

    def predict_single_image(self, image_data, detection_threshold=0.5):
        """
        Run inference on a single image.

        Parameters:
        -----------
        image_data : dict
            Image data dictionary from ImageProcessor
        detection_threshold : float
            Probability threshold for detections

        Returns:
        --------
        dict
            Inference results containing:
            - 'filename': Image filename
            - 'pred_x', 'pred_y': Best candidate coordinates
            - 'rank': Rank of best candidate (always 1 for best)
            - 'probability': BCG probability
            - 'uncertainty': Uncertainty estimate
            - 'n_candidates': Total candidates
            - 'n_detections': Candidates above threshold
            - 'all_candidates': List of all candidate coords
            - 'all_probabilities': List of all probabilities
            - 'all_uncertainties': List of all uncertainties
        """
        image = image_data['image']
        filename = image_data['filename']
        candidates_df = image_data.get('candidates')

        # Initialize result
        result = {
            'filename': filename,
            'pred_x': np.nan,
            'pred_y': np.nan,
            'rank': np.nan,
            'probability': np.nan,
            'uncertainty': np.nan,
            'n_candidates': 0,
            'n_detections': 0,
            'all_candidates': [],
            'all_probabilities': [],
            'all_uncertainties': []
        }

        if candidates_df is None or len(candidates_df) == 0:
            print(f"No candidates for {filename}")
            return result

        # Extract features
        try:
            features = self._extract_features_for_candidates(image, candidates_df)
        except Exception as e:
            print(f"Error extracting features for {filename}: {e}")
            return result

        # Scale features
        scaled_features = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(scaled_features).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_with_uncertainty'):
                # UQ model
                probabilities, uncertainties = self.model.predict_with_uncertainty(features_tensor)
                probabilities = probabilities.cpu().numpy()
                uncertainties = uncertainties.cpu().numpy()
            else:
                # Standard model
                logits = self.model(features_tensor).squeeze(-1)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                uncertainties = np.zeros_like(probabilities)

        # Get candidate coordinates
        all_candidates = candidates_df[['x', 'y']].values

        # Find best candidate
        best_idx = np.argmax(probabilities)
        best_candidate = all_candidates[best_idx]

        # Count detections above threshold
        n_detections = np.sum(probabilities >= detection_threshold)

        # Update result
        result.update({
            'pred_x': best_candidate[0],
            'pred_y': best_candidate[1],
            'rank': 1,  # Best candidate is always rank 1
            'probability': probabilities[best_idx],
            'uncertainty': uncertainties[best_idx],
            'n_candidates': len(all_candidates),
            'n_detections': int(n_detections),
            'all_candidates': all_candidates.tolist(),
            'all_probabilities': probabilities.tolist(),
            'all_uncertainties': uncertainties.tolist()
        })

        return result

    def run_inference(self, model_name='best_probabilistic_classifier',
                     detection_threshold=0.5, save_images=True):
        """
        Run inference on all images.

        Parameters:
        -----------
        model_name : str
            Name of model files
        detection_threshold : float
            Probability threshold for detections
        save_images : bool
            Whether to save annotated images

        Returns:
        --------
        dict
            Dictionary containing:
            - 'results': List of per-image results
            - 'output_files': Paths to generated output files
        """
        # Load models
        self.load_models(model_name)

        # Validate images
        print("\nValidating images...")
        validation_report = self.image_processor.validate_images()
        print(f"Found {validation_report['n_images']} images")

        if validation_report['size_issues']:
            print(f"Warning: {len(validation_report['size_issues'])} images have size issues")

        if validation_report['missing_candidates']:
            print(f"Warning: {len(validation_report['missing_candidates'])} images have no candidates")

        # Run inference
        print("\nRunning inference...")
        results = []
        images_dict = {}

        for image_data in self.image_processor.load_all_images():
            filename = image_data['filename']
            print(f"Processing: {filename}")

            result = self.predict_single_image(image_data, detection_threshold)
            results.append(result)

            # Store image for visualization
            if save_images:
                images_dict[filename] = image_data['image']

        # Generate outputs
        print("\nGenerating outputs...")
        output_files = self.output_generator.generate_all_outputs(
            results, images_dict, detection_threshold
        )

        # Print summary
        self.output_generator.print_summary(results)

        return {
            'results': results,
            'output_files': output_files
        }


def run_inference_from_config(config):
    """
    Run inference from a configuration dictionary.

    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - 'model_dir': Path to model directory
        - 'image_dir': Path to image directory
        - 'candidates_csv': Path to candidates CSV
        - 'output_dir': Path to output directory
        - 'model_name': Name of model files (optional)
        - 'detection_threshold': Detection threshold (optional)
        - 'use_gpu': Whether to use GPU (optional)
        - 'save_images': Whether to save annotated images (optional)

    Returns:
    --------
    dict
        Inference results and output files
    """
    # Extract config
    model_dir = config['model_dir']
    image_dir = config['image_dir']
    candidates_csv = config['candidates_csv']
    output_dir = config['output_dir']

    model_name = config.get('model_name', 'best_probabilistic_classifier')
    detection_threshold = config.get('detection_threshold', 0.5)
    use_gpu = config.get('use_gpu', True)
    save_images = config.get('save_images', True)

    # Create inference engine
    inference = BCGInference(
        model_dir=model_dir,
        image_dir=image_dir,
        candidates_csv=candidates_csv,
        output_dir=output_dir,
        use_gpu=use_gpu
    )

    # Run inference
    return inference.run_inference(
        model_name=model_name,
        detection_threshold=detection_threshold,
        save_images=save_images
    )
