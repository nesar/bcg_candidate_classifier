"""
Model Loading Utilities

Handles loading of trained models, scalers, and preprocessing components.
"""

import os
import torch
import joblib
import sys
from pathlib import Path


class ModelLoader:
    """Loads trained BCG classification models and preprocessing components."""

    def __init__(self, model_dir):
        """
        Initialize model loader.

        Parameters:
        -----------
        model_dir : str
            Directory containing trained model files:
            - best_probabilistic_classifier.pth (or .pt)
            - best_probabilistic_classifier_scaler.pkl
            - best_probabilistic_classifier_color_extractor.pkl (if color features used)
        """
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.model = None
        self.scaler = None
        self.color_extractor = None
        self.device = None

    def _get_parent_module_path(self):
        """Get path to parent bcg_candidate_classifier module."""
        # Assume bcg_deployment is a subdirectory of bcg_candidate_classifier
        parent_dir = self.model_dir.parent.parent
        if not parent_dir.exists():
            raise RuntimeError(
                f"Cannot find parent module directory. "
                f"Expected bcg_candidate_classifier at: {parent_dir}"
            )
        return parent_dir

    def _setup_module_paths(self):
        """Add parent module to Python path for imports."""
        parent_path = str(self._get_parent_module_path())
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)

    def load_model(self, model_name='best_probabilistic_classifier', use_gpu=True):
        """
        Load trained model from disk.

        Parameters:
        -----------
        model_name : str
            Name of model file (without extension)
        use_gpu : bool
            Whether to use GPU if available

        Returns:
        --------
        model : torch.nn.Module
            Loaded PyTorch model in eval mode
        device : torch.device
            Device the model is loaded on
        """
        # Setup module paths for imports
        self._setup_module_paths()

        # Import model class
        try:
            from ml_models.uq_classifier import BCGProbabilisticClassifier
        except ImportError as e:
            raise ImportError(
                f"Failed to import BCGProbabilisticClassifier: {e}\n"
                f"Make sure the bcg_candidate_classifier module is in the parent directory."
            )

        # Find model file
        model_path = None
        for ext in ['.pth', '.pt']:
            candidate = self.model_dir / f"{model_name}{ext}"
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Model file not found: {model_name}.[pth|pt] in {self.model_dir}"
            )

        # Determine device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Load model checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Determine feature dimension from first layer
        if 'network.0.weight' in state_dict:
            feature_dim = state_dict['network.0.weight'].shape[1]
            print(f"Model expects {feature_dim} input features")
        else:
            raise ValueError("Cannot determine feature dimension from model")

        # Create model instance
        self.model = BCGProbabilisticClassifier(
            feature_dim=feature_dim,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2
        )

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully (feature_dim={feature_dim})")
        return self.model, self.device

    def load_scaler(self, model_name='best_probabilistic_classifier'):
        """
        Load feature scaler.

        Parameters:
        -----------
        model_name : str
            Name of model (scaler file should be {model_name}_scaler.pkl)

        Returns:
        --------
        scaler : sklearn.preprocessing.StandardScaler
            Fitted feature scaler
        """
        scaler_path = self.model_dir / f"{model_name}_scaler.pkl"

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        print(f"Loading scaler from: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully")

        return self.scaler

    def load_color_extractor(self, model_name='best_probabilistic_classifier'):
        """
        Load color feature extractor (optional, for models trained with color features).

        Parameters:
        -----------
        model_name : str
            Name of model (color extractor file should be {model_name}_color_extractor.pkl)

        Returns:
        --------
        color_extractor : ColorFeatureExtractor or None
            Fitted color feature extractor, or None if not found
        """
        color_extractor_path = self.model_dir / f"{model_name}_color_extractor.pkl"

        if not color_extractor_path.exists():
            print("Color extractor not found (model may not use color features)")
            self.color_extractor = None
            return None

        print(f"Loading color extractor from: {color_extractor_path}")
        self.color_extractor = joblib.load(color_extractor_path)
        print("Color extractor loaded successfully")

        return self.color_extractor

    def load_all(self, model_name='best_probabilistic_classifier', use_gpu=True):
        """
        Load all model components (model, scaler, color extractor).

        Parameters:
        -----------
        model_name : str
            Name of model files (without extension)
        use_gpu : bool
            Whether to use GPU if available

        Returns:
        --------
        dict
            Dictionary containing:
            - 'model': Loaded model
            - 'scaler': Feature scaler
            - 'color_extractor': Color feature extractor (or None)
            - 'device': torch.device
        """
        model, device = self.load_model(model_name, use_gpu)
        scaler = self.load_scaler(model_name)
        color_extractor = self.load_color_extractor(model_name)

        return {
            'model': model,
            'scaler': scaler,
            'color_extractor': color_extractor,
            'device': device
        }
