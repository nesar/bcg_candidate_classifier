"""
BCG Deployment Package

A pip-installable package for deploying trained BCG (Brightest Cluster Galaxy)
classification models on new astronomical cluster images.

This package provides tools for:
- Loading trained models and preprocessing pipelines
- Processing cluster images (3.8 arcmin × 3.8 arcmin)
- Running inference with BCG candidate ranking
- Generating results with uncertainties and visualizations
"""

__version__ = "0.1.0"

from .inference import BCGInference
from .model_loader import ModelLoader
from .image_processor import ImageProcessor
from .output_generator import OutputGenerator

__all__ = [
    'BCGInference',
    'ModelLoader',
    'ImageProcessor',
    'OutputGenerator',
]
