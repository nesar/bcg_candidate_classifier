"""
Image Processing Utilities

Handles loading and validation of cluster images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


class ImageProcessor:
    """Processes and validates cluster images for BCG inference."""

    # Expected image dimensions for 3.8 arcmin × 3.8 arcmin images
    EXPECTED_SIZE = (400, 400)  # Standard size in pixels
    ARCMIN_SIZE = 3.8  # Physical size in arcminutes

    def __init__(self, image_dir, candidates_csv=None):
        """
        Initialize image processor.

        Parameters:
        -----------
        image_dir : str
            Directory containing cluster images (.tif files)
        candidates_csv : str, optional
            Path to BCG candidates CSV file (DESprior format)
        """
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.candidates_csv = candidates_csv
        self.candidates_df = None

        if candidates_csv is not None:
            self._load_candidates()

    def _load_candidates(self):
        """Load BCG candidates from CSV file."""
        if not Path(self.candidates_csv).exists():
            raise FileNotFoundError(f"Candidates CSV not found: {self.candidates_csv}")

        print(f"Loading candidates from: {self.candidates_csv}")
        self.candidates_df = pd.read_csv(self.candidates_csv)

        # Validate required columns
        required_cols = ['filename', 'x', 'y', 'delta_mstar', 'starflag']
        missing_cols = [col for col in required_cols if col not in self.candidates_df.columns]
        if missing_cols:
            raise ValueError(
                f"Candidates CSV missing required columns: {missing_cols}\n"
                f"Expected format: filename, x, y, delta_mstar, starflag"
            )

        print(f"Loaded {len(self.candidates_df)} candidates for {self.candidates_df['filename'].nunique()} images")

    def get_image_list(self):
        """
        Get list of image files in directory.

        Returns:
        --------
        list
            List of image file paths (.tif files)
        """
        image_files = sorted(self.image_dir.glob("*.tif"))
        if len(image_files) == 0:
            raise FileNotFoundError(f"No .tif files found in {self.image_dir}")

        print(f"Found {len(image_files)} images in {self.image_dir}")
        return image_files

    def load_image(self, image_path):
        """
        Load and validate a single image.

        Parameters:
        -----------
        image_path : str or Path
            Path to image file

        Returns:
        --------
        dict
            Dictionary containing:
            - 'image': numpy array (H, W, C) for RGB or (H, W) for grayscale
            - 'filename': Image filename
            - 'path': Full image path
            - 'shape': Image shape
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = Image.open(image_path)
        image_array = np.array(img)

        # Validate size
        if image_array.shape[:2] != self.EXPECTED_SIZE:
            print(
                f"Warning: Image {image_path.name} has size {image_array.shape[:2]}, "
                f"expected {self.EXPECTED_SIZE} for {self.ARCMIN_SIZE} arcmin × {self.ARCMIN_SIZE} arcmin"
            )

        # Normalize to [0, 1] if needed
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0
        elif image_array.dtype == np.uint16:
            image_array = image_array.astype(np.float32) / 65535.0

        return {
            'image': image_array,
            'filename': image_path.name,
            'path': str(image_path),
            'shape': image_array.shape
        }

    def get_candidates_for_image(self, filename):
        """
        Get BCG candidates for a specific image.

        Parameters:
        -----------
        filename : str
            Image filename

        Returns:
        --------
        pandas.DataFrame or None
            DataFrame with candidates for this image, or None if no candidates CSV loaded
        """
        if self.candidates_df is None:
            return None

        file_candidates = self.candidates_df[self.candidates_df['filename'] == filename]
        return file_candidates if len(file_candidates) > 0 else None

    def load_all_images(self):
        """
        Load all images in directory with their candidates.

        Returns:
        --------
        list
            List of dictionaries, each containing:
            - 'image': Image array
            - 'filename': Filename
            - 'path': Full path
            - 'shape': Image shape
            - 'candidates': DataFrame of candidates (if candidates CSV provided)

        Yields:
        -------
        dict
            Image data dictionary
        """
        image_files = self.get_image_list()

        for image_path in image_files:
            image_data = self.load_image(image_path)

            # Add candidates if available
            if self.candidates_df is not None:
                candidates = self.get_candidates_for_image(image_data['filename'])
                image_data['candidates'] = candidates
            else:
                image_data['candidates'] = None

            yield image_data

    def validate_images(self):
        """
        Validate all images in directory.

        Returns:
        --------
        dict
            Validation report with:
            - 'n_images': Number of images found
            - 'size_issues': List of images with size issues
            - 'format_issues': List of images with format issues
            - 'missing_candidates': List of images without candidates (if candidates CSV provided)
        """
        image_files = self.get_image_list()
        report = {
            'n_images': len(image_files),
            'size_issues': [],
            'format_issues': [],
            'missing_candidates': []
        }

        for image_path in image_files:
            try:
                image_data = self.load_image(image_path)

                # Check size
                if image_data['shape'][:2] != self.EXPECTED_SIZE:
                    report['size_issues'].append(image_data['filename'])

                # Check for candidates
                if self.candidates_df is not None:
                    candidates = self.get_candidates_for_image(image_data['filename'])
                    if candidates is None or len(candidates) == 0:
                        report['missing_candidates'].append(image_data['filename'])

            except Exception as e:
                report['format_issues'].append({
                    'filename': image_path.name,
                    'error': str(e)
                })

        return report
