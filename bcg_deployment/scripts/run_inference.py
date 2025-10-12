#!/usr/bin/env python3
"""
BCG Inference CLI

Command-line interface for running BCG inference on cluster images.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for bcg_deploy import
sys.path.insert(0, str(Path(__file__).parent.parent))

from bcg_deploy.inference import run_inference_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Run BCG inference on cluster images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_inference.py \\
    --model_dir /path/to/models \\
    --image_dir /path/to/images \\
    --candidates_csv /path/to/candidates.csv \\
    --output_dir /path/to/output

  # With custom detection threshold
  python run_inference.py \\
    --model_dir /path/to/models \\
    --image_dir /path/to/images \\
    --candidates_csv /path/to/candidates.csv \\
    --output_dir /path/to/output \\
    --detection_threshold 0.7

  # CPU-only mode
  python run_inference.py \\
    --model_dir /path/to/models \\
    --image_dir /path/to/images \\
    --candidates_csv /path/to/candidates.csv \\
    --output_dir /path/to/output \\
    --no-gpu

Image Requirements:
  - Format: .tif files
  - Size: 3.8 arcmin × 3.8 arcmin (typically 400×400 pixels)
  - Type: RGB or grayscale

Model Requirements:
  model_dir should contain:
  - best_probabilistic_classifier.pth (or .pt)
  - best_probabilistic_classifier_scaler.pkl
  - best_probabilistic_classifier_color_extractor.pkl (optional)

Candidates CSV Requirements:
  Required columns: filename, x, y, delta_mstar, starflag
  Format: DESprior candidates format
"""
    )

    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model files')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing cluster images (.tif files)')
    parser.add_argument('--candidates_csv', type=str, required=True,
                       help='Path to BCG candidates CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save inference results')

    # Optional arguments
    parser.add_argument('--model_name', type=str, default='best_probabilistic_classifier',
                       help='Name of model files (without extension) (default: best_probabilistic_classifier)')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Probability threshold for BCG detection (0.0-1.0) (default: 0.5)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU usage (use CPU only)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip saving annotated images (only save CSV results)')

    args = parser.parse_args()

    # Validate arguments
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)

    candidates_csv = Path(args.candidates_csv)
    if not candidates_csv.exists():
        print(f"Error: Candidates CSV not found: {args.candidates_csv}")
        sys.exit(1)

    if not (0.0 <= args.detection_threshold <= 1.0):
        print(f"Error: Detection threshold must be between 0.0 and 1.0, got {args.detection_threshold}")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("BCG INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Image directory: {args.image_dir}")
    print(f"Candidates CSV: {args.candidates_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model name: {args.model_name}")
    print(f"Detection threshold: {args.detection_threshold}")
    print(f"GPU enabled: {not args.no_gpu}")
    print(f"Save annotated images: {not args.no_images}")
    print("=" * 60)
    print()

    # Create configuration
    config = {
        'model_dir': args.model_dir,
        'image_dir': args.image_dir,
        'candidates_csv': args.candidates_csv,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'detection_threshold': args.detection_threshold,
        'use_gpu': not args.no_gpu,
        'save_images': not args.no_images
    }

    # Run inference
    try:
        results = run_inference_from_config(config)

        print("\n" + "=" * 60)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nOutput directory: {args.output_dir}")
        print("\nGenerated files:")
        print(f"  - evaluation_results.csv")
        print(f"  - probability_analysis.csv")
        if not args.no_images:
            print(f"  - annotated_images/ (PNG files)")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("INFERENCE FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
