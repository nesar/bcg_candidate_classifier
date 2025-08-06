# BCG Candidate Classifier

A streamlined implementation of candidate-based Brightest Cluster Galaxy (BCG) detection using deep learning. This repository focuses exclusively on the candidate-based classification approach, where local bright spots are first identified and then ranked/classified to find the best BCG.

## Overview

Instead of direct coordinate regression, this approach:

1. **Finds Candidates**: Identifies bright spots (local maxima) in astronomical images
2. **Extracts Features**: Computes rich feature vectors around each candidate location
3. **Classifies/Ranks**: Uses a neural network to score candidates and select the best BCG

This method is more robust as it constrains predictions to actual bright regions in the image.

## Repository Structure

```
bcg_candidate_classifier/
├── data/                          # Data handling modules
│   ├── data_read.py              # Image loading and coordinate conversion  
│   ├── candidate_dataset.py      # Dataset classes for candidate-based training
│   └── __init__.py
├── ml_models/                     # Machine learning models
│   ├── candidate_classifier.py   # BCG candidate classifier network
│   └── __init__.py
├── utils/                         # Core utilities
│   ├── candidate_based_bcg.py    # Candidate finding and feature extraction
│   ├── viz_bcg.py                # Visualization functions
│   └── __init__.py
├── train.py                      # Training script
├── test.py                       # Evaluation script  
├── full_run.py                   # Complete workflow demonstration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

### Core Components

- **Candidate Detection**: Robust local maxima finding with non-maximum suppression
- **Feature Extraction**: Rich 30-dimensional feature vectors including:
  - Intensity statistics (mean, std, max, min, median)
  - Central vs peripheral intensity ratios
  - Gradient/edge features  
  - Position features relative to image center
  - Shape/symmetry measures via image moments
  - Multi-scale contextual features

- **Neural Classification**: Configurable MLP for candidate ranking
- **Robust Training**: Handles variable numbers of candidates per image
- **Comprehensive Evaluation**: Distance metrics and success rate analysis

### Data Handling

- **WCS Support**: Automatic conversion from RA/Dec to pixel coordinates using embedded WCS headers
- **Multi-format Support**: Works with SPT3G_1500d and megadeep500 dataset formats
- **Quality Filtering**: Basic validation of coordinates and image data
- **Data Splitting**: Reproducible train/validation/test splits

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Training a Model

```bash
python train.py \
    --image_dir /path/to/tif/images/ \
    --truth_table /path/to/truth_table.csv \
    --dataset_type SPT3G_1500d \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --plot
```

**Key Parameters:**
- `--min_distance 15`: Minimum pixels between candidates
- `--threshold_rel 0.12`: Relative intensity threshold for candidate detection
- `--max_candidates 25`: Maximum candidates per image

### 2. Evaluating a Trained Model

```bash
python test.py \
    --model_path ./trained_models/best_candidate_classifier.pth \
    --scaler_path ./trained_models/best_candidate_classifier_scaler.pkl \
    --image_dir /path/to/tif/images/ \
    --truth_table /path/to/truth_table.csv \
    --show_samples 5 \
    --show_failures 3 \
    --save_results
```

### 3. Complete Workflow Demo

```bash
python full_run.py
```

The demo script will:
1. Prompt you to select dataset (SPT3G_1500d or megadeep500)  
2. Allow customization of candidate detection and training parameters
3. Automatically train the candidate-based classifier
4. Test the model and generate comprehensive visualizations
5. Save all results with timestamps

## Data Format

### Input Images
- **Format**: Multi-frame TIFF files with WCS headers
- **Size**: 512×512 pixels
- **Channels**: RGB (3 channels)
- **Frame**: Uses frame 1 (not frame 0) from multi-frame TIFFs
- **WCS**: World Coordinate System information embedded in TIFF headers

### Truth Tables
CSV files with BCG coordinates:

**SPT3G_1500d format:**
```csv
Cluster name,RA,Dec,z,other_columns...
SPT-CLJ0000-0000,123.456,-45.678,0.5,...
```

**megadeep500 format:**
```csv
Cluster name,BCG RA,BCG Dec,other_columns...
cluster001,123.456,-45.678,...
```

## Model Architecture

### Candidate Detection
- Local maxima detection with configurable parameters
- Non-maximum suppression to avoid crowded candidates
- Border exclusion to avoid edge artifacts

### Feature Engineering
30-dimensional feature vectors per candidate including:
- **Intensity**: Mean, std, max, min, median of patch
- **Structure**: Central/peripheral intensity ratios
- **Edges**: Gradient magnitude statistics
- **Position**: Relative location within image
- **Shape**: Symmetry measures via image moments
- **Context**: Multi-scale neighborhood features

### Classification Network
- Configurable MLP (default: 128→64→32→1)
- ReLU activations with dropout (0.2)
- Output: Single score per candidate
- Loss: Cross-entropy over candidates per image

## Performance Metrics

The system reports:
- **Distance Errors**: Mean, median, std of pixel distances
- **Success Rates**: Percentage within 10, 20, 30, 50 pixel thresholds
- **Candidate Statistics**: Average candidates found per image
- **Classification Accuracy**: Correct candidate selection rate

## Advantages of Candidate-Based Approach

1. **Spatial Constraints**: Predictions limited to bright regions
2. **Interpretable Features**: Rich, astronomy-motivated feature set
3. **Robust Training**: Less sensitive to coordinate system issues
4. **Failure Analysis**: Easy to identify and diagnose failure modes
5. **Extensible**: Easy to add new features or change candidate detection

## Output Files

### Training
- `best_candidate_classifier.pth`: Best model weights
- `best_candidate_classifier_scaler.pkl`: Feature scaler
- `training_curves.png`: Loss and accuracy plots

### Evaluation
- `CandidateBasedTesting_prediction_sample_*.png`: Visualization of best predictions
- `CandidateBasedTesting_failure_sample_*.png`: Visualization of worst cases
- `evaluation_results.csv`: Detailed per-sample results

## Comparison with Original Repository

This repository maintains the same structure as `bcg_with_ml` but focuses exclusively on:
- ✅ Candidate-based classification (no coordinate regression)
- ✅ Same data/, ml_models/, utils/ structure
- ✅ Compatible train.py and test.py scripts
- ✅ Similar full_run.py workflow
- ❌ No ResNet18/ResNet34 coordinate regression models
- ❌ No redshift/SN map multi-input support
- ❌ No uncertainty quantification via dropout

## Dependencies

- `torch>=2.0.0`: Deep learning framework
- `scikit-learn>=1.0.0`: Feature scaling and metrics
- `astropy>=5.0.0`: WCS coordinate conversion
- `matplotlib>=3.4.0`: Plotting and visualization
- `numpy`, `pandas`, `scipy`: Standard scientific computing

## License

[Add appropriate license information]