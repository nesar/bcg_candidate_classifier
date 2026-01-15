# CCG Candidate Classifier

Implementation of candidate-based Cluster Central Galaxy (CCG) detection using deep learning. This repository focuses exclusively on the candidate-based classification approach, where local bright spots are first identified and then ranked/classified to find the best BCG.

## Overview

Instead of direct coordinate regression, this approach:

1. **Finds Candidates**: Identifies bright spots (local maxima) in astronomical images
2. **Extracts Features**: Computes rich feature vectors around each candidate location
3. **Classifies/Ranks**: Uses a neural network to score candidates and select the best CCG


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

The architecture consists of three main stages: candidate detection, feature extraction, and neural classification.

### 1. Candidate Detection Pipeline

**Local Maxima Detection:**
- Uses 3×3 maximum filter to find bright spots in grayscale image
- Applies relative intensity threshold (default: 12% of max intensity)
- Excludes border regions (default: 30 pixels) to avoid edge artifacts

**Non-Maximum Suppression:**
- Sorts candidates by brightness (descending)
- Enforces minimum separation distance (default: 15 pixels)
- Limits total candidates per image (default: 25 max)
- Returns candidates sorted by brightness

### 2. Feature Extraction

**Morphological Features (30 dimensions):**

1. **Intensity Statistics (5 features)**
   - Mean, std, max, min, median intensity of 64×64 patch

2. **Structural Features (3 features)**
   - Central region mean intensity (16×16 center)
   - Peripheral region mean intensity (annulus around center)
   - Central/peripheral ratio (log-ratio for robustness)

3. **Gradient Features (3 features)**
   - Mean gradient magnitude (edge strength)
   - Std of gradient magnitude (texture variation)
   - Maximum gradient magnitude (sharp edges)

4. **Position Features (3 features)**
   - Normalized x position relative to image center (-1 to 1)
   - Normalized y position relative to image center (-1 to 1)
   - Euclidean distance from image center

5. **Shape/Symmetry Features (3 features)**
   - Normalized centroid offset x (asymmetry measure)
   - Normalized centroid offset y (asymmetry measure)
   - Eccentricity from second-order image moments

6. **Multi-Scale Context Features (9 features)**
   - Mean/std/pixel count at 3 radii: 32px, 64px, 128px
   - Captures neighborhood characteristics at multiple scales

7. **Directional Context Features (4 features)**
   - Mean intensity sampled along 4 directions: up, right, down, left
   - Detects asymmetric brightness patterns

**Optional Color Features (30-56 dimensions before PCA, 8 after):**

When enabled (`include_color=True`), extracts:

1. **Basic Color Statistics (9 features)**
   - Mean R, G, B values
   - Std R, G, B values
   - Relative R, G, B contributions

2. **Color Ratio Features (7 features)**
   - R/G ratio and normalized difference
   - R/B ratio and normalized difference
   - G/B ratio
   - Color magnitude (deviation from white)
   - Red-sequence score (redness indicator)

3. **Spatial Color Features (3 features)**
   - Spatial variation of R/G and R/B ratios
   - Central vs peripheral color difference

4. **Color Gradient Features (8 features)**
   - Mean and std of gradient magnitude per channel (R, G, B)
   - Cross-channel gradient correlations (coherence indicators)

5. **Convolution-Based Features (27 features)**
   - Edge, smooth, and Laplacian kernel responses
   - Per-channel statistics (mean, std, max response)

**PCA Reduction (optional):**
- Reduces 30-56 raw color features to 8 principal components
- Preserves ~90%+ of variance while reducing dimensionality
- Fitted on training set, applied to validation/test

### 3. Classification Network

**Architecture: BCGCandidateClassifier**
- Input: 30D (morphological only) or 38D (with color PCA)
- Hidden layers: [128, 64, 32] (configurable)
- Activations: ReLU
- Regularization: Dropout (rate=0.2) after each hidden layer
- Output: Single score per candidate (unbounded)

**Forward Pass:**
```
Input features → Linear(in, 128) → ReLU → Dropout(0.2) →
Linear(128, 64) → ReLU → Dropout(0.2) →
Linear(64, 32) → ReLU → Dropout(0.2) →
Linear(32, 1) → Raw score
```

**Training Details:**
- Loss: Cross-entropy over candidates per image
  - Softmax applied across all candidates in same image
  - True label: index of candidate closest to ground truth BCG
  - Handles variable numbers of candidates per image
- Optimizer: Adam (default lr=0.001)
- Feature scaling: StandardScaler (fitted on training set)
- Batch processing: Processes all candidates from one image together

**Inference:**
- Computes scores for all candidates in test image
- Selects candidate with highest score as predicted BCG
- Returns: (x, y) coordinates of best candidate

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
- `evaluation_results.csv`: Detailed per-sample results including:
  - `cluster_name`: Original cluster identifier
  - `z`: Redshift (if available in truth table)
  - `bcg_prob`: BCG probability score (if available in truth table)
  - `pred_x`, `pred_y`: Predicted BCG coordinates
  - `true_x`, `true_y`: True BCG coordinates
  - `distance_error`: Pixel distance between prediction and truth
  - `n_candidates`: Number of candidates found in image

## Dependencies

- `torch>=2.0.0`: Deep learning framework
- `scikit-learn>=1.0.0`: Feature scaling and metrics
- `astropy>=5.0.0`: WCS coordinate conversion
- `matplotlib>=3.4.0`: Plotting and visualization
- `numpy`, `pandas`, `scipy`: Standard scientific computing

