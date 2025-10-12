# BCG Deployment Demo

This demo folder contains everything needed to run BCG inference on sample cluster images.

## Contents

### 1. `models/` - Trained Model Files
Pre-trained models from the October 10, 2025 experiment:
- `best_probabilistic_classifier.pth` - PyTorch model weights (88 KB)
- `best_probabilistic_classifier_scaler.pkl` - Feature scaler (2.7 KB)
- `best_probabilistic_classifier_color_extractor.pkl` - Color feature extractor (142 B)

**Model Configuration**:
- Dataset: BCG 3.8 arcmin
- Features: Color features enabled, additional features (redshift, delta_mstar_z)
- Candidates: DESprior catalog candidates
- Training: Uncertainty quantification with probabilistic outputs
- Detection threshold: 0.5

### 2. `images/` - Sample Cluster Images (5 images)
Test set images from the 3.8 arcmin × 3.8 arcmin BCG dataset:
- `SPT-CLJ0228.5-5658.6_4.06_sigma_grz.tif` (z=0.2268)
- `SPT-CLJ0035.7-4423.9_5.61_sigma_grz.tif` (z=0.2443)
- `SPT-CLJ0157.2-5821.0_14.82_sigma_grz.tif` (z=0.2191)
- `SPT-CLJ0100.1-6109.5_6.05_sigma_grz.tif` (z=0.6074)
- `SPT-CLJ2113.4-5310.4_4.67_sigma_grz.tif` (z=0.3646)

**Image Properties**:
- Format: TIFF (RGB color)
- Size: ~400×400 pixels (3.8 arcmin × 3.8 arcmin)
- Total size: ~3 MB

**Note**: These images are from the **test set** (not training set), ensuring unbiased evaluation.

### 3. `demo_candidates.csv` - BCG Candidate Catalog
DESprior candidates for the 5 demo images (63 total candidates):
- Columns: cluster, filename, candidate_ra, candidate_dec, x, y, delta_mstar, starflag, is_inside_image
- Candidates per image: 8-17 candidates each

## Quick Start

### Option 1: Using the CLI Script

From the `bcg_deployment` directory:

```bash
python scripts/run_inference.py \
  --model_dir demo/models \
  --image_dir demo/images \
  --candidates_csv demo/demo_candidates.csv \
  --output_dir demo/results
```

### Option 2: Using Python API

```python
import sys
sys.path.insert(0, 'bcg_deployment')

from bcg_deploy import BCGInference

# Initialize inference
inference = BCGInference(
    model_dir='bcg_deployment/demo/models',
    image_dir='bcg_deployment/demo/images',
    candidates_csv='bcg_deployment/demo/demo_candidates.csv',
    output_dir='bcg_deployment/demo/results',
    use_gpu=True  # Set to False for CPU-only
)

# Run inference
results = inference.run_inference(
    model_name='best_probabilistic_classifier',
    detection_threshold=0.5,
    save_images=True
)

# Print summary
print(f"Processed {len(results['results'])} images")
for result in results['results']:
    print(f"{result['filename']}: prob={result['probability']:.3f}")
```

## Expected Output

After running inference, you should see these files in `demo/results/`:

### 1. `evaluation_results.csv`
Summary results for each image:
```csv
filename,pred_x,pred_y,rank,max_probability,max_uncertainty,n_candidates,n_detections
SPT-CLJ0228.5-5658.6_4.06_sigma_grz.tif,265.18,268.26,1,0.687,0.060,6,1
SPT-CLJ0035.7-4423.9_5.61_sigma_grz.tif,237.95,212.74,1,0.830,0.062,5,1
...
```

### 2. `probability_analysis.csv`
Per-candidate probabilities and uncertainties:
```csv
filename,candidate_idx,x,y,probability,uncertainty,is_best_candidate
SPT-CLJ0228.5-5658.6_4.06_sigma_grz.tif,0,265.18,268.26,0.687,0.060,True
SPT-CLJ0228.5-5658.6_4.06_sigma_grz.tif,1,123.45,234.56,0.234,0.045,False
...
```

### 3. `annotated_images/`
PNG visualizations showing:
- Top 10 ranked candidates marked with colored circles
- Rank 1 (best): Red circle
- Ranks 2-3: Orange circles
- Ranks 4-10: Yellow circles
- Probability labels with uncertainties

## Expected Performance

Based on the training results, these demo images should show:
- **All 5 images**: Rank 1 predictions (best candidate matches true BCG)
- **Distance errors**: ~0 pixels (perfect matches)
- **Probabilities**: 0.5-0.8 range
- **Uncertainties**: 0.05-0.09 range

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`:
```bash
# Make sure you're in the parent directory
cd /Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier

# Then run with relative paths
python bcg_deployment/scripts/run_inference.py \
  --model_dir bcg_deployment/demo/models \
  --image_dir bcg_deployment/demo/images \
  --candidates_csv bcg_deployment/demo/demo_candidates.csv \
  --output_dir bcg_deployment/demo/results
```

### GPU Memory Issues
If you run out of GPU memory:
```bash
python scripts/run_inference.py \
  --model_dir demo/models \
  --image_dir demo/images \
  --candidates_csv demo/demo_candidates.csv \
  --output_dir demo/results \
  --no-gpu
```

## Demo Dataset Statistics

- **Total images**: 5
- **Total candidates**: 63
- **Average candidates per image**: 12.6
- **Redshift range**: 0.22-0.61
- **Expected inference time**: <1 minute (GPU), ~2-3 minutes (CPU)

## Model Provenance

- **Training experiment**: `candidate_classifier_color_uq_run_20251010_162255`
- **Training date**: October 10, 2025
- **Source**: `/Users/nesar/Projects/HEP/IMGmarker/best_runs/oct10/`
- **Test accuracy**: Rank-1 success rate ~80%
- **Mean distance error**: ~25 pixels (on full test set)

## Next Steps

After running the demo:
1. Examine `results/evaluation_results.csv` for quantitative results
2. View `results/annotated_images/` for visual inspection
3. Compare with ground truth BCG locations (provided in candidates CSV)
4. Try adjusting `--detection_threshold` to see how it affects detections

## Using Your Own Data

To run inference on your own cluster images:
1. Prepare images in 3.8 arcmin × 3.8 arcmin format (.tif files)
2. Create a candidates CSV in DESprior format
3. Point to your own model directory (if using a different trained model)
4. Run the inference script with your paths

See the main README.md for detailed instructions.
