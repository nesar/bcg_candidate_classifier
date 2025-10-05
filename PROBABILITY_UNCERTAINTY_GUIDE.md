# BCG Classifier: Probability and Uncertainty Guide

## Overview

This document clarifies the different probability and uncertainty metrics used in the BCG classifier evaluation pipeline.

## Key Terminology

### 1. **RedMapper BCG Probability** (`bcg_prob`)
- **Source**: RedMapper catalog (external truth/reference data)
- **Meaning**: RedMapper's assessment that a galaxy is the true BCG of a cluster
- **Range**: [0, 1], where 1.0 = RedMapper is certain this is the BCG
- **Location in data**:
  - Column `bcg_prob` in `evaluation_results.csv`
  - From the BCG catalog CSV files
- **Purpose**: Ground truth reference for comparison with ML predictions

### 2. **ML Predictive Confidence** (`max_probability`, `probability`)
- **Source**: Our ML model's uncertainty quantification (UQ) system
- **Meaning**: The ML model's confidence that a candidate is the true BCG
- **Range**: [0, 1], where 1.0 = ML model is highly confident this is the BCG
- **Calculation**:
  - Uses Monte Carlo Dropout with temperature scaling
  - `logits = model.forward_with_temperature(features)`
  - `probabilities = sigmoid(logits / temperature)`
  - Mean over multiple MC samples: `mean_probs = predictions.mean(dim=0)`
- **Location in data**:
  - Column `max_probability` in `evaluation_results.csv` (highest prob among all candidates)
  - Column `avg_probability` in `evaluation_results.csv` (average prob across all candidates)
  - Column `probability` in `probability_analysis.csv` (per-candidate probabilities)
- **Purpose**: Model's prediction confidence for BCG identification

### 3. **ML Uncertainty Estimate** (`max_uncertainty`, `uncertainty`)
- **Source**: Our ML model's epistemic uncertainty via MC Dropout
- **Meaning**: How uncertain the model is about its prediction (variability in predictions)
- **Range**: [0, ∞), typically [0, 0.2], where higher = more uncertain
- **Calculation**:
  - Standard deviation across MC Dropout samples
  - `uncertainty = predictions.std(dim=0)`
  - Represents model's epistemic (knowledge) uncertainty
- **Location in data**:
  - Column `max_uncertainty` in `evaluation_results.csv` (highest uncertainty among candidates)
  - Column `avg_uncertainty` in `evaluation_results.csv` (average uncertainty across candidates)
  - Column `uncertainty` in `probability_analysis.csv` (per-candidate uncertainties)
- **Purpose**: Confidence interval / reliability indicator for predictions

## Relationship Between Metrics

### Confidence vs Uncertainty
- **High Confidence, Low Uncertainty**: Model is confident and certain (ideal case)
  - Example: `probability = 0.95, uncertainty = 0.02`
  - Interpretation: "95% confident, very stable prediction"

- **High Confidence, High Uncertainty**: Model predicts high probability but is inconsistent
  - Example: `probability = 0.85, uncertainty = 0.15`
  - Interpretation: "High average prediction but varies widely across MC samples"

- **Low Confidence, Low Uncertainty**: Model consistently predicts low probability
  - Example: `probability = 0.25, uncertainty = 0.03`
  - Interpretation: "Consistently low - probably not a BCG"

- **Low Confidence, High Uncertainty**: Model is unsure
  - Example: `probability = 0.40, uncertainty = 0.18`
  - Interpretation: "Uncertain prediction - model needs more information"

### RedMapper vs ML Comparison
- **Goal**: Compare RedMapper's assessment (`bcg_prob`) with our ML confidence (`max_probability`)
- **Perfect agreement**: `bcg_prob ≈ max_probability`
- **Disagreement cases**:
  - RedMapper confident, ML uncertain: Model might need more training
  - RedMapper uncertain, ML confident: Model might have learned additional features
  - Both uncertain: Genuinely ambiguous case

## CSV File Structure

### evaluation_results.csv
```
cluster_name: Cluster identifier
z: Redshift
bcg_prob: RedMapper BCG probability (EXTERNAL REFERENCE)
pred_x, pred_y: ML predicted BCG coordinates
true_x, true_y: True BCG coordinates
distance_error: Pixel distance between prediction and truth
bcg_rank: Rank of true BCG among all candidates (1 = best)
max_probability: Highest ML confidence among all candidates
avg_probability: Average ML confidence across all candidates
n_detections: Number of candidates above detection threshold (typically 0.5)
detection_threshold: Threshold used for detections (typically 0.5)
max_uncertainty: Highest ML uncertainty among candidates
avg_uncertainty: Average ML uncertainty across candidates
n_candidates: Total number of candidate galaxies evaluated
```

### probability_analysis.csv
```
sample_name: Cluster identifier
candidate_idx: Candidate galaxy index (0, 1, 2, ...)
probability: ML confidence for this specific candidate
uncertainty: ML uncertainty for this specific candidate
is_detection: Boolean, True if probability >= detection_threshold
is_best_candidate: Boolean, True if this is the highest-probability candidate
distance_error: Distance error for this cluster (same for all candidates in cluster)
```

## Visualization Terminology

### Correct Labels for Plots

**BEFORE (Confusing)**:
- "BCG Probability" (ambiguous - which one?)
- "Uncertainty" (unclear what kind)
- "Max Probability" (unclear whose probability)

**AFTER (Clear)**:
- "ML Predictive Confidence" or "ML BCG Probability"
- "ML Uncertainty Estimate" or "ML Epistemic Uncertainty"
- "RedMapper BCG Probability" (when referring to ground truth)
- "ML Max Confidence" (when referring to best candidate)

### Plot Title Examples

**Probability Distribution**:
- ❌ "Distribution of BCG Probabilities"
- ✅ "Distribution of ML Predictive Confidence"

**Uncertainty Analysis**:
- ❌ "Uncertainty Distribution"
- ✅ "Distribution of ML Uncertainty Estimates (MC Dropout)"

**Comparison**:
- ❌ "Probability vs Uncertainty"
- ✅ "ML Predictive Confidence vs ML Uncertainty Estimate"

**RedMapper Comparison**:
- ❌ "RedMapper vs ML Probability"
- ✅ "RedMapper BCG Probability vs ML Predictive Confidence"

## Implementation Notes

### In test.py
- `probabilities, uncertainties = model.predict_with_uncertainty(features_tensor)`
  - `probabilities`: ML predictive confidence (mean of MC samples)
  - `uncertainties`: ML uncertainty estimate (std of MC samples)

### In plot_eval_results.py
- `bcg_prob`: RedMapper BCG probability (from catalog)
- `max_probability`: ML predictive confidence (highest among candidates)
- `max_uncertainty`: ML uncertainty estimate (highest among candidates)

### Detection Logic
- A "detection" occurs when `ML_confidence >= detection_threshold` (typically 0.5)
- This is NOT based on RedMapper probability
- RedMapper probability is only used for evaluation/comparison

## Best Practices

1. **Always specify the source** when mentioning probability:
   - "RedMapper BCG probability" (external reference)
   - "ML predictive confidence" (our model's output)

2. **Distinguish confidence from uncertainty**:
   - Confidence = how likely the model thinks this is a BCG (probability)
   - Uncertainty = how sure the model is about that assessment (variability)

3. **Use consistent terminology** across all plots, tables, and documentation

4. **Clarify in figure captions**:
   - Example: "ML Predictive Confidence (via MC Dropout with temperature scaling)"
   - Example: "RedMapper BCG Probability (from catalog, for reference only)"
