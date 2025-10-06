# Visualization Labels Update Summary

## Question Answered
**Q: What is the probability mentioned in each ranked candidate in images like `ProbabilisticTesting_prediction_sample_best_rank1_prediction_sample_7_enhanced.png`? Is it the ML Predictive Confidence or ML Uncertainty Estimate?**

**A: It is the ML Predictive Confidence** (NOT the uncertainty estimate).

## Technical Details

### What's Displayed in the Images

1. **Numbers on candidates (e.g., "0.85", "0.72")**
   - Source: `probabilities[cand_idx]` from `viz_bcg.py` line 554
   - Meaning: ML model's confidence that this candidate is the BCG
   - Range: 0.0 to 1.0
   - Higher = more confident this is the true BCG

2. **Legend entries**
   - Format: `"Rank {rank} (ML conf: {prob})"`
   - Example: "Rank 1 (ML conf: 0.85)"
   - Shows both the ranking and the confidence value

3. **Subtitle information**
   - Format: `"ML Max Conf: {max_prob} | Detections: {n} (≥{threshold})"`
   - Shows the maximum confidence and number of detections

### How ML Predictive Confidence is Calculated

```python
# In ml_models/uq_classifier.py, predict_with_uncertainty()
# Step 1: Multiple forward passes with MC Dropout enabled
for _ in range(n_samples):  # typically 10 samples
    logits = model.forward_with_temperature(features)
    probs = sigmoid(logits / temperature)
    predictions.append(probs)

# Step 2: Aggregate predictions
mean_probs = predictions.mean(dim=0)      # ML Predictive Confidence
uncertainty = predictions.std(dim=0)       # ML Uncertainty Estimate
```

### Key Distinction

| Metric | What's shown in images? | Range | Interpretation |
|--------|------------------------|-------|----------------|
| **ML Predictive Confidence** | ✅ YES (the numbers) | 0-1 | How confident model is this is BCG |
| **ML Uncertainty Estimate** | ❌ NO | 0-0.2 | How much predictions vary (epistemic uncertainty) |

## Files Updated

### 1. `utils/viz_bcg.py`

**Line 562**: Added clarifying comment
```python
# Add ML confidence label
ax.text(candidate[0] + 8, candidate[1] - 8, f'{prob:.2f}', ...)
```

**Line 570**: Updated legend label
```python
# Before: label=f'Rank {rank+1} ({prob:.2f})'
# After:  label=f'Rank {rank+1} (ML conf: {prob:.2f})'
```

**Line 132**: Updated comment in `show_predictions_with_candidates`
```python
# Add ML confidence label
```

**Line 437**: Updated subtitle text
```python
# Before: f' | Max Prob: {max_prob:.3f} | ...'
# After:  f' | ML Max Conf: {max_prob:.3f} | ...'
```

## Visual Example Interpretation

For an image showing:
- Rank 1 candidate with "0.85"
- Rank 2 candidate with "0.72"
- Rank 3 candidate with "0.65"

**Interpretation:**
- ML model is **85% confident** Rank 1 is the BCG
- ML model is **72% confident** Rank 2 is the BCG
- ML model is **65% confident** Rank 3 is the BCG
- Rank 1 has highest confidence → it becomes the prediction

**These are NOT:**
- ❌ Uncertainty values (which would be ~0.02-0.08)
- ❌ RedMapper probabilities (which come from catalog)
- ❌ Raw scores or logits

## Where to Find Uncertainty

If you need the **ML Uncertainty Estimate** (not shown in standard images), look at:

1. **CSV files:**
   - `evaluation_results.csv`: columns `max_uncertainty`, `avg_uncertainty`
   - `probability_analysis.csv`: column `uncertainty`

2. **Analysis plots:**
   - `analysis_plots/uncertainty_analysis.png`
   - Shows uncertainty distributions and correlations

3. **Console output:**
   - When running `plot_eval_results.py`
   - Section "ML UNCERTAINTY QUANTIFICATION"

## Summary

✅ **Candidates in images show ML Predictive Confidence (0-1 scale)**
- Higher value = more confident this is the BCG
- Used for ranking and detection
- Displayed as numbers on/near candidates

❌ **Candidates do NOT show ML Uncertainty Estimate**
- Uncertainty represents prediction variability
- Not directly visualized in candidate images
- Available in CSV files and analysis plots

The updated labels now make this distinction crystal clear with "ML conf:" prefix in legends and "ML Max Conf:" in subtitles.
