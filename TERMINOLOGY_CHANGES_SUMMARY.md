# Terminology Changes Summary

## Changes Made to `plot_eval_results.py`

### Overview
Updated all plot labels, titles, and console output to clearly distinguish between:
1. **RedMapper BCG Probability** - External reference from catalog
2. **ML Predictive Confidence** - Our model's confidence in BCG identification
3. **ML Uncertainty Estimate** - Our model's epistemic uncertainty (via MC Dropout)

### Specific Changes

#### Plot Axis Labels

| Location | Before | After |
|----------|--------|-------|
| RedMapper correlation plot | "ML Max Probability" | "ML Predictive Confidence" |
| Rank distribution plot | "ML Max Probability" | "ML Predictive Confidence" |
| Detection statistics plot | "Max ML Probability" | "ML Predictive Confidence" |
| Uncertainty patterns plot | "Uncertainty" | "ML Uncertainty Estimate" |
| Uncertainty vs accuracy | "Max Uncertainty" | "ML Uncertainty Estimate" |
| Calibration plot | "Mean Uncertainty (binned)" | "Mean ML Uncertainty (binned)" |
| Confidence-uncertainty scatter | "Max Uncertainty" | "ML Uncertainty Estimate" |
| Candidate rank probability | "ML Probability" | "ML Predictive Confidence" |

#### Plot Titles

| Location | Before | After |
|----------|--------|-------|
| Main correlation | "Probability Correlation" | "RedMapper vs ML Confidence Correlation" |
| Rank distribution | "ML Probability Distribution by Rank" | "ML Confidence Distribution by Rank" |
| Uncertainty patterns | "Uncertainty Patterns by RedMapper Category" | "ML Uncertainty Patterns by RedMapper Category" |
| Uncertainty vs accuracy | "Uncertainty vs Accuracy" | "ML Uncertainty vs Accuracy" |
| Calibration | "Uncertainty Calibration" | "ML Uncertainty Calibration" |
| Distribution by performance | "Uncertainty Distribution by Performance" | "ML Uncertainty Distribution by Performance" |
| Confidence vs uncertainty | "Confidence vs Uncertainty" | "ML Confidence vs Uncertainty" |
| Number of detections | "Uncertainty vs Number of Detections" | "ML Uncertainty vs Number of Detections" |
| Failure predictor | "Uncertainty as Failure Predictor" | "ML Uncertainty as Failure Predictor" |
| Candidate rank | "Probability Distribution by Candidate Rank" | "ML Confidence Distribution by Candidate Rank" |

#### Console Output Text

| Section | Before | After |
|---------|--------|-------|
| Correlation stats | "Overall Pearson correlation" | "RedMapper vs ML Predictive Confidence: Pearson correlation" |
| Summary section 3 | "REDMAPPER-ML CORRELATION" | "REDMAPPER BCG PROBABILITY vs ML PREDICTIVE CONFIDENCE" |
| Summary section 6 | "UNCERTAINTY QUANTIFICATION" | "ML UNCERTAINTY QUANTIFICATION" |
| Uncertainty details | "Mean max uncertainty" | "Mean ML uncertainty (max)" |
| Uncertainty correlation | "Uncertainty-error correlation" | "ML uncertainty-error correlation" |
| Key findings | "RedMapper and ML probabilities" | "RedMapper BCG probability and ML predictive confidence" |
| Key findings | "Uncertainty does/does not" | "ML uncertainty estimate does/does not" |
| Key findings | "high confidence from both methods" | "high confidence from both RedMapper and ML" |
| Recommendations | "differences between RedMapper and ML probability" | "differences between RedMapper BCG probability and ML predictive confidence" |
| Recommendations | "probability calibration" | "temperature calibration" |
| Recommendations | "Uncertainty estimates" | "ML uncertainty estimates" |
| Analysis summary | "Uncertainty Analysis Summary" | "ML Uncertainty Analysis Summary" |
| Analysis details | "Mean max uncertainty" | "Mean ML uncertainty estimate" |
| Analysis ROC | "Uncertainty ROC AUC" | "ML uncertainty ROC AUC" |
| Analysis threshold | "Optimal uncertainty threshold" | "Optimal ML uncertainty threshold" |

### Key Improvements

1. **Eliminated Ambiguity**: Every mention of "probability" now specifies whether it's RedMapper's or ML's
2. **Clear Source Attribution**: All uncertainty metrics explicitly labeled as "ML Uncertainty Estimate"
3. **Consistent Terminology**: Uses "ML Predictive Confidence" throughout instead of mixing "probability", "max probability", etc.
4. **Better Comparisons**: When comparing RedMapper to ML, both sources are explicitly named

### Files Created

1. **PROBABILITY_UNCERTAINTY_GUIDE.md** - Comprehensive documentation explaining:
   - What each metric means
   - How it's calculated
   - Where it appears in the data
   - How to interpret it
   - Relationship between metrics
   - Best practices for terminology

2. **TERMINOLOGY_CHANGES_SUMMARY.md** (this file) - Quick reference for all changes made

### Backward Compatibility

- CSV file structure unchanged (column names remain the same)
- Only visualization labels and console output updated
- All existing code functionality preserved
- No changes to calculation methods

### Testing

Verify with existing experiment data at:
`/Users/nesar/Projects/HEP/IMGmarker/best_runs/oct5/candidate_classifier_color_uq_run_20251005_014607/`

Run:
```bash
python plot_eval_results.py /path/to/evaluation_results/evaluation_results.csv
```

Expected output: Plots with clear, unambiguous labels distinguishing RedMapper vs ML metrics.
