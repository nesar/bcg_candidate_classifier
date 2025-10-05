# Quick Reference: Probability & Uncertainty Terminology

## The Three Key Metrics

### 1. RedMapper BCG Probability
```
SOURCE:     RedMapper catalog (external truth/reference)
CSV COLUMN: bcg_prob
RANGE:      [0, 1]
MEANING:    RedMapper's confidence that this galaxy is the cluster's BCG
PURPOSE:    Ground truth for evaluating our ML model
```

### 2. ML Predictive Confidence
```
SOURCE:     Our ML uncertainty quantification model
CSV COLUMN: max_probability, avg_probability, probability
RANGE:      [0, 1]
MEANING:    ML model's confidence that a candidate is the BCG
CALCULATION: Mean of sigmoid(logits/temperature) over MC Dropout samples
PURPOSE:    Our model's prediction
```

### 3. ML Uncertainty Estimate
```
SOURCE:     Our ML model's epistemic uncertainty (MC Dropout)
CSV COLUMN: max_uncertainty, avg_uncertainty, uncertainty
RANGE:      [0, ∞), typically [0, 0.2]
MEANING:    How much the ML predictions vary across MC samples
CALCULATION: Standard deviation of probabilities over MC Dropout samples
PURPOSE:    Reliability indicator / confidence interval
```

## Quick Decision Tree

**Q: Is this from the catalog or our model?**
- Catalog → "RedMapper BCG Probability"
- Our model → Continue below

**Q: Is it a probability or an uncertainty?**
- Probability (0-1, higher = more confident it's a BCG) → "ML Predictive Confidence"
- Uncertainty (0-0.2, higher = less certain) → "ML Uncertainty Estimate"

## Common Plot Labels

| What you're plotting | X-axis label | Y-axis label |
|----------------------|--------------|--------------|
| RedMapper vs ML comparison | "RedMapper BCG Probability" | "ML Predictive Confidence" |
| ML confidence distribution | "ML Predictive Confidence" | "Count" or "Frequency" |
| ML uncertainty distribution | "ML Uncertainty Estimate" | "Count" or "Frequency" |
| Confidence vs uncertainty | "ML Predictive Confidence" | "ML Uncertainty Estimate" |
| Uncertainty vs error | "ML Uncertainty Estimate" | "Distance Error (pixels)" |

## CSV File Quick Reference

### evaluation_results.csv
- `bcg_prob` = RedMapper BCG Probability (reference)
- `max_probability` = ML Predictive Confidence (highest among candidates)
- `max_uncertainty` = ML Uncertainty Estimate (highest among candidates)

### probability_analysis.csv
- `probability` = ML Predictive Confidence (for this specific candidate)
- `uncertainty` = ML Uncertainty Estimate (for this specific candidate)

## Remember
✅ Always specify "RedMapper" or "ML"
✅ Use "ML Predictive Confidence" not just "probability"
✅ Use "ML Uncertainty Estimate" not just "uncertainty"
✅ RedMapper is for **reference/comparison**, not model input during testing
✅ ML metrics come from **MC Dropout + Temperature Scaling**

❌ Don't use ambiguous terms like "BCG Probability" or "Uncertainty"
❌ Don't confuse confidence (prediction) with uncertainty (variability)
