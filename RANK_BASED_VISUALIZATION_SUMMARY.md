# Rank-Based Sample Visualization System

## New UQ Model Sample Files

When running `enhanced_full_run.py` with `--use_uq`, instead of generic "prediction samples" and "failure samples", you'll now get **rank-based sample categories**:

### Generated Files

#### **Rank-Based Sample Categories:**
- `ProbabilisticTesting_prediction_sample_best_rank1_*.png` - Cases where **best candidate** matches true BCG
- `ProbabilisticTesting_prediction_sample_best_rank2_*.png` - Cases where **2nd best candidate** matches true BCG  
- `ProbabilisticTesting_prediction_sample_best_rank3_*.png` - Cases where **3rd best candidate** matches true BCG
- `ProbabilisticTesting_prediction_sample_best_rest_*.png` - Cases where true BCG found at rank >3 or not detected

#### **True Failures Only:**
- `ProbabilisticTesting_failure_sample_*.png` - Only **genuine failures** (rank >5 or not detected)

### Key Improvements

#### **1. More Informative Categorization**
Instead of binary success/failure, you can see:
- How often your model gets it right on the first try (rank 1)
- How often the true BCG is in the top 2-3 candidates (valuable for interactive use)
- What genuine failures look like (rank >5 or completely missed)

#### **2. Better Sample Distribution**
- Samples are distributed across rank categories (not just "best" by distance)
- Within each category, samples are selected by highest probability/score
- Shows representative examples from each performance tier

#### **3. Interactive Application Insights**
Perfect for evaluating models intended for:
- Interactive BCG selection tools
- Multi-candidate review workflows  
- Confidence-based decision making

### Sample Selection Logic

```
Total samples requested: N (e.g., 20 via --show_samples)
Samples per category: N/4 (e.g., 5 each)

For each rank category:
1. Find all samples in that category
2. If more samples than needed, select highest scoring within category
3. Generate visualization with category-specific filename
```

### Console Output Example

```
Generating rank-based sample visualizations...
  Generating 5 samples for rank1 (ranks 1)
  Generating 3 samples for rank2 (ranks 2)  
  Generating 2 samples for rank3 (ranks 3)
  Generating 4 samples for rest (ranks >3 or None)

Showing true prediction failures (not detected or rank >5)...
```

### Comparison: Before vs After

#### **Before (Distance-Based):**
- `ProbabilisticTesting_prediction_sample_*.png` - Best predictions by distance
- `ProbabilisticTesting_failure_sample_*.png` - **Included rank 2/3 successes** ❌

#### **After (Rank-Based):**  
- `ProbabilisticTesting_prediction_sample_best_rank1_*.png` - Pure rank 1 successes
- `ProbabilisticTesting_prediction_sample_best_rank2_*.png` - Pure rank 2 successes  
- `ProbabilisticTesting_prediction_sample_best_rank3_*.png` - Pure rank 3 successes
- `ProbabilisticTesting_prediction_sample_best_rest_*.png` - Lower ranks + not detected
- `ProbabilisticTesting_failure_sample_*.png` - **Only genuine failures** ✅

### Non-UQ Models

For models without `--use_uq`, the traditional system remains:
- `EnhancedTesting_prediction_sample_*.png` - Best predictions by distance
- `EnhancedTesting_failure_sample_*.png` - Worst predictions by distance

This maintains backward compatibility while providing enhanced analysis for UQ-enabled models.

## Benefits

1. **Clearer Performance Understanding**: See exactly where your model succeeds/fails
2. **Interactive Use Case Evaluation**: Perfect for applications where users examine multiple candidates
3. **Reduced Noise in Failure Analysis**: Focus on true failures, not acceptable near-misses
4. **Better Model Comparison**: Compare models by rank-based performance, not just distance
5. **Actionable Insights**: Understand if model finds right regions but ranks them incorrectly