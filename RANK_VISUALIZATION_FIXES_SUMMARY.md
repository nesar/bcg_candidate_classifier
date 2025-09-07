# Rank-Based Visualization Fixes Summary

## Issues Fixed

### **1. ✅ Removed Redundant Failure Plots for UQ Models**

**Problem:** UQ models were still generating `ProbabilisticTesting_failure_sample_*.png` files even though the "rest" category already covers genuine failures.

**Solution:** 
- Modified condition to skip failure visualization for UQ models: `if args.show_failures and len(distances) > 0 and not args.use_uq:`
- UQ failures are now exclusively handled by the "rest" category
- Updated `enhanced_full_run.py` output description to remove mention of failure files for UQ

**Result:** No more redundant failure sample files for UQ models.

---

### **2. ✅ Fixed Sample Distribution Per Rank**

**Problem:** Only 1 sample per rank was being generated instead of the requested number (e.g., 5 samples per rank when using `--show_samples 5`).

**Solution:** 
- Changed: `samples_per_category = max(1, args.show_samples // 4)` (divided across categories)
- To: `samples_per_category = args.show_samples` (full number per category)

**Result:** Each rank category now shows up to the full requested number of samples.

**Example with `--show_samples 5`:**
- **Before:** 1-2 samples total across all ranks
- **After:** Up to 5 samples per rank (rank1, rank2, rank3, rest)

---

### **3. ✅ Fixed Rank 3 Circle Display Issues**

**Problem:** Rank 3 candidates sometimes didn't appear or weren't visible in visualizations due to adaptive candidate display logic that limited visibility based on max probability.

**Solution:** Enhanced `utils/viz_bcg.py` with rank-aware candidate display:

```python
# Rank-aware candidate display logic
if phase and 'rank3' in phase:
    n_candidates_to_show = max(3, min(5, len(candidates)))  # Show at least 3 for rank3
elif phase and ('rank2' in phase or 'rank1' in phase):
    n_candidates_to_show = max(2, min(3, len(candidates)))  # Show at least 2 for rank2/rank1
else:
    # Default probability-based logic with more conservative limits
    n_candidates_to_show = min(5, len(candidates))  # Show more candidates when uncertain
```

**Additional Improvements:**
- Expanded color palette: `['red', 'orange', 'yellow', 'lime', 'cyan']` for up to 5 visible ranks
- Guaranteed visibility of relevant rank candidates based on filename/phase

**Result:** Rank 3 candidates are now always visible when viewing rank 3 sample files.

---

## Summary of File Changes

### **test.py:**
- Removed UQ failure case generation 
- Fixed sample distribution logic
- Each rank category gets full requested sample count

### **utils/viz_bcg.py:**
- Added rank-aware candidate display logic
- Expanded color palette for 5 ranks
- Guaranteed minimum candidate visibility based on rank category

### **enhanced_full_run.py:**
- Updated output file descriptions
- Removed mention of failure files for UQ models

---

## Expected User Experience

When running `enhanced_full_run.py --use_uq --show_samples 5`:

### **Generated Files:**
- `ProbabilisticTesting_prediction_sample_best_rank1_*.png` (up to 5 samples)
- `ProbabilisticTesting_prediction_sample_best_rank2_*.png` (up to 5 samples)  
- `ProbabilisticTesting_prediction_sample_best_rank3_*.png` (up to 5 samples)
- `ProbabilisticTesting_prediction_sample_best_rest_*.png` (up to 5 samples)
- **No more:** `ProbabilisticTesting_failure_sample_*.png` (redundant)

### **Visual Improvements:**
- **Rank 3 files:** Always show at least 3 candidate circles, ensuring rank 3 candidate is visible
- **Rank 2 files:** Always show at least 2 candidate circles
- **Better colors:** Up to 5 distinct colors for different rank candidates
- **Clear rank identification:** Each circle colored according to its probability rank

### **Console Output:**
```
Generating rank-based sample visualizations...
  Generating 3 samples for rank1 (ranks 1)
  Generating 5 samples for rank2 (ranks 2)
  Generating 5 samples for rank3 (ranks 3)
  Generating 5 samples for rest (ranks >3 or None)
```

All three issues have been resolved, providing a much cleaner and more informative rank-based evaluation system!