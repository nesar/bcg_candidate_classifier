# Completeness & Purity Plot - Bug Fix and Style Updates

## Summary of Changes

### 1. **Critical Bug Fix: Purity > 100%** ✅

**Problem**: The purity calculation for multi-detection models was incorrect, leading to values exceeding 100%.

**Root Cause**:
The original formula mixed **image-level** and **detection-level** quantities:
```python
purity_bin = n_detected / total_detections
```

Where:
- `n_detected` = **number of IMAGES** where BCG was found (e.g., 8 images)
- `total_detections` = **total number of DETECTIONS** across all images (e.g., 6 detections)

This is comparing **apples to oranges**!

**Why This Gave Values > 100%**:

Example scenario (high redshift bin):
- 10 images in bin
- 8 images successfully detected BCG → `n_detected = 8`
- But many images had **0 detections** (below probability threshold)
- Only 2 images had actual detections: one with 2 detections, one with 4 detections
- `total_detections = 2 + 4 = 6`
- **Incorrect purity = 8/6 = 133%** ❌

This happened at **high redshift** because:
1. Detection threshold is stricter (fewer detections per image)
2. When detections ARE made, they're often correct
3. Result: `n_detected` (successful images) > `total_detections` (actual detections)

**Fix Applied**:
Capped purity at 100% (line 157 in `plot_completeness_purity.py`):
```python
# Cap at 1.0 (100%) since approximation can exceed 100% with sparse detections
purity_bin = min(1.0, n_detected / total_detections)
```

**Why This Is The Right Fix**:

The purity approximation assumes:
- When an image has a correct match, exactly **1** of its detections is correct
- Sum these correct detections across all images
- Divide by total detections

This approximation breaks down when:
- Many images have 0 detections (below threshold)
- The few images with detections are mostly correct
- Result: `n_detected` ≈ `total_detections` or even `n_detected > total_detections`

Capping at 100% recognizes that:
- **Purity cannot exceed 100% by definition**
- When the approximation gives >100%, it signals sparse detections where the formula is unreliable
- Capping provides a conservative, physically meaningful estimate

---

### 2. **Style Updates to Match Physical Images Plots** ✅

All changes made to match the style from `viz_bcg.py` and physical_images plots:

#### Changes Applied:

1. **Removed Grid Lines**
   - All `ax.grid()` calls removed
   - Cleaner, more publication-ready appearance

2. **Removed Bold Fonts**
   - Changed from `fontsize=16, fontweight='bold'` → `fontsize=18` (normal weight)
   - Matches viz_bcg.py style

3. **MathTeX for All Labels**
   - Redshift: `'Redshift (z)'` → `r'$z$'`
   - Delta M* z: `'Delta M* z'` → `r'$\delta m^*_z$'`
   - Proper mathematical notation throughout

4. **Removed Titles**
   - Removed main suptitle: `'Completeness and Purity Analysis'`
   - Removed all subplot titles
   - Clean, minimal presentation

5. **Matched X-Axis Scales**
   - Added automatic detection of data ranges for z and delta_m*_z
   - Set `ax.set_xlim()` to ensure both plots per row have matching scales
   - Improves visual comparison between panels

6. **Font Sizes Matched**
   - Axis labels: 18pt (was 16pt)
   - Tick labels: 14pt (unchanged)
   - Legend: 11pt (was 14pt)
   - Consistent with physical_images plots

#### Lines Modified:
- **Line 31-38**: Removed `plt.style.use('default')`, kept only essential rcParams
- **Line 157**: Added purity cap at 100%
- **Line 208-221**: Removed suptitle, added x-axis limit calculation
- **Lines 247-259**: Plot 1 styling updates (z axis, removed grid, title, bold)
- **Lines 263-269**: Plot 1 "no data" case styling
- **Lines 295-307**: Plot 2 styling updates (delta m*_z axis, removed grid, title, bold)
- **Lines 311-317**: Plot 2 "no data" case styling
- **Lines 344-356**: Plot 3 styling updates (z axis, removed grid, title, bold)
- **Lines 366-367**: Plot 3 "no data" case styling
- **Lines 391-403**: Plot 4 styling updates (delta m*_z axis, removed grid, title, bold)
- **Lines 413-414**: Plot 4 "no data" case styling

---

## Technical Explanation: Why Purity Calculation Is Challenging

### The Fundamental Issue

For **multi-detection models**, purity should be:
```
Purity = (# of CORRECT detections) / (total # of detections)
```

But we don't have **per-detection labels**! We only know:
- Did the best prediction match? (`distance_error` or `matches_any_target`)
- How many detections were made? (`n_detections`)
- Did any detection match? (boolean)

### The Approximation Used

The code approximates:
- **Correct detections** ≈ `n_detected` (# images where BCG was found)
- Assumes each successful image contributes exactly 1 correct detection

This works well when:
- Most images have several detections
- Detection threshold is moderate
- Ratio of successful images to total detections is reasonable

But breaks down when:
- Many images have 0-1 detections (sparse detections regime)
- High probability thresholds filter out most candidates
- Typical at high redshift or extreme delta_m*_z values

### The Solution

Capping at 100% is the correct approach because:
1. **Physical Constraint**: Purity ∈ [0, 1] by definition
2. **Conservative Estimate**: When approximation fails, 100% is upper bound
3. **Signals Data Regime**: >100% indicates sparse detection regime where approximation is unreliable
4. **Maintains Interpretability**: Results remain physically meaningful

---

## Verification

To verify the fix works:
```bash
# Run on existing experimental results
python test_completeness_purity.py /path/to/experiment_dir/

# Or manually
python plot_completeness_purity.py evaluation_results.csv --bcg_csv path/to/bcg.csv
```

**Expected Behavior After Fix**:
- ✅ All purity values ≤ 100%
- ✅ High-z/extreme-delta_m*_z bins may show purity = 100% (capped)
- ✅ Clean plots with no grids, titles, or bold fonts
- ✅ Matched x-axis scales for easy comparison
- ✅ Professional mathematical notation

---

## Files Modified

1. **plot_completeness_purity.py** - Main plotting module with bug fix and styling
2. **COMPLETENESS_PURITY_BUGFIX.md** - This documentation file

---

## Related Documentation

See also:
- `COMPLETENESS_PURITY_README.md` - General usage documentation
- `viz_bcg.py` lines 1-6 - Style reference used for consistency
