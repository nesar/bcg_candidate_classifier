# Sectors Plot Integration - Summary

## Overview
Successfully integrated the dynamic sectors plot generation into the `enhanced_full_run.py` workflow. All hardcoded values have been removed and the plot now works automatically with every experiment run.

## Changes Made

### 1. `plot_sectors_hardcoded.py` - Complete Rewrite
**Status:** ✅ Transformed from hardcoded to fully dynamic

**Key Changes:**
- Removed all hardcoded values (`single_values`, `multi_values`, accuracy percentages)
- Added `load_evaluation_results()` function to read CSV data
- Added `calculate_rank_statistics()` function to compute rank percentages dynamically
- Modified `improved_donut()` to accept labels as parameters
- Created `create_sectors_plot()` main function (similar to `create_diagnostic_plots()`)
- Added command-line interface with argparse
- Added comprehensive docstrings and error handling

**New Features:**
- Automatically calculates rank distributions from `evaluation_results.csv`
- Supports both single-target (`bcg_rank`) and multi-target (`multi_target_rank`) analysis
- Prints detailed statistics to console
- Saves plots as both PNG (300 DPI) and PDF for publication quality
- Follows the same pattern as `utils/diagnostic_plots.py`

### 2. `enhanced_full_run.py` - Integration
**Status:** ✅ Fully integrated into workflow

**Changes at Line 715-721:**
```python
# Generate sectors plot (donut charts for rank analysis)
sectors_command = f"python -c \"from plot_sectors_hardcoded import create_sectors_plot; create_sectors_plot('{evaluation_csv}', '{output_dir}')\""

if not run_command(sectors_command, "Generating sectors plot"):
    print("Sectors plotting failed, but continuing...")
else:
    print(f"Sectors plot saved to: {output_dir}/diagnostic_plots_sectors.png")
```

**Changes at Line 798-799 (Summary Output):**
```python
print(f"  Sectors plot: {output_dir}/diagnostic_plots_sectors.png")
print(f"  Sectors plot (PDF): {output_dir}/diagnostic_plots_sectors.pdf")
```

## Output Files

When you run `enhanced_full_run.py`, the following files will be automatically generated:

1. **diagnostic_plots_sectors.png** - High-resolution PNG (300 DPI) with donut charts
2. **diagnostic_plots_sectors.pdf** - Publication-quality PDF version

These files are saved in the experiment's main output directory (same location as `diagnostic_plots.png`).

## Plot Contents

The sectors plot displays:
- **Left panel:** Single-target rank analysis
  - Rank 1, Rank 2, Rank 3, and Rest percentages
  - Top-3 accuracy in the center
- **Right panel:** Multi-target rank analysis (if data available)
  - Same rank breakdown for multi-target scenarios
  - Top-3 accuracy in the center

## Testing

✅ All tests passed successfully:
- Import test: `create_sectors_plot` function imports correctly
- Syntax check: Both modified files pass Python syntax validation
- Functional test: Generated plots from mock data successfully
- File generation: Both PNG and PDF files created correctly

**Test Results:**
```
Created test CSV: /tmp/test_sectors_plot/evaluation_results.csv
Number of samples: 100
Rank distribution:
  Rank 1: 87 (87.0%)
  Rank 2: 8 (8.0%)
  Rank 3: 4 (4.0%)
  Rest: 1 (1.0%)

Sectors Plot Statistics:
  Single-target Top-3 Accuracy: 99.0%
  Multi-target Top-3 Accuracy: 100.0%

✓ Sectors plot PNG generated successfully
✓ Sectors plot PDF generated successfully
```

## Usage

### Automatic (via enhanced_full_run.py)
Simply run `enhanced_full_run.py` as usual - the sectors plot will be generated automatically after the diagnostic plots in Step 4.

### Manual (command line)
```bash
python plot_sectors_hardcoded.py <path_to_evaluation_results.csv> --output_dir <output_directory>
```

### Programmatic
```python
from plot_sectors_hardcoded import create_sectors_plot

# Generate sectors plot
fig = create_sectors_plot(
    results_file='path/to/evaluation_results.csv',
    output_dir='path/to/output',
    figsize=(14, 7)  # optional
)
```

## Requirements

The sectors plot requires the following columns in `evaluation_results.csv`:
- `bcg_rank` - for single-target analysis
- `multi_target_rank` - for multi-target analysis (optional)

At least one of these rank columns must be present.

## Next Steps

The integration is complete and ready to use. When you run your next experiment with `enhanced_full_run.py`, the sectors plot will be automatically generated alongside the other diagnostic plots.

## Files Modified

1. `/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/plot_sectors_hardcoded.py` (complete rewrite)
2. `/Users/nesar/Projects/HEP/IMGmarker/bcg_candidate_classifier/enhanced_full_run.py` (integration added)

---

**Date:** 2025-10-14
**Status:** ✅ Complete and tested
