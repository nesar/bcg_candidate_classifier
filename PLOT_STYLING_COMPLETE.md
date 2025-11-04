# Plot Styling Standardization - Complete

## Summary

All plotting routines in the BCG classifier repository have been updated to use consistent styling through a centralized `plot_config.py` module.

## Files Updated

### Core Configuration
- **plot_config.py** - NEW: Central configuration with consistent colors, fonts, sizes

### Plotting Scripts Called by `enhanced_full_run.py`
1. **utils/diagnostic_plots.py** - diagnostic_plots.png/pdf
2. **plot_sectors_hardcoded.py** - diagnostic_plots_sectors.png/pdf
3. **plot_rank_histograms.py** - rank_histograms.png/pdf
4. **plot_completeness_purity.py** - completeness_purity_plots.png/pdf
5. **plot_physical_results.py** - feature_breakdown_with_group_totals.png/pdf
6. **plot_eval_results.py** - Various analysis plots
7. **train.py** - training_loss_curves.png

### Additional Plotting Files
- **analysis/importance_plots.py** - Feature importance visualizations

## Consistent Settings Applied

### Fonts
- **Title**: 20pt
- **Labels**: 18pt
- **Ticks**: 14pt
- **Legend**: 14pt
- **Font family**: Serif (Computer Modern for math)

### Sizes
- **Linewidth**: 2.0 (main), 1.2 (edges)
- **Markersize**: 8
- **DPI**: 300 (high quality)
- **Axes linewidth**: 1.2

### Colors
- **Completeness**: #2ecc71 (green)
- **Purity**: #3498db (blue)
- **Training**: #2E86AB (blue)
- **Validation**: #E63946 (red)
- **Rank colors**: Paired colormap (consistent across all plots)

### Style Elements
- Ticks on all four sides (in direction)
- Grid with 30% alpha
- Consistent legend styling

## To See Changes on Remote Cluster

Since you're running on a remote cluster, you need to sync these changes:

```bash
# From your local machine, push to your repository
git add plot_config.py
git add utils/diagnostic_plots.py
git add plot_sectors_hardcoded.py
git add plot_physical_results.py
git add plot_completeness_purity.py
git add plot_rank_histograms.py
git add plot_eval_results.py
git add train.py
git add analysis/importance_plots.py

git commit -m "Standardize plot styling across all visualization routines"
git push

# Then on the remote cluster
cd /path/to/bcg_candidate_classifier
git pull

# Run your workflow
python enhanced_full_run.py
```

## Usage in New Scripts

```python
from plot_config import setup_plot_style, COLORS, FONTS, SIZES

# Apply style at the start of your plotting function
setup_plot_style()

# Use consistent values
plt.plot(x, y, color=COLORS['primary'], linewidth=SIZES['linewidth'])
ax.set_xlabel('X Label', fontsize=FONTS['label'])
ax.set_title('My Plot', fontsize=FONTS['title'])
plt.savefig('plot.png', dpi=SIZES['dpi'])
```

## Key Benefits

1. **Professional appearance** - Larger, more readable fonts
2. **Publication quality** - 300 DPI, serif fonts, consistent styling
3. **Easy maintenance** - Change one config file to update all plots
4. **Consistency** - All plots now have matching fonts, colors, and sizes
5. **Reusability** - Simple import and use in any new plotting code

## Verification

Run the verification script to confirm all files are properly updated:

```bash
python verify_plot_config.py
```

All files should show ✓ with imports and usage of plot_config values.
