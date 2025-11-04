#!/usr/bin/env python3
"""
Verification script to check if plot_config is properly integrated
"""

import sys

print("="*70)
print("PLOT CONFIG VERIFICATION")
print("="*70)
print()

# Test 1: Import plot_config
print("1. Testing plot_config import...")
try:
    from plot_config import setup_plot_style, COLORS, FONTS, SIZES
    print("   ✓ plot_config imports successfully")
    print(f"   - Title font size: {FONTS['title']}")
    print(f"   - Label font size: {FONTS['label']}")
    print(f"   - Linewidth: {SIZES['linewidth']}")
    print(f"   - DPI: {SIZES['dpi']}")
except Exception as e:
    print(f"   ✗ Failed to import plot_config: {e}")
    sys.exit(1)

print()

# Test 2: Check individual plotting files
print("2. Checking individual plotting files...")
files_to_check = [
    'plot_sectors_hardcoded.py',
    'plot_physical_results.py',
    'plot_completeness_purity.py',
    'train.py',
    'plot_rank_histograms.py'
]

for filename in files_to_check:
    try:
        with open(filename, 'r') as f:
            content = f.read()
            has_import = 'from plot_config import' in content
            has_setup = 'setup_plot_style()' in content
            uses_fonts = 'FONTS[' in content
            uses_sizes = 'SIZES[' in content
            uses_colors = 'COLORS[' in content

            status = "✓" if (has_import and (has_setup or uses_fonts or uses_sizes)) else "✗"
            print(f"   {status} {filename}")
            if has_import:
                print(f"      - Imports plot_config: Yes")
            if has_setup:
                print(f"      - Calls setup_plot_style(): Yes")
            if uses_fonts or uses_sizes or uses_colors:
                print(f"      - Uses config values: Yes")
    except FileNotFoundError:
        print(f"   ? {filename} - File not found")

print()
print("="*70)
print("NEXT STEPS TO SEE THE CHANGES:")
print("="*70)
print()
print("The code has been updated with consistent styling, but you need to")
print("REGENERATE the plots to see the changes. Here's how:")
print()
print("For training plots (training_loss_curves.png):")
print("  → Re-run your training script")
print()
print("For other diagnostic plots:")
print("  → plot_sectors_hardcoded.py <results_file>")
print("  → plot_physical_results.py <csv_dir>")
print("  → plot_completeness_purity.py <evaluation_csv>")
print("  → plot_rank_histograms.py <results_file>")
print()
print("=" * 70)
print()
print("KEY CHANGES THAT WILL APPEAR:")
print("  - Larger, more readable fonts (title: 20, labels: 18, ticks: 14)")
print("  - Serif font family (more professional)")
print("  - Thicker lines (2.0 instead of 1.5)")
print("  - Higher resolution (DPI: 300 instead of lower values)")
print("  - Consistent colors across all plots")
print("  - Ticks on all sides (in direction)")
print("="*70)
