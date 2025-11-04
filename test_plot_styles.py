#!/usr/bin/env python3
"""
Test script to compare old vs new plotting styles
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_config import setup_plot_style, COLORS, FONTS, SIZES

# Generate test data
x = np.linspace(0, 10, 100)
y_train = np.exp(-x/10) * np.sin(x) + 0.5
y_val = np.exp(-x/10) * np.cos(x) + 0.5

# Create figure with OLD style (default matplotlib)
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(x, y_train, 'b-', label='Training', linewidth=1.5)
ax1.plot(x, y_val, 'r-', label='Validation', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('OLD STYLE: Training Loss', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_old_style.png', dpi=150, bbox_inches='tight')
print("Saved: test_old_style.png (OLD style)")
plt.close()

# Create figure with NEW style (using plot_config)
setup_plot_style()
fig2, ax2 = plt.subplots(1, 1, figsize=SIZES['figsize_single'])
ax2.plot(x, y_train, '-', color=COLORS['train'], label='Training', linewidth=SIZES['linewidth'])
ax2.plot(x, y_val, '-', color=COLORS['validation'], label='Validation', linewidth=SIZES['linewidth'])
ax2.set_xlabel('Epoch', fontsize=FONTS['label'])
ax2.set_ylabel('Loss', fontsize=FONTS['label'])
ax2.set_title('NEW STYLE: Training Loss', fontsize=FONTS['title'])
ax2.legend(fontsize=FONTS['legend'])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('test_new_style.png', dpi=SIZES['dpi'], bbox_inches='tight')
print("Saved: test_new_style.png (NEW style)")
plt.close()

print("\nComparison:")
print("="*60)
print("OLD STYLE:")
print("  - Font sizes: title=14, label=12, legend=10")
print("  - Linewidth: 1.5")
print("  - DPI: 150")
print("  - Font family: default (sans-serif)")
print("")
print("NEW STYLE:")
print(f"  - Font sizes: title={FONTS['title']}, label={FONTS['label']}, legend={FONTS['legend']}")
print(f"  - Linewidth: {SIZES['linewidth']}")
print(f"  - DPI: {SIZES['dpi']}")
print(f"  - Font family: {FONTS['family']}")
print("  - Tick direction: in (on all sides)")
print("="*60)
