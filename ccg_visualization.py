#!/usr/bin/env python3
"""
CCG Visualization Module

This module provides visualization functions for p_{CCG} analysis:
1. Physical images with members showing p_{CCG} values (matching physical_images style)
2. Diagnostic plots comparing p_{CCG} vs bar_p
3. Sectors plots (like diagnostic_plots_sectors.png)
4. Completeness/Purity plots (like completeness_purity_plots.png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Wedge
from PIL import Image as pillow

from ccg_probability import (
    read_wcs_from_tif, pixel_to_radec, radec_to_pixel,
    load_rm_member_catalog, find_cluster_image,
    count_members_around_candidate, physical_to_angular_arcsec,
    get_data_paths, CCGProbabilityCalculator
)

# Set style consistent with other plots
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.2
})


# ============================================================================
# Physical Images with Members Visualization (matching physical_images style)
# ============================================================================
def plot_cluster_with_members_pccg(cluster_name, image_path, candidates_pixel,
                                   candidate_probs, p_ccg_values, member_counts,
                                   redshift, radius_kpc=300.0, wcs=None,
                                   members_df=None, rm_member_dir=None,
                                   save_path=None, dataset_type='3p8arcmin',
                                   target_coords=None, target_prob=None,
                                   pmem_cutoff=0.2, show_top_n=5):
    """
    Create enhanced visualization showing candidates with both bar_p and p_{CCG}.
    Uses SQUARE markers for candidates and actual RA/Dec coordinates on axes.

    Args:
        cluster_name: Cluster name
        image_path: Path to cluster image
        candidates_pixel: Array of (N, 2) candidate pixel coordinates
        candidate_probs: Array of bar_p values
        p_ccg_values: Array of p_{CCG} values
        member_counts: Array of member counts around each candidate
        redshift: Cluster redshift
        radius_kpc: Physical radius used for member counting
        wcs: WCS object (optional, will load from image if not provided)
        members_df: DataFrame with cluster members (optional)
        rm_member_dir: Directory with member catalogs
        save_path: Path to save plot
        dataset_type: '2p2arcmin' or '3p8arcmin'
        target_coords: Target BCG coordinates (optional)
        target_prob: Target BCG RedMapper probability (optional)
        pmem_cutoff: Minimum pmem for display
        show_top_n: Number of top candidates to show with labels
    """
    # Load image
    pil_image = pillow.open(image_path)
    pil_image.seek(0)
    image_array = np.array(pil_image)
    img_height = image_array.shape[0]
    img_width = image_array.shape[1] if len(image_array.shape) > 1 else img_height
    pil_image.close()

    # Load WCS if needed
    if wcs is None:
        wcs = read_wcs_from_tif(image_path)

    # Load members if needed
    if members_df is None:
        if rm_member_dir is None:
            rm_member_dir = get_data_paths()['rm_member_dir']
        members_df = load_rm_member_catalog(cluster_name, rm_member_dir)

    # Pixel scale
    if "2p2arcmin" in dataset_type:
        arcmin_per_pixel = 2.2 / 512
    else:
        arcmin_per_pixel = 3.8 / 512

    # Get center RA/Dec for coordinate display
    center_ra, center_dec = pixel_to_radec(256, 256, wcs, img_height)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display image
    ax.imshow(image_array)

    # Plot cluster members with pmem coloring (circle markers)
    n_members_shown = 0
    if members_df is not None and len(members_df) > 0:
        # Apply pmem cutoff for display
        if 'pmem' in members_df.columns and pmem_cutoff > 0:
            display_members = members_df[members_df['pmem'] >= pmem_cutoff]
        else:
            display_members = members_df

        # Convert member RA/Dec to pixel
        member_x, member_y = [], []
        member_pmem = []
        for _, member in display_members.iterrows():
            x, y = radec_to_pixel(member['ra'], member['dec'], wcs, img_height)
            member_x.append(x)
            member_y.append(y)
            if 'pmem' in member:
                member_pmem.append(member['pmem'])

        member_x = np.array(member_x)
        member_y = np.array(member_y)
        member_pmem = np.array(member_pmem) if member_pmem else None

        # Filter to members within image bounds
        in_bounds = (member_x >= 0) & (member_x <= img_width) & (member_y >= 0) & (member_y <= img_height)
        member_x = member_x[in_bounds]
        member_y = member_y[in_bounds]
        if member_pmem is not None:
            member_pmem = member_pmem[in_bounds]

        n_members_shown = len(member_x)

        # Plot members colored by pmem (CIRCLES for members)
        if member_pmem is not None and len(member_pmem) > 0:
            norm = Normalize(vmin=0, vmax=1)
            cmap = plt.cm.viridis
            sm = ScalarMappable(norm=norm, cmap=cmap)

            for i in range(len(member_x)):
                edge_color = cmap(norm(member_pmem[i]))
                ax.scatter(member_x[i], member_y[i], marker='o', s=120,
                          facecolors='none', edgecolors=[edge_color],
                          linewidths=1.5, alpha=0.7, zorder=3)

            # Add colorbar for pmem
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('$p_{\\mathrm{mem}}$', fontsize=14)
        else:
            ax.scatter(member_x, member_y, marker='o', s=120,
                      facecolors='none', edgecolors='green',
                      linewidths=1.5, alpha=0.7, zorder=3)

    # Plot radius circles around top candidates
    if redshift > 0 and not np.isnan(redshift):
        radius_arcsec = physical_to_angular_arcsec(radius_kpc, redshift)
        radius_pixels = radius_arcsec / (arcmin_per_pixel * 60)

        # Only draw for top candidates
        sorted_indices = np.argsort(candidate_probs)[::-1]
        for rank, idx in enumerate(sorted_indices[:show_top_n]):
            x, y = candidates_pixel[idx]
            circle = plt.Circle((x, y), radius_pixels, fill=False,
                               color='gray', linestyle='--', alpha=0.4, linewidth=1)
            ax.add_patch(circle)

    # Color scheme for ranked candidates
    colors = ["#FF0000", "#FF6600", "#FFAA00", "#FFD700", "#9ACD32",
             "#32CD32", "#00CED1", "#4169E1", "#9370DB", "#FF69B4"]

    legend_elements = []

    # Sort candidates by bar_p and show top candidates as SQUARES
    sorted_indices = np.argsort(candidate_probs)[::-1]

    for rank, idx in enumerate(sorted_indices[:show_top_n]):
        x, y = candidates_pixel[idx]
        bar_p = candidate_probs[idx]
        p_ccg = p_ccg_values[idx] if idx < len(p_ccg_values) else np.nan
        n_mem = member_counts[idx] if idx < len(member_counts) else 0

        color = colors[rank % len(colors)]

        # Plot candidate as SQUARE marker (like original physical_images)
        ax.scatter(x, y, marker='s', s=600, facecolors='none',
                  edgecolors=color, linewidths=3, alpha=0.95, zorder=10)

        # Add probability labels
        if not np.isnan(p_ccg):
            label_text = f'$\\bar{{p}}$={bar_p:.2f}\n$p_{{CCG}}$={p_ccg:.2f}\n$n_{{mem}}$={int(n_mem)}'
        else:
            label_text = f'$\\bar{{p}}$={bar_p:.2f}'

        # Position label offset from marker
        ax.text(x + 15, y - 15, label_text, fontsize=9, color='black',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.85),
               zorder=11)

        # Legend entry
        legend_label = f'Rank {rank+1}: $\\bar{{p}}$={bar_p:.2f}'
        if not np.isnan(p_ccg):
            legend_label += f', $p_{{CCG}}$={p_ccg:.2f}'
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                         markeredgecolor=color, markersize=12,
                                         markeredgewidth=3, linestyle='None',
                                         markerfacecolor='None', label=legend_label))

    # Plot target BCG if provided (CYAN dashed square)
    if target_coords is not None:
        tx, ty = target_coords
        if not np.isnan(tx) and not np.isnan(ty):
            target_label = 'Target BCG'
            if target_prob is not None and not np.isnan(target_prob):
                target_label = f'Target ($p_{{RM}}$={target_prob:.2f})'

            ax.scatter(tx, ty, marker='s', s=700, facecolors='none',
                      edgecolors="#00FFFF", linewidths=3, alpha=1.0,
                      linestyle='dashed', zorder=9)
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                             markeredgecolor="#00FFFF", markersize=12,
                                             markeredgewidth=3, linestyle='None',
                                             markerfacecolor='None', label=target_label))

    # Add member legend entry
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markeredgecolor='green', markersize=10,
                                     markeredgewidth=1.5, linestyle='None',
                                     markerfacecolor='None',
                                     label=f'Members (N={n_members_shown}, $p_{{mem}}\\geq${pmem_cutoff})'))

    # Add radius legend entry
    if redshift > 0 and not np.isnan(redshift):
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='gray',
                                         alpha=0.5, label=f'r={radius_kpc:.0f} kpc'))

    # =========================================================================
    # Set up RA/Dec axis labels (like original physical_images)
    # =========================================================================
    # Get corner coordinates
    corner_pixels = [(0, 0), (img_width, 0), (0, img_height), (img_width, img_height)]
    corner_radec = [pixel_to_radec(px, py, wcs, img_height) for px, py in corner_pixels]

    # Create tick positions and labels using actual RA/Dec
    n_ticks = 5
    x_ticks = np.linspace(0, img_width, n_ticks)
    y_ticks = np.linspace(0, img_height, n_ticks)

    x_labels = []
    y_labels = []
    for xt in x_ticks:
        ra, _ = pixel_to_radec(xt, img_height/2, wcs, img_height)
        x_labels.append(f'{ra:.4f}')
    for yt in y_ticks:
        _, dec = pixel_to_radec(img_width/2, yt, wcs, img_height)
        y_labels.append(f'{dec:.4f}')

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("RA (deg)", fontsize=14)
    ax.set_ylabel("Dec (deg)", fontsize=14)

    # Cluster info text box
    display_text = cluster_name
    if redshift > 0 and not np.isnan(redshift):
        display_text = f"{cluster_name}\nz={redshift:.3f}"

    ax.text(0.02, 0.98, display_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

    # Legend
    ncol = min(2, len(legend_elements))
    ax.legend(handles=legend_elements, loc='lower left',
             bbox_to_anchor=(0.02, 0.02), ncol=ncol, fontsize=9,
             frameon=True, fancybox=True, shadow=False, framealpha=0.9,
             columnspacing=0.5, handletextpad=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


# ============================================================================
# Sectors Plot (like diagnostic_plots_sectors.png)
# ============================================================================
def plot_pccg_sectors(p_ccg_results_df, output_dir, dataset_type='3p8arcmin'):
    """
    Create sectors plot showing p_{CCG} agreement distribution.

    Args:
        p_ccg_results_df: DataFrame with p_{CCG} results
        output_dir: Directory to save plots
        dataset_type: Dataset type for labeling
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_mask = ~p_ccg_results_df['p_ccg'].isna() & ~p_ccg_results_df['bar_p'].isna()
    valid_df = p_ccg_results_df[valid_mask].copy()

    if len(valid_df) == 0:
        print("Warning: No valid p_CCG results for sectors plot")
        return

    # Define agreement categories
    categories = {
        'High Agreement\n($\\bar{p}>0.5, p_{CCG}>0.5$)': (valid_df['bar_p'] > 0.5) & (valid_df['p_ccg'] > 0.5),
        'High $\\bar{p}$\nLow $p_{CCG}$': (valid_df['bar_p'] > 0.5) & (valid_df['p_ccg'] <= 0.5),
        'Low $\\bar{p}$\nHigh $p_{CCG}$': (valid_df['bar_p'] <= 0.5) & (valid_df['p_ccg'] > 0.5),
        'Low Agreement\n($\\bar{p}\\leq0.5, p_{CCG}\\leq0.5$)': (valid_df['bar_p'] <= 0.5) & (valid_df['p_ccg'] <= 0.5),
    }

    counts = [mask.sum() for mask in categories.values()]
    labels = list(categories.keys())
    total = sum(counts)
    percentages = [c/total*100 for c in counts]

    # Colors for the sectors
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # =========================================================================
    # Left: Donut chart
    # =========================================================================
    ax = axes[0]

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct='',
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )

    # Add percentage labels
    for i, (wedge, pct, cnt) in enumerate(zip(wedges, percentages, counts)):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        x = 0.75 * np.cos(np.radians(ang))
        y = 0.75 * np.sin(np.radians(ang))

        if pct > 5:  # Only show label if slice is big enough
            ax.text(x, y, f'{pct:.1f}%\n({cnt})', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')

    # Center text
    ax.text(0, 0, f'N={total}', ha='center', va='center', fontsize=16, fontweight='bold')

    ax.set_title('$p_{CCG}$ vs $\\bar{p}$ Agreement\n(Threshold: 0.5)', fontsize=14, fontweight='bold')

    # Custom legend
    legend_labels = [f'{l}: {c} ({p:.1f}%)' for l, c, p in zip(labels, counts, percentages)]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
             fontsize=10, frameon=True)

    # =========================================================================
    # Right: Bar chart with details
    # =========================================================================
    ax = axes[1]

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, cnt, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{cnt}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('\n', ' ') for l in labels], fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Number of Clusters', fontsize=14)
    ax.set_title('$p_{CCG}$ Agreement Categories', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, 'pccg_sectors.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved sectors plot to: {save_path}")

    save_path_pdf = os.path.join(output_dir, 'pccg_sectors.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Saved sectors plot (PDF) to: {save_path_pdf}")

    plt.close()


# ============================================================================
# Completeness/Purity Plots (like completeness_purity_plots.png)
# ============================================================================
def plot_pccg_completeness_purity(p_ccg_results_df, output_dir, dataset_type='3p8arcmin'):
    """
    Create completeness and purity plots for p_{CCG} analysis.

    Args:
        p_ccg_results_df: DataFrame with p_{CCG} results
        output_dir: Directory to save plots
        dataset_type: Dataset type for labeling
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_mask = ~p_ccg_results_df['p_ccg'].isna() & ~p_ccg_results_df['bar_p'].isna()
    valid_df = p_ccg_results_df[valid_mask].copy()

    if len(valid_df) == 0:
        print("Warning: No valid p_CCG results for completeness/purity plot")
        return

    # Check if we have bcg_rank information for "true positives"
    has_rank = 'bcg_rank' in valid_df.columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # =========================================================================
    # Plot 1: Completeness vs p_{CCG} threshold
    # =========================================================================
    ax = axes[0, 0]

    thresholds = np.linspace(0, 1, 51)

    # Completeness: fraction of high bar_p cases also identified by p_CCG
    completeness_barp = []
    for thresh in thresholds:
        high_barp = valid_df['bar_p'] > 0.5
        high_pccg = valid_df['p_ccg'] >= thresh
        if high_barp.sum() > 0:
            completeness_barp.append((high_barp & high_pccg).sum() / high_barp.sum())
        else:
            completeness_barp.append(np.nan)

    ax.plot(thresholds, completeness_barp, 'b-', linewidth=2, label='High $\\bar{p}$ Recovery')

    if has_rank:
        # Completeness for Rank-1 cases
        completeness_rank1 = []
        for thresh in thresholds:
            rank1 = valid_df['bcg_rank'] == 1
            high_pccg = valid_df['p_ccg'] >= thresh
            if rank1.sum() > 0:
                completeness_rank1.append((rank1 & high_pccg).sum() / rank1.sum())
            else:
                completeness_rank1.append(np.nan)
        ax.plot(thresholds, completeness_rank1, 'g-', linewidth=2, label='Rank-1 Recovery')

    ax.set_xlabel('$p_{CCG}$ Threshold', fontsize=14)
    ax.set_ylabel('Completeness', fontsize=14)
    ax.set_title('Completeness vs $p_{CCG}$ Threshold', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.9, color='r', linestyle='--', alpha=0.5, label='90% line')

    # =========================================================================
    # Plot 2: Purity vs p_{CCG} threshold
    # =========================================================================
    ax = axes[0, 1]

    # Purity: of high p_CCG cases, what fraction are also high bar_p?
    purity_barp = []
    n_selected = []
    for thresh in thresholds:
        high_pccg = valid_df['p_ccg'] >= thresh
        high_barp = valid_df['bar_p'] > 0.5
        n_selected.append(high_pccg.sum())
        if high_pccg.sum() > 0:
            purity_barp.append((high_pccg & high_barp).sum() / high_pccg.sum())
        else:
            purity_barp.append(np.nan)

    ax.plot(thresholds, purity_barp, 'b-', linewidth=2, label='High $\\bar{p}$ Purity')

    if has_rank:
        purity_rank1 = []
        for thresh in thresholds:
            high_pccg = valid_df['p_ccg'] >= thresh
            rank1 = valid_df['bcg_rank'] == 1
            if high_pccg.sum() > 0:
                purity_rank1.append((high_pccg & rank1).sum() / high_pccg.sum())
            else:
                purity_rank1.append(np.nan)
        ax.plot(thresholds, purity_rank1, 'g-', linewidth=2, label='Rank-1 Purity')

    ax.set_xlabel('$p_{CCG}$ Threshold', fontsize=14)
    ax.set_ylabel('Purity', fontsize=14)
    ax.set_title('Purity vs $p_{CCG}$ Threshold', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Number selected vs threshold
    # =========================================================================
    ax = axes[1, 0]

    ax.plot(thresholds, n_selected, 'k-', linewidth=2)
    ax.fill_between(thresholds, 0, n_selected, alpha=0.3)

    ax.set_xlabel('$p_{CCG}$ Threshold', fontsize=14)
    ax.set_ylabel('Number of Clusters Selected', fontsize=14)
    ax.set_title('Selection Count vs Threshold', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add vertical lines at key thresholds
    for thresh, color, label in [(0.5, 'r', '0.5'), (0.8, 'orange', '0.8')]:
        idx = int(thresh * 50)
        ax.axvline(thresh, color=color, linestyle='--', alpha=0.7)
        ax.text(thresh + 0.02, ax.get_ylim()[1] * 0.9, f'N={n_selected[idx]}',
               fontsize=10, color=color)

    # =========================================================================
    # Plot 4: Completeness vs Purity trade-off
    # =========================================================================
    ax = axes[1, 1]

    ax.plot(purity_barp, completeness_barp, 'b-', linewidth=2, marker='o', markersize=3)

    # Annotate some threshold points
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        idx = int(thresh * 50)
        if idx < len(purity_barp) and not np.isnan(purity_barp[idx]) and not np.isnan(completeness_barp[idx]):
            ax.annotate(f't={thresh}', (purity_barp[idx], completeness_barp[idx]),
                       textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel('Purity', fontsize=14)
    ax.set_ylabel('Completeness', fontsize=14)
    ax.set_title('Completeness vs Purity Trade-off', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, 'pccg_completeness_purity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved completeness/purity plot to: {save_path}")

    save_path_pdf = os.path.join(output_dir, 'pccg_completeness_purity.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Saved completeness/purity plot (PDF) to: {save_path_pdf}")

    plt.close()


# ============================================================================
# Diagnostic Plots: p_{CCG} vs bar_p Comparison
# ============================================================================
def plot_pccg_vs_barp_diagnostic(p_ccg_results_df, output_dir, dataset_type='3p8arcmin'):
    """
    Create diagnostic plots comparing p_{CCG} with bar_p.

    Args:
        p_ccg_results_df: DataFrame with p_{CCG} results
        output_dir: Directory to save plots
        dataset_type: Dataset type for labeling
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Filter valid results
    valid_mask = ~p_ccg_results_df['p_ccg'].isna() & ~p_ccg_results_df['bar_p'].isna()
    valid_df = p_ccg_results_df[valid_mask].copy()

    if len(valid_df) == 0:
        print("Warning: No valid p_CCG results to plot")
        return

    # =========================================================================
    # Plot 1: p_{CCG} vs bar_p scatter
    # =========================================================================
    ax = axes[0, 0]
    scatter = ax.scatter(valid_df['bar_p'], valid_df['p_ccg'],
                        c=valid_df['z'], cmap='viridis',
                        alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='Redshift')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='1:1 line')
    ax.set_xlabel('$\\bar{p}$ (ML Probability)', fontsize=14)
    ax.set_ylabel('$p_{CCG}$ (Member Density)', fontsize=14)
    ax.set_title('$p_{CCG}$ vs $\\bar{p}$', fontsize=16)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Correlation
    if len(valid_df) > 2:
        corr = np.corrcoef(valid_df['bar_p'], valid_df['p_ccg'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}\nN = {len(valid_df)}',
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # =========================================================================
    # Plot 2: Histograms of p_{CCG} and bar_p
    # =========================================================================
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(valid_df['bar_p'], bins=bins, alpha=0.6, label='$\\bar{p}$',
           edgecolor='black', color='steelblue')
    ax.hist(valid_df['p_ccg'], bins=bins, alpha=0.6, label='$p_{CCG}$',
           edgecolor='black', color='orange')
    ax.set_xlabel('Probability', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 3: Member count distribution
    # =========================================================================
    ax = axes[0, 2]
    if 'n_members' in valid_df.columns:
        member_counts = valid_df['n_members'].values
        ax.hist(member_counts, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(np.median(member_counts), color='red', linestyle='--',
                  label=f'Median: {np.median(member_counts):.0f}')
        ax.set_xlabel('Number of Members within Radius', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Member Count Distribution', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No member count data', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)

    # =========================================================================
    # Plot 4: p_{CCG} vs Member Count
    # =========================================================================
    ax = axes[1, 0]
    if 'n_members' in valid_df.columns:
        scatter = ax.scatter(valid_df['n_members'], valid_df['p_ccg'],
                  c=valid_df['bar_p'], cmap='RdYlGn',
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='$\\bar{p}$')
        ax.set_xlabel('Number of Members', fontsize=14)
        ax.set_ylabel('$p_{CCG}$', fontsize=14)
        ax.set_title('$p_{CCG}$ vs Member Count', fontsize=16)
        ax.grid(True, alpha=0.3)

    # =========================================================================
    # Plot 5: Agreement Analysis
    # =========================================================================
    ax = axes[1, 1]
    # Define agreement categories
    high_both = (valid_df['bar_p'] > 0.7) & (valid_df['p_ccg'] > 0.7)
    high_barp_low_pccg = (valid_df['bar_p'] > 0.7) & (valid_df['p_ccg'] < 0.3)
    low_barp_high_pccg = (valid_df['bar_p'] < 0.3) & (valid_df['p_ccg'] > 0.7)
    low_both = (valid_df['bar_p'] < 0.3) & (valid_df['p_ccg'] < 0.3)

    categories = ['High Both', 'High $\\bar{p}$,\nLow $p_{CCG}$',
                 'Low $\\bar{p}$,\nHigh $p_{CCG}$', 'Low Both']
    counts = [high_both.sum(), high_barp_low_pccg.sum(),
             low_barp_high_pccg.sum(), low_both.sum()]
    colors = ['green', 'orange', 'blue', 'gray']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Agreement Analysis', fontsize=16)
    ax.tick_params(axis='x', rotation=15)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{count}\n({count/len(valid_df)*100:.1f}%)',
               ha='center', va='bottom', fontsize=10)

    # =========================================================================
    # Plot 6: p_{CCG} by Redshift Bins
    # =========================================================================
    ax = axes[1, 2]
    z_bins = [0, 0.3, 0.5, 0.7, 1.0, 2.0]
    z_labels = ['0-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0', '>1.0']

    mean_pccg_by_z = []
    mean_barp_by_z = []

    for i in range(len(z_bins) - 1):
        z_mask = (valid_df['z'] >= z_bins[i]) & (valid_df['z'] < z_bins[i+1])
        if z_mask.sum() > 0:
            mean_pccg_by_z.append(valid_df[z_mask]['p_ccg'].mean())
            mean_barp_by_z.append(valid_df[z_mask]['bar_p'].mean())
        else:
            mean_pccg_by_z.append(np.nan)
            mean_barp_by_z.append(np.nan)

    x = np.arange(len(z_labels))
    width = 0.35

    ax.bar(x - width/2, mean_barp_by_z, width, label='$\\bar{p}$',
          color='steelblue', edgecolor='black')
    ax.bar(x + width/2, mean_pccg_by_z, width, label='$p_{CCG}$',
          color='orange', edgecolor='black')

    ax.set_xlabel('Redshift Bin', fontsize=14)
    ax.set_ylabel('Mean Probability', fontsize=14)
    ax.set_title('Mean Probabilities by Redshift', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(z_labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, 'pccg_diagnostic_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic plots to: {save_path}")

    # Also save PDF
    save_path_pdf = os.path.join(output_dir, 'pccg_diagnostic_plots.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Saved diagnostic plots (PDF) to: {save_path_pdf}")

    plt.close()


def plot_pccg_summary_scatter(p_ccg_results_df, output_dir):
    """
    Create a single summary scatter plot for p_{CCG} vs bar_p with annotations.

    Args:
        p_ccg_results_df: DataFrame with p_{CCG} results
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_mask = ~p_ccg_results_df['p_ccg'].isna() & ~p_ccg_results_df['bar_p'].isna()
    valid_df = p_ccg_results_df[valid_mask].copy()

    if len(valid_df) == 0:
        print("Warning: No valid p_CCG results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    # Scatter plot with size based on member count
    sizes = valid_df['n_members'].values * 5 + 20 if 'n_members' in valid_df.columns else 50

    scatter = ax.scatter(valid_df['bar_p'], valid_df['p_ccg'],
                        c=valid_df['z'], cmap='viridis',
                        s=sizes, alpha=0.7,
                        edgecolors='black', linewidths=0.5)

    # Reference lines
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='1:1 line')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Redshift', fontsize=14)

    # Labels
    ax.set_xlabel('$\\bar{p}$ (ML Probability)', fontsize=16)
    ax.set_ylabel('$p_{CCG}$ (Member Density Probability)', fontsize=16)
    ax.set_title('Comparison: ML Probability vs Member-based CCG Probability', fontsize=18)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add statistics
    if len(valid_df) > 2:
        corr = np.corrcoef(valid_df['bar_p'], valid_df['p_ccg'])[0, 1]
        agree_high = ((valid_df['bar_p'] > 0.5) & (valid_df['p_ccg'] > 0.5)).sum()
        agree_pct = agree_high / len(valid_df) * 100

        stats_text = f'N = {len(valid_df)}\nCorrelation: {corr:.3f}\nAgreement (>0.5): {agree_pct:.1f}%'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               ha='left', va='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Size legend
    if 'n_members' in valid_df.columns:
        # Create size legend
        for n in [5, 20, 50]:
            ax.scatter([], [], s=n*5+20, c='gray', alpha=0.5,
                      label=f'{n} members')
        ax.legend(loc='lower right', fontsize=10, title='Member count')

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'pccg_vs_barp_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to: {save_path}")

    plt.close()


# ============================================================================
# Image Selection Functions
# ============================================================================
def select_diverse_images(detailed_results, n_images=20):
    """
    Select diverse images showing best matches, mismatches, and edge cases.

    Args:
        detailed_results: List of detailed result dictionaries
        n_images: Total number of images to select

    Returns:
        List of selected result dictionaries
    """
    valid_results = [r for r in detailed_results
                    if r.get('error') is None and
                    len(r.get('p_ccg', [])) > 0 and
                    len(r.get('candidate_probs', [])) > 0 and
                    not np.isnan(r.get('redshift', np.nan))]

    if len(valid_results) == 0:
        return []

    # Calculate agreement score for each result
    for r in valid_results:
        bar_p = r['candidate_probs'][0] if len(r['candidate_probs']) > 0 else 0
        p_ccg = r['p_ccg'][0] if len(r['p_ccg']) > 0 else 0
        r['_agreement'] = 1 - abs(bar_p - p_ccg)
        r['_bar_p'] = bar_p
        r['_p_ccg'] = p_ccg

    # Sort by agreement
    sorted_by_agreement = sorted(valid_results, key=lambda r: r['_agreement'], reverse=True)

    selected = []

    # Category 1: Best matches (high agreement, high both)
    best_matches = [r for r in sorted_by_agreement if r['_bar_p'] > 0.7 and r['_p_ccg'] > 0.7]
    n_best = min(n_images // 4, len(best_matches))
    selected.extend(best_matches[:n_best])

    # Category 2: High bar_p, low p_ccg (ML confident, but few members)
    high_ml_low_mem = [r for r in valid_results if r['_bar_p'] > 0.7 and r['_p_ccg'] < 0.3]
    n_high_ml = min(n_images // 4, len(high_ml_low_mem))
    for r in high_ml_low_mem[:n_high_ml]:
        if r not in selected:
            selected.append(r)

    # Category 3: Low bar_p, high p_ccg (ML uncertain, but many members)
    low_ml_high_mem = [r for r in valid_results if r['_bar_p'] < 0.5 and r['_p_ccg'] > 0.7]
    n_low_ml = min(n_images // 4, len(low_ml_high_mem))
    for r in low_ml_high_mem[:n_low_ml]:
        if r not in selected:
            selected.append(r)

    # Category 4: Fill with diverse redshift range
    remaining = [r for r in valid_results if r not in selected]
    if remaining:
        # Sort by redshift and pick evenly spaced
        remaining_sorted = sorted(remaining, key=lambda r: r.get('redshift', 0))
        n_remaining = n_images - len(selected)
        if n_remaining > 0 and len(remaining_sorted) > 0:
            step = max(1, len(remaining_sorted) // n_remaining)
            for i in range(0, len(remaining_sorted), step):
                if len(selected) >= n_images:
                    break
                selected.append(remaining_sorted[i])

    # Clean up temporary keys
    for r in selected:
        r.pop('_agreement', None)
        r.pop('_bar_p', None)
        r.pop('_p_ccg', None)

    return selected[:n_images]


# ============================================================================
# Batch Visualization for physical_images_with_members
# ============================================================================
def generate_physical_images_with_members(p_ccg_detailed_results, image_dir,
                                          output_dir, dataset_type='3p8arcmin',
                                          rm_member_dir=None, max_images=None,
                                          pmem_cutoff=0.2):
    """
    Generate physical images with member overlays and p_{CCG} values.

    Args:
        p_ccg_detailed_results: List of dictionaries from CCGProbabilityCalculator
        image_dir: Directory containing cluster images
        output_dir: Directory to save plots
        dataset_type: '2p2arcmin' or '3p8arcmin'
        rm_member_dir: Directory with member catalogs
        max_images: Maximum number of images to generate (None for all)
        pmem_cutoff: Minimum pmem for member display
    """
    # Create output directory
    physical_dir = os.path.join(output_dir, 'physical_images_with_members')
    os.makedirs(physical_dir, exist_ok=True)

    if rm_member_dir is None:
        rm_member_dir = get_data_paths()['rm_member_dir']

    # Select diverse images
    if max_images:
        selected_results = select_diverse_images(p_ccg_detailed_results, max_images)
    else:
        selected_results = [r for r in p_ccg_detailed_results if r.get('error') is None]

    n_generated = 0
    for result in selected_results:
        cluster_name = result.get('cluster_name', 'unknown')
        if cluster_name == 'unknown':
            continue

        # Find image
        image_path = find_cluster_image(cluster_name, image_dir)
        if image_path is None:
            continue

        # Get candidate data
        candidates_pixel = np.array(result.get('candidates_pixel', []))
        candidate_probs = np.array(result.get('candidate_probs', []))
        p_ccg_values = np.array(result.get('p_ccg', []))
        member_counts = np.array(result.get('member_counts', []))
        redshift = result.get('redshift', np.nan)
        radius_kpc = result.get('radius_kpc', 300.0)

        if len(candidates_pixel) == 0:
            continue

        # Target info
        target_coords = result.get('target_coords')
        target_prob = result.get('target_prob')

        # Output path
        save_path = os.path.join(physical_dir, f'{cluster_name}_pccg.png')

        try:
            plot_cluster_with_members_pccg(
                cluster_name, image_path, candidates_pixel,
                candidate_probs, p_ccg_values, member_counts,
                redshift, radius_kpc, wcs=None,
                members_df=None, rm_member_dir=rm_member_dir,
                save_path=save_path, dataset_type=dataset_type,
                target_coords=target_coords, target_prob=target_prob,
                pmem_cutoff=pmem_cutoff
            )
            n_generated += 1
        except Exception as e:
            print(f"Warning: Failed to generate plot for {cluster_name}: {e}")

    print(f"Generated {n_generated} physical images with members")
    return physical_dir


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate p_{CCG} visualizations")
    parser.add_argument('--pccg_csv', type=str, required=True,
                       help='Path to p_ccg_results.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--dataset_type', type=str, default='3p8arcmin')

    args = parser.parse_args()

    # Load results
    results_df = pd.read_csv(args.pccg_csv)

    # Generate all diagnostic plots
    plot_pccg_vs_barp_diagnostic(results_df, args.output_dir, args.dataset_type)
    plot_pccg_summary_scatter(results_df, args.output_dir)
    plot_pccg_sectors(results_df, args.output_dir, args.dataset_type)
    plot_pccg_completeness_purity(results_df, args.output_dir, args.dataset_type)

    print("Visualization complete!")
