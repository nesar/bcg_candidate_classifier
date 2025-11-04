import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from plot_config import setup_plot_style, COLORS, FONTS, SIZES

def load_csvs(csv_dir):
    files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    breakdown_file = [f for f in files if "breakdown" in f.lower()][0]
    group_file     = [f for f in files if "importance" in f.lower()][0]
    breakdown_df   = pd.read_csv(os.path.join(csv_dir, breakdown_file))
    group_df       = pd.read_csv(os.path.join(csv_dir, group_file))
    return breakdown_df, group_df

def plot_feature_breakdown(breakdown_df, group_df, output="feature_breakdown"):
    # Apply consistent plot style
    setup_plot_style()

    # Normalize column names
    breakdown_df = breakdown_df.rename(columns={
        "group_title":"Group",
        "physical_feature_name":"Feature",
        "importance":"Importance"
    })
    group_df = group_df.rename(columns={
        "group_title":"Group",
        "total_importance":"Importance"
    })

    # Color palette - use consistent colors from plot_config
    if "color" in group_df.columns:
        group_colors = dict(zip(group_df["Group"], group_df["color"]))
    else:
        palette = COLORS['group_palette']
        group_colors = {g:palette[i%len(palette)] for i,g in enumerate(group_df["Group"])}

    group_totals = dict(zip(group_df["Group"], group_df["Importance"]))

    # Order features
    order=[]
    for g in group_df.sort_values("Importance", ascending=False)["Group"]:
        sub = breakdown_df[breakdown_df["Group"]==g].sort_values("Importance", ascending=True)
        for f in sub["Feature"]:
            order.append((g,f))

    # labels = [f for g,f in order]
    # vals   = [breakdown_df[(breakdown_df["Group"]==g)&(breakdown_df["Feature"]==f)]["Importance"].values[0] for g,f in order]
    # cols   = [group_colors[g] for g,f in order]

    # # Styling
    # plt.rcParams.update({"text.usetex":False,"font.family":"serif","mathtext.fontset":"cm","axes.linewidth":1.2})
    # fig,ax = plt.subplots(figsize=(12,24))
    # bars = ax.barh(labels, vals, color=cols, edgecolor="black", height=1)

    labels = [f for g, f in order]
    vals   = [breakdown_df[(breakdown_df["Group"]==g)&(breakdown_df["Feature"]==f)]["Importance"].values[0] for g,f in order]
    cols   = [group_colors[g] for g,f in order]

    fig, ax = plt.subplots(figsize=(12, 24))

    # Define manual y positions
    height =1.0        # bar thickness
    gap = 0.4           # gap between bars
    y = np.arange(len(labels)) * (height + gap)

    bars = ax.barh(y, vals, color=cols, edgecolor="black", height=height)

    for b,v in zip(bars,vals):
        ax.text(b.get_width()+0.002, b.get_y()+b.get_height()/2,
                f"{v:.3f}", va="center", fontsize=FONTS['annotation'])

    ax.set_xlim(0, max(vals)*1.19)
    ax.set_xlabel(r"Importance Score", fontsize=FONTS['label'])
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(axis='x', labelsize=FONTS['tick'])
    ax.tick_params(axis='y', labelsize=FONTS['tick'])

    for s in ["top","right","left","bottom"]: ax.spines[s].set_visible(True)
    ax.xaxis.set_ticks_position("both"); ax.yaxis.set_ticks_position("both")
    ax.tick_params(axis="both", which="both", direction="in")

    legend = [Patch(facecolor=group_colors[g], edgecolor="black",
                    label=f"{g}\n(total={group_totals[g]:.3f})") for g in group_df["Group"]]
    ax.legend(handles=legend, title=r"Groups (Total Importance)",
              fontsize=FONTS['legend'], title_fontsize=FONTS['legend'], loc="upper right",
              bbox_to_anchor=(0.98,0.65), frameon=True)
    
    # Match y positions to labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    plt.tight_layout()
    for ext in ["png","pdf"]:
        plt.savefig(f"{output}.{ext}", dpi=SIZES['dpi'], bbox_inches="tight")
    print(f"Saved: {output}.{ext}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_physical_results.py <csv_dir>")
        sys.exit(1)

    csv_dir = sys.argv[1]
    breakdown_df, group_df = load_csvs(csv_dir)
    plot_feature_breakdown(breakdown_df, group_df, output=os.path.join(csv_dir, "feature_breakdown_with_group_totals"))