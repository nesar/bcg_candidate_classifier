import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

labels = ["Rank 1", "Rank 2", "Rank 3", "Rest"]  # updated legend label
single_values = [84.4, 11.4, 2.1, 2.1]
multi_values  = [92.7, 5.2, 1.3, 0.8]

def improved_donut(ax, values, title, success, if_legend):
    colors = plt.cm.Paired.colors
    wedges, _ = ax.pie(values, startangle=90, colors=colors, radius=1.05,
                       wedgeprops={'edgecolor':'white','linewidth':1.2})
    ax.add_artist(plt.Circle((0,0),0.65,fc='white'))
    ax.text(0,0, title + "\n" + f"Acc: {success}%",ha='center',va='center',fontsize=28)

    for i,(w,v) in enumerate(zip(wedges,values)):
        ang = (w.theta2 + w.theta1)/2
        x,y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        r = 1.01

        # Longer arrow for "Rest", slightly shorter for others
        if labels[i] == "Rest":
            arrow_len = 1.75

        elif labels[i] == "Rank 2":
            arrow_len = 1.2

        elif v < 12:
            arrow_len = 1.45
        else:
            arrow_len = None

        if arrow_len:

            if labels[i] == "Rest":

                x_edit = x - 0.15
                y_edit = y - 0.25

                ax.annotate(f"{labels[i]} ({v:.1f}%)",
                            xy=(r*x, r*y), xytext=(arrow_len*x_edit, arrow_len*y_edit),
                            ha='center', va='center', fontsize=18,
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.8,
                                            shrinkA=0, shrinkB=2))


            else: 
                ax.annotate(f"{labels[i]} ({v:.1f}%)",
                            xy=(r*x, r*y), xytext=(arrow_len*x, arrow_len*y),
                            ha='center', va='center', fontsize=18,
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.8,
                                            shrinkA=0, shrinkB=2))
            

        else:
            ax.text(0.5*x, 0.85*y, f"{labels[i]} ({v:.1f}%)", ha='center', va='center', fontsize=18)

    legend_elems = [Patch(facecolor=colors[i], label=l) for i,l in enumerate(labels)]
    if if_legend:
        ax.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1,0.5),
                  fontsize=18, frameon=True)

fig,axs = plt.subplots(1,2,figsize=(14,7))
improved_donut(axs[0], single_values, "Single-target", 97.9, True)
improved_donut(axs[1], multi_values, "Multi-target", 99.2, False)
plt.tight_layout()
plt.savefig('/lcrc/project/cosmo_ai/nramachandra/Projects/BCGs_swing/codes/bcg_candidate_classifier/trained_models/plots/sectors.png', bbox_inches='tight')
plt.show()