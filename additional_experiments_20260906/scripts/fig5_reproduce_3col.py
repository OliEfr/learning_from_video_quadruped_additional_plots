"""Variant of the reproduced Fig. 5: imitation-score column dropped, remaining three
columns 20% narrower each (so the whole figure is narrower too).

Same data and styling as fig5_reproduce.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig5_reproduce as base
import plot_DEFINITIONS

COL_WIDTH_SCALE = 0.8
METRIC_FIELDS = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot"]

plt.rcParams.update({"font.size": 20, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "legend.fontsize": 18})


def main():
    n_cols, n_rows = len(METRIC_FIELDS), len(base.EXPERIMENT_DIRS)
    # square cells: narrowing a column shortens it too, so the height scales with it
    fig = plt.figure(figsize=(n_cols * 6 * COL_WIDTH_SCALE, n_rows * 5 * COL_WIDTH_SCALE))
    experiment_names = plot_DEFINITIONS.ExperimentNames()

    for i, (experiment_dir, dataset_key) in enumerate(zip(base.EXPERIMENT_DIRS, base.DATASET_KEYS)):
        for j, metric_field in enumerate(METRIC_FIELDS):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            base.draw_panel(base.load_mean(dataset_key, metric_field), metric_field, ax)

            ax.set_title("")
            if i < n_rows - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field],
                             fontweight="bold", y=1.4)

            if j == 0:
                name = experiment_names.map_experiment_dir_to_experiment_name(experiment_dir)
                ax.text(-0.7 / COL_WIDTH_SCALE, 0.5,
                        name.replace(" ", "\n", 1).replace(" Camera", "\nCamera", 1).replace(" (", "\n("),
                        transform=ax.transAxes, rotation=0, verticalalignment="center",
                        horizontalalignment="left", fontweight="bold")

            if hasattr(ax, "collections") and ax.collections:
                if ax.collections[0].colorbar:
                    ax.collections[0].colorbar.remove()
                if i == 0:
                    cax = ax.inset_axes([0.1, 1.05, 0.8, 0.05])
                    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
                    cbar.ax.xaxis.set_ticks_position("top")
                    cbar.ax.xaxis.set_label_position("top")

    fig.subplots_adjust(left=0.19, right=1, top=0.9, bottom=0.5, hspace=0.1, wspace=0.05)

    out = os.path.join(base.ROOT, "fig5_reproduced_3col.pdf")
    fig.savefig(out, bbox_inches="tight", format="pdf")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
