"""Re-render the paper's Fig. 5 (combined_heatmap_grid.pdf).

The original generator is
    project_repos/isaac_lab/IsaacLab/plot_errorOnTargetDistribution.py::main()
which reads per-command evaluation YAMLs from
    logs/rsl_rl/unitree_go2_AMPflat/<experiment>_SEED_*/TargetXYDistributionEvaluation/*.yaml

Those logs are DVC-tracked on a Nextcloud remote and are not present locally, so this
script substitutes the cell values recovered from the shipped vector PDF by
`fig5_extract_from_pdf.py`.  Everything else -- layout, styling, colour limits, titles --
is a faithful copy of the original code path.

Output: fig5_reproduced.pdf (+ .png) in the parent folder.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "/home/admin_07/project_repos/isaac_lab/IsaacLab")
import plot_DEFINITIONS  # noqa: E402  (read-only import of the paper's own definitions)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "fig5_recovered")

COLORBAR_LIMITS = {
    "mean_mechanical_cot": (0.8, 2.0),
    "error_vel_xy": (0.00, 0.1),
    "agent_expert_distances": (2.0, 4.0),
    "heading_error": (0.5, 1.5),
    "error_vel_yaw": (0.0, 0.4),
}

EXPERIMENT_DIRS = [
    "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_*",
    "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_*",
    "2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_*",
]
DATASET_KEYS = [
    "mocap_AMP_for_hardware",
    "fromVision_motions_DepthCam",
    "fromVision_motions_DepthCam_extendedWithoutReverse",
]
METRIC_FIELDS = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot", "agent_expert_distances"]

X_FIELD, Y_FIELD = "target_velocity_x", "target_velocity_y"

# NOTE these are the parameters the original script uses for the grid plot.
plt.rcParams.update(
    {
        "font.size": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
)


def load_mean(dataset_key, metric_field):
    fn = os.path.join(DATA_DIR, f"{dataset_key}__{metric_field}.csv")
    df = pd.read_csv(fn, index_col=0)
    df.index = np.round(df.index.astype(float), 2)
    df.columns = np.round(df.columns.astype(float), 2)
    df.index.name = Y_FIELD
    df.columns.name = X_FIELD
    return df


def draw_panel(df_mean, metric_field, ax):
    vmin, vmax = COLORBAR_LIMITS.get(metric_field, (None, None))
    x_label = plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING[X_FIELD]
    y_label = plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING[Y_FIELD].replace(" [m", "\n[m")
    plot_title = plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field]

    sns.heatmap(
        data=df_mean,
        annot=False,
        fmt="",
        cmap="viridis",
        cbar_kws={"label": "", "shrink": 0.3},
        square=True,
        annot_kws={"size": 18},
        linewidths=0.5,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
        center=(vmin + vmax) / 2 if vmin is not None and vmax is not None else None,
        ax=ax,
    )
    ax.set_title(plot_title, pad=25)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)


def main():
    n_cols, n_rows = len(METRIC_FIELDS), len(EXPERIMENT_DIRS)
    combined_fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
    experiment_names = plot_DEFINITIONS.ExperimentNames()

    for i, (experiment_dir, dataset_key) in enumerate(zip(EXPERIMENT_DIRS, DATASET_KEYS)):
        for j, metric_field in enumerate(METRIC_FIELDS):
            ax = combined_fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
            draw_panel(load_mean(dataset_key, metric_field), metric_field, ax)

            # Styling (verbatim from plot_errorOnTargetDistribution.py::main)
            ax.set_title("")
            if i < n_rows - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(
                    plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field],
                    fontweight="bold", y=1.4,
                )

            if j == 0:
                name = experiment_names.map_experiment_dir_to_experiment_name(experiment_dir)
                ax.text(
                    -0.7, 0.5,
                    name.replace(" ", "\n", 1).replace(" Camera", "\nCamera", 1).replace(" (", "\n("),
                    transform=ax.transAxes, rotation=0, verticalalignment="center",
                    horizontalalignment="left", fontweight="bold",
                )

            if hasattr(ax, "collections") and ax.collections:
                if ax.collections[0].colorbar:
                    ax.collections[0].colorbar.remove()
                if i == 0:
                    cax = ax.inset_axes([0.1, 1.05, 0.8, 0.05])
                    cbar = combined_fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
                    cbar.ax.xaxis.set_ticks_position("top")
                    cbar.ax.xaxis.set_label_position("top")

    combined_fig.subplots_adjust(left=0.19, right=1, top=0.9, bottom=0.5, hspace=0.1, wspace=0.05)

    out_pdf = os.path.join(ROOT, "fig5_reproduced.pdf")
    combined_fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
    combined_fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=110)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
