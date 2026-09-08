"""Combined replacement for the paper's Fig. 5.

Merges two existing figures into a single double-column (7.16 in) figure that shares one
row axis -- the three AMP datasets, top to bottom, in the paper's Fig. 5 order:

    cols 1-2: fig_exp2_state_coverage_2col_log
                  what the expert dataset *contains* (base-velocity histogram, colour =
                  number of data points per bin, log scale)
    cols 3-5: fig5_reproduced_3col
                  how the trained policy *performs* over the target-command grid
                  (tracking error vel. / yaw, cost of transport)

Layout follows fig5_reproduced.pdf: bold block header, column title, colour bar directly
above the panels; dataset names run vertically, centred on their row.  Both blocks use the same panel height, so every row lines up top and
bottom; that fixes the panel height independently of the 21 x 7 Fig. 5 grid, so its cells
end up slightly taller than wide (~1.2:1) instead of exactly square.

The Fig. 5 heatmaps keep the *picture* of the original (top row = the CSV row labelled
v_y = -0.3, as seaborn drew it) while the axis reads v_y increasing upwards like the coverage
panels, i.e. the rows are flipped relative to the CSV index (author's request).  Cell values,
colormap, colour limits and the white cell separators are unchanged; the separators use the
same stroke-to-cell ratio as the original at print size.  Nothing is rasterized, so the PDF
keeps the thin vector separators rather than a 100 dpi image of them.

Data sources are unchanged: coverage from exp2_coverage.dataset_frames, Fig. 5 cell values
from data/fig5_recovered/*.csv (recovered from the shipped vector PDF, see
../../additional_experiments_20260906/scripts/fig5_extract_from_pdf.py).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from common import CMD_RANGES, DATA, DOUBLE_W, savefig
from exp2_coverage import SOURCES, dataset_frames
from exp2_state_coverage_2col import COLS, hist

sys.path.insert(0, "/home/admin_07/project_repos/isaac_lab/IsaacLab")
import plot_DEFINITIONS  # noqa: E402  (read-only import of the paper's own definitions)

# --------------------------------------------------------------------------------------
# Fig. 5 half: data, colour limits and titles exactly as in plot_errorOnTargetDistribution
# --------------------------------------------------------------------------------------
FIG5_DIR = os.path.join(DATA, "fig5_recovered")
DATASET_KEYS = [
    "mocap_AMP_for_hardware",
    "fromVision_motions_DepthCam",
    "fromVision_motions_DepthCam_extendedWithoutReverse",
]
METRICS = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot"]
CBAR_LIMITS = {"error_vel_xy": (0.0, 0.1), "error_vel_yaw": (0.0, 0.4), "mean_mechanical_cot": (0.8, 2.0), "agent_expert_distances": (2.0, 4.0)}
CBAR_TICKS = {"error_vel_xy": [0.0, 0.05, 0.1], "error_vel_yaw": [0.0, 0.2, 0.4], "mean_mechanical_cot": [0.8, 1.4, 2.0],
              "agent_expert_distances": [2.0, 3.0, 4.0]}
# 4-column variant (Exp. 4): adds the paper's own AMP-space metric "agent_expert_distances" (play.py), i.e. the per-step
# nearest-neighbour distance of the agent's (joint pos, joint vel) to the expert set, averaged over the episode.
METRICS_4COL = METRICS + ["agent_expert_distances"]
TITLE_OVERRIDE = {"agent_expert_distances": "Agent–Expert Dist. ↓\n(AMP obs. space)"}
FIG5_X = np.arange(-1.0, 1.001, 0.1)  # target_velocity_x, 21 cells
FIG5_Y = np.arange(-0.3, 0.301, 0.1)  # target_velocity_y, 7 cells
ROW_LABELS = ["MoCap", "Video w.\nDepth Cam.", "Video w.\nDepth Cam.\n(extended)"]
COV_LABEL = "Number of Datapoints (log)"

# --------------------------------------------------------------------------------------
# Geometry, in inches on a DOUBLE_W wide canvas (explicit add_axes: the 2+3 column split is
# not a regular grid, and the panel aspects of the two halves differ)
# --------------------------------------------------------------------------------------
H = 2.79
M_L, M_R = 0.05, 0.04
GUTTER = 0.36            # dataset names, vertical text
COV_YLAB = [0.32, 0.32]  # room for ytick labels + ylabel, per coverage column
COV_W = 0.70
BLOCK_GAP = 0.10
MET_YLAB = 0.30
MET_GAP = 0.09
PANEL_H = 0.52           # identical in both blocks, so the rows align top and bottom

# top to bottom: bold block header, column title, colour bar (ticks above it), panels
Y_HDR, Y_TITLE_BOT, Y_CB, Y_ROW0 = 0.05, 0.36, 0.49, 0.60
CB_H, ROW_GAP, XLABEL_DROP = 0.05, 0.13, 0.22
CELL_EDGE_LW = 0.17      # same stroke-to-cell ratio as the original Fig. 5 at print size

COV_X = [M_L + GUTTER + COV_YLAB[0]]
COV_X.append(COV_X[0] + COV_W + COV_YLAB[1])
MET_X0 = COV_X[1] + COV_W + BLOCK_GAP + MET_YLAB
MET_W = (DOUBLE_W - M_R - MET_X0 - 2 * MET_GAP) / 3
MET_X = [MET_X0 + i * (MET_W + MET_GAP) for i in range(3)]
ROW_Y = [Y_ROW0 + r * (PANEL_H + ROW_GAP) for r in range(3)]
PANELS_TOP, PANELS_BOT = ROW_Y[0], ROW_Y[2] + PANEL_H
COV_MID = (COV_X[0] + COV_X[1] + COV_W) / 2
MET_MID = (MET_X[0] + MET_X[2] + MET_W) / 2


def rect(x, y, w, h):
    """inches from the top-left corner -> matplotlib figure-fraction rect"""
    return [x / DOUBLE_W, 1.0 - (y + h) / H, w / DOUBLE_W, h / H]


def fx(x):
    return x / DOUBLE_W


def fy(y):
    return 1.0 - y / H


def load_fig5(dataset_key, metric):
    df = pd.read_csv(os.path.join(FIG5_DIR, f"{dataset_key}__{metric}.csv"), index_col=0)
    df.index = np.round(df.index.astype(float), 2)      # target_velocity_y
    df.columns = np.round(df.columns.astype(float), 2)  # target_velocity_x
    return df


def cell_edges(centres):
    c = np.asarray(centres, dtype=float)
    step = np.median(np.diff(c))
    return np.concatenate([c - step / 2, [c[-1] + step / 2]])


def colour_bar(fig, mappable, x, w, ticks, labels=None):
    cax = fig.add_axes(rect(x, Y_CB, w, CB_H))
    cb = Colorbar(cax, mappable, orientation="horizontal")
    cb.set_ticks(ticks)
    if labels is not None:
        cax.set_xticklabels(labels)
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)


def main(metrics=METRICS, name="fig5_combined_data_and_performance"):
    n_met = len(metrics)
    MET_W = (DOUBLE_W - M_R - MET_X0 - (n_met - 1) * MET_GAP) / n_met
    MET_X = [MET_X0 + i * (MET_W + MET_GAP) for i in range(n_met)]
    MET_MID = (MET_X[0] + MET_X[-1] + MET_W) / 2
    D = {src: dataset_frames(src)[0] for src in SOURCES}

    # one shared log colour scale for both coverage columns; empty bins are masked, so it starts at 1
    occ = np.concatenate([hist(D[s], c[0], c[1], c[4], c[5], c[6], c[7])[2].ravel() for s in SOURCES for c in COLS])
    occ = occ[occ > 0]
    vmax = float(occ.max())
    cov_norm = LogNorm(vmin=1, vmax=vmax)
    print(f"coverage: {occ.size} occupied bins, counts 1..{vmax:.0f} (median {np.median(occ):.0f})")

    fig = plt.figure(figsize=(DOUBLE_W, H))
    cov_mesh, met_mesh = None, {}

    for r, src in enumerate(SOURCES):
        # ---------------- Expert Dataset ----------------
        for c, (kx, ky, xl, yl, xr, yr, bx, by, cmd, _t) in enumerate(COLS):
            ax = fig.add_axes(rect(COV_X[c], ROW_Y[r], COV_W, PANEL_H))
            hx, hy, Hh = hist(D[src], kx, ky, xr, yr, bx, by)
            cov_mesh = ax.pcolormesh(hx, hy, np.ma.masked_where(Hh <= 0, Hh).T,
                                     cmap="viridis", norm=cov_norm, edgecolors="face", linewidth=0)
            cx, cy = CMD_RANGES[cmd[0]], CMD_RANGES[cmd[1]]
            ax.add_patch(Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0],
                                   fill=False, ec="black", lw=0.7, ls="--"))
            if r == 0:  # annotate the box once, not in every panel
                ax.text(cx[0], cy[0] - 0.02 * (yr[1] - yr[0]), "target cmd range",
                        ha="left", va="top", fontsize=4.6, color="black")
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax.grid(alpha=0.25, lw=0.35)
            ax.tick_params(length=1.8, pad=1.5, labelsize=5.2)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-0.5, 0, 0.5] if c == 0 else [-2, 0, 2])
            ax.set_yticklabels(["-0.5", "0", "0.5"] if c == 0 else ["-2", "0", "2"])
            if r < 2:
                ax.set_xticklabels([])

        # ---------------- Trained Policies (paper Fig. 5) ----------------
        for m, metric in enumerate(metrics):
            ax = fig.add_axes(rect(MET_X[m], ROW_Y[r], MET_W, PANEL_H))
            df = load_fig5(DATASET_KEYS[r], metric)
            vmin, vmax_m = CBAR_LIMITS[metric]
            met_mesh[metric] = ax.pcolormesh(
                cell_edges(df.columns.values), cell_edges(df.index.values), df.values[::-1],  # rows flipped (author's request): same picture as the original Fig. 5, axis unchanged
                cmap="viridis", vmin=vmin, vmax=vmax_m,
                edgecolors="white", linewidth=CELL_EDGE_LW)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.35, 0.35)  # v_y upwards, matching the coverage panels
            ax.tick_params(length=1.8, pad=1.5, labelsize=5.2)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
            ax.set_yticks([-0.3, 0.0, 0.3])
            ax.set_yticklabels(["-0.3", "0", "0.3"])
            if r < 2:
                ax.set_xticklabels([])
            if m > 0:
                ax.set_yticklabels([])

        # dataset name, once for all five panels of the row
        fig.text(fx(M_L + GUTTER / 2), fy(ROW_Y[r] + PANEL_H / 2), ROW_LABELS[r], fontsize=5.8,
                 fontweight="bold", ha="center", va="center", rotation=90, linespacing=1.25)

    # ---------------- column titles and colour bars, directly above the panels ----------
    fig.text(fx(COV_MID), fy(Y_TITLE_BOT), COV_LABEL, fontsize=6.2, ha="center", va="bottom")
    cb_cov_w = 0.62 * (COV_X[1] + COV_W - COV_X[0])
    colour_bar(fig, cov_mesh, COV_MID - cb_cov_w / 2, cb_cov_w,
               [1, 2, 5, 10, vmax], ["1", "2", "5", "10", f"{int(vmax)}"])

    for m, metric in enumerate(metrics):
        fig.text(fx(MET_X[m] + MET_W / 2), fy(Y_TITLE_BOT),
                 TITLE_OVERRIDE.get(metric, plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric]),
                 fontsize=6.2, ha="center", va="bottom", linespacing=1.15)
        colour_bar(fig, met_mesh[metric], MET_X[m] + 0.1 * MET_W, 0.8 * MET_W, CBAR_TICKS[metric])

    # ---------------- block headers and shared axis labels ----------------
    fig.text(fx(COV_MID), fy(Y_HDR), "Expert Dataset", fontsize=7,
             fontweight="bold", ha="center", va="top")
    fig.text(fx(MET_MID), fy(Y_HDR), "Trained Policies", fontsize=7,
             fontweight="bold", ha="center", va="top")

    fig.text(fx(COV_MID), fy(PANELS_BOT + XLABEL_DROP), "$v_x$ [m/s]",
             fontsize=6.2, ha="center", va="center")
    fig.text(fx(MET_MID), fy(PANELS_BOT + XLABEL_DROP),
             plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING["target_velocity_x"],
             fontsize=6.2, ha="center", va="center")
    for c, lab in enumerate(["$v_y$ [m/s]", "$\\omega_z$ [rad/s]"]):
        fig.text(fx(COV_X[c] - COV_YLAB[c] + 0.06), fy((PANELS_TOP + PANELS_BOT) / 2), lab,
                 fontsize=6.2, rotation=90, ha="left", va="center")
    fig.text(fx(MET_X[0] - MET_YLAB + 0.06), fy((PANELS_TOP + PANELS_BOT) / 2),
             plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING["target_velocity_y"],
             fontsize=6.2, rotation=90, ha="left", va="center")

    savefig(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    main()
    main(METRICS_4COL, "fig5_combined_data_and_performance_4col")
