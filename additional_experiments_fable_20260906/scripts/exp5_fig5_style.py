"""Exp. 5: Fig.-5-style figure (expert-data coverage | policy heat maps over the vx-vy command grid) with the paper's three sets
plus ablation arms that have a TargetXY grid evaluation (147 cells x 5000 envs, real yamls).

Rows: MoCap, Video, Video (extended) exactly as fig5_replacement_combined.py (coverage from the expert files, policy values recovered
from the paper's Fig. 5), followed by one row per ablation arm with the real per-cell yamls (seed mean). Geometry and colour scales are
those of fig5_replacement_combined.py, re-derived for the number of rows.
Usage: python exp5_fig5_style.py [folder ...]   default: fromVision_motions_DepthCam_extHalf
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig5_replacement_combined as F5
from common import CMD_RANGES, DOUBLE_W, load_amp_dir, savefig
from exp2_coverage import SOURCES, dataset_frames
from exp2_state_coverage_2col import COLS, hist
from exp5_build_video_subsets import frames_of
from exp5_grid_analysis import DT, LOGROOT, W, load_grid

ARM_LABELS = {
    "fromVision_motions_DepthCam_extHalf": "Video (ext.)\nhalf\n(1 window/clip)",
    "fromVision_motions_DepthCam_extHalfCov": "Video (ext.)\nhalf, cov.-\nmatched",
    "fromVision_motions_DepthCam_extAddedOnly": "Video\n(added\nclips only)",
    "fromVision_motions_DepthCam_extNoTurn": "Video (ext.)\nminus\nturning",
}


def arm_grid_mean(folder):
    grids = [g for run in sorted(glob.glob(os.path.join(LOGROOT, f"exp5_{DT}_{folder}_SEED_*"))) if (g := load_grid(run)) is not None]
    if not grids:
        raise SystemExit(f"no complete TargetXY grid for {folder}")
    return {m: np.mean([g[m] for g in grids], axis=0) for m in grids[0]}, len(grids)


def main(folders, metrics=F5.METRICS, name="fig_exp5_fig5_style"):
    rows = [(F5.ROW_LABELS[i], dataset_frames(src)[0], ("csv", F5.DATASET_KEYS[i])) for i, src in enumerate(SOURCES)]
    for f in folders:
        g, n = arm_grid_mean(f)
        rows.append((ARM_LABELS.get(f, f) + f"\n[{n} seed{'s' if n > 1 else ''}]", frames_of(load_amp_dir(os.path.join(W, "datasets", f))), ("grid", g)))
    n_rows = len(rows)
    # geometry: same constants as F5, rows extended
    ROW_Y = [F5.Y_ROW0 + r * (F5.PANEL_H + F5.ROW_GAP) for r in range(n_rows)]
    PANELS_TOP, PANELS_BOT = ROW_Y[0], ROW_Y[-1] + F5.PANEL_H
    H = PANELS_BOT + F5.XLABEL_DROP + 0.12
    n_met = len(metrics)
    MET_W = (DOUBLE_W - F5.M_R - F5.MET_X0 - (n_met - 1) * F5.MET_GAP) / n_met
    MET_X = [F5.MET_X0 + i * (MET_W + F5.MET_GAP) for i in range(n_met)]
    MET_MID = (MET_X[0] + MET_X[-1] + MET_W) / 2

    def rect(x, y, w, h):
        return [x / DOUBLE_W, 1.0 - (y + h) / H, w / DOUBLE_W, h / H]

    def fx(x):
        return x / DOUBLE_W

    def fy(y):
        return 1.0 - y / H

    def colour_bar(fig, mappable, x, w, ticks, labels=None):
        from matplotlib.colorbar import Colorbar
        cax = fig.add_axes(rect(x, F5.Y_CB, w, F5.CB_H))
        cb = Colorbar(cax, mappable, orientation="horizontal")
        cb.set_ticks(ticks)
        if labels is not None:
            cax.set_xticklabels(labels)
        cax.xaxis.set_ticks_position("top")
        cax.tick_params(labelsize=5, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)

    occ = np.concatenate([hist(d, c[0], c[1], c[4], c[5], c[6], c[7])[2].ravel() for _, d, _ in rows for c in COLS])
    occ = occ[occ > 0]
    cov_norm = LogNorm(vmin=1, vmax=float(occ.max()))
    fig = plt.figure(figsize=(DOUBLE_W, H))
    cov_mesh, met_mesh = None, {}
    for r, (label, d, (kind, payload)) in enumerate(rows):
        for c, (kx, ky, xl, yl, xr, yr, bx, by, cmd, _t) in enumerate(COLS):
            ax = fig.add_axes(rect(F5.COV_X[c], ROW_Y[r], F5.COV_W, F5.PANEL_H))
            hx, hy, Hh = hist(d, kx, ky, xr, yr, bx, by)
            cov_mesh = ax.pcolormesh(hx, hy, np.ma.masked_where(Hh <= 0, Hh).T, cmap="viridis", norm=cov_norm, edgecolors="face", linewidth=0)
            cx, cy = CMD_RANGES[cmd[0]], CMD_RANGES[cmd[1]]
            ax.add_patch(Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0], fill=False, ec="black", lw=0.7, ls="--"))
            if r == 0:
                ax.text(cx[0], cy[0] - 0.02 * (yr[1] - yr[0]), "target cmd range", ha="left", va="top", fontsize=4.6, color="black")
            ax.set_xlim(*xr); ax.set_ylim(*yr)
            ax.grid(alpha=0.25, lw=0.35); ax.tick_params(length=1.8, pad=1.5, labelsize=5.2)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-0.5, 0, 0.5] if c == 0 else [-2, 0, 2])
            ax.set_yticklabels(["-0.5", "0", "0.5"] if c == 0 else ["-2", "0", "2"])
            if r < n_rows - 1:
                ax.set_xticklabels([])
        for m, metric in enumerate(metrics):
            ax = fig.add_axes(rect(MET_X[m], ROW_Y[r], MET_W, F5.PANEL_H))
            if kind == "csv":
                df = F5.load_fig5(payload, metric)
                xe, ye, Z = F5.cell_edges(df.columns.values), F5.cell_edges(df.index.values), df.values[::-1]
            else:  # real grid: axis 0 = vy ascending, axis 1 = vx ascending (explicit targets in the yamls, no orientation ambiguity)
                xe, ye, Z = F5.cell_edges(F5.FIG5_X), F5.cell_edges(F5.FIG5_Y), payload[metric]
            vmin, vmax_m = F5.CBAR_LIMITS[metric]
            met_mesh[metric] = ax.pcolormesh(xe, ye, Z, cmap="viridis", vmin=vmin, vmax=vmax_m, edgecolors="white", linewidth=F5.CELL_EDGE_LW)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.35, 0.35)
            ax.tick_params(length=1.8, pad=1.5, labelsize=5.2)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
            ax.set_yticks([-0.3, 0.0, 0.3]); ax.set_yticklabels(["-0.3", "0", "0.3"])
            if r < n_rows - 1:
                ax.set_xticklabels([])
            if m > 0:
                ax.set_yticklabels([])
        fig.text(fx(F5.M_L + F5.GUTTER / 2), fy(ROW_Y[r] + F5.PANEL_H / 2), label, fontsize=5.3 if kind == "grid" else 5.8,
                 fontweight="bold", ha="center", va="center", rotation=90, linespacing=1.2)
    fig.text(fx(F5.COV_MID), fy(F5.Y_TITLE_BOT), F5.COV_LABEL, fontsize=6.2, ha="center", va="bottom")
    cb_cov_w = 0.62 * (F5.COV_X[1] + F5.COV_W - F5.COV_X[0])
    vmax = float(occ.max())
    colour_bar(fig, cov_mesh, F5.COV_MID - cb_cov_w / 2, cb_cov_w, [1, 2, 5, 10, vmax], ["1", "2", "5", "10", f"{int(vmax)}"])
    for m, metric in enumerate(metrics):
        fig.text(fx(MET_X[m] + MET_W / 2), fy(F5.Y_TITLE_BOT), F5.TITLE_OVERRIDE.get(metric, F5.plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric]),
                 fontsize=6.2, ha="center", va="bottom", linespacing=1.15)
        colour_bar(fig, met_mesh[metric], MET_X[m] + 0.1 * MET_W, 0.8 * MET_W, F5.CBAR_TICKS[metric])
    fig.text(fx(F5.COV_MID), fy(F5.Y_HDR), "Expert Dataset", fontsize=7, fontweight="bold", ha="center", va="top")
    fig.text(fx(MET_MID), fy(F5.Y_HDR), "Trained Policies", fontsize=7, fontweight="bold", ha="center", va="top")
    fig.text(fx(F5.COV_MID), fy(PANELS_BOT + F5.XLABEL_DROP), "$v_x$ [m/s]", fontsize=6.2, ha="center", va="center")
    fig.text(fx(MET_MID), fy(PANELS_BOT + F5.XLABEL_DROP), F5.plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING["target_velocity_x"], fontsize=6.2, ha="center", va="center")
    for c, lab in enumerate(["$v_y$ [m/s]", "$\\omega_z$ [rad/s]"]):
        fig.text(fx(F5.COV_X[c] - F5.COV_YLAB[c] + 0.06), fy((PANELS_TOP + PANELS_BOT) / 2), lab, fontsize=6.2, rotation=90, ha="left", va="center")
    fig.text(fx(MET_X[0] - F5.MET_YLAB + 0.06), fy((PANELS_TOP + PANELS_BOT) / 2), F5.plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING["target_velocity_y"],
             fontsize=6.2, rotation=90, ha="left", va="center")
    savefig(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fromVision_motions_DepthCam_extHalf"])
