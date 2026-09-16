"""Exp. 5: Fig.-4-style comparison of the video-only ablation arms (single-column version).

Style mirrors IsaacLab/plot_selected_metrics.py (paper Fig. 4): bold panel titles with the unit on a second line, bars at alpha 0.8,
black min-max whiskers with caps, dashed y-grid, no x tick labels, colour legend below the panels. Colours of the two paper sets are the
paper colours (tab10 1 = Video, 6 = Video (extended)); the new arms use colours that no Fig.-4 method uses (tab10 9, tab10 3, tab20b 0).
Bars = seed mean, whiskers = min/max over seeds (paper sets: published three-seed means from Fig. 4, no range). Values from
data/exp5_metrics.json (exp5_analyze.py). Labels above the bars give the mean and the change vs Video (extended).
Panels: velocity and yaw tracking error (CoT dropped 2026-09-15).
Usage: python exp5_fig4_style.py            -> figures/fig_exp5_fig4_style(.pdf/.png)
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COL_W, COLORS, DATA, NAME_VIDEO, NAME_VIDEO_EXT, savefig

# Fig.-4 panel titles (IsaacLab/plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING)
METRICS = [("error_vel_xy", "Tracking Error Vel.\n[m/s]", "{:.3f}"), ("error_vel_yaw", "Tracking Error Yaw\n[rad]", "{:.3f}")]
TAB10 = plt.get_cmap("tab10", 10)
TAB20B = plt.get_cmap("tab20b", 20)
# (key in exp5_metrics.json, legend label, colour) in plotting order
ARMS = [
    ("Video (paper)", "Video w. Depth Camera (AMP)", COLORS[NAME_VIDEO]),
    ("Video (extended) (paper)", "Video w. Depth Camera (extended) (AMP)", COLORS[NAME_VIDEO_EXT]),
    ("E1b Video (extended, half, coverage-matched)", "Video w. Depth Camera (ext., half, cov.-matched) (AMP)", TAB10(9)),
    ("E2 Video (added clips only)", "Video w. Depth Camera (added clips only) (AMP)", TAB10(3)),
    ("E3 Video (extended minus turning)", "Video w. Depth Camera (ext. minus turning) (AMP)", TAB20B(0)),
]
REF = "Video (extended) (paper)"


def main(name="fig_exp5_fig4_style"):
    S = json.load(open(os.path.join(DATA, "exp5_metrics.json")))
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.75))
    x = np.arange(len(ARMS))
    ref = S[REF]["mean"]
    for ax, (m, title, fmt) in zip(axes, METRICS):
        for i, (key, _label, col) in enumerate(ARMS):
            s = S[key]
            mean = s["mean"][m]
            vals = [v[m] for v in s["per_seed"].values()] if "per_seed" in s else [mean]
            yerr = [[mean - min(vals)], [max(vals) - mean]] if len(vals) > 1 else None
            ax.bar(i, mean, width=0.8, color=col, alpha=0.8, yerr=yerr, capsize=2, error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="black"), zorder=2)
            rel = "" if key == REF else f"\n{100 * (mean / ref[m] - 1):+.0f}%"
            ax.text(i, max(vals), fmt.format(mean) + rel, ha="center", va="bottom", fontsize=5.2, linespacing=1.0, zorder=6, clip_on=False)
        ax.set_title(title, fontweight="bold", fontsize=7, pad=3)
        ax.set_xticks([])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.7, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=6, length=2, pad=1.5)
    handles = [Patch(facecolor=a[2], alpha=0.8, edgecolor="none", label=a[1]) for a in ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=5.2, handlelength=1.4, handleheight=0.8,
               columnspacing=1.2, labelspacing=0.3, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(wspace=0.4, bottom=0.08, top=0.84, left=0.12, right=0.99)
    savefig(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    main()
