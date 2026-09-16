"""Sine yaw-rate tracking trial (~/Downloads/sine_representative (1).pdf) redrawn in the paper's figure style.

The original plotting script and its data are not available, so the two curves are recovered once from the vector paths of
the source PDF (matplotlib output -> pdftocairo SVG) and cached in data/sine_tracking.csv. The axis calibration is read
from the tick marks of the source figure, paired with the tick values listed in X_TICK_VALUES / Y_TICK_VALUES below, so a
new source PDF only needs those two lists updated.

Trial: sinusoidal yaw-rate command (amplitude 0.5 rad/s, period 10 s) at a constant forward command v_x = 0.3 m/s; the
actual yaw rate is the motion-capture measurement.

Outputs figures/fig_sine_tracking.{pdf,png} (single column) and figures/fig_sine_tracking_2col.{pdf,png} (double column),
both using roughly half the vertical space of the original figure.
"""
import csv
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import COL_W, COLORS, DATA, DOUBLE_W, FIG, NAME_VIDEO_EXT  # noqa: E402

SRC_PDF = os.path.expanduser("~/Downloads/sine_representative (1).pdf")
CSV = os.path.join(DATA, "sine_tracking.csv")
X_TICK_VALUES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]          # time [s], as labelled in the source figure
Y_TICK_VALUES = [-0.50, -0.25, 0.00, 0.25, 0.50, 0.75]                # yaw rate [rad/s]
TARGET_RGB, ACTUAL_RGB = "14.509583%, 38.822937%, 92.155457%", "86.273193%, 14.901733%, 14.901733%"
V_X = 0.3  # constant forward velocity command during the trial [m/s]


def _paths(svg_text):
    """[(attributes, points)] of every stroked path; all paths share one transform, so raw path coordinates suffice
    (that transform has a negative y scale, i.e. y already points up as in PDF space)."""
    out = []
    for m in re.finditer(r"<path([^>]*?)d=\"(M [^\"]*)\"([^>]*)>", svg_text):
        pts = np.array([[float(a), float(b)] for a, b in re.findall(r"([\d.-]+) ([\d.-]+)", m.group(2))])
        out.append((m.group(1) + m.group(3), pts))
    return out


def _axis_map(paths, values, axis):
    """Linear pixel -> data map for one axis, fitted on the short tick marks (axis=0: x ticks, axis=1: y ticks)."""
    ticks = sorted({round(p[0, axis], 4) for a, p in paths
                    if len(p) == 2 and abs(p[0, axis] - p[1, axis]) < 1e-6 and 2.0 < abs(p[0, 1 - axis] - p[1, 1 - axis]) < 5.0
                    and "0%, 0%, 0%" in a})
    if len(ticks) != len(values):
        raise RuntimeError(f"found {len(ticks)} ticks on axis {axis}, expected {len(values)}: {ticks}")
    slope, intercept = np.polyfit(ticks, values, 1)
    return lambda v: slope * v + intercept


def extract():
    """Recover (t, target, actual) from the vector paths of the source PDF."""
    svg = os.path.join(os.path.dirname(CSV), "_sine_tmp.svg")
    subprocess.run(["pdftocairo", "-svg", SRC_PDF, svg], check=True)
    paths = _paths(open(svg).read())
    os.remove(svg)
    to_t, to_y = _axis_map(paths, X_TICK_VALUES, 0), _axis_map(paths, Y_TICK_VALUES, 1)
    curves = {}
    for attrs, pts in paths:
        col = re.search(r"stroke=\"rgb\(([^)]*)\)\"", attrs)
        if col is not None and len(pts) > 50:
            curves[col.group(1)] = np.c_[to_t(pts[:, 0]), to_y(pts[:, 1])]
    tgt, act = curves[TARGET_RGB], curves[ACTUAL_RGB]
    # the target is a clean sine that matplotlib's path simplification thinned out -> resample it on the actual's time base
    target_on_act = np.interp(act[:, 0], tgt[:, 0], tgt[:, 1])
    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "target_yaw_rate_rad_s", "actual_yaw_rate_rad_s"])
        w.writerows([[f"{t:.4f}", f"{c:.5f}", f"{a:.5f}"] for (t, a), c in zip(act, target_on_act)])
    print(f"wrote {CSV}: {len(act)} samples, target drawn with {len(tgt)} points")


def load():
    if not os.path.exists(CSV):
        extract()
    d = np.genfromtxt(CSV, delimiter=",", names=True)
    return d["time_s"], d["target_yaw_rate_rad_s"], d["actual_yaw_rate_rad_s"]


def figure(t, target, actual, width, height_frac, name):
    fig, ax = plt.subplots(figsize=(width, height_frac * width))
    ax.plot(t, actual, color=COLORS[NAME_VIDEO_EXT], lw=0.6, label="Actual (Motion Capture)", zorder=3, solid_joinstyle="round")
    ax.plot(t, target, color="black", lw=1.0, ls="--", dashes=(3, 1.6), label="Target (commanded)", zorder=4)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-0.72, 0.85)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticks([2.5, 7.5, 12.5, 17.5], minor=True)
    ax.set_xlabel("Time [s]", fontsize=6.2, labelpad=1.2)
    ax.set_ylabel("Yaw Rate [rad/s]", fontsize=6.2, labelpad=1.5)
    ax.tick_params(labelsize=5.2, length=1.8, pad=1.5)
    ax.tick_params(which="minor", length=1.0)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # the trial runs at a constant forward command; the lower right of the panel is free
    ax.text(0.995, 0.03, f"$v_x = {V_X}$ m/s", transform=ax.transAxes, fontsize=5.4, ha="right", va="bottom")
    h, la = ax.get_legend_handles_labels()
    fig.legend(h[::-1], la[::-1], loc="lower center", bbox_to_anchor=(0.55, 0.907), ncol=2, fontsize=5.8, frameon=False,
               handlelength=1.8, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.0, borderpad=0.0)
    fig.subplots_adjust(top=0.9, bottom=0.26, left=0.11, right=0.995)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.012, **kw)
    plt.close(fig)
    print("saved", os.path.join(FIG, f"{name}.pdf"))


def main():
    t, target, actual = load()
    err = actual - target
    print(f"{len(t)} samples, {t[-1]:.2f} s; RMSE {np.sqrt((err ** 2).mean()):.3f} rad/s, "
          f"MAE {np.abs(err).mean():.3f} rad/s, command amplitude {np.abs(target).max():.2f} rad/s, "
          f"period {2 * abs(t[np.argmax(target)] - t[np.argmin(target)]):.1f} s")
    figure(t, target, actual, COL_W, 0.27, "fig_sine_tracking")
    figure(t, target, actual, DOUBLE_W, 0.135, "fig_sine_tracking_2col")


if __name__ == "__main__":
    main()
