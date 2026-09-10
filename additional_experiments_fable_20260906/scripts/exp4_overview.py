"""Experiment 4 overview (author's request): the mutual relationships between data coverage, imitation, tracking error and
cost of transport in ONE double-column figure with three panels, per command cell of the Fig. 5 evaluation grid
(vx-vy grid at zero yaw rate, 147 cells per expert set):
  (a) tracking error vs distance of the target command to the closest expert frame, colour = expert imitation   (= Fig. 8)
  (b) agent-expert distance vs the same command distance, colour = commanded speed                              (coverage -> imitation)
  (c) cost of transport vs agent-expert distance, colour = commanded speed, cells with |v| >= V_MIN only        (imitation -> efficiency)
The |v| -> 0 cells are left out of (c) because CoT diverges there (small |v| in the denominator; clipped at 2.0 in the colour map).
Least-squares line and Spearman rho per set in the legends, marker shape per set. Values are the ones recovered from the Fig. 5
colour maps (exp4_amp_evidence.py). Two files: the two paper sets (MoCap, Video (extended)) and all three sets ("_3sets").
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import exp4_amp_evidence as E
from common import COLORS, DATA, DOUBLE_W, EVAL_GRID, FIG

V_MIN = 0.3
FS_LEG = 5.3


def load_points():
    D = {}
    for src in E.SOURCES:
        D[src], _ = E.dataset_frames(src)
    cov = dict(np.load(os.path.join(DATA, "exp2_coverage_maps.npz")))
    _, _, P = E.analysis_cells(D, cov)
    gx, gy = EVAL_GRID["xy"]
    VX, VY = np.meshgrid(gx, gy)
    P["speed"] = np.concatenate([np.sqrt(VX ** 2 + VY ** 2).ravel()] * len(E.SOURCES))
    P["imit"] = -P["aed"]
    return P


def panel(ax, P, kx, ky, kc, sources, clim, cmap, ylim, xlim, sel=None):
    sc, rhos = None, {}
    for src in sources:
        m = (P["src"] == src) & (sel if sel is not None else True)
        x, y = P[kx][m], P[ky][m]
        sc = ax.scatter(x, y, c=P[kc][m], cmap=cmap, vmin=clim[0], vmax=clim[1], s=9, marker=E.MARKERS[src],
                        edgecolors=COLORS[src], linewidths=0.4, alpha=0.9, zorder=2)
        a, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        yy = a * xx + b
        keep = (yy >= ylim[0]) & (yy <= ylim[1])
        ax.plot(xx[keep], yy[keep], color=COLORS[src], lw=1.5, zorder=3)
        rhos[src] = float(spearmanr(x, y)[0])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.35)
    ax.tick_params(length=2, pad=1.5)
    return sc, rhos


def rho_box(ax, rhos, loc):
    """Spearman rho per set, one line each in the set colour, in a small white box at a corner (loc = 'lr', 'ur', 'ul', 'll')."""
    x, ha = (0.97, "right") if loc[1] == "r" else (0.03, "left")
    y0, va, step = (0.04, "bottom", 1) if loc[0] == "l" else (0.96, "top", -1)
    lines = [(src, r) for src, r in rhos.items()]
    if va == "bottom":
        lines = lines[::-1]
    for i, (src, r) in enumerate(lines):
        ax.text(x, y0 + step * i * 0.115, f"ρ = {r:.2f}", transform=ax.transAxes, ha=ha, va=va, fontsize=FS_LEG + 0.4,
                color=COLORS[src], fontweight="bold", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))


def colorbar(fig, ax, sc, label, ticks):
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06, ticks=ticks)
    cb.set_label(label, fontsize=plt.rcParams["axes.labelsize"])
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], length=2)
    cb.outline.set_linewidth(0.4)


def build(P, sources, fname):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.15), gridspec_kw=dict(wspace=0.78))
    lab_cmd = "Distance of Target Command\nto Closest Expert Frame [1]"
    lab_aed = "Agent-Expert Distance\nin AMP obs. space (q, q̇) [1]"
    sc, r = panel(axes[0], P, "cmd", "comb", "imit", sources, (-4.0, -2.0), "viridis_r", (0, 1.12), (-0.03, 1.55))
    rho_box(axes[0], r, "lr")
    axes[0].set_xlabel(lab_cmd)
    axes[0].set_ylabel(r"$\bf{Tracking\ Error}$" + "\nYaw and Vel. Normalized [1]")
    colorbar(fig, axes[0], sc, "Expert Imitation [1]\n↑ better", [-4, -3, -2])
    sc, r = panel(axes[1], P, "cmd", "aed", "speed", sources, (0.0, 1.05), "viridis", (1.85, 4.15), (-0.03, 1.55))
    rho_box(axes[1], r, "lr")
    axes[1].set_xlabel(lab_cmd)
    axes[1].set_ylabel(r"$\bf{Agent\text{-}Expert\ Distance}$" + "\nin AMP obs. space (q, q̇) [1]")
    colorbar(fig, axes[1], sc, "Commanded Speed\n|v| [m/s]", [0, 0.5, 1])
    sel = P["speed"] >= V_MIN
    sc, r = panel(axes[2], P, "aed", "cot", "speed", sources, (0.0, 1.05), "viridis", (0.78, 2.05), (1.95, 4.05), sel=sel)
    rho_box(axes[2], r, "ul")
    axes[2].set_xlabel(lab_aed)
    axes[2].set_ylabel(r"$\bf{Cost\ of\ Transport}$" + " [1]")
    colorbar(fig, axes[2], sc, "Commanded Speed\n|v| [m/s]", [0, 0.5, 1])
    titles = ["(a) coverage → tracking", "(b) coverage → imitation", f"(c) imitation → efficiency (|v| ≥ {V_MIN:.1f} m/s)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=7, pad=3)
    handles = [Line2D([], [], color=COLORS[s_], lw=1.5, marker=E.MARKERS[s_], markersize=4, markerfacecolor="white",
                      markeredgecolor=COLORS[s_], markeredgewidth=0.8) for s_ in sources]
    labels = [E.SHORT[s_].replace("ext.", "extended") for s_ in sources]
    fig.legend(handles, labels, loc="lower center", ncol=len(sources), frameon=False, fontsize=6.2, handlelength=2.6,
               columnspacing=1.8, handletextpad=0.6, bbox_to_anchor=(0.5, -0.16))
    pad = 2 / 72
    fig.savefig(os.path.join(FIG, fname + ".pdf"), bbox_inches="tight", pad_inches=pad)
    fig.savefig(os.path.join(FIG, fname + ".png"), bbox_inches="tight", pad_inches=pad, dpi=200)
    print("saved", os.path.join(FIG, fname + ".pdf"))
    plt.close(fig)


def main():
    P = load_points()
    build(P, E.PLOT_SOURCES, "fig_exp4_overview")
    build(P, E.SOURCES, "fig_exp4_overview_3sets")


if __name__ == "__main__":
    main()
