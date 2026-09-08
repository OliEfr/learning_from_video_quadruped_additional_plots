"""Reduced version of fig_exp2_state_coverage (author's request):
only the two base-velocity columns, black dashed target-cmd-range box labelled underneath, no row labels.

Colour = plain number of expert frames per bin (linear scale), i.e. the raw distribution of the
available expert data points -- one stage simpler than fig_exp2_state_coverage, which colours the
AMP-sampling-weighted *fraction* of samples per bin on a log scale.

Rows are MoCap / Video / Video (extended), top to bottom -- same order as the paper's Fig. 5.
Bins, axis ranges and the underlying velocities are identical to exp2_coverage.fig_state_coverage.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle

from common import CMD_RANGES, DOUBLE_W, savefig
from exp2_coverage import SOURCES, dataset_frames

COLS = [
    ("vx", "vy", "$v_x$ [m/s]", "$v_y$ [m/s]", (-1.2, 1.8), (-0.65, 0.65), 0.1, 0.05, ("vx", "vy"), "base velocity $v_x$ vs $v_y$"),
    ("vx", "wz", "$v_x$ [m/s]", "$\\omega_z$ [rad/s]", (-1.2, 1.8), (-2.6, 2.6), 0.1, 0.2, ("vx", "wz"), "base velocity $v_x$ vs $\\omega_z$"),
]


def hist(d, kx, ky, xr, yr, bx, by):
    hx = np.arange(xr[0], xr[1] + 1e-9, bx)
    hy = np.arange(yr[0], yr[1] + 1e-9, by)
    H, _, _ = np.histogram2d(d[kx], d[ky], bins=[hx, hy])  # unweighted: one count per expert frame
    return hx, hy, H


def draw(D, vmax, norm, fname):
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_W * 0.45, 4.3), gridspec_kw=dict(hspace=0.25, wspace=0.42))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.09)
    mesh = None
    for r, src in enumerate(SOURCES):
        d = D[src]
        for c_, (kx, ky, xl, yl, xr, yr, bx, by, cmd, title) in enumerate(COLS):
            ax = axes[r, c_]
            hx, hy, H = hist(d, kx, ky, xr, yr, bx, by)
            H = np.ma.masked_where(H <= 0, H)
            mesh = ax.pcolormesh(hx, hy, H.T, cmap="viridis", norm=norm, rasterized=True)
            cx, cy = CMD_RANGES[cmd[0]], CMD_RANGES[cmd[1]]
            ax.add_patch(Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0], fill=False, ec="black", lw=0.9, ls="--"))
            ax.text(cx[0], cy[0] - 0.015 * (yr[1] - yr[0]), "target cmd range", ha="left", va="top", fontsize=5, color="black")
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax.grid(alpha=0.25, lw=0.4)
            ax.tick_params(length=2)
            if r == 0:
                ax.set_title(title, fontsize=6.8, fontweight="bold", pad=3)
            if r == 2:
                ax.set_xlabel(xl)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(yl, labelpad=1)

    cax = fig.add_axes([0.2, 0.955, 0.6, 0.015])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("number of expert frames per bin", fontsize=6, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2)
    if isinstance(norm, LogNorm):  # integer counts: label the decades and the endpoints
        cb.set_ticks([1, 2, 5, 10, vmax])
        cb.ax.set_xticklabels(["1", "2", "5", "10", f"{int(vmax)}"])
    savefig(fig, fname)
    plt.close(fig)


def main():
    D = {src: dataset_frames(src)[0] for src in SOURCES}
    for src in SOURCES:
        print(f"{src:16s} n_frames={len(D[src]['vx']):5d}")

    # empty bins are masked and never drawn, so the colour scale starts at 1
    counts = np.concatenate([hist(D[src], c[0], c[1], c[4], c[5], c[6], c[7])[2].ravel() for src in SOURCES for c in COLS])
    occ = counts[counts > 0]
    vmax = float(occ.max())
    print(f"occupied bins: {occ.size}, counts min={occ.min():.0f} median={np.median(occ):.0f} max={vmax:.0f}")
    print("count histogram (n frames in bin -> n bins):", {int(v): int((occ == v).sum()) for v in np.unique(occ)})

    draw(D, vmax, Normalize(vmin=1, vmax=vmax), "fig_exp2_state_coverage_2col")
    draw(D, vmax, LogNorm(vmin=1, vmax=vmax), "fig_exp2_state_coverage_2col_log")


if __name__ == "__main__":
    main()
