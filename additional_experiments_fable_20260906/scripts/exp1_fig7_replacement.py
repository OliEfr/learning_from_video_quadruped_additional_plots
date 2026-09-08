"""Experiment 1, Fig. 7 replacement: trajectories and keypoint noise in ONE single-column figure.

Standalone drawing code (only data loading and metrics are imported); the panels re-plot the data of
    fig_exp1_trajectory_grid_all_clips_corrected_filtered  ->  Trajectories: Top View, Side View
    fig_exp1_noise_paper_robust                             ->  Paw height, Trajectory noise

Top     MoCap trot2 (labelled "MoCap (walk)" at the author's request) and Video walk, one rigid transform per clip (origin = torso centre of frame 0, mean
        heading -> +x, ground = 5th percentile of paw heights); the video clip re-lifted with the corrected
        640x360 intrinsics and passed through the proposed filter (depth Hampel + FC Hz zero-phase low-pass),
        MoCap unchanged.  Identical axis ranges in every panel, equal metric aspect.
Bottom  left: paw height of the video walk clip over the first T_MAX s, raw (thin) vs filtered (thick).
        right: stance-height scatter, high-frequency part, robust std (labelled "std") (1.4826 * MAD) in % hip height, one dot per clip
        (6 MoCap, 8 video) and the set mean as a bar; video UNFILTERED (the noise as it stands in the data).

All panel edges are placed explicitly in inches, so the left edges of the two left panels and the right edges of
the two right panels line up.  Colours: black torso, four viridis greens for the paws (shared by the grid and (c)), and one
colour per source in the row labels of the grid and the dots of (d): the paper's per-method colours
(plot_DEFINITIONS.py, Fig. 4) in the main file, viridis indigo (MoCap) and viridis blue (Video, kept off the green paw shades) in "_viridis".
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

from common import COL_W, COLORS, FIG, NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR
from exp1_filter import FC, STANCE_Z, build_variants
from exp1_noise import FS, get_clips
from exp1_noise_paper_figure import T_MAX, Y_MAX_HF, stance_scatter, walk_traces

SOURCES = [NAME_MOCAP, NAME_VIDEO_CORR]
LABEL = {NAME_MOCAP: "MoCap", NAME_VIDEO_CORR: "Video"}
SRC_COLORS = {
    "paper": {NAME_MOCAP: COLORS[NAME_MOCAP], NAME_VIDEO_CORR: COLORS[NAME_VIDEO]},
    "viridis": {NAME_MOCAP: plt.get_cmap("viridis")(0.15), NAME_VIDEO_CORR: plt.get_cmap("viridis")(0.30)},  # Fig. 5 blue, away from the green paws (author's request)
}
TORSO = "black"
PAWS = [plt.get_cmap("viridis")(v) for v in (0.55, 0.68, 0.80, 0.92)]  # green segment of the Fig. 5 colormap

# ---- geometry of the trajectory panels (metres), as in the corrected grids
X_SPAN, X_PAD = 2.4, 0.15
Y_HALF = 0.45
Z_LIM = (-0.08, 0.82)
ASPECT = (2 * Y_HALF) / X_SPAN  # panel height / panel width for an equal metric aspect

# ---- page geometry (inches)
W = COL_W
LEFT, RIGHT = 0.62, 0.06                # axes span = [LEFT, W - RIGHT]; vertical row labels need less room
COL_GAP = 0.40                          # room for the y label of the right column
ROW_GAP = 0.07
GRID_W = (W - LEFT - RIGHT - COL_GAP) / 2
GRID_H = GRID_W * ASPECT * 1.10  # 10 % taller than the equal metric aspect (author's request); same ranges in both rows
TOP, HEAD = 0.03, 0.26                  # margin, heading + sub-heading rows
XLAB, LEG = 0.24, 0.13                  # x tick labels + x label, one legend row
GAP_BLOCKS = 0.05
TITLE2 = 0.14                           # one-line panel heading
BOT_H = 0.95                            # height of the bottom panels
D_W = 0.98                              # width of (d); (c) takes the rest
BOTTOM = 0.02
H = TOP + HEAD + 2 * GRID_H + ROW_GAP + XLAB + GAP_BLOCKS + TITLE2 + BOT_H + XLAB + LEG + BOTTOM

FS_HEAD, FS_TITLE, FS_LAB, FS_TICK, FS_LEG, FS_ROW = 7.2, 6.8, 6.2, 5.2, 5.6, 6.2
ROW_LABEL = {NAME_MOCAP: "MoCap\n(walk)", NAME_VIDEO_CORR: "Video\n(walk)"}


class PawsHandle:
    """Legend key placeholder for the four paw shades."""


class PawsHandler(HandlerBase):
    """Draws the legend line as four consecutive segments, one per paw shade."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        n = len(PAWS)
        seg = width / n
        y = height / 2 - ydescent
        return [Line2D([-xdescent + i * seg, -xdescent + (i + 1) * seg], [y, y], color=PAWS[i], lw=1.2,
                       solid_capstyle="butt", transform=trans) for i in range(n)]


def ax_in(fig, x, y, w, h):
    """Axes from a rectangle given in inches (x, y = lower-left corner)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def style(ax):
    ax.grid(alpha=0.3, lw=0.4)
    ax.tick_params(length=2, width=0.5, labelsize=FS_TICK, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def draw_trajectory_row(ax_top, ax_side, clip, last):
    body, feet = clip["body"], clip["feet"]
    x0 = np.concatenate([body, feet], axis=0)[..., 0].min() - X_PAD
    yc = body[..., 1].mean()
    for ax, j in ((ax_top, 1), (ax_side, 2)):
        for i in range(4):
            ax.plot(feet[i, :, 0], feet[i, :, j], color=PAWS[i], lw=0.8)
        for i in range(2):
            ax.plot(body[i, :, 0], body[i, :, j], color=TORSO, lw=0.9)
        for f in np.linspace(0, body.shape[1] - 1, 5).astype(int):  # torso segment at 5 instants
            ax.plot(body[:, f, 0], body[:, f, j], color=TORSO, lw=1.3, alpha=0.4)
        ax.set_xlim(x0, x0 + X_SPAN)
        ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
        style(ax)
        if last:
            ax.set_xlabel("x [m]", fontsize=FS_LAB, labelpad=1.5)
        else:
            ax.set_xticklabels([])
    ax_top.set_ylim(yc - Y_HALF, yc + Y_HALF)
    ax_top.set_yticks([t for t in (-0.6, -0.3, 0, 0.3, 0.6) if yc - Y_HALF < t < yc + Y_HALF])
    ax_top.set_ylabel("y [m]", fontsize=FS_LAB, labelpad=1.5)
    ax_side.set_ylim(*Z_LIM)
    ax_side.set_yticks([0, 0.3, 0.6])
    ax_side.set_ylabel("z [m]", fontsize=FS_LAB, labelpad=1.5)
    ax_side.axhline(0, color="0.55", lw=0.5, ls="--", zorder=0)


def draw_trace(ax):
    raw, fil = walk_traces()
    t = np.arange(raw.shape[1]) / FS
    keep = t <= T_MAX
    g0, g1 = np.percentile(raw[..., 2], 5), np.percentile(fil[..., 2], 5)
    for k in range(4):
        ax.plot(t[keep], raw[k, keep, 2] - g0, color=PAWS[k], lw=0.55, alpha=0.5)
        ax.plot(t[keep], fil[k, keep, 2] - g1, color=PAWS[k], lw=1.15)
    ax.axhline(STANCE_Z, color="0.5", lw=0.5, ls="--", zorder=0)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.02, 0.135)
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    style(ax)
    ax.set_xlabel("time [s]", fontsize=FS_LAB, labelpad=1.5)
    ax.set_ylabel("paw height [m]", fontsize=FS_LAB, labelpad=1.5)
    handles = [plt.Line2D([], [], color="0.35", lw=0.55, alpha=0.6, label="raw"),
               plt.Line2D([], [], color="0.35", lw=1.15, label="filtered")]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=FS_LEG, handlelength=1.6,
              columnspacing=0.9, borderaxespad=0.15, handletextpad=0.5)


def draw_scatter(ax, clips, sc, color):
    """Dots = clips, bar = set mean; pairs (MoCap, Video) for the torso and for the paws."""
    xs = {("body", NAME_MOCAP): 0.0, ("body", NAME_VIDEO_CORR): 1.0, ("paw", NAME_MOCAP): 2.6, ("paw", NAME_VIDEO_CORR): 3.6}
    means = {}
    for (part, src), x in xs.items():
        v = np.asarray([s[f"{part}_hf"] for c, s in zip(clips, sc) if c["source"] == src], float)
        v = v[~np.isnan(v)]
        jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.06
        ax.scatter(x + jit, v, s=7, color=color[src], alpha=0.85, lw=0, zorder=3)
        ax.hlines(v.mean(), x - 0.32, x + 0.32, color=color[src], lw=1.4, zorder=4)
        means[(part, src)] = float(v.mean())
    ax.set_xlim(-0.65, 4.25)
    ax.set_ylim(0, Y_MAX_HF)
    ax.set_xticks([0.5, 3.1])
    ax.set_xticklabels(["Torso", "Paws"])
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, labelsize=FS_LAB)
    ax.set_ylabel("std [% hip height]", fontsize=FS_LAB, labelpad=1.5)
    for i, part in enumerate(["body", "paw"]):
        r = means[(part, NAME_VIDEO_CORR)] / means[(part, NAME_MOCAP)]
        ax.text(0.5 + 2.6 * i, 0.96, f"{r:.1f}x", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_LAB)
    return means


def build(grid_clips, noise_clips, sc, fname, color):
    fig = plt.figure(figsize=(W, H))
    x_left, x_right = LEFT, LEFT + GRID_W + COL_GAP
    x_mid = (LEFT + W - RIGHT) / 2  # centre of the axes span, used for the legends

    # ---- (a)/(b) trajectory grid
    y = H - TOP - HEAD - GRID_H
    for r, c in enumerate(grid_clips):
        ax_t, ax_s = ax_in(fig, x_left, y, GRID_W, GRID_H), ax_in(fig, x_right, y, GRID_W, GRID_H)
        draw_trajectory_row(ax_t, ax_s, c, last=(r == len(grid_clips) - 1))
        fig.text(0.13 / W, (y + GRID_H / 2) / H, ROW_LABEL[c["source"]], ha="center", va="center", rotation=90,
                 fontsize=FS_ROW, fontweight="bold", color="black", linespacing=1.15)
        if r == 0:
            fig.text(x_mid / W, (y + GRID_H + 0.16) / H, "Trajectories", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold")
            for x, lab in ((x_left, "Top View"), (x_right, "Side View")):
                fig.text((x + GRID_W / 2) / W, (y + GRID_H + 0.03) / H, lab, ha="center", va="bottom", fontsize=FS_TITLE)
        y -= GRID_H + ROW_GAP

    # ---- (c)/(d) noise panels
    y_bot = BOTTOM + LEG + XLAB
    c_w = W - LEFT - RIGHT - COL_GAP - D_W
    ax_c = ax_in(fig, x_left, y_bot, c_w, BOT_H)
    ax_d = ax_in(fig, W - RIGHT - D_W, y_bot, D_W, BOT_H)
    draw_trace(ax_c)
    means = draw_scatter(ax_d, noise_clips, sc, color)
    for x, w_, lab in ((x_left, c_w, "Paw height"), (W - RIGHT - D_W, D_W, "Trajectory noise")):
        fig.text((x + w_ / 2) / W, (y_bot + BOT_H + 0.03) / H, lab, ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold")
    # one legend row for the whole figure: keypoints of the grid / trace, sources of the scatter
    handles = [plt.Line2D([], [], color=TORSO, lw=1.2), PawsHandle()]
    handles += [plt.Line2D([], [], marker="o", ls="-", color=color[s], ms=3, lw=1.2) for s in SOURCES]
    fig.legend(handles=handles, labels=["Torso", "Paws"] + [LABEL[s] for s in SOURCES], handler_map={PawsHandle: PawsHandler()},
               loc="upper center", bbox_to_anchor=(x_mid / W, (y_bot - XLAB) / H), ncol=4, frameon=False, fontsize=FS_LEG,
               handlelength=2.2, columnspacing=1.4, handletextpad=0.5, borderaxespad=0)

    savefig_flush(fig, fname)
    plt.close(fig)
    return means


def savefig_flush(fig, name):
    """Vector PDF (+ PNG preview) cropped to the ink with a 0.02 in margin."""
    pdf = os.path.join(FIG, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    print("saved", pdf)


def main():
    V = build_variants()
    filt_corr = [v[1] for v in V[NAME_VIDEO_CORR]]  # corrected intrinsics + A+B filter, as in the "_filtered" grids
    all_clips = get_clips()
    grid_clips = [c for c in all_clips if c["source"] == NAME_MOCAP and c["name"] == "trot2"] + [c for c in filt_corr if c["name"] == "walk"]
    noise_clips = [c for c in all_clips if c["source"] in SOURCES]  # MoCap + unfiltered corrected video (fig_exp1_noise_paper_robust)
    sc = [stance_scatter(c, robust=True) for c in noise_clips]
    print(f"figure {W:.2f} x {H:.2f} in; video row of the grid: corrected intrinsics + depth Hampel + {FC:.0f} Hz zero-phase low-pass")
    for fname, key in [("fig_exp1_fig7_replacement", "paper"), ("fig_exp1_fig7_replacement_viridis", "viridis")]:
        means = build(grid_clips, noise_clips, sc, fname, SRC_COLORS[key])
        for part in ["body", "paw"]:
            m, v = means[(part, NAME_MOCAP)], means[(part, NAME_VIDEO_CORR)]
            print(f"{fname} (d) {part}: MoCap {m:.2f}  Video {v:.2f}  ratio {v / m:.2f}x  [% hip height, robust std]")


if __name__ == "__main__":
    main()
