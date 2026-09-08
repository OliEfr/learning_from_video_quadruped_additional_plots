"""Experiment 1 (author's follow-up): top-view overlay of the torso centre (COM proxy) trajectories.

Left panel : Video clips L-R turn, turn R, turn L, walk (overlaid).  Right panel: all six MoCap segments (overlaid).
Only the centre between the two torso keypoints is drawn (no paws). Global coordinates, ONE rigid transform per clip:
origin = torso centre of frame 0, yaw such that the torso heading averaged over the first 0.2 s points to +x, i.e.
every movement starts in positive x direction. Both sources at 30 Hz (MoCap sub-sampled from 60 Hz).
Colours: shades sampled from the viridis colormap of the paper's Fig. 5 - greens for the video clips, blues for MoCap.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import COL_W, COLORS, NAME_MOCAP, NAME_VIDEO, rot_z, savefig
from exp1_noise import get_clips

VIDEO_SEL = ["L-R turn", "turn R", "turn L", "walk"]
MOCAP_SEL = ["pace", "trot", "trot2", "canter", "right turn0", "left turn0"]
VIR = plt.get_cmap("viridis")
GREENS = [VIR(v) for v in np.linspace(0.55, 0.85, len(VIDEO_SEL))]  # green part of viridis
BLUES = [VIR(v) for v in np.linspace(0.18, 0.42, len(MOCAP_SEL))]  # blue part of viridis
INIT_WINDOW_S = 0.2


def com_start_aligned(c):
    """Torso centre trajectory (T, 3) with frame 0 at the origin and the initial heading rotated onto +x."""
    body = c["body"]
    com = body.mean(axis=0)
    k = max(2, int(round(INIT_WINDOW_S * c["fs"])))
    heading = (body[1] - body[0])[:k, :2].mean(axis=0)
    R = rot_z(-np.arctan2(heading[1], heading[0]))
    return (com - com[0]) @ R.T


def panel(ax, clips, names, colors, title, title_color):
    for name, col in zip(names, colors):
        c = [c for c in clips if c["name"] == name][0]
        p = com_start_aligned(c)
        ax.plot(p[:, 0], p[:, 1], color=col, lw=1.1, label=f"{name} ({p.shape[0] / c['fs']:.1f} s)")
        ax.plot(p[0, 0], p[0, 1], marker="o", color=col, ms=3, lw=0)
        ax.plot(p[-1, 0], p[-1, 1], marker=">", color=col, ms=3, lw=0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 1.9)
    ax.set_ylim(-0.8, 0.7)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.grid(alpha=0.3, lw=0.4)
    ax.tick_params(length=2)
    ax.set_xlabel("x [m]")
    ax.set_title(title, fontsize=7, fontweight="bold", color=title_color, pad=3)
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, handlelength=1.3, columnspacing=0.8, labelspacing=0.25)


def main():
    clips = get_clips()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw=dict(wspace=0.12, left=0.1, right=0.99, top=0.9, bottom=0.42))
    panel(axes[0], [c for c in clips if c["source"] == NAME_VIDEO], VIDEO_SEL, GREENS, "Video w. Depth Camera", COLORS[NAME_VIDEO])
    panel(axes[1], [c for c in clips if c["source"] == NAME_MOCAP], MOCAP_SEL, BLUES, "MoCap", COLORS[NAME_MOCAP])
    axes[0].set_ylabel("y [m]", labelpad=1)
    axes[1].set_yticklabels([])
    fig.text(0.5, 0.005, "torso centre, top view; each clip starts at the origin heading +x (o start, > end)", ha="center", fontsize=5.5, color="0.3")
    savefig(fig, "fig_exp1_com_topview_overlay")
    plt.close(fig)


if __name__ == "__main__":
    main()
