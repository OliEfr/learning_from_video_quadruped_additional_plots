"""Experiment 1 (author's follow-up): quantitative trajectory comparison over ALL clips.

Revision 3/4 (author): 2-row (MoCap trot2 + Video walk) and 14-row variants, each with the video rows either as
used in the paper or re-lifted with the corrected 640x360 intrinsics (exp1_relift_keypoints.py). Two columns:
left = top view (x-y), right = side view (x-z). Global coordinates, ONE rigid transform per clip
(origin = body centre of frame 0, mean heading -> +x, ground = 5th percentile of paw heights); nothing is
re-centred per frame. Every panel uses the same axis ranges, so lengths are directly comparable.

Colours: torso keypoints black; the four paws in four green shades sampled from the viridis colormap used by the
paper's Fig. 5 heatmaps; row labels use the per-method colours of plot_DEFINITIONS.py (MoCap = tab10[5],
Video = tab10[1], clips that only belong to Video (extended) = tab10[6]).
No per-row annotations (author request); the per-clip numbers are in data/exp1_per_clip_table.md.
Both sources at 30 Hz (MoCap sub-sampled from 60 Hz).
"""
import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, DOUBLE_W, NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR, NAME_VIDEO_EXT, VIDEO_BASE_CLIPS, savefig
from exp1_noise import get_clips

BODY_COLOR = "black"
FEET_COLORS = [plt.get_cmap("viridis")(v) for v in (0.55, 0.68, 0.80, 0.92)]  # green segment of viridis (Fig. 5 colormap)
FEET_LABELS = ["paw 1", "paw 2", "paw 3", "paw 4"]
X_SPAN, X_PAD = 2.4, 0.15  # common metric x range [m]
Y_TOP = 0.35  # top view: y in [yc - Y_TOP, yc + Y_TOP]  (0.70 m span, same as the side view)
Z_SIDE = (-0.08, 0.62)
KEEP_TWO = [(NAME_MOCAP, "trot2"), ("video", "walk")]  # author: "keep only the mocap trot2 and video walk"


def row_style(c):
    """Row-label colour and source name: the per-method colours of plot_DEFINITIONS.py."""
    if c["source"] == NAME_MOCAP:
        return COLORS[NAME_MOCAP], "MoCap"
    if c.get("clip") in VIDEO_BASE_CLIPS:
        return COLORS[NAME_VIDEO], "Video"
    return COLORS[NAME_VIDEO_EXT], "Video (ext.)"


def legend_handles():
    handles = [plt.Line2D([], [], color=BODY_COLOR, lw=1.2, label="torso keypoints (rear / front)")]
    handles += [plt.Line2D([], [], color=FEET_COLORS[i], lw=1.0, label=FEET_LABELS[i]) for i in range(4)]
    return handles


def draw_clip_row(axes_row, c, r, n, y_top=Y_TOP, z_side=Z_SIDE, mode="panel"):
    """One clip as top view (axes_row[0]) and side view (axes_row[1]); r is the row index of n rows
    (x tick labels only on the last row, y labels on every row in "panel" mode)."""
    body, feet = c["body"], c["feet"]
    allp = np.concatenate([body, feet], axis=0).reshape(-1, 3)
    x0 = allp[:, 0].min() - X_PAD
    yc = body[..., 1].mean()
    for ax, j in zip(axes_row, (1, 2)):
        for i in range(4):
            ax.plot(feet[i, :, 0], feet[i, :, j], color=FEET_COLORS[i], lw=0.8)
        for i in range(2):
            ax.plot(body[i, :, 0], body[i, :, j], color=BODY_COLOR, lw=0.9)
        for f in np.linspace(0, body.shape[1] - 1, 5).astype(int):  # torso segment at 5 instants
            ax.plot(body[:, f, 0], body[:, f, j], color=BODY_COLOR, lw=1.3, alpha=0.45)
        ax.set_xlim(x0, x0 + X_SPAN)
        if mode == "panel":
            ax.set_aspect("equal")
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(length=2, labelsize=5 if mode == "panel" else 4)
        if j == 1:
            ax.set_ylim(yc - y_top, yc + y_top)
            ax.set_yticks([t for t in np.arange(-0.9, 0.91, 0.3) if yc - y_top < t < yc + y_top])
            if mode == "panel" or r == n // 2:
                ax.set_ylabel("y [m]", labelpad=1)
        else:
            ax.set_ylim(*z_side)
            ax.set_yticks([0, 0.3, 0.6])
            if mode == "panel" or r == n // 2:
                ax.set_ylabel("z [m]", labelpad=1)
            ax.axhline(0, color="0.55", lw=0.5, ls="--")
        if r < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x [m]")


def main(video_source=NAME_VIDEO, rows="two", mode="panel", video_override=None, suffix="", footer=None):
    """video_source: NAME_VIDEO (pipeline output, as used in the paper) or NAME_VIDEO_CORR (re-lifted with the 640x360 intrinsics).
    rows: "two" (MoCap trot2 + Video walk) or "all" (6 MoCap segments, then the 8 flat-walking video clips).
    mode="panel": every panel has the size and equal metric aspect of the panels in fig_exp1_all_clips_*.
    The corrected reconstruction is ~2x larger, so the corrected variants use a wider y/z span (0.9 m); MoCap and video
    rows always share identical axis ranges within one figure."""
    corrected = video_source == NAME_VIDEO_CORR
    y_top, z_side = (0.45, (-0.08, 0.82)) if corrected else (Y_TOP, Z_SIDE)
    all_clips = get_clips()
    video_clips = video_override if video_override is not None else [c for c in all_clips if c["source"] == video_source]
    if rows == "two":
        clips = [c for c in all_clips if c["source"] == NAME_MOCAP and c["name"] == "trot2"] + [c for c in video_clips if c["name"] == "walk"]
    else:
        clips = [c for c in all_clips if c["source"] == NAME_MOCAP] + video_clips
    n = len(clips)
    fig, axes = plt.subplots(n, 2, figsize=(3.5, 0.42 * n + 1.0), gridspec_kw=dict(hspace=0.35, wspace=0.42, left=0.22 if n <= 2 else 0.3, right=0.99, top=0.9 if n <= 2 else 0.955, bottom=0.33 if n <= 2 else 0.09))
    for r, c in enumerate(clips):
        draw_clip_row(axes[r], c, r, n, y_top, z_side, mode)
        col, src_lab = row_style(c)
        b = axes[r, 0].get_position()
        fig.text(0.005, (b.y0 + b.y1) / 2, f"{src_lab}\n{c['name']}", ha="left", va="center", fontsize=6 if mode == "panel" else 4.5, fontweight="bold", color=col)
    for ax, lab in zip(axes[0], ("top view (x-y)", "side view (x-z)")):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.99, lab, ha="center", va="top", fontsize=7.5, fontweight="bold")
    fig.legend(handles=legend_handles(), loc="lower center", frameon=False, bbox_to_anchor=(0.6, 0.035 if (n <= 2 and corrected) else (0.014 if corrected else 0.0)), fontsize=5.5, ncol=3 if n <= 2 else 5)
    if footer is None and corrected:
        footer = "video rows re-lifted with the 640x360 intrinsics (corrected); MoCap unchanged"
    if footer:
        fig.text(0.5, 0.0, footer, ha="center", va="bottom", fontsize=5.2, color="0.3")
    savefig(fig, "fig_exp1_trajectory_grid_all_clips" + ("_corrected" if corrected else "") + suffix + ("" if rows == "two" else "_14rows"))
    plt.close(fig)


if __name__ == "__main__":
    main(NAME_VIDEO, "two")  # shipped grid (unchanged)
    main(NAME_VIDEO_CORR, "two")  # corrected video rows, same 2-row selection
    main(NAME_VIDEO, "all")  # full clip set, pipeline output
    main(NAME_VIDEO_CORR, "all")  # full clip set, corrected video rows
