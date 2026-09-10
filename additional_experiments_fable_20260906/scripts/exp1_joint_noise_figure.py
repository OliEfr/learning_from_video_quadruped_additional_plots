"""Joint-space noise of the retargeted expert data (author's request): ONE single-column figure.
Top     joint angles of the front-right thigh and calf vs time, MoCap trot2 vs Video walk (the clips of Fig. 7), same axes.
Bottom  direction reversals of the joint motion per second, per joint type (hip / thigh / calf): one dot per clip, bar = set mean,
        MoCap (6 clips) vs Video (extended) (8 clips).  A reversal = sign change of the frame-to-frame joint-angle change; a walking
        joint reverses ~2x per stride, everything beyond that is noise turning the signal around.  Per second, so the different
        frame durations (MoCap 0.021 s, video 0.0333 s) do not matter.
Joint order in the expert files: URDF FL, FR, RL, RR x (hip, thigh, calf) (notes.md 9b).
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from common import AMP_DIRS, COL_W, FIG, NAME_MOCAP, NAME_VIDEO_EXT, load_amp_dir

SRC = [NAME_MOCAP, NAME_VIDEO_EXT]
LABEL = {NAME_MOCAP: "MoCap", NAME_VIDEO_EXT: "Video"}
ROW = {NAME_MOCAP: "MoCap\n(trot)", NAME_VIDEO_EXT: "Video\n(walk)"}
TRACE_CLIP = {NAME_MOCAP: "trot2", NAME_VIDEO_EXT: "walk_869488000"}
COL = {NAME_MOCAP: plt.get_cmap("viridis")(0.15), NAME_VIDEO_EXT: plt.get_cmap("viridis")(0.30)}
JOINTS = {"hip": [0, 3, 6, 9], "thigh": [1, 4, 7, 10], "calf": [2, 5, 8, 11]}
FR_THIGH, FR_CALF = 4, 5
T_MAX = 0.9
FS_LAB, FS_TICK, FS_LEG, FS_HEAD = 6.2, 5.2, 5.6, 7.2


def style(ax):
    ax.grid(alpha=0.3, lw=0.4)
    ax.tick_params(length=2, width=0.5, labelsize=FS_TICK, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def reversals_per_s(q, fd):
    """Direction reversals per second per joint, q (T, 12)."""
    v = np.diff(q, axis=0)
    rev = (np.sign(v[1:]) * np.sign(v[:-1]) < 0).sum(axis=0)
    return rev / (len(q) * fd)


def main():
    clips = {s: load_amp_dir(AMP_DIRS[s]) for s in SRC}
    fig = plt.figure(figsize=(COL_W, 2.9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.42, 1.35], hspace=0.12, left=0.21, right=0.98, top=0.93, bottom=0.10)  # row 2 = spacer
    # ---- (a) traces
    axes = []
    for i, s in enumerate(SRC):
        ax = fig.add_subplot(gs[i])
        c = next(c for c in clips[s] if c["name"] == TRACE_CLIP[s])
        q = np.degrees(c["frames"][:, 7:19])
        t = np.arange(len(q)) * c["fd"]
        ax.plot(t, q[:, FR_THIGH], color=COL[s], lw=1.0, label="thigh")
        ax.plot(t, q[:, FR_CALF], color=COL[s], lw=1.0, ls="--", label="calf")
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-125, 95)
        ax.set_yticks([-90, -45, 0, 45, 90])
        style(ax)
        ax.text(-0.30, 0.5, ROW[s], transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=FS_LAB, fontweight="bold", linespacing=1.15)
        if i == 0:
            ax.set_title("Joint angles, front-right leg", fontsize=FS_HEAD, fontweight="bold", pad=3)
            ax.set_xticklabels([])
            ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=FS_LEG, handlelength=1.8, columnspacing=0.9, borderaxespad=0.15)
        else:
            ax.set_xlabel("time [s]", fontsize=FS_LAB, labelpad=1.5)
        axes.append(ax)
    fig.text(0.115, 0.5 * (axes[0].get_position().y1 + axes[1].get_position().y0), "angle [deg]", rotation=90, ha="center", va="center", fontsize=FS_LAB)
    # ---- (b) reversals per second per joint type
    ax = fig.add_subplot(gs[3])
    xs = {"hip": 0, "thigh": 1.3, "calf": 2.6}
    off = {NAME_MOCAP: -0.22, NAME_VIDEO_EXT: 0.22}
    means = {}
    for s in SRC:
        per_clip = np.array([reversals_per_s(c["frames"][:, 7:19], c["fd"]) for c in clips[s]])  # (n_clips, 12)
        for jt, idx in JOINTS.items():
            v = per_clip[:, idx].mean(axis=1)  # per clip, mean over the 4 legs
            x = xs[jt] + off[s]
            jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.035
            ax.scatter(x + jit, v, s=7, color=COL[s], alpha=0.85, lw=0, zorder=3)
            ax.hlines(v.mean(), x - 0.16, x + 0.16, color=COL[s], lw=1.4, zorder=4)
            means[(jt, s)] = float(v.mean())
    for jt in JOINTS:
        r = means[(jt, NAME_VIDEO_EXT)] / means[(jt, NAME_MOCAP)]
        ax.text(xs[jt], 0.97, f"{r:.1f}x", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_LAB)
    ax.set_xlim(-0.6, 3.2)
    ax.set_ylim(0, 14)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["Hip", "Thigh", "Calf"])
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, labelsize=FS_LAB)
    ax.set_ylabel("reversals [1/s]", fontsize=FS_LAB, labelpad=1.5)
    ax.set_title("Direction reversals of the joint motion", fontsize=FS_HEAD, fontweight="bold", pad=3)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=COL[s], ms=3, lw=1.2) for s in SRC]
    ax.legend(handles, [LABEL[s] for s in SRC], loc="lower right", ncol=2, frameon=False, fontsize=FS_LEG, handlelength=1.8, columnspacing=1.0, borderaxespad=0.15)
    for k, v in means.items():
        print(f"reversals/s {k[0]:6s} {LABEL[k[1]]:6s} {v:5.2f}")
    for name in ["fig_exp1_joint_noise"]:
        fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
        fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
        print("saved", os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
