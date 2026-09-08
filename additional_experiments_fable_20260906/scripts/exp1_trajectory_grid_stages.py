"""Experiment 1 (author's follow-up): the corrected 2-row grid, extended by the two later stages of the video pipeline.

Rows (same layout / panel geometry / colours as fig_exp1_trajectory_grid_all_clips_corrected.pdf):
  1  MoCap trot2            - raw Zhang et al. markers (pelvis, neck, 4 toes), 60 -> 30 Hz
  2  Video walk, corrected  - 2D tracks re-lifted with the 640x360 intrinsics (exp1_relift_keypoints.py, not used in the paper)
  3  Video walk, post-proc. - 3D keypoints exactly as fed to retarget_motion_fromVision.py (data_fromVision_depth_cam/walk_869488000)
  4  Video walk, retargeted - Unitree Go2 root position (black) and paw positions from forward kinematics of the 12 joint
                             angles in datasets/fromVision_motions_DepthCam/walk_869488000_amp.txt (the paper's expert file).
                             Leg order of the joint columns = URDF order FL, FR, RL, RR (verified against the saved MoCap toe
                             targets, see checks_report.md); Go2 geometry from assets/go2/go2.urdf (hip +-0.1934/+-0.0465 m,
                             thigh offset 0.0955 m, thigh and calf 0.213 m, foot at the calf tip).
Global coordinates, one rigid transform per row (origin = torso/root at frame 0, mean heading -> +x, ground = 5th percentile
of paw heights). No annotations. Paws in viridis greens, torso/root black, row labels in the Fig. 4/5 method colours.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common import AMP_DIRS, COLORS, NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR, align_clip, rot_z, savefig, yaw_from_quat_xyzw
from exp1_noise import get_clips
from exp1_trajectory_grid import BODY_COLOR, FEET_COLORS, FEET_LABELS, X_PAD, X_SPAN

HIP = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
LEG_ORDER = ["FL", "FR", "RL", "RR"]  # URDF order = pybullet joint order in the expert files
THIGH_OFF, THIGH, CALF = 0.0955, 0.213, 0.213
Y_TOP, Z_SIDE = 0.45, (-0.08, 0.82)  # as in the corrected grid


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def go2_feet_world(frame):
    """Paw positions (4, 3) in world coordinates for one expert frame [root(3), quat xyzw(4), 12 joints]."""
    root, R, q = frame[:3], quat_R(frame[3:7]), frame[7:19].reshape(4, 3)
    feet = []
    for leg, (h, t, c) in zip(LEG_ORDER, q):
        hx, hy = HIP[leg]
        side = 1.0 if leg[1] == "L" else -1.0
        Rh = Rx(h)
        p_th = np.array([hx, hy, 0.0]) + Rh @ np.array([0, side * THIGH_OFF, 0])
        Rt = Rh @ Ry(t)
        p_ca = p_th + Rt @ np.array([0, 0, -THIGH])
        feet.append(root + R @ (p_ca + Rt @ Ry(c) @ np.array([0, 0, -CALF])))
    return np.array(feet)


def retargeted_clip(path, fs=30.0):
    F = np.array(json.load(open(path))["Frames"])
    root = F[:, :3]
    yaw = yaw_from_quat_xyzw(F[:, 3:7])
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)
    body = np.stack([root - 0.1 * fwd, root + 0.1 * fwd])  # two pseudo torso points 0.1 m behind/ahead of the root (heading only)
    feet = np.stack([go2_feet_world(f) for f in F], axis=1)  # (4, T, 3)
    body, feet, _ = align_clip(body, feet)
    return dict(body=body, feet=feet, fs=fs, root_only=True)


def main():
    clips = get_clips()
    pick = lambda src, name: [c for c in clips if c["source"] == src and c["name"] == name][0]  # noqa: E731
    rows = [
        (pick(NAME_MOCAP, "trot2"), "MoCap\ntrot2", COLORS[NAME_MOCAP]),
        (pick(NAME_VIDEO_CORR, "walk"), "Video walk\ncorrected re-lift", COLORS[NAME_VIDEO]),
        (pick(NAME_VIDEO, "walk"), "Video walk\npost-processed\n(retarget. input)", COLORS[NAME_VIDEO]),
        (retargeted_clip(os.path.join(AMP_DIRS[NAME_VIDEO], "walk_869488000_amp.txt")), "Video walk\nretargeted Go2\n(root + FK paws)", COLORS[NAME_VIDEO]),
    ]
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(3.5, 0.5 * n + 1.0), gridspec_kw=dict(hspace=0.35, wspace=0.42, left=0.34, right=0.99, top=0.94, bottom=0.2))
    for r, (c, label, col) in enumerate(rows):
        body, feet = c["body"], c["feet"]
        allp = np.concatenate([body, feet], axis=0).reshape(-1, 3)
        x0 = allp[:, 0].min() - X_PAD
        yc = body[..., 1].mean()
        for ax, j in zip(axes[r], (1, 2)):
            for i in range(4):
                ax.plot(feet[i, :, 0], feet[i, :, j], color=FEET_COLORS[i], lw=0.8)
            if c.get("root_only"):
                root = body.mean(axis=0)
                ax.plot(root[:, 0], root[:, j], color=BODY_COLOR, lw=1.1)
            else:
                for i in range(2):
                    ax.plot(body[i, :, 0], body[i, :, j], color=BODY_COLOR, lw=0.9)
            for f in np.linspace(0, body.shape[1] - 1, 5).astype(int):
                ax.plot(body[:, f, 0], body[:, f, j], color=BODY_COLOR, lw=1.3, alpha=0.45)
            ax.set_xlim(x0, x0 + X_SPAN)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3, lw=0.4)
            ax.tick_params(length=2, labelsize=5)
            if j == 1:
                ax.set_ylim(yc - Y_TOP, yc + Y_TOP)
                ax.set_yticks([t for t in np.arange(-0.9, 0.91, 0.3) if yc - Y_TOP < t < yc + Y_TOP])
                ax.set_ylabel("y [m]", labelpad=1)
            else:
                ax.set_ylim(*Z_SIDE)
                ax.set_yticks([0, 0.3, 0.6])
                ax.set_ylabel("z [m]", labelpad=1)
                ax.axhline(0, color="0.55", lw=0.5, ls="--")
            if r < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("x [m]")
        b = axes[r, 0].get_position()
        fig.text(0.005, (b.y0 + b.y1) / 2, label, ha="left", va="center", fontsize=5.5, fontweight="bold", color=col)
    for ax, lab in zip(axes[0], ("top view (x-y)", "side view (x-z)")):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.99, lab, ha="center", va="top", fontsize=7.5, fontweight="bold")
    handles = [plt.Line2D([], [], color=BODY_COLOR, lw=1.2, label="torso keypoints / robot root")]
    handles += [plt.Line2D([], [], color=FEET_COLORS[i], lw=1.0, label=FEET_LABELS[i]) for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.6, 0.0), fontsize=5.5)
    savefig(fig, "fig_exp1_trajectory_grid_walk_stages")
    plt.close(fig)


if __name__ == "__main__":
    main()
