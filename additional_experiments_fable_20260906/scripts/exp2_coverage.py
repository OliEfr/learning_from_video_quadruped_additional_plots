"""Experiment 2: Which part of the target (command) distribution do the expert datasets cover?

Datasets (exactly the AMP expert files used for the paper's flat-walking policies):
  MoCap = datasets/mocap_AMP_for_hardware, Video = datasets/fromVision_motions_DepthCam,
  Video (extended) = datasets/fromVision_motions_DepthCam_extendedWithoutReverse

Per expert frame we compute the quantities the RL command refers to:
  vx, vy : base linear velocity in the base-yaw frame, wz : yaw rate.
  Primary ("smoothed"): root position and unwrapped yaw are Savitzky-Golay smoothed (7 frames, poly 3, i.e.
  0.23 s for video / 0.15 s for MoCap) before central differencing with the dataset's FrameDuration.
  Secondary ("raw"): the stored linear velocity (columns 31:33 = diff(root_pos)/FrameDuration) and the raw
  finite-difference yaw rate. The angular-velocity columns 34:37 are all-zero in every dataset, so wz must be
  derived from the root quaternion in any case.
Frames are weighted like the AMP loader samples them (trajectory MotionWeight / trajectory length).

Target distribution (velocity_env_cfg.py, CommandsCfg): uniform vx in [-1, 1] m/s, vy in [-0.3, 0.3] m/s,
wz in [-1.57, 1.57] rad/s, heading_command=False, 2 % standing environments.
Evaluation grids (targetDistributions.sh / Fig. 5): vx x vy at wz=0 (21 x 7, step 0.1), vx x wz at vy=0 (21 x 21).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from common import (
    AMP_DIRS,
    CMD_RANGES,
    COLORS,
    DATA,
    DOUBLE_W,
    EVAL_GRID,
    NAME_MOCAP,
    NAME_VIDEO,
    NAME_VIDEO_EXT,
    PAPER_LABEL,
    REL_STANDING_ENVS,
    load_amp_dir,
    md_table,
    rot_z,
    savefig,
    sg_residual,
    yaw_from_quat_xyzw,
)

SOURCES = [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_EXT]
SMOOTH_WINDOW, SMOOTH_POLY = 7, 3
# coverage rule: a target command is covered if some expert frame lies in the evaluation cell around it
TOL_V = 0.05  # half the evaluation-grid spacing [m/s]
TOL_W = 0.05  # half the evaluation-grid spacing [rad/s]
SLICE_W = 0.25  # |wz| tolerance for the vx-vy grid (evaluated at wz = 0) [rad/s]
SLICE_VY = 0.10  # |vy| tolerance for the vx-wz grid (evaluated at vy = 0) [m/s]
BOX_TOL = (0.1, 0.1, 0.2)  # tolerance box for the 3-D Monte-Carlo coverage (one eval-grid step per axis)


# ----------------------------------------------------------------------------------------------
def dataset_frames(src):
    """Concatenate all clips of a dataset -> dict of per-frame arrays + AMP sampling weights."""
    clips = load_amp_dir(AMP_DIRS[src])
    keys = ["vx", "vy", "wz", "vx_raw", "vy_raw", "wz_raw", "q", "w_amp", "clip", "fd", "v_speed"]
    out = {k: [] for k in keys}
    total_w = sum(c["weight"] for c in clips)
    for c in clips:
        F, fd = c["frames"], c["fd"]
        T = len(F)
        yaw = np.unwrap(yaw_from_quat_xyzw(F[:, 3:7]))
        # raw (as stored)
        v_w = F[:, 31:34]
        v_b = np.stack([rot_z(-y) @ v for y, v in zip(yaw, v_w)])
        wz_raw = np.gradient(yaw) / fd
        # smoothed
        pos_s = F[:, :3] - sg_residual(F[:, :3], SMOOTH_WINDOW, SMOOTH_POLY)
        yaw_s = yaw - sg_residual(yaw[:, None], SMOOTH_WINDOW, SMOOTH_POLY)[:, 0]
        v_ws = np.gradient(pos_s, axis=0) / fd
        v_bs = np.stack([rot_z(-y) @ v for y, v in zip(yaw_s, v_ws)])
        wz_s = np.gradient(yaw_s) / fd
        out["vx"].append(v_bs[:, 0])
        out["vy"].append(v_bs[:, 1])
        out["wz"].append(wz_s)
        out["vx_raw"].append(v_b[:, 0])
        out["vy_raw"].append(v_b[:, 1])
        out["wz_raw"].append(wz_raw)
        out["q"].append(F[:, 7:19])
        out["w_amp"].append(np.full(T, c["weight"] / total_w / T))  # AMPLoader: p(traj) = w_i / sum w, time uniform within traj
        out["clip"].append(np.array([c["name"]] * T))
        out["fd"].append(np.full(T, fd))
        out["v_speed"].append(np.linalg.norm(v_bs[:, :2], axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}, clips


def covered_cells_xy(d, grid_x, grid_y, tol_v=TOL_V, slice_w=SLICE_W, suffix=""):
    """(ny, nx) bool: eval cell (cx, cy) at wz=0 contains an expert frame."""
    vx, vy, wz = d["vx" + suffix], d["vy" + suffix], d["wz" + suffix]
    cov = np.zeros((len(grid_y), len(grid_x)), bool)
    sel = np.abs(wz) <= slice_w
    for j, cy in enumerate(grid_y):
        for i, cx in enumerate(grid_x):
            cov[j, i] = np.any(sel & (np.abs(vx - cx) <= tol_v) & (np.abs(vy - cy) <= tol_v))
    return cov


def covered_cells_xyaw(d, grid_x, grid_w, tol_v=TOL_V, tol_w=TOL_W, slice_vy=SLICE_VY, suffix=""):
    vx, vy, wz = d["vx" + suffix], d["vy" + suffix], d["wz" + suffix]
    cov = np.zeros((len(grid_w), len(grid_x)), bool)
    sel = np.abs(vy) <= slice_vy
    for j, cw in enumerate(grid_w):
        for i, cx in enumerate(grid_x):
            cov[j, i] = np.any(sel & (np.abs(vx - cx) <= tol_v) & (np.abs(wz - cw) <= tol_w))
    return cov


def nearest_distance_map(d, grid_a, grid_b, plane):
    """Normalised distance (each axis / command half-range) from every eval grid point to the nearest expert frame."""
    scale = np.array([1.0, 0.3, 1.57])
    P = np.stack([d["vx"], d["vy"], d["wz"]], 1) / scale
    out = np.zeros((len(grid_b), len(grid_a)))
    for j, b in enumerate(grid_b):
        for i, a in enumerate(grid_a):
            c = np.array([a, b, 0.0]) if plane == "xy" else np.array([a, 0.0, b])
            out[j, i] = np.min(np.linalg.norm(P - c / scale, axis=1))
    return out


def box_coverage(d, n=40000, seed=0, suffix=""):
    rng = np.random.default_rng(seed)
    lo = np.array([CMD_RANGES["vx"][0], CMD_RANGES["vy"][0], CMD_RANGES["wz"][0]])
    hi = np.array([CMD_RANGES["vx"][1], CMD_RANGES["vy"][1], CMD_RANGES["wz"][1]])
    C = rng.uniform(lo, hi, size=(n, 3))
    P = np.stack([d["vx" + suffix], d["vy" + suffix], d["wz" + suffix]], 1)
    tol = np.array(BOX_TOL)
    hit = np.zeros(n, bool)
    for k in range(0, n, 2000):
        diff = np.abs(C[k : k + 2000, None, :] - P[None]) <= tol
        hit[k : k + 2000] = diff.all(-1).any(-1)
    return float(hit.mean())


def wstats(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o] / w.sum()
    cw = np.cumsum(w)
    q = lambda p: float(np.interp(p, cw, x))  # noqa: E731
    return dict(mean=float((x * w).sum()), min=float(x.min()), max=float(x.max()), p05=q(0.05), p50=q(0.5), p95=q(0.95))


def row_label(fig, ax, text, color, x=0.005):
    b = ax.get_position()
    fig.text(x, (b.y0 + b.y1) / 2, text, ha="left", va="center", fontweight="bold", fontsize=7, color=color)


# ----------------------------------------------------------------------------------------------
def fig_target_coverage(D, fname):
    gx, gy = EVAL_GRID["xy"]
    gx2, gw = EVAL_GRID["xyaw"]
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_W * 0.8, 5.0), gridspec_kw=dict(hspace=0.35, wspace=0.42))
    fig.subplots_adjust(left=0.27, right=0.99, top=0.86, bottom=0.08)
    vmax = 0.6
    ims = []
    for r, src in enumerate(SOURCES):
        for c_, (plane, ga, gb, cov, ylab) in enumerate(
            [("xy", gx, gy, D[src]["cov_xy"], "Target Vel. Y [m/s]"), ("xyaw", gx2, gw, D[src]["cov_xyaw"], "Target Ang. Vel. Z [rad/s]")]
        ):
            ax = axes[r, c_]
            dist = D[src]["dist_" + plane]
            im = ax.imshow(dist, origin="lower", cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto", extent=[ga[0] - 0.05, ga[-1] + 0.05, gb[0] - 0.05, gb[-1] + 0.05])
            ims.append(im)
            yy, xx = np.where(cov)
            ax.scatter(ga[xx], gb[yy], s=3.5, color="white", lw=0, zorder=3)
            ax.set_xlim(ga[0] - 0.05, ga[-1] + 0.05)
            ax.set_ylim(gb[0] - 0.05, gb[-1] + 0.05)
            ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
            ax.set_yticks(np.arange(gb[0], gb[-1] + 1e-6, 0.5 if plane == "xyaw" else 0.1))
            ax.tick_params(length=2)
            n_cov, n_tot = int(cov.sum()), cov.size
            ax.set_title(f"{n_cov}/{n_tot} cells covered ({100 * n_cov / n_tot:.0f} %)", fontsize=6.5, pad=2)
            ax.set_ylabel(ylab)
            if r == 2:
                ax.set_xlabel("Target Vel. X [m/s]")
            if r == 0:
                ax.text(0.5, 1.2, "$v_x$ - $v_y$ evaluation grid ($\\omega_z=0$)" if plane == "xy" else "$v_x$ - $\\omega_z$ evaluation grid ($v_y=0$)", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=7)
        row_label(fig, axes[r, 0], PAPER_LABEL[src].replace(" w. ", " w.\n").replace(" (", "\n("), COLORS[src])
    cax = fig.add_axes([0.42, 0.955, 0.4, 0.015])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label("normalised distance to nearest expert frame\n(white dot: target cell contains expert data)", fontsize=6, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2)
    savefig(fig, fname)
    plt.close(fig)


def fig_state_coverage(D, fname):
    """Fig.-5-like grid: rows = datasets, columns = state-space heatmaps (AMP sampling weights, log colour)."""
    cols = [
        ("vx", "vy", "$v_x$ [m/s]", "$v_y$ [m/s]", (-1.2, 1.8), (-0.65, 0.65), 0.1, 0.05, ("vx", "vy"), "base velocity $v_x$ vs $v_y$"),
        ("vx", "wz", "$v_x$ [m/s]", "$\\omega_z$ [rad/s]", (-1.2, 1.8), (-2.6, 2.6), 0.1, 0.2, ("vx", "wz"), "base velocity $v_x$ vs $\\omega_z$"),
        ("thigh_f", "calf_f", "thigh [rad]", "calf [rad]", (0.0, 2.1), (-2.9, -0.4), 0.07, 0.083, None, "front legs: thigh vs calf"),
        ("thigh_r", "calf_r", "thigh [rad]", "calf [rad]", (0.0, 2.1), (-2.9, -0.4), 0.07, 0.083, None, "rear legs: thigh vs calf"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_W, 4.3), gridspec_kw=dict(hspace=0.25, wspace=0.38))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.86, bottom=0.09)
    for r, src in enumerate(SOURCES):
        d = D[src]["frames"]
        q = d["q"]
        # joint order in the expert files: 4 legs x (hip, thigh, calf); legs 0,1 = front, 2,3 = rear
        feats = dict(vx=d["vx"], vy=d["vy"], wz=d["wz"], thigh_f=np.r_[q[:, 1], q[:, 4]], calf_f=np.r_[q[:, 2], q[:, 5]], thigh_r=np.r_[q[:, 7], q[:, 10]], calf_r=np.r_[q[:, 8], q[:, 11]])
        for c_, (kx, ky, xl, yl, xr, yr, bx, by, cmd, title) in enumerate(cols):
            ax = axes[r, c_]
            x, y = feats[kx], feats[ky]
            w = d["w_amp"] if len(x) == len(d["w_amp"]) else np.r_[d["w_amp"], d["w_amp"]] / 2
            hx = np.arange(xr[0], xr[1] + 1e-9, bx)
            hy = np.arange(yr[0], yr[1] + 1e-9, by)
            Hh, _, _ = np.histogram2d(x, y, bins=[hx, hy], weights=w)
            Hh = np.ma.masked_where(Hh <= 0, Hh)
            ax.pcolormesh(hx, hy, Hh.T, cmap="viridis", norm=LogNorm(vmin=1e-4, vmax=0.3), rasterized=True)
            if cmd is not None:
                cx, cy = CMD_RANGES[cmd[0]], CMD_RANGES[cmd[1]]
                ax.add_patch(Rectangle((cx[0], cy[0]), cx[1] - cx[0], cy[1] - cy[0], fill=False, ec="red", lw=0.9, ls="--"))
                out = float(w[(x < cx[0]) | (x > cx[1]) | (y < cy[0]) | (y > cy[1])].sum() / w.sum())
                ax.text(0.03, 0.03, f"{100 * out:.0f} % outside\ncommand range", transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color="red")
            else:  # Go2 default pose (IsaacLab): thigh 0.8 (front) / 1.0 (rear), calf -1.5
                ax.plot(0.8 if kx == "thigh_f" else 1.0, -1.5, marker="+", color="red", ms=5, mew=0.8)
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
        row_label(fig, axes[r, 0], PAPER_LABEL[src].replace(" w. ", " w.\n").replace(" (", "\n("), COLORS[src])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=LogNorm(vmin=1e-4, vmax=0.3))
    cax = fig.add_axes([0.4, 0.955, 0.35, 0.015])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("fraction of AMP expert samples per bin (log);  red dashed box = command range,  red + = Go2 default pose", fontsize=6, labelpad=3)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, length=2)
    savefig(fig, fname)
    plt.close(fig)


def fig_velocity_distributions(D, fname):
    cols = [("vx", "Vel. X [m/s]", CMD_RANGES["vx"], (-1.3, 1.9), 0.1), ("vy", "Vel. Y [m/s]", CMD_RANGES["vy"], (-0.7, 0.7), 0.05), ("wz", "Ang. Vel. Z [rad/s]", CMD_RANGES["wz"], (-2.8, 2.8), 0.2)]
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_W, 3.9), gridspec_kw=dict(hspace=0.3, wspace=0.3))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.93, bottom=0.16)
    for r, src in enumerate(SOURCES):
        d = D[src]["frames"]
        for c_, (k, xl, cr, xr, bw) in enumerate(cols):
            ax = axes[r, c_]
            bins = np.arange(xr[0], xr[1] + 1e-9, bw)
            ax.hist(d[k], bins=bins, weights=d["w_amp"], density=True, color=COLORS[src], alpha=0.85, lw=0, label="expert dataset")
            u = 1.0 / (cr[1] - cr[0]) * (1 - REL_STANDING_ENVS)
            ax.fill_between([cr[0], cr[1]], 0, u, step="pre", color="0.35", alpha=0.25, lw=0, label="target distribution (uniform)")
            ax.plot([cr[0], cr[0], cr[1], cr[1]], [0, u, u, 0], color="0.25", lw=0.8)
            ax.plot([0], [u], marker="^", color="0.25", ms=3, lw=0, clip_on=False, label="2 % standing envs (spike at 0)")
            ax.set_xlim(*xr)
            ax.set_ylim(0, None)
            ax.tick_params(length=2)
            ax.grid(alpha=0.25, lw=0.4)
            ax.axvline(0, color="0.5", lw=0.4)
            if r == 2:
                ax.set_xlabel("base / target " + xl)
            else:
                ax.set_xticklabels([])
            if c_ == 0:
                ax.set_ylabel("density")
            st = D[src]["stats"][k]
            ax.text(0.98, 0.95, f"p5-p95: [{st['p05']:.2f}, {st['p95']:.2f}]\nmean {st['mean']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.2)
            if r == 0:
                ax.set_title({"vx": "$v_x$", "vy": "$v_y$", "wz": "$\\omega_z$"}[k], fontsize=7.5, fontweight="bold")
        row_label(fig, axes[r, 0], PAPER_LABEL[src].replace(" w. ", " w.\n").replace(" (", "\n("), COLORS[src])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.6, -0.01))
    savefig(fig, fname)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    gx, gy = EVAL_GRID["xy"]
    gx2, gw = EVAL_GRID["xyaw"]
    D = {}
    for src in SOURCES:
        d, clips = dataset_frames(src)
        D[src] = dict(frames=d, clips=clips)
        D[src]["cov_xy"] = covered_cells_xy(d, gx, gy)
        D[src]["cov_xyaw"] = covered_cells_xyaw(d, gx2, gw)
        D[src]["dist_xy"] = nearest_distance_map(d, gx, gy, "xy")
        D[src]["dist_xyaw"] = nearest_distance_map(d, gx2, gw, "xyaw")
        D[src]["box_cov"] = box_coverage(d)
        D[src]["stats"] = {k: wstats(d[k], d["w_amp"]) for k in ["vx", "vy", "wz", "v_speed", "vx_raw", "vy_raw", "wz_raw"]}
        w = d["w_amp"]
        D[src]["fractions"] = dict(
            backward=float(w[d["vx"] < -0.05].sum()),
            standing=float(w[(d["v_speed"] < 0.1) & (np.abs(d["wz"]) < 0.2)].sum()),
            turning=float(w[np.abs(d["wz"]) > 0.5].sum()),
            lateral=float(w[np.abs(d["vy"]) > 0.15].sum()),
            faster_than_1=float(w[d["vx"] > 1.0].sum()),
            outside_box=float(w[(np.abs(d["vx"]) > 1.0) | (np.abs(d["vy"]) > 0.3) | (np.abs(d["wz"]) > 1.57)].sum()),
        )

    fig_target_coverage(D, "fig_exp2_target_coverage")
    fig_state_coverage(D, "fig_exp2_state_coverage")
    fig_velocity_distributions(D, "fig_exp2_velocity_distributions")

    # ---- three-way comparison numbers
    res = dict(settings=dict(TOL_V=TOL_V, TOL_W=TOL_W, SLICE_W=SLICE_W, SLICE_VY=SLICE_VY, BOX_TOL=BOX_TOL, SMOOTH_WINDOW=SMOOTH_WINDOW, SMOOTH_POLY=SMOOTH_POLY), datasets={})
    for src in SOURCES:
        d = D[src]["frames"]
        res["datasets"][src] = dict(
            n_clips=len(D[src]["clips"]),
            n_frames=int(len(d["vx"])),
            duration_s=float(sum((len(c["frames"]) - 1) * c["fd"] for c in D[src]["clips"])),
            covered_xy=int(D[src]["cov_xy"].sum()),
            covered_xyaw=int(D[src]["cov_xyaw"].sum()),
            covered_xy_raw=int(covered_cells_xy(d, gx, gy, suffix="_raw").sum()),
            covered_xyaw_raw=int(covered_cells_xyaw(d, gx2, gw, suffix="_raw").sum()),
            box_coverage=D[src]["box_cov"],
            box_coverage_raw=box_coverage(d, suffix="_raw"),
            stats=D[src]["stats"],
            fractions=D[src]["fractions"],
            wz_raw_minus_smooth_std=float(np.std(d["wz_raw"] - d["wz"])),
            vx_raw_minus_smooth_std=float(np.std(d["vx_raw"] - d["vx"])),
            clips={c["name"]: dict(frames=len(c["frames"]), weight=c["weight"], fd=c["fd"]) for c in D[src]["clips"]},
        )
    cov = {s: (D[s]["cov_xy"], D[s]["cov_xyaw"]) for s in SOURCES}
    res["set_relations"] = {
        "xy: video_ext minus video": int((cov[NAME_VIDEO_EXT][0] & ~cov[NAME_VIDEO][0]).sum()),
        "xy: video_ext minus mocap": int((cov[NAME_VIDEO_EXT][0] & ~cov[NAME_MOCAP][0]).sum()),
        "xy: mocap minus video_ext": int((cov[NAME_MOCAP][0] & ~cov[NAME_VIDEO_EXT][0]).sum()),
        "xy: video minus mocap": int((cov[NAME_VIDEO][0] & ~cov[NAME_MOCAP][0]).sum()),
        "xy: mocap minus video": int((cov[NAME_MOCAP][0] & ~cov[NAME_VIDEO][0]).sum()),
        "xy: union of all three": int((cov[NAME_MOCAP][0] | cov[NAME_VIDEO_EXT][0]).sum()),
        "xyaw: video_ext minus video": int((cov[NAME_VIDEO_EXT][1] & ~cov[NAME_VIDEO][1]).sum()),
        "xyaw: video_ext minus mocap": int((cov[NAME_VIDEO_EXT][1] & ~cov[NAME_MOCAP][1]).sum()),
        "xyaw: mocap minus video_ext": int((cov[NAME_MOCAP][1] & ~cov[NAME_VIDEO_EXT][1]).sum()),
        "xyaw: video minus mocap": int((cov[NAME_VIDEO][1] & ~cov[NAME_MOCAP][1]).sum()),
        "xyaw: mocap minus video": int((cov[NAME_MOCAP][1] & ~cov[NAME_VIDEO][1]).sum()),
        "xy cells total": int(cov[NAME_MOCAP][0].size),
        "xyaw cells total": int(cov[NAME_MOCAP][1].size),
        "xy cells with vx>=0": int((gx >= -1e-9).sum() * len(gy)),
        "xyaw cells with vx>=0": int((gx2 >= -1e-9).sum() * len(gw)),
        "xy: covered cells with vx<0 (mocap, video, video_ext)": [int(cov[s][0][:, gx < -1e-9].sum()) for s in SOURCES],
        "xyaw: covered cells with vx<0 (mocap, video, video_ext)": [int(cov[s][1][:, gx2 < -1e-9].sum()) for s in SOURCES],
        "xy: covered cells in vx in [0.0,0.5] band (mocap, video, video_ext)": [int(cov[s][0][:, (gx >= -1e-9) & (gx <= 0.5 + 1e-9)].sum()) for s in SOURCES],
        "xyaw: covered cells with |wz|>=0.5 (mocap, video, video_ext)": [int(cov[s][1][np.abs(gw) >= 0.5 - 1e-9].sum()) for s in SOURCES],
        "xyaw: covered cells with wz<=-0.5 (mocap, video, video_ext)": [int(cov[s][1][gw <= -0.5 + 1e-9].sum()) for s in SOURCES],
        "xyaw: covered cells with wz>=0.5 (mocap, video, video_ext)": [int(cov[s][1][gw >= 0.5 - 1e-9].sum()) for s in SOURCES],
    }

    def per_clip_table(src):
        d = D[src]["frames"]
        out = {}
        for c in D[src]["clips"]:
            m = d["clip"] == c["name"]
            sub = {k: v[m] for k, v in d.items()}
            out[c["name"]] = dict(
                frames=int(m.sum()),
                weight=c["weight"],
                covered_xy=int(covered_cells_xy(sub, gx, gy).sum()),
                covered_xyaw=int(covered_cells_xyaw(sub, gx2, gw).sum()),
                vx_mean=float(sub["vx"].mean()),
                vx_p05=float(np.percentile(sub["vx"], 5)),
                vx_p95=float(np.percentile(sub["vx"], 95)),
                vy_p05=float(np.percentile(sub["vy"], 5)),
                vy_p95=float(np.percentile(sub["vy"], 95)),
                wz_mean=float(sub["wz"].mean()),
                wz_p05=float(np.percentile(sub["wz"], 5)),
                wz_p95=float(np.percentile(sub["wz"], 95)),
                wz_raw_p05=float(np.percentile(sub["wz_raw"], 5)),
                wz_raw_p95=float(np.percentile(sub["wz_raw"], 95)),
                speed_min=float(sub["v_speed"].min()),
                speed_max=float(sub["v_speed"].max()),
            )
        return out

    res["per_clip"] = {s: per_clip_table(s) for s in SOURCES}

    # sensitivity of the coverage numbers to the tolerance choices and to raw vs smoothed velocities
    sens = {}
    for tolv, slw, slvy in [(0.05, 0.25, 0.10), (0.05, 0.125, 0.05), (0.05, 0.5, 0.2), (0.1, 0.25, 0.10)]:
        for suf in ["", "_raw"]:
            sens[f"tol={tolv},slice_w={slw},slice_vy={slvy},velocities={'smoothed' if suf == '' else 'raw'}"] = {
                s: dict(
                    xy=int(covered_cells_xy(D[s]["frames"], gx, gy, tol_v=tolv, slice_w=slw, suffix=suf).sum()),
                    xyaw=int(covered_cells_xyaw(D[s]["frames"], gx2, gw, tol_v=tolv, tol_w=tolv, slice_vy=slvy, suffix=suf).sum()),
                )
                for s in SOURCES
            }
    res["sensitivity"] = sens

    with open(os.path.join(DATA, "exp2_coverage.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez(os.path.join(DATA, "exp2_coverage_maps.npz"), **{f"{s}_cov_xy": D[s]["cov_xy"] for s in SOURCES}, **{f"{s}_cov_xyaw": D[s]["cov_xyaw"] for s in SOURCES})

    rows = []
    for src in SOURCES:
        r = res["datasets"][src]
        rows.append([src, r["n_clips"], r["n_frames"], r["duration_s"], f"{r['covered_xy']}/{gy.size * gx.size}", f"{r['covered_xyaw']}/{gw.size * gx2.size}", r["box_coverage"] * 100, r["stats"]["vx"]["p05"], r["stats"]["vx"]["p95"], r["stats"]["vy"]["p05"], r["stats"]["vy"]["p95"], r["stats"]["wz"]["p05"], r["stats"]["wz"]["p95"], r["fractions"]["standing"] * 100, r["fractions"]["turning"] * 100, r["fractions"]["outside_box"] * 100])
    table = md_table(["dataset", "clips", "frames", "dur [s]", "vx-vy cells", "vx-wz cells", "3D box cov [%]", "vx p5", "vx p95", "vy p5", "vy p95", "wz p5", "wz p95", "standing [%]", "turning [%]", "outside box [%]"], rows, "{:.2f}")
    with open(os.path.join(DATA, "exp2_summary_table.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(json.dumps(res["set_relations"], indent=1))
    for s in SOURCES:
        print(s, "raw-vs-smooth std: wz", round(res["datasets"][s]["wz_raw_minus_smooth_std"], 3), "vx", round(res["datasets"][s]["vx_raw_minus_smooth_std"], 3), "| covered raw:", res["datasets"][s]["covered_xy_raw"], res["datasets"][s]["covered_xyaw_raw"], "box raw", round(res["datasets"][s]["box_coverage_raw"], 3))
    print(json.dumps(res["sensitivity"], indent=1))
    for s in SOURCES:
        print("##", s)
        for k, v in res["per_clip"][s].items():
            print(f"  {k:30s} T={v['frames']:3d} w={v['weight']} xy={v['covered_xy']:2d} xyaw={v['covered_xyaw']:3d} vx[{v['vx_p05']:.2f},{v['vx_p95']:.2f}] mean {v['vx_mean']:.2f} | vy[{v['vy_p05']:.2f},{v['vy_p95']:.2f}] | wz[{v['wz_p05']:.2f},{v['wz_p95']:.2f}] (raw [{v['wz_raw_p05']:.2f},{v['wz_raw_p95']:.2f}]) | speed [{v['speed_min']:.2f},{v['speed_max']:.2f}]")


if __name__ == "__main__":
    main()
