"""Experiment 4: AMP-specific evidence that insufficient expert coverage limits the trained policies.

Fig. 5 already shows *that* the expert sets cover the command box poorly and *that* the policies track
poorly in some cells. This experiment adds the AMP mechanism in between, using only quantities that
follow from the AMP implementation in the IsaacLab repo (read-only):

  (a) The discriminator sees joint positions + joint velocities only (manager_based_rl_env.py
      get_amp_observations, AMPLoader amp_data default ["JOINT_POS", "JOINT_VEL"]). The commanded base
      velocity never enters the discriminator; it only sees the gait that produces it. Hence a command
      without expert support forces the policy to produce (q, qdot) transitions that are out of the
      expert distribution, the style reward r_style = max(0, 1 - 0.25 (D - 1)^2) drops, and the
      task reward (60 exp(-e^2/0.22^2) + 20 exp(...)) and the style reward pull in different directions
      (amp_task_reward_lerp = 0.5, amp_reward_coef = 2.0 in rsl_rl_ppo_cfg.py).

  (b) The paper's own evaluation logs this mechanism per target cell: play.py accumulates the Euclidean
      nearest-neighbour distance between the agent's AMP observation (q, qdot; 24-D) and the expert set
      (interpolated 10x) at every step and divides by the episode length ("agent_expert_distances",
      plotted as "Imitation score" in plot_errorOnTargetDistribution.py). The values here are the ones
      recovered from the shipped Fig. 5 PDF (data/fig5_recovered, colour-bar range 2..4, so they are
      clipped at both ends -- the clipping fractions are reported).

  Analyses
    1. Cell-wise link on the 7 x 21 vx-vy evaluation grid (yaw rate 0), per dataset and pooled:
       command-space distance to the nearest expert frame (Exp. 2)  ->  agent-expert distance (AMP space)
       ->  tracking error (vel, yaw). Spearman rank correlations + covered vs. uncovered cell means.
    2. Scale calibration of the agent-expert distance in the expert data themselves: leave-one-clip-out
       nearest-neighbour distance within a set, cross-set distances (MoCap <-> Video), the distance of the
       Go2 default standing pose (qdot = 0) to each set, and the share of the distance carried by the joint
       velocities (the metric is dominated by qdot).
    3. What the discriminator sees vs. the command: per-clip base speed vs. RMS joint velocity, AMP
       sampling mass per clip (MotionWeight / clip length per frame), and the AMP-weighted joint-speed
       distribution per dataset.

Outputs: data/exp4_amp_evidence.json, data/exp4_tables.md,
         figures/fig_exp4_coverage_vs_performance.pdf, figures/fig_exp4_agent_expert_distance_grid.pdf,
         figures/fig_exp4_joint_speed_vs_base_speed.pdf
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from common import COL_W, DOUBLE_W, COLORS, DATA, EVAL_GRID, FIG, NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_EXT, load_amp_dir, md_table, savefig, AMP_DIRS, CLIP_SHORT
from exp2_coverage import SLICE_W, SOURCES, dataset_frames, nearest_distance_map

FIG5_DIR = os.path.join(DATA, "fig5_recovered")
DATASET_KEYS = {
    NAME_MOCAP: "mocap_AMP_for_hardware",
    NAME_VIDEO: "fromVision_motions_DepthCam",
    NAME_VIDEO_EXT: "fromVision_motions_DepthCam_extendedWithoutReverse",
}
CBAR_LIMITS = {"error_vel_xy": (0.0, 0.1), "error_vel_yaw": (0.0, 0.4), "mean_mechanical_cot": (0.8, 2.0), "agent_expert_distances": (2.0, 4.0)}
MARKERS = {NAME_MOCAP: "o", NAME_VIDEO: "s", NAME_VIDEO_EXT: "^"}
SHORT = {NAME_MOCAP: "MoCap", NAME_VIDEO: "Video", NAME_VIDEO_EXT: "Video (ext.)"}
PLOT_SOURCES = [NAME_MOCAP, NAME_VIDEO_EXT]  # figures show the two paper-relevant sets; the analysis keeps all three

# AMP observation = joint pos (cols 7:19) + joint vel (cols 37:49) of the expert file (AMPLoader index constants)
AMP_COLS = np.r_[7:19, 37:49]
NUM_INTERP = 10  # play.py: interpolate_trajectory(trajectory, num_interpolations=10)
# Go2 default joint pose (IsaacLab unitree.py UNITREE_GO2_CFG init_state): hips +-0.1, thighs 0.8, calves -1.5.
# The expert files are in URDF leg order FL, FR, RL, RR (notes.md 9b); hip sign only matters at the 0.1 rad level.
Q_DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])  # rear thighs 1.0 (constants.yaml DEFAULT_JOINT_POS_ISAAC_LAB)


# ------------------------------------------------------------------------------------------------
def fig5_grid(src, metric):
    """Recovered Fig. 5 cell values as (7, 21) array with v_y ascending along axis 0, v_x ascending along axis 1.

    Follows the orientation of fig5_replacement_combined.py (author's request): the picture of the original
    Fig. 5 is kept and the axis reads v_y increasing upwards, i.e. the CSV row labelled -0.3 (top row as
    seaborn drew it) is the v_y = +0.3 row. The alternative orientation is evaluated as a sensitivity check.
    """
    df = pd.read_csv(os.path.join(FIG5_DIR, f"{DATASET_KEYS[src]}__{metric}.csv"), index_col=0)
    return df.values[::-1].astype(float)


def clip_fraction(a, metric, eps=0.01):
    lo, hi = CBAR_LIMITS[metric]
    return float(np.mean(a <= lo + eps)), float(np.mean(a >= hi - eps))


def interpolate(traj, k=NUM_INTERP):
    """Linear interpolation with k extra points between consecutive frames (as play.py interpolate_trajectory)."""
    T = traj.shape[0]
    t_new = np.linspace(0, T - 1, (T - 1) * (k + 1) + 1)
    return np.stack([np.interp(t_new, np.arange(T), traj[:, j]) for j in range(traj.shape[1])], 1)


def nn_dist(Q, R, split=None):
    """Per-row Euclidean distance from Q to its nearest row in R; optionally also the squared share of dims >= split."""
    out = np.empty(len(Q))
    share = np.empty(len(Q))
    for k in range(0, len(Q), 512):
        d2 = ((Q[k : k + 512, None, :] - R[None]) ** 2)
        tot = d2.sum(-1)
        j = tot.argmin(1)
        out[k : k + 512] = np.sqrt(tot[np.arange(len(j)), j])
        if split is not None:
            share[k : k + 512] = d2[np.arange(len(j)), j, split:].sum(-1) / tot[np.arange(len(j)), j]
    return (out, share) if split is not None else out


def amp_obs(clip):
    return clip["frames"][:, AMP_COLS]


PAIRS = [("cmd", "aed", "cmd_vs_aed"), ("cmd", "evel", "cmd_vs_evel"), ("cmd", "eyaw", "cmd_vs_eyaw"), ("cmd", "comb", "cmd_vs_comb"),
         ("cmd", "cot", "cmd_vs_cot"), ("aed", "evel", "aed_vs_evel"), ("aed", "eyaw", "aed_vs_eyaw"), ("aed", "comb", "aed_vs_comb"),
         ("aed", "cot", "aed_vs_cot")]


def nearest_distance_map_ms(d, grid_x, grid_y, slice_w=SLICE_W):
    """Unnormalised planar distance [m/s] from every vx-vy grid point to the nearest expert frame with |yaw rate| <= slice_w."""
    sel = np.abs(d["wz"]) <= slice_w
    P = np.stack([d["vx"][sel], d["vy"][sel]], 1)
    out = np.zeros((len(grid_y), len(grid_x)))
    for j, b in enumerate(grid_y):
        for i, a in enumerate(grid_x):
            out[j, i] = np.min(np.linalg.norm(P - np.array([a, b]), axis=1))
    return out


def combined_error(evel, eyaw):
    """Equal-weight combination of the two Fig. 5 tracking errors, each normalised by its colour-bar maximum (0.1 m/s, 0.4 rad) -> 0..1."""
    return 0.5 * (evel / CBAR_LIMITS["error_vel_xy"][1] + eyaw / CBAR_LIMITS["error_vel_yaw"][1])


# ------------------------------------------------------------------------------------------------
def analysis_cells(D, cov):
    gx, gy = EVAL_GRID["xy"]
    res = {"per_dataset": {}, "pooled": {}, "orientation_check": {}}
    rows = []
    pooled = {k: [] for k in ["cmd", "cmd_ms", "aed", "evel", "eyaw", "cot", "comb", "covered", "src"]}
    for src in SOURCES:
        cmd = nearest_distance_map(D[src], gx, gy, "xy")  # (7, 21), normalised command-space distance
        cmd_ms = nearest_distance_map_ms(D[src], gx, gy)  # (7, 21), planar distance in m/s
        aed = fig5_grid(src, "agent_expert_distances")
        evel = fig5_grid(src, "error_vel_xy")
        eyaw = fig5_grid(src, "error_vel_yaw")
        cot = fig5_grid(src, "mean_mechanical_cot")
        comb = combined_error(evel, eyaw)
        c = cov[f"{src}_cov_xy"]
        r = {}
        for a, b, name in PAIRS:
            a, b = {"cmd": cmd, "aed": aed, "evel": evel, "eyaw": eyaw, "cot": cot, "comb": comb}[a], \
                   {"cmd": cmd, "aed": aed, "evel": evel, "eyaw": eyaw, "cot": cot, "comb": comb}[b]
            rho, p = spearmanr(a.ravel(), b.ravel())
            r[name] = {"rho": float(rho), "p": float(p)}
        for name, g in [("aed", aed), ("evel", evel), ("eyaw", eyaw), ("comb", comb)]:
            u = mannwhitneyu(g[~c], g[c], alternative="greater")
            r[f"{name}_covered_mean"] = float(g[c].mean())
            r[f"{name}_uncovered_mean"] = float(g[~c].mean())
            r[f"{name}_uncovered_gt_covered_p"] = float(u.pvalue)
        r["n_covered"] = int(c.sum())
        r["aed_clip_lo_hi"] = clip_fraction(aed, "agent_expert_distances")
        r["evel_clip_lo_hi"] = clip_fraction(evel, "error_vel_xy")
        r["eyaw_clip_lo_hi"] = clip_fraction(eyaw, "error_vel_yaw")
        # the cell with the best imitation score vs the cell with the best tracking
        j = np.unravel_index(aed.argmin(), aed.shape)
        r["aed_min_cell"] = {"vx": float(gx[j[1]]), "vy": float(gy[j[0]]), "aed": float(aed[j])}
        r["aed_row_vy0"] = aed[np.argmin(np.abs(gy))].tolist()
        res["per_dataset"][src] = r
        rows.append([SHORT[src], r["cmd_vs_aed"]["rho"], r["cmd_vs_evel"]["rho"], r["cmd_vs_eyaw"]["rho"],
                     r["aed_vs_evel"]["rho"], r["aed_vs_eyaw"]["rho"],
                     r["aed_covered_mean"], r["aed_uncovered_mean"], r["evel_covered_mean"], r["evel_uncovered_mean"],
                     r["eyaw_covered_mean"], r["eyaw_uncovered_mean"]])
        for k, g in [("cmd", cmd), ("cmd_ms", cmd_ms), ("aed", aed), ("evel", evel), ("eyaw", eyaw), ("cot", cot), ("comb", comb)]:
            pooled[k].append(g.ravel())
        pooled["covered"].append(c.ravel())
        pooled["src"].append(np.array([src] * cmd.size))
        # orientation sensitivity: correlate with the un-flipped CSV orientation
        alt = {}
        for m in ["agent_expert_distances", "error_vel_xy", "error_vel_yaw"]:
            g_alt = fig5_grid(src, m)[::-1]
            alt[m] = float(spearmanr(cmd.ravel(), g_alt.ravel())[0])
        res["orientation_check"][src] = alt
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    for a, b, name in PAIRS:
        rho, p = spearmanr(P[a], P[b])
        res["pooled"][name] = {"rho": float(rho), "p": float(p)}
    c = P["covered"]
    for name in ["aed", "evel", "eyaw"]:
        res["pooled"][f"{name}_covered_mean"] = float(P[name][c].mean())
        res["pooled"][f"{name}_uncovered_mean"] = float(P[name][~c].mean())
        res["pooled"][f"{name}_uncovered_gt_covered_p"] = float(mannwhitneyu(P[name][~c], P[name][c], alternative="greater").pvalue)
    rows.append(["pooled (441 cells)", res["pooled"]["cmd_vs_aed"]["rho"], res["pooled"]["cmd_vs_evel"]["rho"],
                 res["pooled"]["cmd_vs_eyaw"]["rho"], res["pooled"]["aed_vs_evel"]["rho"], res["pooled"]["aed_vs_eyaw"]["rho"],
                 res["pooled"]["aed_covered_mean"], res["pooled"]["aed_uncovered_mean"],
                 res["pooled"]["evel_covered_mean"], res["pooled"]["evel_uncovered_mean"],
                 res["pooled"]["eyaw_covered_mean"], res["pooled"]["eyaw_uncovered_mean"]])
    table = md_table(
        ["dataset", "rho(cmd dist, agent-expert dist)", "rho(cmd dist, err vel)", "rho(cmd dist, err yaw)",
         "rho(agent-expert dist, err vel)", "rho(agent-expert dist, err yaw)",
         "agent-expert dist covered", "uncovered", "err vel covered", "uncovered", "err yaw covered", "uncovered"],
        rows, floatfmt="{:.2f}")
    return res, table, P


def analysis_calibration(clips_by_src):
    res = {"leave_one_clip_out": {}, "cross_set": {}, "default_pose": {}, "qdot_share": {}, "within_set_mean": {}}
    rows = []
    for src in SOURCES:
        clips = clips_by_src[src]
        obs = {c["name"]: amp_obs(c) for c in clips}
        interp = {n: interpolate(o) for n, o in obs.items()}
        per_clip = {}
        shares = []
        for n in obs:
            others = np.vstack([interp[m] for m in obs if m != n])
            d, sh = nn_dist(obs[n], others, split=12)
            per_clip[n] = float(d.mean())
            shares.append(sh)
            rows.append([SHORT[src], CLIP_SHORT.get(n, n), len(obs[n]), float(d.mean()), float(np.median(d)), float(d.max())])
        res["leave_one_clip_out"][src] = per_clip
        res["within_set_mean"][src] = float(np.mean(list(per_clip.values())))
        res["qdot_share"][src] = float(np.concatenate(shares).mean())
        # standing pose
        allinterp = np.vstack(list(interp.values()))
        q0 = np.concatenate([Q_DEFAULT, np.zeros(12)])[None]
        q0_alt = np.concatenate([-Q_DEFAULT * np.r_[[1, 0, 0] * 4] + Q_DEFAULT * np.r_[[0, 1, 1] * 4], np.zeros(12)])[None]
        res["default_pose"][src] = {"dist": float(nn_dist(q0, allinterp)[0]), "dist_hip_sign_flipped": float(nn_dist(q0_alt, allinterp)[0]),
                                    "nearest_qdot_norm": float(np.linalg.norm(allinterp[np.argmin(np.linalg.norm(allinterp - q0, axis=1)), 12:]))}
        # slowest expert frame: min |qdot| in the set
        qd = np.linalg.norm(np.vstack(list(obs.values()))[:, 12:], axis=1)
        res["default_pose"][src]["min_expert_qdot_norm"] = float(qd.min())
        res["default_pose"][src]["p5_expert_qdot_norm"] = float(np.percentile(qd, 5))
    # cross-set distances (frames of A to interpolated set B)
    for a, b in [(NAME_MOCAP, NAME_VIDEO_EXT), (NAME_VIDEO_EXT, NAME_MOCAP), (NAME_VIDEO, NAME_MOCAP), (NAME_MOCAP, NAME_VIDEO)]:
        A = np.vstack([amp_obs(c) for c in clips_by_src[a]])
        B = np.vstack([interpolate(amp_obs(c)) for c in clips_by_src[b]])
        d = nn_dist(A, B)
        res["cross_set"][f"{a} -> {b}"] = {"mean": float(d.mean()), "median": float(np.median(d))}
    table = md_table(["dataset", "clip", "frames", "LOO NN dist mean", "median", "max"], rows, floatfmt="{:.2f}")
    return res, table


def analysis_discriminator_view(D, clips_by_src):
    res = {"per_clip": {}, "pooled_rho_speed_qdot": {}, "qdot_weighted_percentiles": {}}
    rows = []
    for src in SOURCES:
        d = D[src]
        qd = np.concatenate([np.linalg.norm(c["frames"][:, 37:49], axis=1) for c in clips_by_src[src]])
        assert len(qd) == len(d["vx"])
        w = d["w_amp"]
        rho = spearmanr(d["v_speed"], qd)[0]
        res["pooled_rho_speed_qdot"][src] = float(rho)
        order = np.argsort(qd)
        cw = np.cumsum(w[order]) / w.sum()
        res["qdot_weighted_percentiles"][src] = {f"p{p}": float(qd[order][np.searchsorted(cw, p / 100)]) for p in [5, 25, 50, 75, 95]}
        per_clip = {}
        for c in clips_by_src[src]:
            m = d["clip"] == c["name"]
            e = {
                "frames": int(m.sum()),
                "duration_s": float((m.sum() - 1) * c["fd"]),
                "motion_weight": c["weight"],
                "amp_mass": float(w[m].sum()),
                "per_frame_density_rel": float(w[m][0] / w.min()),
                "speed_mean": float(d["v_speed"][m].mean()),
                "vx_mean": float(d["vx"][m].mean()),
                "wz_abs_mean": float(np.abs(d["wz"][m]).mean()),
                "qdot_rms": float(np.sqrt((qd[m] ** 2).mean())),
            }
            per_clip[c["name"]] = e
            rows.append([SHORT[src], CLIP_SHORT.get(c["name"], c["name"]), e["frames"], e["motion_weight"], 100 * e["amp_mass"],
                         e["per_frame_density_rel"], e["speed_mean"], e["wz_abs_mean"], e["qdot_rms"]])
        res["per_clip"][src] = per_clip
    table = md_table(["dataset", "clip", "frames", "MotionWeight", "AMP mass [%]", "per-frame density (rel. to min)",
                      "mean speed [m/s]", "mean |yaw rate| [rad/s]", "RMS |qdot| [rad/s]"], rows, floatfmt="{:.2f}")
    return res, table


# ------------------------------------------------------------------------------------------------
AX_LABEL = {
    "cmd": "target command → nearest expert\nframe, distance [norm.]",
    "aed": "agent–expert distance [1]\n(AMP obs. space, q + q̇)",
    "evel": "tracking error vel. [m/s]",
    "eyaw": "tracking error yaw [rad]",
    "comb": "combined tracking error\n(vel. + yaw, normalised) [1]",
    "cot": "cost of transport [1]",
}
AX_LIM = {"cmd": (-0.05, 1.55), "aed": (1.85, 4.15), "evel": (0.0, 0.105), "eyaw": (0.0, 0.42), "comb": (0.0, 1.05), "cot": (0.75, 2.05)}
Y_HEADROOM = 0.30  # extra room above the data for the two-line legend
CLIP_LINES = {"aed": (2.0, 4.0), "evel": (0.1,), "eyaw": (0.4,), "cot": (0.8, 2.0), "comb": (), "cmd": ()}
PANELS = [  # (x, y, key) ; every y is "higher = worse"
    ("cmd", "aed", "cmd_vs_aed"), ("aed", "evel", "aed_vs_evel"), ("aed", "eyaw", "aed_vs_eyaw"),
    ("cmd", "comb", "cmd_vs_comb"), ("aed", "comb", "aed_vs_comb"), ("aed", "cot", "aed_vs_cot"),
]


def fig_coverage_vs_performance(P, res, fname):
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_W, 3.1))
    sel = np.isin(P["src"], PLOT_SOURCES)
    for ax, (kx, ky, key) in zip(axes.ravel(), PANELS):
        for src in PLOT_SOURCES:
            m = P["src"] == src
            ax.scatter(P[kx][m], P[ky][m], s=6, marker=MARKERS[src], facecolors="none", edgecolors=COLORS[src],
                       linewidths=0.5, alpha=0.85, label=f"{SHORT[src]}  ρ = {res['per_dataset'][src][key]['rho']:.2f}")
        rho2 = spearmanr(P[kx][sel], P[ky][sel])[0]
        ax.set_title(f"both sets, {int(sel.sum())} cells: ρ = {rho2:.2f}", fontsize=5.5, pad=2)
        ax.set_xlabel(AX_LABEL[kx])
        ax.set_ylabel(AX_LABEL[ky])
        ax.set_xlim(*AX_LIM[kx])
        lo, hi = AX_LIM[ky]
        ax.set_ylim(lo, hi + Y_HEADROOM * (hi - lo))
        for v in CLIP_LINES[kx]:
            ax.axvline(v, color="0.6", lw=0.4, ls=":")
        for v in CLIP_LINES[ky]:
            ax.axhline(v, color="0.6", lw=0.4, ls=":")
        ax.grid(alpha=0.25, lw=0.35)
        ax.tick_params(length=2, pad=1.5)
        ax.legend(frameon=False, handletextpad=0.1, borderaxespad=0.15, fontsize=4.8, labelspacing=0.2, loc="upper left", ncol=1,
                  columnspacing=0.8)
    fig.subplots_adjust(wspace=0.42, hspace=0.62)
    savefig(fig, fname)
    plt.close(fig)


def _key_panel(P, kx, ky, kc, xlabel, ylabel, clabel, clim, cticks, ylim, xlim, fname, clip_y=(), legend_loc="lower right", cmap="viridis"):
    """Coverage on x, one policy metric on y, another as marker colour; straight least-squares line per dataset,
    Pearson correlation coefficient r of the plotted points in the legend."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.84))  # 20 % lower than the first version (2.3 in)
    sc = None
    out = {}
    for src in PLOT_SOURCES:
        m = P["src"] == src
        x, y = P[kx][m], P[ky][m]
        sc = ax.scatter(x, y, c=P[kc][m], cmap=cmap, vmin=clim[0], vmax=clim[1], s=11, marker=MARKERS[src],
                        edgecolors=COLORS[src], linewidths=0.45, alpha=0.9, zorder=2)
        a, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        yy = a * xx + b
        keep = (yy >= ylim[0]) & (yy <= ylim[1])  # do not draw the fit outside the axes
        r = float(np.corrcoef(x, y)[0, 1])
        rho = float(spearmanr(x, y)[0])
        out[src] = {"spearman_rho": rho, "pearson_r": r, "slope": float(a), "intercept": float(b)}
        ax.plot(xx[keep], yy[keep], color=COLORS[src], lw=1.6, zorder=3,
                label=f"{SHORT[src].replace('ext.', 'extended')} (correlation ρ = {rho:.2f})")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05, ticks=cticks)
    cb.set_label(clabel, fontsize=plt.rcParams["axes.labelsize"])  # same size as the axis labels
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], length=2)
    cb.outline.set_linewidth(0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for yv in clip_y:
        ax.axhline(yv, color="0.6", lw=0.4, ls=":")
    ax.grid(alpha=0.25, lw=0.35)
    ax.tick_params(length=2, pad=1.5)
    ax.legend(frameon=False, fontsize=5.5, loc=legend_loc, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.3)
    # tight crop with 2 px margin (instead of the 0.1 in default of common.savefig)
    pad = 2 / 72
    fig.savefig(os.path.join(FIG, fname + ".pdf"), bbox_inches="tight", pad_inches=pad)
    fig.savefig(os.path.join(FIG, fname + ".png"), bbox_inches="tight", pad_inches=pad, dpi=200)
    print("saved", os.path.join(FIG, fname + ".pdf"))
    plt.close(fig)
    return out


CB_AED = r"$\bf{Agent\text{-}Expert\ Distance}$" + "\nin AMP obs. space (q, q̇)\n[1], ↑ worse"
# flipped sign of the agent-expert distance so that "higher = better" (author's request); same information, easier to read
CB_IMIT = r"$\bf{Expert\ Imitation}$" + "\nin AMP observation\nspace (q, q̇), [1], ↑ better"
CB_COMB = r"$\bf{Tracking\ Error}$" + "\nYaw and Vel. Normalized\n[1], ↑ worse"
XLAB_NORM = "Distance of Target Command to Closest Expert Frame [1]"
XLAB_MS = "Distance of Target Command to Closest Expert Frame [m/s]"


def fig_key_results(P, res):
    """Four single-panel figures: {tracking error, agent-expert distance} on y x {normalised, m/s} on x."""
    fits = {}
    P["imit"] = -P["aed"]  # expert-imitation score = negative agent-expert distance (higher = closer to the expert data)
    for kx, xlab, xlim, suffix in [("cmd", XLAB_NORM, (-0.03, 1.55), ""), ("cmd_ms", XLAB_MS, (-0.02, 1.25), "_ms")]:
        fits[f"key_result{suffix}"] = _key_panel(P, kx, "comb", "imit", xlab, r"$\bf{Tracking\ Error}$" + "\nYaw and Vel. Normalized [1]", CB_IMIT,
                                                 (-4.0, -2.0), [-4, -3, -2], (0, 1.12), xlim, f"fig_exp4_key_result{suffix}", cmap="viridis_r")
        fits[f"key_result_aed{suffix}"] = _key_panel(P, kx, "aed", "comb", xlab, r"$\bf{Agent\text{-}Expert\ Distance}$" + " [1]", CB_COMB,
                                                     (0.0, 1.0), [0, 0.5, 1], (1.85, 4.3), xlim, f"fig_exp4_key_result_aed{suffix}",
                                                     clip_y=(2.0, 4.0), legend_loc="upper left")
    res["key_result_fits"] = fits
    return fits


def fig_aed_grid(cov, fname):
    gx, gy = EVAL_GRID["xy"]
    ex = np.concatenate([gx - 0.05, [gx[-1] + 0.05]])
    ey = np.concatenate([gy - 0.05, [gy[-1] + 0.05]])
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 2.3), sharex=True)
    mesh = None
    for ax, src in zip(axes, SOURCES):
        aed = fig5_grid(src, "agent_expert_distances")
        mesh = ax.pcolormesh(ex, ey, aed, cmap="viridis", vmin=2.0, vmax=4.0, edgecolors="white", linewidth=0.17)
        c = cov[f"{src}_cov_xy"]
        yy, xx = np.nonzero(c)
        ax.scatter(gx[xx], gy[yy], s=3, c="white", edgecolors="black", linewidths=0.3, zorder=3)
        ax.set_ylabel(SHORT[src], fontsize=6, color=COLORS[src], fontweight="bold")
        ax.set_yticks([-0.3, 0, 0.3])
        ax.set_yticklabels(["-0.3", "0", "0.3"])
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.35, 0.35)
        ax.tick_params(length=1.8, pad=1.5, labelsize=5.2)
        ax.text(1.01, 0.5, f"{aed.mean():.2f}", transform=ax.transAxes, fontsize=5, va="center", ha="left")
    axes[-1].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[-1].set_xlabel("Target Vel. X [m/s]", fontsize=6.2)
    fig.text(0.0, 0.5, "Target Vel. Y [m/s]", rotation=90, va="center", ha="left", fontsize=6.2)
    cax = fig.add_axes([0.25, 0.97, 0.5, 0.03])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks([2, 3, 4])
    cax.xaxis.set_ticks_position("top")
    cax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cax.set_title("Agent–expert distance in AMP observation space (joint pos. + vel.)  ↓   [white dot = target covered by expert data]",
                  fontsize=5.2, pad=9)
    fig.subplots_adjust(left=0.13, right=0.94, top=0.88, bottom=0.14, hspace=0.18)
    savefig(fig, fname)
    plt.close(fig)


def fig_joint_speed(D, clips_by_src, res_view, fname):
    """Per-frame cloud (faint) + per-clip mean (marker, size = AMP sampling mass within the set). MoCap and Video (extended) only."""
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    ax.axvspan(0, 1.0, color="0.92", zorder=0)
    ax.text(0.5, 0.985, "command range |v| ≤ 1 m/s", transform=ax.get_xaxis_transform(), fontsize=5, ha="center", va="top", color="0.35")
    for src in PLOT_SOURCES:
        d = D[src]
        qd = np.concatenate([np.linalg.norm(c["frames"][:, 37:49], axis=1) for c in clips_by_src[src]])
        ax.scatter(d["v_speed"], qd, s=1.2, color=COLORS[src], alpha=0.2, linewidths=0, rasterized=True)
        for n, e in res_view["per_clip"][src].items():
            ax.scatter(e["speed_mean"], e["qdot_rms"], s=14 + 350 * e["amp_mass"], marker=MARKERS[src], color=COLORS[src],
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.annotate(CLIP_SHORT.get(n, n), (e["speed_mean"], e["qdot_rms"]), fontsize=4.3, xytext=(3, 2), textcoords="offset points")
    ax.scatter([0], [0], marker="*", s=30, color="black", zorder=4)
    ax.annotate("Go2 default pose (standing)", (0, 0), fontsize=4.3, xytext=(4, 3), textcoords="offset points")
    for src in PLOT_SOURCES:
        ax.scatter([], [], marker=MARKERS[src], color=COLORS[src], edgecolors="black", linewidths=0.4, s=14, label=SHORT[src])
    ax.legend(loc="lower right", frameon=False, fontsize=5, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3)
    ax.set_xlabel("expert base speed |v| [m/s]")
    ax.set_ylabel("joint speed ‖q̇‖ [rad/s]\n(discriminator input)")
    ax.set_xlim(-0.05, 2.4)
    ax.set_ylim(-0.5, 22)
    ax.grid(alpha=0.25, lw=0.35)
    ax.tick_params(length=2, pad=1.5)
    savefig(fig, fname)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def main():
    D, clips_by_src = {}, {}
    for src in SOURCES:
        D[src], clips_by_src[src] = dataset_frames(src)
    cov = dict(np.load(os.path.join(DATA, "exp2_coverage_maps.npz")))

    res_cells, t_cells, P = analysis_cells(D, cov)
    res_cal, t_cal = analysis_calibration(clips_by_src)
    res_view, t_view = analysis_discriminator_view(D, clips_by_src)

    fits = fig_key_results(P, res_cells)
    out = {"cells": res_cells, "calibration": res_cal, "discriminator_view": res_view,
           "notes": {
               "amp_observation": "joint_pos (12) + joint_vel (12); manager_based_rl_env.get_amp_observations, AMPLoader amp_data default",
               "agent_expert_distances": "play.py: per-step Euclidean NN distance of the agent AMP observation to the 10x interpolated expert set, averaged over the episode; recovered from the Fig. 5 PDF with colour-bar range 2..4",
               "reward": "r = 0.5 * task + 0.5 * style (amp_task_reward_lerp=0.5); style = 2 * max(0, 1 - 0.25 (D-1)^2) (amp_reward_coef=2); task = 60 exp(-|v_cmd - v|^2 / 0.22^2) + 20 exp(-(w_cmd - w)^2 / 0.22^2) (parameters.set_velocity_rewards_amp)",
               "orientation": "Fig. 5 grids read with v_y ascending as in fig5_replacement_combined.py; orientation_check gives rho for the un-flipped CSV",
           }}
    with open(os.path.join(DATA, "exp4_amp_evidence.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(DATA, "exp4_tables.md"), "w") as f:
        f.write("## Cell-wise link (7 x 21 vx-vy grid at yaw rate 0), Spearman rho and covered/uncovered means\n\n" + t_cells + "\n\n")
        f.write("## Expert-set calibration of the agent-expert distance (leave-one-clip-out NN distance, 24-D AMP space)\n\n" + t_cal + "\n\n")
        f.write("## What the discriminator sees per clip, and how AMP samples it\n\n" + t_view + "\n")

    fig_coverage_vs_performance(P, res_cells, "fig_exp4_coverage_vs_performance")
    print("key-result fits (Pearson r):", json.dumps(fits, indent=1))
    fig_aed_grid(cov, "fig_exp4_agent_expert_distance_grid")
    fig_joint_speed(D, clips_by_src, res_view, "fig_exp4_joint_speed_vs_base_speed")

    # console summary
    print(t_cells)
    print()
    for src in SOURCES:
        r = res_cells["per_dataset"][src]
        print(f"{src}: covered cells {r['n_covered']}/147; agent-expert dist clipped lo/hi {r['aed_clip_lo_hi']}, "
              f"err vel clipped lo/hi {r['evel_clip_lo_hi']}, err yaw clipped lo/hi {r['eyaw_clip_lo_hi']}; "
              f"best imitation cell vx={r['aed_min_cell']['vx']:.1f} vy={r['aed_min_cell']['vy']:.1f} ({r['aed_min_cell']['aed']:.2f})")
        print(f"   orientation check (un-flipped CSV) rho(cmd, .): {res_cells['orientation_check'][src]}")
    print()
    print(t_cal)
    print("within-set LOO mean:", res_cal["within_set_mean"])
    print("cross-set:", json.dumps(res_cal["cross_set"], indent=1))
    print("default standing pose:", json.dumps(res_cal["default_pose"], indent=1))
    print("share of squared NN distance from joint velocities:", res_cal["qdot_share"])
    print()
    print(t_view)
    print("rho(speed, |qdot|):", res_view["pooled_rho_speed_qdot"])
    print("AMP-weighted |qdot| percentiles:", json.dumps(res_view["qdot_weighted_percentiles"], indent=1))


if __name__ == "__main__":
    main()
