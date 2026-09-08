"""Experiment 4b: 2-D projection of the state space the AMP discriminator consumes.

The discriminator input is the 24-D vector (q, qdot) of the 12 joints (get_amp_observations; AMPLoader amp_data default).
We standardise every dimension by the pooled expert mean/std (the AMP Normalizer does the same before the discriminator)
and project with PCA fitted on the pooled expert frames of the plotted datasets. Three panels: PCA of (q, qdot), of q only,
and of qdot only. Convex hull per dataset = the region the discriminator has ever seen as "expert"; star = Go2 default
standing pose with qdot = 0.

Policy side (no re-running here): if joint-position logs of trained policies are dropped into data/policy_rollouts/, they are
projected with the *same* standardisation and PCA and overlaid in grey/black. Accepted formats
  * <name>.th   torch.save of play.py's jpos_log, shape (T, num_envs, 12) in Isaac Lab joint order at step_dt = 0.02 s
                (eval_configurator.record_episode_jpos, files like x_0.4_y_0.0_yaw_0.0.th in the eval folder)
  * <name>.npy  (T, 12) or (T, N, 12) joint positions, Isaac Lab order, 50 Hz
  * <name>.npz  keys q (T, 12) [rad], qd (T, 12) [rad/s]
qdot is obtained by central differences at 50 Hz where not stored. The file stem is the legend label; a stem starting with
"mocap"/"video" is coloured like that dataset, anything else black.

Joint order: the expert files are in URDF/pybullet order and are reordered to Isaac Lab order with the loader's mapping
(source/constants.yaml JOINT_UNITREE_TO_ISAAC_LAB_MAPPING) so that policy logs can be overlaid one-to-one. The projection of the
expert data alone is invariant to this permutation.

Outputs: figures/fig_exp4_amp_state_projection.pdf, data/exp4_state_projection.json
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import ConvexHull

from common import COLORS, DATA, DOUBLE_W, ISAAC, NAME_MOCAP, NAME_VIDEO_EXT, savefig
from exp2_coverage import dataset_frames

PLOT_SOURCES = [NAME_MOCAP, NAME_VIDEO_EXT]
SHORT = {NAME_MOCAP: "MoCap", NAME_VIDEO_EXT: "Video (ext.)"}
MARKERS = {NAME_MOCAP: "o", NAME_VIDEO_EXT: "^"}
POLICY_DIR = os.path.join(DATA, "policy_rollouts")
STEP_DT = 0.02  # Isaac Lab env step (decimation 4 x sim dt 0.005)

with open(os.path.join(ISAAC, "source/constants.yaml")) as f:
    _c = yaml.safe_load(f)
TO_ISAAC = np.array(_c["JOINT_UNITREE_TO_ISAAC_LAB_MAPPING"])
Q_DEFAULT_ISAAC = np.array(_c["DEFAULT_JOINT_POS_ISAAC_LAB"], float)


def expert_amp_obs(clips):
    """(q, qdot) of every frame in Isaac Lab joint order (AMPLoader.reorder_from_pybullet_to_isaac_lab)."""
    q = np.vstack([c["frames"][:, 7:19] for c in clips])[:, TO_ISAAC]
    qd = np.vstack([c["frames"][:, 37:49] for c in clips])[:, TO_ISAAC]
    return q, qd


def load_policy_rollouts():
    out = {}
    for f in sorted(glob.glob(os.path.join(POLICY_DIR, "*"))):
        stem, ext = os.path.splitext(os.path.basename(f))
        try:
            if ext == ".th":
                import torch

                a = torch.load(f, map_location="cpu").numpy()
            elif ext == ".npy":
                a = np.load(f)
            elif ext == ".npz":
                z = np.load(f)
                out[stem] = (np.asarray(z["q"], float), np.asarray(z["qd"], float))
                continue
            else:
                continue
        except Exception as e:  # noqa: BLE001
            print(f"skip {f}: {e}")
            continue
        a = np.asarray(a, float)
        if a.ndim == 3:  # (T, N, 12): concatenate envs, differentiate per env
            q = a.transpose(1, 0, 2)  # (N, T, 12)
            keep = np.any(np.abs(q) > 0, axis=(1, 2))  # drop all-zero (unfilled) envs
            q = q[keep]
            qd = np.gradient(q, STEP_DT, axis=1)
            q, qd = q.reshape(-1, 12), qd.reshape(-1, 12)
        else:
            q = a.reshape(-1, 12)
            qd = np.gradient(q, STEP_DT, axis=0)
        out[stem] = (q, qd)
    return out


def pca(X):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S**2 / (len(X) - 1)
    return mu, Vt[:2], var[:2] / var.sum()


def hull(ax, P, color):
    if len(P) < 4:
        return
    h = ConvexHull(P)
    v = np.append(h.vertices, h.vertices[0])
    ax.plot(P[v, 0], P[v, 1], color=color, lw=0.7, alpha=0.9)


def main():
    D, clips = {}, {}
    for src in PLOT_SOURCES:
        D[src], clips[src] = dataset_frames(src)
    Q, QD, W = {}, {}, {}
    for src in PLOT_SOURCES:
        Q[src], QD[src] = expert_amp_obs(clips[src])
        W[src] = D[src]["w_amp"]
    X_all = np.vstack([np.hstack([Q[s], QD[s]]) for s in PLOT_SOURCES])
    mean, std = X_all.mean(0), X_all.std(0) + 1e-8  # pooled expert standardisation (AMP Normalizer analogue)

    def Z(q, qd):
        return (np.hstack([q, qd]) - mean) / std

    policies = load_policy_rollouts()
    if policies:
        print("policy rollouts found:", {k: v[0].shape for k, v in policies.items()})
    else:
        print(f"no policy rollouts in {POLICY_DIR} - expert data only (see module docstring for the accepted formats)")

    panels = [("(q, q̇), 24-D", slice(0, 24)), ("q only, 12-D", slice(0, 12)), ("q̇ only, 12-D", slice(12, 24))]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.15))
    summary = {"standardisation": {"mean": mean.tolist(), "std": std.tolist()}, "panels": {}}
    for ax, (title, sl) in zip(axes, panels):
        Zexp = {s: Z(Q[s], QD[s])[:, sl] for s in PLOT_SOURCES}
        mu, comps, evr = pca(np.vstack(list(Zexp.values())))
        proj = {s: (Zexp[s] - mu) @ comps.T for s in PLOT_SOURCES}
        for s in PLOT_SOURCES:
            ax.scatter(proj[s][:, 0], proj[s][:, 1], s=4, marker=MARKERS[s], color=COLORS[s], alpha=0.45, linewidths=0,
                       label=f"{SHORT[s]} expert ({len(proj[s])} frames)", rasterized=True)
            hull(ax, proj[s], COLORS[s])
        # standing pose
        z0 = ((Z(Q_DEFAULT_ISAAC[None], np.zeros((1, 12)))[:, sl] - mu) @ comps.T)[0]
        ax.scatter(*z0, marker="*", s=45, color="black", zorder=5, label="Go2 default pose, q̇ = 0")
        # policy overlay
        for k, (q, qd) in policies.items():
            zp = (Z(q, qd)[:, sl] - mu) @ comps.T
            col = COLORS[NAME_MOCAP] if k.lower().startswith("mocap") else COLORS[NAME_VIDEO_EXT] if k.lower().startswith("video") else "black"
            ax.scatter(zp[:, 0], zp[:, 1], s=1.5, color=col, alpha=0.25, linewidths=0, rasterized=True)
            hull(ax, zp, col)
            ax.plot([], [], color=col, lw=0.7, label=f"policy: {k}")
        ax.set_title(f"PCA of standardised {title}", fontsize=6.5, pad=3)
        ax.set_xlabel(f"PC 1 ({100 * evr[0]:.0f} % var.)")
        ax.set_ylabel(f"PC 2 ({100 * evr[1]:.0f} % var.)")
        ax.grid(alpha=0.25, lw=0.35)
        ax.tick_params(length=2, pad=1.5)
        ax.set_aspect("equal", adjustable="datalim")
        # overlap statistic: share of one set's frames inside the other's hull (2-D), and hull areas
        from matplotlib.path import Path

        stats = {}
        for a in PLOT_SOURCES:
            for b in PLOT_SOURCES:
                if a == b:
                    continue
                hb = ConvexHull(proj[b])
                inside = Path(proj[b][hb.vertices]).contains_points(proj[a])
                stats[f"{SHORT[a]} frames inside {SHORT[b]} hull"] = float(inside.mean())
        for s in PLOT_SOURCES:
            stats[f"{SHORT[s]} hull area"] = float(ConvexHull(proj[s]).volume)
        stats["explained_var"] = evr.tolist()
        stats["loadings_pc1"] = comps[0].tolist()
        summary["panels"][title] = stats
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.2, loc="lower center", ncol=len(l), handletextpad=0.3, columnspacing=1.5, markerscale=1.6,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.35, bottom=0.3, top=0.88)
    savefig(fig, "fig_exp4_amp_state_projection")
    plt.close(fig)

    with open(os.path.join(DATA, "exp4_state_projection.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for t, s in summary["panels"].items():
        print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("loadings_pc1",)})


if __name__ == "__main__":
    main()
