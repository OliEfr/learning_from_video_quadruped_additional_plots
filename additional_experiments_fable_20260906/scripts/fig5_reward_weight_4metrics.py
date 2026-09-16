"""Paper Fig. 5 with four metrics: yaw-tracking reward weight sweep for MoCap and Video (extended).

Data: the paper's own evaluation files, extracted from logs.zip into data/paper_logs/ (DefaultEvalConfig, 10 000 envs x 2 episodes).
* weight 20 (paper default): the three Fig.-4 seeds, <run>/metrics.yaml
* weights 30/40/50: two seeds each, <run>_trackAnVelRewWeight_<w>_SEED_<n>/None_metrics.yaml
  (the metrics.yaml inside those folders is a stale copy of another run's evaluation and must not be used).

Outputs figures/fig5_reward_weight_4metrics.{pdf,png} (2 x 2 panels, single column) and data/reward_weight_sweep.{md,json}.
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml

from common import COL_W, COLORS, DATA, NAME_MOCAP, NAME_VIDEO_EXT, md_table, savefig

LOGS = os.path.join(DATA, "paper_logs", "rsl_rl", "unitree_go2_AMPflat")
METRICS = ["error_vel_yaw", "error_vel_xy", "mean_mechanical_cot", "agent_expert_distances"]
TITLES = {
    "error_vel_yaw": "Tracking Error Yaw\n[rad]",
    "error_vel_xy": "Tracking Error Vel.\n[m/s]",
    "mean_mechanical_cot": "Cost of Transport\n[1]",
    "agent_expert_distances": "Agent-Expert Distance\n[1] (lower = better)",
}
LINESTYLE = {"error_vel_xy": ("-", "o", "Tracking Error Vel."), "agent_expert_distances": ("--", "s", "Agent-Expert Distance")}
SETS = {
    NAME_MOCAP: ("2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_*", "2025-07-13_13-22-15_mocap_AMP_for_hardware_trackAnVelRewWeight_*"),
    NAME_VIDEO_EXT: (
        "2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_*",
        "2025-07-11_21-14-15_fromVision_motions_DepthCam_extendedWithoutReverse_trackAnVelRewWeight_*",
    ),
}
LEGEND = {NAME_MOCAP: "MoCap (AMP)", NAME_VIDEO_EXT: "Video w. Depth Camera (extended) (AMP)"}
DEFAULT_W = 20


def load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def collect():
    out = {}
    for name, (base, sweep) in SETS.items():
        byw = {DEFAULT_W: [load(os.path.join(d, "metrics.yaml")) for d in sorted(glob.glob(os.path.join(LOGS, base)))]}
        for d in sorted(glob.glob(os.path.join(LOGS, sweep))):
            w = int(re.search(r"trackAnVelRewWeight_(\d+)", d).group(1))
            p = os.path.join(d, "None_metrics.yaml")
            if os.path.exists(p):
                byw.setdefault(w, []).append(load(p))
        out[name] = {w: {m: [r[m] for r in runs] for m in METRICS} | {"n": len(runs)} for w, runs in sorted(byw.items())}
    return out


MAX_W_2P = 40  # two-panel variant shows weights 20-40 only
# right panel of the two-panel variant: metric -> (linestyle, marker, label, sign); sign=-1 turns the agent-expert distance into the
# paper's expert-imitation score (Fig. 7: negative distance in AMP observation space, higher = better)
RIGHT = {
    "error_vel_xy": ("-", "o", "Tracking Error Vel.", 1),
    "mean_mechanical_cot": (":", "^", "CoT", 1),
    "agent_expert_distances": ("--", "s", "Expert Imitation (↓ worse)", -1),
}


def two_panels(data, weights):
    """Two panels (height as the paper's Fig. 5): yaw tracking error (absolute) | velocity tracking error, CoT and expert
    imitation as change relative to the dataset's own default weight. Seed means only (ranges: table / 2x2 figure). Weights 20-40."""
    weights = [w for w in weights if w <= MAX_W_2P]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 0.36 * COL_W))
    ax = axes[0]
    for name, byw in data.items():
        ws = [w for w in sorted(byw) if w <= MAX_W_2P]
        mean = np.array([np.mean(byw[w]["error_vel_yaw"]) for w in ws])
        lo = np.array([np.min(byw[w]["error_vel_yaw"]) for w in ws])
        hi = np.array([np.max(byw[w]["error_vel_yaw"]) for w in ws])
        # linear axis; whiskers shorter than MIN_WHISKER (axis units) are drawn at that length so every seed range is visible.
        # Author's choice (2026-09-16); the true ranges are in data/reward_weight_sweep.md. Say so in the caption.
        MIN_WHISKER = 0.012
        yerr = [np.maximum(mean - lo, MIN_WHISKER), np.maximum(hi - mean, MIN_WHISKER)]
        ax.errorbar(ws, mean, yerr=yerr, color=COLORS[name], marker="o", ms=2.0, lw=1.2, capsize=2.0, elinewidth=0.8, capthick=0.8, label=LEGEND[name], zorder=3)
    ax.set_title("Tracking Err. Yaw [rad/s]", fontsize=6, fontweight="bold", pad=2)
    ax.set_ylim(bottom=0)
    h_sets, l_sets = ax.get_legend_handles_labels()
    ax = axes[1]
    for name, byw in data.items():
        ws = [w for w in sorted(byw) if w <= MAX_W_2P]
        for m, (ls, _, _, sign) in RIGHT.items():
            ref = sign * np.mean(byw[DEFAULT_W][m])
            mean = np.array([100 * (sign * np.mean(byw[w][m]) - ref) / abs(ref) for w in ws])
            ax.plot(ws, mean, color=COLORS[name], ls=ls, lw=1.3 if name == NAME_MOCAP else 1.0, zorder=3)
            if name == NAME_MOCAP:  # direct end labels for the lines that change
                ax.text(ws[-1] + 0.6, mean[-1], f"{mean[-1]:+.0f}", color=COLORS[name], fontsize=5, va="center", ha="left")
    ax.axhline(0, color="0.3", lw=0.5, zorder=1)
    for w in weights:  # vertical guides at the evaluated weights instead of markers
        ax.axvline(w, color="0.75", lw=0.5, zorder=0)
    ax.set_title("Change [%]", fontsize=6, fontweight="bold", pad=2)
    ax.set_yticks([0, 20, 40])
    ax.set_xlim(19, 43.5)
    ax.legend(h_sets, [l.replace(" (AMP)", "").replace("Video w. Depth Camera", "Video") for l in l_sets],
              fontsize=5.0, frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.1, labelspacing=0.25, handletextpad=0.4)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="0.25", ls=ls, lw=1.2, label=lab) for ls, _, lab, _ in RIGHT.values()]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=5.2, frameon=False, bbox_to_anchor=(0.5, 0.935), handlelength=2.6, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    for ax in axes:
        ax.set_xticks(weights)
        ax.tick_params(labelsize=5.2, length=2, pad=1.2)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.7)
        ax.set_xlabel("Yaw Tracking Reward Weight", fontsize=5.6, labelpad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.subplots_adjust(top=0.80, bottom=0.25, left=0.1, right=0.99, wspace=0.3)
    import os as _os
    from common import FIG
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):  # tight crop without the default 0.1 in padding
        fig.savefig(_os.path.join(FIG, f"fig5_reward_weight_2panels.{ext}"), bbox_inches="tight", pad_inches=0.005, **kw)
    print("saved", _os.path.join(FIG, "fig5_reward_weight_2panels.pdf"))


def main():
    data = collect()
    weights = sorted({w for s in data.values() for w in s})
    two_panels(data, weights)

    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 0.72 * COL_W), sharex=True)
    for ax, m in zip(axes.ravel(), METRICS):
        for name, byw in data.items():
            ws = sorted(byw)
            mean = np.array([np.mean(byw[w][m]) for w in ws])
            lo = np.array([np.min(byw[w][m]) for w in ws])
            hi = np.array([np.max(byw[w][m]) for w in ws])
            ax.errorbar(ws, mean, yerr=[mean - lo, hi - mean], color=COLORS[name], marker="o", ms=2.6, lw=1.1, capsize=1.8, elinewidth=0.7, label=LEGEND[name], zorder=3)
        ax.set_title(TITLES[m], fontsize=6.5, fontweight="bold", pad=3)
        ax.set_xticks(weights)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.7)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Yaw Tracking Reward Weight", fontsize=6)
    axes[1, 1].set_ylim(1.9, 2.6)  # imitation score: differences are small, zoom in (noted in caption)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=5.8, frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.11, right=0.99, hspace=0.55, wspace=0.32)
    savefig(fig, "fig5_reward_weight_4metrics")

    # table + json
    header = ["set", "weight", "seeds"] + METRICS
    rows = []
    js = {}
    for name, byw in data.items():
        for w, d in byw.items():
            rows.append([name, w, d["n"]] + [f"{np.mean(d[m]):.4f} [{min(d[m]):.4f}, {max(d[m]):.4f}]" for m in METRICS])
            js.setdefault(name, {})[w] = {m: dict(mean=float(np.mean(d[m])), min=float(min(d[m])), max=float(max(d[m])), per_seed=[float(x) for x in d[m]]) for m in METRICS} | {"n": d["n"]}
    ref = {m: np.mean(data[NAME_VIDEO_EXT][DEFAULT_W][m]) for m in METRICS}
    rel = [[name, w] + [f"{100 * (np.mean(d[m]) / ref[m] - 1):+.0f} %" for m in METRICS] for name, byw in data.items() for w, d in byw.items()]
    with open(os.path.join(DATA, "reward_weight_sweep.md"), "w") as fh:
        fh.write("# Yaw-tracking reward weight sweep (paper logs, DefaultEvalConfig; mean [min, max] over seeds)\n\n")
        fh.write("Weight 20 = paper default (Fig.-4 runs, metrics.yaml); 30/40/50 from *_trackAnVelRewWeight_* runs (None_metrics.yaml).\n\n")
        fh.write(md_table(header, rows) + "\n\nRelative to Video (extended) at weight 20:\n\n" + md_table(["set", "weight"] + METRICS, rel) + "\n")
    with open(os.path.join(DATA, "reward_weight_sweep.json"), "w") as fh:
        json.dump(js, fh, indent=1)
    print(open(os.path.join(DATA, "reward_weight_sweep.md")).read())


if __name__ == "__main__":
    main()
