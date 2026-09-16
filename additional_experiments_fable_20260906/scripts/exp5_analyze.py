"""Experiment 5: collect the evaluation metrics of the ablation runs and compare with the paper's Fig.-4 values.

Runs: <WORKTREE>/logs/rsl_rl/unitree_go2_AMPflat/exp5_<DT>_<folder>_SEED_<n>/metrics.yaml (DefaultEvalConfig: 10 000 envs x 2 episodes
of 10 s on the training command distribution). Metrics as in Fig. 4: error_vel_xy [m/s], error_vel_yaw [rad/s], mean_mechanical_cot.
Paper reference values (seed means read from Fig. 4, paper_fig4_numbers.md) are included for MoCap, Video and Video (extended).
Optionally the TargetXYDistribution grid (147 cells) of a run is summarised and correlated with the Exp.-2 command distance.
"""
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import DATA, md_table  # noqa: E402

WORKTREE = "/home/admin_07/project_repos/isaac_lab/IsaacLab_paper_e3df3c0b"
LOGROOT = os.path.join(WORKTREE, "logs/rsl_rl/unitree_go2_AMPflat")
METRICS = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot"]
LABEL = {
    "fromVision_motions_DepthCam_extendedWithoutReverse": "E0 Video (extended), re-run seed",
    "fromVision_motions_DepthCam_extHalf": "E1 Video (extended, half)",
    "fromVision_motions_DepthCam_extHalfCov": "E1b Video (extended, half, coverage-matched)",
    "fromVision_motions_DepthCam_extAddedOnly": "E2 Video (added clips only)",
    "fromVision_motions_DepthCam_extNoTurn": "E3 Video (extended minus turning)",
    "fromVision_motions_DepthCam_extNoStartStop": "E4 Video (extended minus start-stop)",
}
PAPER = {  # Fig. 4 seed means (paper_fig4_numbers.md); per-seed values are read from data/paper_logs (logs.zip) when present
    "MoCap (paper)": dict(error_vel_xy=0.0624, error_vel_yaw=0.645, mean_mechanical_cot=1.51),
    "Video (paper)": dict(error_vel_xy=0.0565, error_vel_yaw=0.213, mean_mechanical_cot=1.59),
    "Video (extended) (paper)": dict(error_vel_xy=0.0481, error_vel_yaw=0.129, mean_mechanical_cot=1.31),
}
PAPER_RUNS = {  # the paper's own evaluation files (DefaultEvalConfig), extracted from logs.zip
    "MoCap (paper)": "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_*",
    "Video (paper)": "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_*",
    "Video (extended) (paper)": "2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_*",
}
PAPER_LOGS = os.path.join(DATA, "paper_logs", "rsl_rl", "unitree_go2_AMPflat")


def paper_per_seed(label):
    """{seed: {metric: value}} from the paper's metrics.yaml files, or {} if logs.zip was not extracted."""
    out = {}
    for d in sorted(glob.glob(os.path.join(PAPER_LOGS, PAPER_RUNS[label]))):
        p = os.path.join(d, "metrics.yaml")
        if os.path.exists(p):
            with open(p) as fh:
                m = yaml.safe_load(fh)
            out[int(d.rsplit("_SEED_", 1)[1])] = {k: float(m[k]) for k in METRICS}
    return out


# Runs excluded from the means (kept in the JSON under "excluded"): training diverged (PPO critic loss exploded at
# iteration ~9640, policy collapsed at 9880 and never recovered; eval shows falls: 23760 instead of 20000 episodes).
EXCLUDE = {("fromVision_motions_DepthCam_extAddedOnly", 1): "diverged at iter ~9.7k (critic blow-up), collapsed policy"}


def collect():
    runs = {}
    excluded = {}
    for d in sorted(glob.glob(os.path.join(LOGROOT, "exp5_*_SEED_*"))):
        f = os.path.join(d, "metrics.yaml")
        if not os.path.exists(f):
            continue
        name = os.path.basename(d)
        folder = name.split("_", 3)[3].rsplit("_SEED_", 1)[0]  # exp5_<date>_<time>_<folder>_SEED_n
        seed = int(name.rsplit("_SEED_", 1)[1])
        with open(f) as fh:
            m = yaml.safe_load(fh)
        rec = {k: float(m[k]) for k in METRICS} | {"episodes": int(m.get("total_episodes (real)", 0))}
        if (folder, seed) in EXCLUDE:
            excluded[f"{folder}_SEED_{seed}"] = rec | {"reason": EXCLUDE[(folder, seed)]}
            continue
        runs.setdefault(folder, {})[seed] = rec
    return runs, excluded


def main():
    runs, excluded = collect()
    rows = []
    summary = {}
    for label, v in PAPER.items():
        ps = paper_per_seed(label)
        if ps:  # exact per-seed values from logs.zip -> mean [min, max] like the new arms
            arr = {k: np.array([ps[s][k] for s in sorted(ps)]) for k in METRICS}
            v = {k: float(arr[k].mean()) for k in METRICS}
            rows.append([label, f"{len(ps)} (paper logs)"] + [f"{arr[k].mean():.4f} [{arr[k].min():.4f}, {arr[k].max():.4f}]" if k != "mean_mechanical_cot" else f"{arr[k].mean():.2f} [{arr[k].min():.2f}, {arr[k].max():.2f}]" for k in METRICS] + ["", "", ""])
            summary[label] = dict(mean=v, n=len(ps), source="paper logs.zip metrics.yaml", per_seed=ps)
            PAPER[label] = v
        else:
            rows.append([label, "3 (paper)"] + [f"{v[k]:.4f}" if k != "mean_mechanical_cot" else f"{v[k]:.2f}" for k in METRICS] + ["", "", ""])
            summary[label] = dict(mean=v, n=3, source="paper Fig. 4")
    ref = PAPER["Video (extended) (paper)"]
    for folder, seeds in runs.items():
        arr = {k: np.array([seeds[s][k] for s in sorted(seeds)]) for k in METRICS}
        mean = {k: float(arr[k].mean()) for k in METRICS}
        rng = {k: (float(arr[k].min()), float(arr[k].max())) for k in METRICS}
        rel = {k: 100 * (mean[k] - ref[k]) / ref[k] for k in METRICS}
        rows.append([LABEL.get(folder, folder), f"{len(seeds)} ({','.join(str(s) for s in sorted(seeds))})"]
                    + [f"{mean['error_vel_xy']:.4f} [{rng['error_vel_xy'][0]:.4f}, {rng['error_vel_xy'][1]:.4f}]",
                       f"{mean['error_vel_yaw']:.3f} [{rng['error_vel_yaw'][0]:.3f}, {rng['error_vel_yaw'][1]:.3f}]",
                       f"{mean['mean_mechanical_cot']:.2f} [{rng['mean_mechanical_cot'][0]:.2f}, {rng['mean_mechanical_cot'][1]:.2f}]"]
                    + [f"{rel[k]:+.0f} %" for k in METRICS])
        summary[LABEL.get(folder, folder)] = dict(mean=mean, range=rng, per_seed={s: seeds[s] for s in sorted(seeds)}, n=len(seeds),
                                                  rel_to_video_ext_paper_pct=rel)
    hdr = ["set", "seeds", "vel. error [m/s] (mean [min, max])", "yaw error [rad/s]", "CoT", "d vel vs Video(ext)", "d yaw", "d CoT"]
    table = md_table(hdr, rows)
    if excluded:
        table += "\n\nExcluded from the means (diverged training runs):\n" + "\n".join(
            f"- {k}: vel {v['error_vel_xy']:.4f}, yaw {v['error_vel_yaw']:.3f}, CoT {v['mean_mechanical_cot']:.2f}, "
            f"{v['episodes']} episodes; {v['reason']}" for k, v in excluded.items())
        summary["excluded"] = excluded
    print(table)
    with open(os.path.join(DATA, "exp5_metrics.md"), "w") as fh:
        fh.write("# Exp. 5: evaluation metrics (DefaultEvalConfig, target command distribution, as Fig. 4)\n\n" + table + "\n")
    with open(os.path.join(DATA, "exp5_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
