"""Extract mechanical Cost of Transport for the yaw-reward-weight sweep.

Companion to IsaacLab/plot_AngVelRew.py, which plots error_vel_xy and
error_vel_yaw for the same runs.  Every eval metrics YAML written by
source/standalone/workflows/rsl_rl/play.py --evaluate already contains
"mean_mechanical_cot" alongside the two plotted errors, so no policy has to be
re-run: this script just re-reads the same files.

Usage:
    python extract_cot_vs_yawweight.py [LOGS_ROOT]

LOGS_ROOT defaults to the IsaacLab logs dir; it must contain
rsl_rl/unitree_go2_AMPflat/<run>_trackAnVelRewWeight_<N>_SEED_<S>/.
"""

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

DEFAULT_LOGS_ROOT = os.path.expanduser(
    "~/project_repos/isaac_lab/IsaacLab/logs"
)

EXPERIMENTS = {
    "Video w. Depth Camera (extended) (AMP)": (
        "2025-07-11_21-14-15_fromVision_motions_DepthCam_"
        "extendedWithoutReverse_*trackAnVelRewWeight*"
    ),
    "MoCap (AMP)": "2025-07-13_13-22-15_mocap_AMP_for_hardware_*trackAnVelRewWeight*",
}

# error_vel_* are re-extracted too, so the recovered data can be checked against
# the published figures before the CoT numbers are trusted.
METRICS = ["mean_mechanical_cot", "error_vel_xy", "error_vel_yaw"]

METRICS_FILENAMES = ("None_metrics.yaml", "metrics.yaml")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_metrics_file(run_dir):
    for name in METRICS_FILENAMES:
        candidate = os.path.join(run_dir, name)
        if os.path.isfile(candidate):
            return candidate
    hits = sorted(glob.glob(os.path.join(run_dir, "*metrics.yaml")))
    return hits[0] if hits else None


def weight_from_path(path):
    match = re.search(r"trackAnVelRewWeight_(\d+)", path)
    return int(match.group(1)) if match else None


def weight_from_env_cfg(run_dir):
    """Cross-check: read the reward weight actually used, from params/env.yaml."""
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        return None
    try:
        cfg = load_yaml(env_yaml)
        return cfg["rewards"]["track_ang_vel_z_exp"]["weight"]
    except (KeyError, TypeError, yaml.YAMLError):
        return None


def collect(logs_root):
    results = {}
    for exp_name, pattern in EXPERIMENTS.items():
        by_weight = defaultdict(list)
        full = os.path.join(
            logs_root, "rsl_rl", "unitree_go2_AMPflat", f"{pattern}_SEED_*"
        )
        for run_dir in sorted(glob.glob(full)):
            weight = weight_from_path(run_dir)
            if weight is None:
                continue
            metrics_file = find_metrics_file(run_dir)
            if metrics_file is None:
                print(f"  no metrics yaml in {run_dir}")
                continue
            data = load_yaml(metrics_file)
            cfg_weight = weight_from_env_cfg(run_dir)
            if cfg_weight is not None and float(cfg_weight) != float(weight):
                print(
                    f"  WARNING weight mismatch in {run_dir}: "
                    f"dir says {weight}, env.yaml says {cfg_weight}"
                )
            by_weight[weight].append((run_dir, data))
        results[exp_name] = by_weight
    return results


def report(results):
    any_data = False
    for exp_name, by_weight in results.items():
        print(f"\n=== {exp_name} ===")
        if not by_weight:
            print("  no run directories found")
            continue
        any_data = True
        header = f"{'weight':>7} {'seeds':>6}"
        for metric in METRICS:
            header += f" {metric + ' mean':>28} {'min':>10} {'max':>10}"
        print(header)
        for weight in sorted(by_weight):
            runs = by_weight[weight]
            row = f"{weight:>7} {len(runs):>6}"
            for metric in METRICS:
                values = [d[metric] for _, d in runs if metric in d]
                if not values:
                    row += f" {'MISSING':>28} {'-':>10} {'-':>10}"
                    continue
                arr = np.asarray(values, dtype=float)
                row += (
                    f" {arr.mean():>28.6f} {arr.min():>10.6f} {arr.max():>10.6f}"
                )
            print(row)
    if not any_data:
        print(
            "\nNo runs found. The eval metrics YAMLs are the only thing needed; "
            "restore the run directories (DVC remote 'nextcloud') and re-run."
        )
    return any_data


def main():
    logs_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOGS_ROOT
    print(f"logs root: {logs_root}")
    if not os.path.isdir(logs_root):
        print("logs root does not exist")
        return 1
    return 0 if report(collect(logs_root)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
