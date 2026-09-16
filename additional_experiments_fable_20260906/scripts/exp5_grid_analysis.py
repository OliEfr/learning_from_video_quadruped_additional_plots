"""Exp. 5, grid evaluation of E1 (Video (extended, half)): coverage-vs-error link with REAL per-cell yamls.

Reads <run>/TargetXYDistributionEvaluation/x_<vx>_y_<vy>_yaw_0.0.yaml (147 cells, 5000 envs x 1 episode each) for every
E1 seed, averages the seeds, and computes the same cell-level statistics as Exp. 4 (exp4_amp_evidence.analysis_cells):
Spearman rho between the distance of the command to the nearest expert frame and the tracking errors, covered/uncovered
means, and the key-result regression (combined normalised tracking error vs command distance).
The paper sets (MoCap, Video, Video (extended)) enter with the Fig.-5 values recovered from the figure colours (Exp. 4),
so their statistics carry the recovery noise; the E1 numbers are exact.
Outputs: data/exp5_grid.md, data/exp5_grid.json, figures/fig_exp5_grid_key_result(.pdf/.png)
"""
import glob
import json
import os
import re
import sys

import numpy as np
import yaml
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import exp4_amp_evidence as e4
from common import DATA, EVAL_GRID, NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_EXT, load_amp_dir, md_table
from exp2_coverage import SLICE_W, covered_cells_xy, dataset_frames, nearest_distance_map
from exp5_build_video_subsets import frames_of

W = "/home/admin_07/project_repos/isaac_lab/IsaacLab_paper_e3df3c0b"
LOGROOT = os.path.join(W, "logs/rsl_rl/unitree_go2_AMPflat")
DT = open(os.path.join(W, "exp5_state_DT.txt")).read().strip()
E1_FOLDER = "fromVision_motions_DepthCam_extHalf"
E1 = "E1 Video (ext., half)"
METRICS = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot", "agent_expert_distances"]


def load_grid(run_dir):
    """(7, 21) arrays indexed [vy, vx] from the per-cell yamls; None if the grid is incomplete."""
    gx, gy = EVAL_GRID["xy"]
    files = glob.glob(os.path.join(run_dir, "TargetXYDistributionEvaluation", "x_*_y_*_yaw_0.0.yaml"))
    if len(files) < len(gx) * len(gy):
        return None
    out = {m: np.full((len(gy), len(gx)), np.nan) for m in METRICS}
    for f in files:
        x, y = map(float, re.match(r".*x_(-?[\d.]+)_y_(-?[\d.]+)_yaw", os.path.basename(f)).groups())
        i, j = int(np.argmin(np.abs(gx - x))), int(np.argmin(np.abs(gy - y)))
        d = yaml.safe_load(open(f))
        for m in METRICS:
            out[m][j, i] = float(d[m])
    assert not any(np.isnan(v).any() for v in out.values())
    return out


def cell_stats(cmd, cmd_ms, cov, evel, eyaw, cot, aed):
    comb = e4.combined_error(evel, eyaw)
    r = {}
    for name, g in [("evel", evel), ("eyaw", eyaw), ("comb", comb), ("cot", cot), ("aed", aed)]:
        r[f"rho_cmd_{name}"] = float(spearmanr(cmd.ravel(), g.ravel())[0])
        r[f"{name}_covered_mean"] = float(g[cov].mean())
        r[f"{name}_uncovered_mean"] = float(g[~cov].mean())
        r[f"{name}_uncovered_gt_covered_p"] = float(mannwhitneyu(g[~cov], g[cov], alternative="greater").pvalue)
    r["rho_aed_evel"] = float(spearmanr(aed.ravel(), evel.ravel())[0])
    r["rho_aed_eyaw"] = float(spearmanr(aed.ravel(), eyaw.ravel())[0])
    a, b = np.polyfit(cmd.ravel(), comb.ravel(), 1)
    r["fit_comb_vs_cmd"] = {"slope": float(a), "intercept": float(b), "pearson_r": float(np.corrcoef(cmd.ravel(), comb.ravel())[0, 1])}
    a, b = np.polyfit(cmd_ms.ravel(), comb.ravel(), 1)
    r["fit_comb_vs_cmd_ms"] = {"slope": float(a), "intercept": float(b)}
    r["n_covered"] = int(cov.sum())
    r["grid_mean"] = {"evel": float(evel.mean()), "eyaw": float(eyaw.mean()), "cot": float(cot.mean()), "aed": float(aed.mean()), "comb": float(comb.mean())}
    return r, comb


def main():
    gx, gy = EVAL_GRID["xy"]
    res = {"note": __doc__.strip()}
    # --- E1: expert frames of the ablation set, real grids of all complete seeds
    d_e1 = frames_of(load_amp_dir(os.path.join(W, "datasets", E1_FOLDER)))
    cmd_e1 = nearest_distance_map(d_e1, gx, gy, "xy")
    cmd_ms_e1 = e4.nearest_distance_map_ms(d_e1, gx, gy)
    cov_e1 = covered_cells_xy(d_e1, gx, gy)
    grids = {}
    for run in sorted(glob.glob(os.path.join(LOGROOT, f"exp5_{DT}_{E1_FOLDER}_SEED_*"))):
        g = load_grid(run)
        if g is not None:
            grids[int(run.rsplit("_SEED_", 1)[1])] = g
    if not grids:
        print("no complete E1 grid yet"); return
    seeds = sorted(grids)
    per_seed = {}
    for s in seeds:
        g = grids[s]
        per_seed[s], _ = cell_stats(cmd_e1, cmd_ms_e1, cov_e1, g["error_vel_xy"], g["error_vel_yaw"], g["mean_mechanical_cot"], g["agent_expert_distances"])
    mean_g = {m: np.mean([grids[s][m] for s in seeds], axis=0) for m in METRICS}
    r_e1, comb_e1 = cell_stats(cmd_e1, cmd_ms_e1, cov_e1, mean_g["error_vel_xy"], mean_g["error_vel_yaw"], mean_g["mean_mechanical_cot"], mean_g["agent_expert_distances"])
    r_e1["seeds"] = seeds
    r_e1["per_seed"] = per_seed
    res[E1] = r_e1
    # seed-to-seed spread of the key numbers
    for k in ["rho_cmd_evel", "rho_cmd_eyaw", "rho_cmd_comb"]:
        v = [per_seed[s][k] for s in seeds]
        r_e1[f"{k}_seed_range"] = [float(min(v)), float(max(v))]
    # --- paper sets from the recovered Fig.-5 grids (Exp. 4)
    P = {k: [] for k in ["cmd", "cmd_ms", "comb", "aed", "src"]}
    for src in [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_EXT]:
        d, _ = dataset_frames(src)
        cmd = nearest_distance_map(d, gx, gy, "xy"); cmd_ms = e4.nearest_distance_map_ms(d, gx, gy); cov = covered_cells_xy(d, gx, gy)
        evel, eyaw, cot, aed = (e4.fig5_grid(src, m) for m in METRICS)
        r, comb = cell_stats(cmd, cmd_ms, cov, evel, eyaw, cot, aed)
        r["source"] = "Fig. 5 recovered from figure colours (Exp. 4)"
        res[e4.SHORT[src] + " (paper, recovered)"] = r
        for k, v in [("cmd", cmd), ("cmd_ms", cmd_ms), ("comb", comb), ("aed", aed)]:
            P[k].append(v.ravel())
        P["src"].append(np.array([src] * cmd.size))
    for k, v in [("cmd", cmd_e1), ("cmd_ms", cmd_ms_e1), ("comb", comb_e1), ("aed", mean_g["agent_expert_distances"])]:
        P[k].append(v.ravel())
    P["src"].append(np.array([E1] * cmd_e1.size))
    P = {k: np.concatenate(v) for k, v in P.items()}
    # --- table
    rows = []
    for name, r in res.items():
        if name == "note": continue
        rows.append([name, r["n_covered"], r["rho_cmd_evel"], r["rho_cmd_eyaw"], r["rho_cmd_comb"], r["rho_cmd_aed"],
                     r["fit_comb_vs_cmd"]["slope"], r["fit_comb_vs_cmd"]["intercept"],
                     r["evel_covered_mean"], r["evel_uncovered_mean"], r["eyaw_covered_mean"], r["eyaw_uncovered_mean"],
                     r["aed_covered_mean"], r["aed_uncovered_mean"]])
    table = md_table(["set", "covered cells /147", "rho(cmd dist, err vel)", "rho(cmd dist, err yaw)", "rho(cmd dist, comb. err)",
                      "rho(cmd dist, agent-expert dist)", "slope comb. err vs cmd dist", "intercept",
                      "err vel covered", "uncovered", "err yaw covered", "uncovered", "agent-expert dist covered", "uncovered"], rows, floatfmt="{:.3f}")
    ps = md_table(["E1 seed", "rho(cmd, err vel)", "rho(cmd, err yaw)", "rho(cmd, comb)", "slope", "grid mean err vel", "grid mean err yaw", "grid mean CoT"],
                  [[s, per_seed[s]["rho_cmd_evel"], per_seed[s]["rho_cmd_eyaw"], per_seed[s]["rho_cmd_comb"], per_seed[s]["fit_comb_vs_cmd"]["slope"],
                    per_seed[s]["grid_mean"]["evel"], per_seed[s]["grid_mean"]["eyaw"], per_seed[s]["grid_mean"]["cot"]] for s in seeds], floatfmt="{:.3f}")
    text = ("# Exp. 5: E1 grid evaluation (TargetXYDistribution, 7 x 21 vx-vy cells at yaw rate 0)\n\n"
            f"E1 seeds with complete grids: {seeds}. Paper rows use the Fig.-5 values recovered from the figure (Exp. 4); E1 rows are exact yaml values "
            "(seed mean). Command distance = normalised distance of the cell's command to the nearest expert frame of the set the policy was trained on.\n\n"
            + table + "\n\nPer E1 seed:\n\n" + ps + "\n")
    print(text)
    open(os.path.join(DATA, "exp5_grid.md"), "w").write(text)
    json.dump(res, open(os.path.join(DATA, "exp5_grid.json"), "w"), indent=1, default=float)
    # --- figure: Exp.-4 key-result panel with E1 added (real grid) next to MoCap and Video (extended) (recovered)
    e4.COLORS[E1] = "#d62728"; e4.MARKERS[E1] = "D"; e4.SHORT[E1] = E1
    P["imit"] = -P["aed"]
    fits = e4._key_panel(P, "cmd", "comb", "imit", e4.XLAB_NORM, r"$\bf{Tracking\ Error}$" + "\nYaw and Vel. Normalized [1]", e4.CB_IMIT,
                         (-4.0, -2.0), [-4, -3, -2], (0, 1.12), (-0.03, 1.55), "fig_exp5_grid_key_result", cmap="viridis_r",
                         sources=[NAME_MOCAP, NAME_VIDEO_EXT, E1])
    res["key_result_fits"] = fits
    json.dump(res, open(os.path.join(DATA, "exp5_grid.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
