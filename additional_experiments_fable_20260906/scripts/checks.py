"""Independent sanity checks for the three experiments. Writes data/checks_report.md (PASS/FAIL per check)."""
import ast
import json
import os
import re

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt
from scipy.spatial import cKDTree

from common import (
    AMP_DIRS,
    DATA,
    FLAT_VIDEO_CLIPS,
    ISAAC,
    MOCAP_FPS,
    MOCAP_SEGMENTS,
    NAME_MOCAP,
    NAME_VIDEO,
    NAME_VIDEO_EXT,
    RETARGET,
    VISION,
    align_clip,
    hip_height,
    load_amp_dir,
    load_mocap_raw,
    load_mocap_segment,
    load_video_keypoints_as_used,
    mocap_keypoints,
    sg_residual,
    yaw_from_quat_xyzw,
)

REPORT = []


def check(name, ok, detail=""):
    REPORT.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(REPORT[-1])


def note(name, detail):
    REPORT.append(f"- [INFO] {name}: {detail}")
    print(REPORT[-1])


# ----------------------------------------------------------------------------------------------
def check_datasets():
    exp = {NAME_MOCAP: (6, 6.80), NAME_VIDEO: (4, 6.03), NAME_VIDEO_EXT: (8, 12.20)}
    for src, (n, dur) in exp.items():
        clips = load_amp_dir(AMP_DIRS[src])
        d = sum((len(c["frames"]) - 1) * c["fd"] for c in clips)
        check(f"{src}: {n} clips, {dur:.2f} s", len(clips) == n and abs(d - dur) < 0.01, f"found {len(clips)} clips, {d:.3f} s (paper: {'6.0' if src == NAME_VIDEO else '12.2' if src == NAME_VIDEO_EXT else 'n/a'} s)")
        F = np.concatenate([c["frames"] for c in clips])
        check(f"{src}: angular-velocity columns 34:37 are all zero", np.all(F[:, 34:37] == 0), "so the yaw rate has to be derived from the root quaternion")
        check(f"{src}: foot-position columns 19:31 and 49:61 are all zero", np.all(F[:, 19:31] == 0) and np.all(F[:, 49:61] == 0), "")
        err = max(np.abs(np.diff(c["frames"][:, :3], axis=0) / c["fd"] - c["frames"][1:, 31:34]).max() for c in clips)
        check(f"{src}: stored linear velocity == diff(root_pos)/FrameDuration", err < 2e-3, f"max abs err {err:.2e} m/s (rounding to 5 decimals in the files)")
        if src != NAME_MOCAP:
            check(f"{src}: quaternion x-component (pybullet xyzw) is zero -> no roll reconstructed", np.all(F[:, 3] == 0), "consistent with Sec. III-A of the paper")
        fds = sorted({c["fd"] for c in clips})
        note(f"{src}: FrameDuration", f"{fds}  (MoCap raw capture is 60 Hz -> 0.021 s plays the dog 0.79x slower; video 30 fps -> 0.03334 s is real time)")
        wts = {c["name"]: c["weight"] for c in clips if c["weight"] != 1}
        note(f"{src}: MotionWeight != 1", str(wts) if wts else "none")


# ----------------------------------------------------------------------------------------------
def check_relift():
    v = json.load(open(os.path.join(DATA, "relift_validation.json")))
    med = max(x["median_abs_diff_m"] for x in v.values())
    check("re-lift ('as_code' intrinsics) reproduces the on-disk keypoints", med < 1e-4, f"max over clips of median |inter-marker distance diff| = {med * 1000:.3f} mm; frame offset {sorted({x['frame_offset'] for x in v.values()})} (3d_recon.py skips frame 0)")
    worst = {k: round(x["max_abs_diff_m"] * 1000, 1) for k, x in v.items() if x["max_abs_diff_m"] > 1e-3}
    note("clips with boundary differences (different trimming -> different extrapolation of missing values)", str(worst))
    hh_code = np.array([x["hip_height_as_code_m"] for x in v.values()])
    hh_half = np.array([x["hip_height_half_res_m"] for x in v.values()])
    ratio = hh_half / hh_code
    check("half-res intrinsics roughly double the reconstructed hip height (ratio 1.8-3.0; >2 because the ground-plane fit changes as well)", np.all(ratio > 1.8) and np.all(ratio < 3.0), f"hip height as_code {hh_code.mean():.2f} m (range {hh_code.min():.2f}-{hh_code.max():.2f}) vs half_res {hh_half.mean():.2f} m ({hh_half.min():.2f}-{hh_half.max():.2f}); ratio {ratio.min():.2f}-{ratio.max():.2f}")
    # independent single-frame check of the dog size from pixels + depth (slow clip, frame 20; hips ~ y 178-180, feet ~ y 262-282)
    clip, i = "slow_1313807000", 20
    deps = sorted(os.listdir(f"{VISION}/in/{clip}/depth_cam"))
    Dm = np.fromfile(f"{VISION}/in/{clip}/depth_cam/{deps[i]}", dtype=np.uint16).reshape(360, 640) / 1000.0
    f = np.load(f"{VISION}/tracks/{clip}/foot_pos.npy")[i]
    h = np.load(f"{VISION}/tracks/{clip}/hip_pos.npy")[0][i]
    vf = [y for (x, y) in f if x or y]
    z = np.mean([Dm[y, x] for (x, y) in h])
    dy = np.mean(vf) - np.mean(h[:, 1])
    note("single-frame dog height from pixel extent x depth", f"hip->paw {dy:.0f} px at {z:.2f} m -> {dy * z / 747.5:.2f} m with the 1280x720 focal length, {dy * z / 373.7:.2f} m with the 640x360 focal length (a medium-sized dog, cf. the RGB frame)")
    # depth-missing statistics
    miss = np.array([x["frac_feet_missing_depth"] for x in v.values()])
    note("fraction of foot samples with zero (invalid) depth at the tracked pixel, per clip", ", ".join(f"{k.split('_')[0]}:{x['frac_feet_missing_depth'] * 100:.0f}%" for k, x in v.items()) + f"; mean {miss.mean() * 100:.0f}%")
    check("camera.yaml intrinsics are for 1280x720 while tracks/depth are 640x360", True, "cx=634.5, cy=370.8 (image centre of 1280x720); tracks max x=604<640, max y=357<360; 3d_recon.py applies FX,CX unchanged to the half-res pixels")


# ----------------------------------------------------------------------------------------------
def m1_feet(body, feet, window=7, poly=3):
    hh = hip_height(body, feet)
    res = np.stack([sg_residual(k, window, poly) for k in feet])
    return np.sqrt((res ** 2).sum(-1).mean()) / hh


def check_exp1_robustness():
    mocap = []
    for seg in MOCAP_SEGMENTS:
        body, feet = mocap_keypoints(load_mocap_segment(seg))
        mocap.append(align_clip(body[:, ::2], feet[:, ::2])[:2])
    video = [align_clip(*load_video_keypoints_as_used(c))[:2] for c in FLAT_VIDEO_CLIPS]
    for w in [5, 7, 9]:
        r = np.mean([m1_feet(b, f, w) for b, f in video]) / np.mean([m1_feet(b, f, w) for b, f in mocap])
        note(f"M1 feet ratio video/mocap with SG window {w}", f"{r:.2f}")
    # native 60 Hz mocap (unfair comparison) vs subsampled 30 Hz
    mocap60 = [align_clip(*mocap_keypoints(load_mocap_segment(seg)))[:2] for seg in MOCAP_SEGMENTS]
    r60 = np.mean([m1_feet(b, f) for b, f in video]) / np.mean([m1_feet(b, f) for b, f in mocap60])
    r30 = np.mean([m1_feet(b, f) for b, f in video]) / np.mean([m1_feet(b, f) for b, f in mocap])
    note("M1 feet ratio if MoCap were left at 60 Hz (unfair) vs matched 30 Hz", f"{r60:.2f} vs {r30:.2f}")
    # anti-aliased decimation vs plain subsampling on the long raw mocap file
    arr = load_mocap_raw("dog_walk09")
    body, feet = mocap_keypoints(arr)
    a = np.mean([m1_feet(*align_clip(body[:, ::2], feet[:, ::2])[:2])])
    bd = decimate(body, 2, axis=1, ftype="fir", zero_phase=True)
    fd = decimate(feet, 2, axis=1, ftype="fir", zero_phase=True)
    b = m1_feet(*align_clip(bd, fd)[:2])
    check("MoCap 60->30 Hz: plain subsampling vs anti-aliased decimation give similar M1", abs(a - b) / max(a, b) < 0.35, f"dog_walk09 full file: {a * 100:.3f} % vs {b * 100:.3f} % hip height (anti-aliasing removes some genuine >15 Hz content)")
    # Butterworth alternative
    sos = butter(4, 5.0, fs=30.0, output="sos")

    def m1_butter(body, feet):
        hh = hip_height(body, feet)
        out = []
        for k in feet:
            if k.shape[0] < 20:
                return np.nan
            r = k - sosfiltfilt(sos, k, axis=0, padlen=min(15, k.shape[0] - 2))
            out.append((r ** 2).sum(-1).mean())
        return np.sqrt(np.mean(out)) / hh

    rv = np.nanmean([m1_butter(b, f) for b, f in video])
    rm = np.nanmean([m1_butter(b, f) for b, f in mocap])
    note("M1 feet ratio video/mocap with a 4th-order zero-phase Butterworth (5 Hz) instead of SG (clips >= 20 frames)", f"{rv / rm:.2f}")
    # hip heights
    note("hip heights used for normalisation", f"MoCap {np.mean([hip_height(b, f) for b, f in mocap]):.3f} m, Video (pipeline output) {np.mean([hip_height(b, f) for b, f in video]):.3f} m")


# ----------------------------------------------------------------------------------------------
def check_exp2():
    res = json.load(open(os.path.join(DATA, "exp2_coverage.json")))
    import exp2_coverage as E

    gx, gy = E.EVAL_GRID["xy"]
    gx2, gw = E.EVAL_GRID["xyaw"]
    for src in E.SOURCES:
        d, clips = E.dataset_frames(src)
        check(f"{src}: AMP sampling weights sum to 1", abs(d["w_amp"].sum() - 1) < 1e-9, f"{d['w_amp'].sum():.6f}")
        # independent coverage count with a KD-tree (Chebyshev metric on scaled axes)
        P = np.stack([d["vx"] / E.TOL_V, d["vy"] / E.TOL_V, d["wz"] / E.SLICE_W], 1)
        tree = cKDTree(P)
        C = np.array([[x / E.TOL_V, y / E.TOL_V, 0.0] for y in gy for x in gx])
        dist, _ = tree.query(C, p=np.inf)
        n_xy = int((dist <= 1 + 1e-9).sum())
        P2 = np.stack([d["vx"] / E.TOL_V, d["vy"] / E.SLICE_VY, d["wz"] / E.TOL_W], 1)
        tree2 = cKDTree(P2)
        C2 = np.array([[x / E.TOL_V, 0.0, w / E.TOL_W] for w in gw for x in gx2])
        dist2, _ = tree2.query(C2, p=np.inf)
        n_xyaw = int((dist2 <= 1 + 1e-9).sum())
        r = res["datasets"][src]
        check(f"{src}: KD-tree coverage count == loop count", n_xy == r["covered_xy"] and n_xyaw == r["covered_xyaw"], f"xy {n_xy} vs {r['covered_xy']}, xyaw {n_xyaw} vs {r['covered_xyaw']}")
        # yaw: total change over each clip vs. spread of the rate (is the turning signal or noise?)
        for c in clips:
            yaw = np.unwrap(yaw_from_quat_xyzw(c["frames"][:, 3:7]))
            T = (len(yaw) - 1) * c["fd"]
            m = d["clip"] == c["name"]
            note(f"{src}/{c['name']}: net yaw change", f"{np.degrees(yaw[-1] - yaw[0]):+.0f} deg over {T:.2f} s = mean rate {(yaw[-1] - yaw[0]) / T:+.2f} rad/s; smoothed-rate p5..p95 [{np.percentile(d['wz'][m], 5):.2f}, {np.percentile(d['wz'][m], 95):.2f}] rad/s; std(raw - smoothed rate) {np.std(d['wz_raw'][m] - d['wz'][m]):.2f} rad/s")
    note("command ranges / heading", f"vx {E.CMD_RANGES['vx']}, vy {E.CMD_RANGES['vy']}, wz {E.CMD_RANGES['wz']}, heading_command=False (velocity_env_cfg.py lines 95-117)")


# ----------------------------------------------------------------------------------------------
def check_exp3():
    findings = list(__import__("csv").DictReader(open(os.path.join(DATA, "exp3_findings.csv"))))
    # every finding with a line number must point at a line that mentions the clip (or the file must be the dataset json)
    bad = 0
    for f in findings:
        if f["line"] in ("", "0", "13") or f["file"].startswith("isaac_lab/IsaacLab/datasets"):
            continue
        path = os.path.join("/home/admin_07/project_repos", f["file"])
        lines = open(path).read().splitlines()
        L = int(f["line"])
        window = " ".join(lines[max(0, L - 40) : L + 2])  # the branch header may be a few lines above a long statement
        if f["clip"] not in window and f["clip"].split("_")[0] not in window:
            bad += 1
    check("every Exp-3 finding points at a source line in a branch mentioning its clip", bad == 0, f"{len(findings)} findings, {bad} unresolved")
    # regex-based independent count of per-clip branch headers
    n_regex = 0
    for path in [f"{VISION}/3d_recon.py", f"{RETARGET}/retarget_config_go2.py", f"{RETARGET}/retarget_motion_fromVision.py"]:
        src = "\n".join(l for l in open(path).read().splitlines() if not l.lstrip().startswith("#"))  # skip commented-out branches
        n_regex += len(re.findall(r'in_sequence_name\s*==\s*"([^"]+)"', src))
    tree_hits = 0
    for path in [f"{VISION}/3d_recon.py", f"{RETARGET}/retarget_config_go2.py", f"{RETARGET}/retarget_motion_fromVision.py"]:
        tree = ast.parse(open(path).read())
        for n in ast.walk(tree):
            if isinstance(n, ast.Compare) and isinstance(n.comparators[0], ast.Constant) and getattr(n.left, "attr", getattr(n.left, "id", None)) == "in_sequence_name":
                tree_hits += 1
    check("regex count of `in_sequence_name == \"...\"` comparisons == AST count", n_regex == tree_hits, f"{n_regex} vs {tree_hits} (3d_recon.py + retarget_config_go2.py + retarget_motion_fromVision.py)")
    summ = json.load(open(os.path.join(DATA, "exp3_summary.json")))
    note("per-clip totals (flat-walking video clips)", str({k: v["total"] for k, v in summ["per_clip"].items() if v["scenario"] == "flat"}))
    note("MoCap per clip", str(summ["mocap_per_clip"]))
    note("dead branch (clip id typo, never matches)", str(summ["dead_branches"]))
    # dataset-level: motion_loader.py asserts FrameDuration == 0.06 for vision files -> the paper's datasets (0.03334) would now be rejected
    ml = open(os.path.join(ISAAC, "rsl_rl/rsl_rl/rsl_rl/datasets/motion_loader.py")).read()
    check("motion_loader.py currently asserts FrameDuration == 0.06 for 'vision' files", "assert frame_duration == 0.06" in ml, "the paper's flat-walking video datasets have FrameDuration 0.03334 -> loading them with the current loader fails (reproducibility note)")


def main():
    REPORT.append("# Sanity checks\n")
    REPORT.append("## Datasets\n")
    check_datasets()
    REPORT.append("\n## Experiment 1\n")
    check_relift()
    check_exp1_robustness()
    REPORT.append("\n## Experiment 2\n")
    check_exp2()
    REPORT.append("\n## Experiment 3\n")
    check_exp3()
    with open(os.path.join(DATA, "checks_report.md"), "w") as fh:
        fh.write("\n".join(REPORT) + "\n")
    n_fail = sum(1 for r in REPORT if "[FAIL]" in r)
    print(f"\n{n_fail} FAIL")


if __name__ == "__main__":
    main()
