"""Experiment 5: build the video-only ablation expert sets and verify their command coverage.

Arms (all from datasets/fromVision_motions_DepthCam_extendedWithoutReverse, the paper's Video (extended) set):
  E1  extHalf      : all 8 clips, each trimmed to ONE contiguous window of ~50 % of its frames; window starts are chosen
                     jointly (greedy coordinate ascent) to retain as many covered evaluation cells as possible.
                     -> same coverage as Video (extended), half the amount (~6 s = the amount of Video).
  E2  extAddedOnly : only the 4 clips that the extension added (slow turn, L-R turn, 2x start-stop).
                     -> same amount as Video (6.2 s), different coverage.
  E3  extNoTurn    : Video (extended) minus the two added turning clips (slow turn, L-R turn).
  E4  extNoStartStop: Video (extended) minus the two start-stop clips.
Header fields (FrameDuration, LoopMode, MotionWeight, cycle offsets) are copied unchanged; only "Frames" is cut.
Coverage is computed with the Exp.-2 code and tolerances (exp2_coverage.py).
Usage: python exp5_build_video_subsets.py --out <IsaacLab>/datasets  [--frac 0.5]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import EVAL_GRID, VIDEO_BASE_CLIPS, VIDEO_EXT_ONLY_CLIPS, load_amp_dir, md_table  # noqa: E402
import exp2_coverage as e2  # noqa: E402

SRC = common.AMP_DIRS[common.NAME_VIDEO_EXT]
ARMS = {
    "fromVision_motions_DepthCam_extHalf": None,  # built by windowing
    "fromVision_motions_DepthCam_extAddedOnly": VIDEO_EXT_ONLY_CLIPS,
    "fromVision_motions_DepthCam_extNoTurn": [c for c in VIDEO_BASE_CLIPS + VIDEO_EXT_ONLY_CLIPS if c not in ("slow_turn_1771233000", "left_right_turn_2058226999")],
    "fromVision_motions_DepthCam_extNoStartStop": [c for c in VIDEO_BASE_CLIPS + VIDEO_EXT_ONLY_CLIPS if not c.startswith("start_stop")],
}


def frames_of(clips):
    """Same per-frame quantities as exp2_coverage.dataset_frames, but from an in-memory clip list."""
    common.AMP_DIRS["__tmp__"] = None
    keys = ["vx", "vy", "wz", "w_amp", "v_speed"]
    out = {k: [] for k in keys}
    total_w = sum(c["weight"] for c in clips)
    for c in clips:
        F, fd = c["frames"], c["fd"]
        T = len(F)
        yaw = np.unwrap(common.yaw_from_quat_xyzw(F[:, 3:7]))
        pos_s = F[:, :3] - common.sg_residual(F[:, :3], e2.SMOOTH_WINDOW, e2.SMOOTH_POLY)
        yaw_s = yaw - common.sg_residual(yaw[:, None], e2.SMOOTH_WINDOW, e2.SMOOTH_POLY)[:, 0]
        v_ws = np.gradient(pos_s, axis=0) / fd
        v_bs = np.stack([common.rot_z(-y) @ v for y, v in zip(yaw_s, v_ws)])
        out["vx"].append(v_bs[:, 0]); out["vy"].append(v_bs[:, 1]); out["wz"].append(np.gradient(yaw_s) / fd)
        out["w_amp"].append(np.full(T, c["weight"] / total_w / T))
        out["v_speed"].append(np.linalg.norm(v_bs[:, :2], axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(clips, box_n=40000):
    d = frames_of(clips)
    cov_xy = e2.covered_cells_xy(d, *EVAL_GRID["xy"])
    cov_xyaw = e2.covered_cells_xyaw(d, *EVAL_GRID["xyaw"])
    standing = float(np.sum(d["w_amp"] * ((d["v_speed"] < 0.1) & (np.abs(d["wz"]) < 0.2))))
    turning = float(np.sum(d["w_amp"] * (np.abs(d["wz"]) > 0.5)))
    fast = float(np.sum(d["w_amp"] * (d["vx"] > 1.0)))
    n = sum(len(c["frames"]) for c in clips)
    dur = sum((len(c["frames"]) - 1) * c["fd"] for c in clips)
    return dict(clips=len(clips), frames=n, dur=dur, xy=int(cov_xy.sum()), xyaw=int(cov_xyaw.sum()),
                box=e2.box_coverage(d, n=box_n), standing=standing, turning=turning, fast=fast,
                cov_xy=cov_xy, cov_xyaw=cov_xyaw)


def cut(clip, start, L):
    c = dict(clip); c["frames"] = clip["frames"][start:start + L]; c["start"] = start; c["L"] = L
    return c


def choose_windows(clips, frac, ref, sweeps=6, restarts=8, seed=0):
    """Greedy coordinate ascent over window starts with random restarts.
    Score = sum over {vx-vy cells, vx-wz cells, 3-D box coverage, standing mass} of min(1, value / reference value),
    i.e. the retained fraction of each coverage measure of the full set (reference = Video (extended))."""
    rng = np.random.default_rng(seed)
    Ls = [max(7, int(round(frac * len(c["frames"])))) for c in clips]

    def score(st):
        r = summarize([cut(c, s, L) for c, s, L in zip(clips, st, Ls)], box_n=20000)
        parts = [r["xy"] / ref["xy"], r["xyaw"] / ref["xyaw"], r["box"] / ref["box"], r["standing"] / max(ref["standing"], 1e-9)]
        return float(sum(min(1.0, p) for p in parts)), r

    best = None
    for k in range(restarts):
        starts = [0 if k == 0 else int(rng.integers(0, len(c["frames"]) - L + 1)) for c, L in zip(clips, Ls)]
        cur, _ = score(starts)
        for _ in range(sweeps):
            improved = False
            for i, c in enumerate(clips):
                for s in range(0, len(c["frames"]) - Ls[i] + 1):
                    if s == starts[i]:
                        continue
                    trial = list(starts); trial[i] = s
                    sc, _ = score(trial)
                    if sc > cur + 1e-9:
                        cur, starts, improved = sc, trial, True
            if not improved:
                break
        print(f"restart {k}: score {cur:.3f} starts {starts}", flush=True)
        if best is None or cur > best[0]:
            best = (cur, list(starts))
    return [cut(c, s, L) for c, s, L in zip(clips, best[1], Ls)]


# E1 window starts as found by choose_windows(frac=0.5, sweeps=6, restarts=8) on 2026-09-11 (best restart 4, score 3.675; the
# search is deterministic, log in data/exp5_build_stdout.txt). Recorded so the trained set is reproduced without the ~3-min search.
E1_STARTS = {"left_right_turn_2058226999": 14, "slow_1313807000": 23, "slow_turn_1771233000": 15, "start_stop_1271493000": 13,
             "start_stop_785558000": 22, "turn_left_1771233000": 2, "turn_right_1771233000": 22, "walk_869488000": 3}

# E1b "coverage-matched half": TWO contiguous windows of 25 % length per clip, each written as its own file with half the
# clip's MotionWeight (so the per-clip AMP sampling mass is unchanged and no artificial junction enters the discriminator's
# transitions). Window starts found by choose_two_windows() (coordinate ascent on the honest per-file coverage score, seed 1,
# 4 restarts) on 2026-09-12; recorded here so that the set is reproducible without the 3-min search.
E1B_STARTS = {"left_right_turn_2058226999": (17, 48), "slow_1313807000": (16, 44), "slow_turn_1771233000": (20, 32),
              "start_stop_1271493000": (0, 12), "start_stop_785558000": (0, 16), "turn_left_1771233000": (2, 14),
              "turn_right_1771233000": (20, 32), "walk_869488000": (3, 14)}


def two_window_set(clips, starts, frac=0.25):
    """16 clips: for every source clip two windows of frac*len frames (>= 4), weight halved, distinct output file names."""
    out = []
    for c in clips:
        L = max(4, int(round(frac * len(c["frames"]))))
        for tag, s in zip("ab", starts[c["name"]]):
            d = cut(c, s, L); d["weight"] = c["weight"] / 2
            d["out_name"] = os.path.basename(c["path"]).replace(".txt", f"_{tag}{s}.txt")
            out.append(d)
    return out


def choose_two_windows(clips, ref, frac=0.25, sweeps=4, restarts=4, seed=1):
    """Coordinate ascent over the two window starts per clip; score on the 16-file set exactly as the coverage table sees it."""
    rng = np.random.default_rng(seed)
    Ls = [max(4, int(round(frac * len(c["frames"])))) for c in clips]

    def score(st):
        r = summarize(two_window_set(clips, {c["name"]: st[i] for i, c in enumerate(clips)}, frac), box_n=20000)
        return r["box"] / ref["box"] + 0.5 * r["xyaw"] / ref["xyaw"] + 0.5 * r["xy"] / ref["xy"] + 0.5 * min(1, r["standing"] / max(ref["standing"], 1e-9))

    best = None
    for k in range(restarts):
        st = []
        for c, L in zip(clips, Ls):
            n = len(c["frames"])
            a = 0 if k == 0 else int(rng.integers(0, n - 2 * L + 1))
            b = n - L if k == 0 else int(rng.integers(a + L, n - L + 1))
            st.append((a, b))
        cur = score(st)
        for _ in range(sweeps):
            improved = False
            for i, c in enumerate(clips):
                n, L = len(c["frames"]), Ls[i]
                for w in (0, 1):
                    for s in range(0, n - L + 1, 2):
                        a, b = st[i]; lo, hi = sorted((s, b) if w == 0 else (a, s))
                        if hi < lo + L:
                            continue
                        trial = list(st); trial[i] = (lo, hi); sc = score(trial)
                        if sc > cur + 1e-9:
                            cur, st, improved = sc, trial, True
            if not improved:
                break
        print(f"two-window restart {k}: score {cur:.3f} starts {st}", flush=True)
        if best is None or cur > best[0]:
            best = (cur, list(st))
    return {c["name"]: best[1][i] for i, c in enumerate(clips)}


def write_set(folder, clips):
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    for c in clips:
        with open(c["path"]) as fh:
            j = json.load(fh)
        j["Frames"] = np.asarray(c["frames"], dtype=float).tolist()
        j["MotionWeight"] = float(c["weight"])
        with open(os.path.join(folder, c.get("out_name", os.path.basename(c["path"]))), "w") as fh:
            json.dump(j, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="IsaacLab datasets/ folder to write the new sets into")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--e1-search", action="store_true", help="re-run the one-window search instead of using the recorded E1_STARTS")
    ap.add_argument("--e1b-search", action="store_true", help="re-run the two-window search instead of using the recorded E1B_STARTS")
    args = ap.parse_args()

    src = load_amp_dir(SRC)
    by_name = {c["name"]: c for c in src}
    rows, report = [], {}

    def add_row(label, clips, extra=""):
        r = summarize(clips)
        rows.append([label, r["clips"], r["frames"], f"{r['dur']:.2f}", f"{r['xy']}/147", f"{r['xyaw']}/441", f"{100*r['box']:.1f}",
                     f"{100*r['standing']:.1f}", f"{100*r['turning']:.1f}", f"{100*r['fast']:.1f}", extra])
        report[label] = {k: v for k, v in r.items() if not k.startswith("cov_")}
        return r

    # references (paper sets)
    add_row("Video (paper)", load_amp_dir(common.AMP_DIRS[common.NAME_VIDEO]))
    r_ext = add_row("Video (extended) (paper)", src)

    # E1
    if args.e1_search:
        e1 = choose_windows(src, args.frac, r_ext, sweeps=args.sweeps, restarts=args.restarts)
    else:
        e1 = [cut(c, E1_STARTS[c["name"]], max(7, int(round(args.frac * len(c["frames"]))))) for c in src]
    windows = {c["name"]: (c["start"], c["start"] + c["L"], len(by_name[c["name"]]["frames"])) for c in e1}
    r_e1 = add_row("E1 extHalf", e1, "windows: " + ", ".join(f"{k} [{a}:{b}) of {n}" for k, (a, b, n) in windows.items()))
    report["E1 windows"] = windows
    report["E1 retained cells"] = dict(xy=int((r_e1["cov_xy"] & r_ext["cov_xy"]).sum()), xyaw=int((r_e1["cov_xyaw"] & r_ext["cov_xyaw"]).sum()))
    write_set(os.path.join(args.out, "fromVision_motions_DepthCam_extHalf"), e1)
    # also a naive first-half variant for the record (not written)
    add_row("(E1 alt: first half of every clip, not used)", [cut(c, 0, max(7, int(round(args.frac * len(c["frames"]))))) for c in src])

    # E1b: coverage-matched half (two windows per clip, 16 files)
    starts = choose_two_windows(src, r_ext) if args.e1b_search else E1B_STARTS
    e1b = two_window_set(src, starts)
    r_e1b = add_row("E1b extHalfCov", e1b, "two windows per clip, weight halved: " + ", ".join(
        f"{k} [{a}:{a + max(4, int(round(0.25 * len(by_name[k]['frames']))))}) + [{b}:{b + max(4, int(round(0.25 * len(by_name[k]['frames']))))}) of {len(by_name[k]['frames'])}"
        for k, (a, b) in starts.items()))
    report["E1b windows"] = starts
    report["E1b retained cells"] = dict(xy=int((r_e1b["cov_xy"] & r_ext["cov_xy"]).sum()), xyaw=int((r_e1b["cov_xyaw"] & r_ext["cov_xyaw"]).sum()))
    write_set(os.path.join(args.out, "fromVision_motions_DepthCam_extHalfCov"), e1b)
    # for the record: uniform every-2nd-frame subsampling keeps the trajectory but halves the frames -> the box measure drops
    add_row("(every 2nd frame of every clip, not used)", [dict(c, frames=c["frames"][::2], fd=c["fd"] * 2) for c in src])

    # E2-E4
    for folder, names in ARMS.items():
        if names is None:
            continue
        clips = [by_name[n] for n in names]
        label = {"fromVision_motions_DepthCam_extAddedOnly": "E2 extAddedOnly", "fromVision_motions_DepthCam_extNoTurn": "E3 extNoTurn",
                 "fromVision_motions_DepthCam_extNoStartStop": "E4 extNoStartStop"}[folder]
        add_row(label, clips)
        write_set(os.path.join(args.out, folder), clips)

    hdr = ["set", "clips", "frames", "dur [s]", "vx-vy cells", "vx-wz cells", "box cov [%]", "standing [%]", "turning [%]", "vx>1 [%]", "note"]
    table = md_table(hdr, rows)
    print(table)
    with open(os.path.join(common.DATA, "exp5_coverage.md"), "w") as fh:
        fh.write("# Exp. 5 ablation sets: command coverage (Exp.-2 rule and tolerances)\n\n" + table + "\n")
    with open(os.path.join(common.DATA, "exp5_sets.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
