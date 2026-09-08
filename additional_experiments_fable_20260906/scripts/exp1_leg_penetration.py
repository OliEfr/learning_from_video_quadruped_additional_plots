"""Leg penetration per reconstruction source (author's follow-up to the "overlapping limbs" statistic).

The first pass counted frames with two paws closer than 0.10 torso lengths on the on-disk keypoints (as-coded
intrinsics). Two follow-ups here:

1. Threshold-free lateral-crossing test (negative control, kept for the record): a L/R paw pair is "crossed"
   when its lateral order in the body frame is inverted relative to the clip's dominant order. MoCap fails
   this test (34 % of frames, 49 % in the turning windows): a trotting dog places its paws close to the
   midline (median L-R offset 0.07-0.13 L) and in turns the pelvis-neck axis bends, so a lateral inversion
   is not a penetration. With six point keypoints there is no leg volume to test directly.

2. Data-set threshold instead of a hand-picked one: the smallest pairwise paw distance MoCap ever reaches
   (over the five full source clips, 60 Hz, thousands of frames) is the physical floor of a real dog; a
   reconstruction frame whose closest paw pair lies below that floor puts two paws where one dog cannot.
   Reported in torso lengths L (median pelvis-neck distance per clip), so the AnimalAvatar scale error does
   not matter. Camera depth is evaluated both as used (on-disk .npy, as-coded intrinsics) and re-lifted with
   the corrected intrinsics (data/relift_half_res_*.npz, camera-depth path only). Also reports the old
   0.10 L criterion for both.

Output: data/exp1_leg_penetration.json, stdout -> data/exp1_leg_penetration_stdout.txt.
"""
import json, os
from itertools import combinations
import numpy as np
from common import (DATA, RETARGET, MOCAP_SEGMENTS,
                    VIDEO_BASE_CLIPS, FLAT_VIDEO_CLIPS, load_mocap_raw, load_mocap_segment, mocap_keypoints)

MOCAP_FULL = ["dog_walk00", "dog_walk03", "dog_run04", "dog_run00", "dog_walk09"]
OLD_THR = 0.10
PAIRS = {"rear": (2, 3), "front": (4, 5)}  # keypoint order [rear hip, front hip, RR, RL, FR, FL]
Z_LIFT = 0.25


def load_disk(clip, folder, prefix=""):
    d = os.path.join(RETARGET, folder, prefix + clip)
    base, feet = np.load(os.path.join(d, "base.npy")), np.load(os.path.join(d, "feet.npy"))
    return np.concatenate([base, feet], axis=0).transpose(1, 0, 2)  # (T, 6, 3), z up


def load_relift(clip):
    return np.load(os.path.join(DATA, f"relift_half_res_{clip}.npz"))["markers_ground"].transpose(1, 0, 2)


def mocap_kp(arr):
    body, feet = mocap_keypoints(arr)
    return np.concatenate([body, feet[[1, 3, 0, 2]]], axis=0).transpose(1, 0, 2)  # feet -> [RR, RL, FR, FL]


def torso_L(kp):
    return float(np.median(np.linalg.norm(kp[:, 1] - kp[:, 0], axis=-1)))


def min_pair_dist(kp):
    """Per frame: smallest of the 6 pairwise paw distances, in torso lengths."""
    f = kp[:, 2:6]
    d = np.stack([np.linalg.norm(f[:, i] - f[:, j], axis=-1) for i, j in combinations(range(4), 2)], axis=1)
    return d.min(axis=1) / torso_L(kp)


def crossing(kp):
    rear, front = kp[:, 0], kp[:, 1]
    L = torso_L(kp)
    fwd = front - rear
    fwd[:, 2] = 0.0
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-12
    left = np.stack([-fwd[:, 1], fwd[:, 0], np.zeros(len(fwd))], axis=-1)
    any_pen = np.zeros(len(kp), bool)
    for a, b in PAIRS.values():
        d = np.einsum("ij,ij->i", kp[:, a] - kp[:, b], left)
        inverted = d * np.sign(np.median(d)) < 0
        lifted = np.abs(kp[:, a, 2] - kp[:, b, 2]) > Z_LIFT * L
        any_pen |= inverted & ~lifted
    return float(any_pen.mean())


def evaluate(clips, floor):
    per = {}
    for name, kp in clips.items():
        m = min_pair_dist(kp)
        per[name] = dict(n=int(len(m)), below_floor=float((m < floor).mean()), below_old=float((m < OLD_THR).mean()),
                         min_L=float(m.min()), p5_L=float(np.percentile(m, 5)), crossing=crossing(kp))
    n = np.array([p["n"] for p in per.values()])
    pooled = lambda k: float(100 * sum(p[k] * p["n"] for p in per.values()) / n.sum())
    worst = lambda k: max(per, key=lambda c: per[c][k])
    return dict(n_clips=len(per), n_frames=int(n.sum()),
                below_floor_pooled_pct=pooled("below_floor"), below_floor_max_clip_pct=float(100 * max(p["below_floor"] for p in per.values())),
                below_floor_worst_clip=worst("below_floor"),
                below_old_pooled_pct=pooled("below_old"), below_old_max_clip_pct=float(100 * max(p["below_old"] for p in per.values())),
                min_L=float(min(p["min_L"] for p in per.values())), crossing_pooled_pct=pooled("crossing"), per_clip=per)


def main():
    mocap_full = {c: mocap_kp(load_mocap_raw(c)) for c in MOCAP_FULL}
    m_all = np.concatenate([min_pair_dist(kp) for kp in mocap_full.values()])
    floor = float(m_all.min())
    print(f"MoCap physical floor: smallest pairwise paw distance over {len(m_all)} frames of {len(MOCAP_FULL)} full clips "
          f"= {floor:.3f} L (p0.1 {np.percentile(m_all, 0.1):.3f}, p1 {np.percentile(m_all, 1):.3f}, median {np.median(m_all):.3f}); "
          f"old criterion {OLD_THR} L\n")
    sources = {
        "MoCap (5 full clips)": mocap_full,
        "MoCap (6 windows)": {s[0]: mocap_kp(load_mocap_segment(s)) for s in MOCAP_SEGMENTS},
        "Camera depth, as used (4)": {c: load_disk(c, "data_fromVision_depth_cam") for c in VIDEO_BASE_CLIPS},
        "Camera depth, corrected intr. (4)": {c: load_relift(c) for c in VIDEO_BASE_CLIPS},
        "Camera depth, as used (8)": {c: load_disk(c, "data_fromVision_depth_cam") for c in FLAT_VIDEO_CLIPS},
        "Camera depth, corrected intr. (8)": {c: load_relift(c) for c in FLAT_VIDEO_CLIPS},
        "DepthAnythingV2 (4)": {c: load_disk(c, "data_fromVision_aligned_video_depth_anything") for c in VIDEO_BASE_CLIPS},
        "AnimalAvatar (4)": {c: load_disk(c, "data_AnimalAvatar", "experiments_maila_") for c in VIDEO_BASE_CLIPS},
    }
    res = {"mocap_floor_L": floor, "old_threshold_L": OLD_THR, "sources": {k: evaluate(v, floor) for k, v in sources.items()}}
    with open(os.path.join(DATA, "exp1_leg_penetration.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'source':36s} {'<floor pooled%':>14s} {'<floor max clip%':>16s} {'<0.10L pooled%':>14s} {'<0.10L max%':>11s} {'min L':>6s} {'crossing%':>10s}  worst clip (<floor)")
    for k, v in res["sources"].items():
        print(f"{k:36s} {v['below_floor_pooled_pct']:14.1f} {v['below_floor_max_clip_pct']:16.1f} {v['below_old_pooled_pct']:14.1f} "
              f"{v['below_old_max_clip_pct']:11.1f} {v['min_L']:6.3f} {v['crossing_pooled_pct']:10.1f}  {v['below_floor_worst_clip']} (n={v['n_frames']})")
    print("\nper clip, % frames below MoCap floor:")
    for k, v in res["sources"].items():
        if "MoCap" in k:
            continue
        print(f"  {k:34s}", {c: round(100 * p["below_floor"], 1) for c, p in v["per_clip"].items()})


if __name__ == "__main__":
    main()
