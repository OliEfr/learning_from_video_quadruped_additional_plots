"""Does the post-processing the paper DESCRIBES reproduce the hand-typed deletions?

Section III of the paper states two filtering steps that do not exist anywhere in
the code: (i) discard depth values differing by more than 0.5 m from the previous
value for that keypoint, and (ii) a moving average over the last three frames.
What the code does instead is linear interpolation of zero-valued keypoints plus
a block of HAND-TYPED per-clip frame ranges in 3d_recon.py:279-298 (e.g.
"markers_3d[29:33,:,:] = 0.0  # There is some noise here in the depth so we
remove it and let interpolate").

This script reconstructs the raw per-keypoint depth series exactly as 3d_recon.py
does (2D tracks + the uint16 .raw depth maps, read-only), applies the described
filter, and measures how much of the hand-typed deletion it recovers. If the
recovery is good, the paper's method text becomes accurate AND the per-clip
hand-coding can be deleted.
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt
import common as C

QFV = C.QFV
THRESH_M = 0.5          # the paper's stated depth-jump rejection threshold
MA = 3                  # the paper's stated moving-average length

# hand-typed deletions, transcribed from 3d_recon.py:279-298
# marker order in markers_3d is [hind hip, front hip, HR, HL, VR, VL]
HAND = {
    "turn_left_1771233000":      [(slice(0, 4), slice(0, 2))],
    "slow_turn_1771233000":      [(slice(29, 33), slice(None)), (slice(40, 42), slice(5, 6))],
    "left_right_turn_2058226999": [(slice(22, 28), slice(0, 1)), (slice(15, 21), slice(5, 6)),
                                   (slice(33, 39), slice(5, 6))],
    "obstacle_2_3126098000":     [(slice(21, 23), slice(5, 6))],
    "stand_up_2431270000":       [(slice(10, -10), slice(4, 6))],
    "box_1_399682000":           [(slice(20, None), slice(4, 6))],
}

def raw_depth_series(clip):
    """-> (T, 6) depth in metres at the tracked pixels, before any filtering."""
    dd = f"{QFV}/in/{clip}/depth_cam"
    files = sorted(f for f in os.listdir(dd) if f.endswith(".raw"))
    foot = C.load_occlusion_mask.__wrapped__ if False else None
    import os as _o
    _cand = [f"{QFV}/tracks/{clip}/foot_pos.npy",
             f"{QFV}/tracks/{clip}/foot_pos_for_depth_cam.npy"]
    fp = np.load(next(c for c in _cand if _o.path.exists(c)))     # (T,4,2)
    hp = np.load(f"{QFV}/tracks/{clip}/hip_pos.npy")             # (1,T,2,2)
    hp = hp[0] if hp.ndim == 4 else hp
    T = min(len(fp), len(hp), len(files))
    markers = np.concatenate([hp[:T], fp[:T]], axis=1).astype(int)  # (T,6,2) x,y
    out = np.zeros((T, 6), np.float32)
    for t in range(T):
        d = np.fromfile(os.path.join(dd, files[t]), dtype=np.uint16)
        d = d.reshape(360, 640).astype(np.float32) / 1000.0
        for k in range(6):
            x, y = markers[t, k]
            if x == 0 and y == 0:
                continue                       # already flagged as not visible
            out[t, k] = d[y, x]                # same indexing as 3d_recon.py
    return out, markers

def paper_filter(depth):
    """The filter Section III describes. Returns (filtered, rejected_mask)."""
    d = depth.copy()
    rej = np.zeros_like(d, bool)
    for k in range(d.shape[1]):
        prev = None
        for t in range(len(d)):
            if d[t, k] == 0.0:
                continue
            if prev is not None and abs(d[t, k] - prev) > THRESH_M:
                rej[t, k] = True
            else:
                prev = d[t, k]
    out = d.copy(); out[rej] = 0.0
    # linear interpolation of the holes (as the pipeline already does)
    for k in range(out.shape[1]):
        col = out[:, k]; ok = col != 0.0
        if ok.sum() >= 2:
            col[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), col[ok])
        out[:, k] = col
    # moving average over the last MA frames
    ker = np.ones(MA) / MA
    for k in range(out.shape[1]):
        out[:, k] = np.convolve(out[:, k], ker, mode="same")
    return out, rej

def hand_mask(clip, T):
    m = np.zeros((T, 6), bool)
    for fs, ms in HAND.get(clip, []):
        m[fs, ms] = True
    return m

# --------------------------------------------------------------- intrinsics
# camera_info/camera.yaml stores [748.0999, 634.5353, 747.5494, 370.8257]. The
# values pattern-match [fx, cx, fy, cy] for the FULL 1280x720 frame (cx~640,
# cy~360). 3d_recon.py:210 unpacks them as "FX, FY, CX, CY" and applies them
# unscaled to 640x360 pixel coordinates, so (a) CX receives fy's value and
# (b) nothing is halved for the half-size images. The result is CX = 747.55 on a
# 640-wide image: (x - CX) is negative everywhere and x_cam collapses onto the
# depth axis, so depth noise leaks straight into the horizontal coordinate.
INTR_ASIS = (748.0999, 634.5353, 747.5494, 370.8257)          # FX, FY, CX, CY as used
INTR_FIX = (748.0999 / 2, 747.5494 / 2, 634.5353 / 2, 370.8257 / 2)   # fx, fy, cx, cy

def _lift(px, d, FX, FY, CX, CY):
    return np.stack([(px[..., 0] - CX) * d / FX, (px[..., 1] - CY) * d / FY, d], -1)

def _interp0(a):
    a = a.copy()
    for k in range(a.shape[1]):
        c = a[:, k]; ok = c != 0
        if ok.sum() >= 2:
            c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok]); a[:, k] = c
    return a

def intrinsics_effect(clips):
    """Like-for-like camera-frame test of the intrinsics fix. No SLAM / ground
    alignment is applied, so absolute CVs are not the shipped pipeline's -- the
    relative change between the two columns is the result."""
    out = {}
    for clip in clips:
        d, px = raw_depth_series(clip)
        d = _interp0(d)
        r = {}
        for tag, I in (("as_is", INTR_ASIS), ("fixed", INTR_FIX)):
            P = _lift(px.astype(float), d, *I)
            L = np.linalg.norm(P[:, 1] - P[:, 0], axis=-1)
            r[tag] = dict(cv=float(100 * L.std() / np.median(L)), L=float(np.median(L)))
        out[clip] = r
    return out


def main():
    rows, detail = [], {}
    for clip in HAND:
        if not os.path.isdir(f"{QFV}/in/{clip}/depth_cam"):
            continue
        depth, _ = raw_depth_series(clip)
        filt, rej = paper_filter(depth)
        hm = hand_mask(clip, len(depth))
        occl = depth == 0.0
        # only samples that actually carry a depth reading can be judged
        valid = ~occl
        tp = int((rej & hm & valid).sum()); fn = int((~rej & hm & valid).sum())
        fp = int((rej & ~hm & valid).sum())
        recall = tp / max(1, tp + fn)
        detail[clip] = dict(T=len(depth), hand=int((hm & valid).sum()),
                            auto=int((rej & valid).sum()), tp=tp, fn=fn, fp=fp,
                            recall=recall,
                            jump_p99=float(np.percentile(
                                np.abs(np.diff(np.where(depth > 0, depth, np.nan), axis=0))
                                [~np.isnan(np.abs(np.diff(np.where(depth > 0, depth, np.nan), axis=0)))], 99)))
        rows.append((clip, detail[clip]))
        print(f"{clip:28s} T={len(depth):3d} hand-deleted={detail[clip]['hand']:4d} "
              f"auto-rejected={detail[clip]['auto']:4d} recall={recall*100:5.1f}% "
              f"extra={fp:4d}")

    tot_hand = sum(d["hand"] for _, d in rows)
    tot_tp = sum(d["tp"] for _, d in rows)
    tot_auto = sum(d["auto"] for _, d in rows)
    print(f"\nOVERALL recall of the hand-typed deletions: {100*tot_tp/max(1,tot_hand):.1f}% "
          f"({tot_tp}/{tot_hand}); automatic filter flags {tot_auto} samples in total")

    # figure: one clip's depth series before/after, plus the recall summary
    clip = "slow_turn_1771233000"
    depth, _ = raw_depth_series(clip)
    filt, rej = paper_filter(depth)
    hm = hand_mask(clip, len(depth))
    intr = intrinsics_effect(list(HAND))
    fig = plt.figure(figsize=(C.IEEE_2COL, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.95, 0.95], wspace=0.60)
    ax = fig.add_subplot(gs[0])
    t = np.arange(len(depth))
    k = 5
    d = np.where(depth[:, k] > 0, depth[:, k], np.nan)
    ax.plot(t, d, color="0.6", lw=1.0, label="raw depth")
    ax.plot(t, filt[:, k], color=C.C_VIDEO, lw=1.3, label="paper filter (0.5 m + 3-frame MA)")
    ax.scatter(t[rej[:, k]], d[rej[:, k]], s=26, facecolors="none",
               edgecolors="red", lw=1.0, label="auto-rejected", zorder=5)
    ax.scatter(t[hm[:, k]], np.full(hm[:, k].sum(), np.nanmin(d)), s=14, marker="|",
               color="k", label="hand-deleted range", zorder=4)
    ax.set_xlabel("frame", fontsize=8); ax.set_ylabel("depth [m]", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.3)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    ax.set_title(f"(a) {clip}, keypoint VL", fontsize=8)

    ax = fig.add_subplot(gs[1])
    names = [c.rsplit("_", 1)[0][:16] for c, _ in rows]
    rec = [100 * d["recall"] for _, d in rows]
    ax.barh(range(len(rows)), rec, color=C.C_VIDEO, height=.7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=5.4)
    ax.set_xlabel("% of hand-deleted samples\nrecovered automatically", fontsize=7.5)
    ax.set_xlim(0, 100); ax.tick_params(labelsize=6.5); ax.grid(alpha=.3, axis="x")
    ax.set_title("(b) recall per clip", fontsize=8)

    ax = fig.add_subplot(gs[2])
    a = [v["as_is"]["cv"] for v in intr.values()]
    b = [v["fixed"]["cv"] for v in intr.values()]
    for i, (x, y) in enumerate(zip(a, b)):
        ax.plot([0, 1], [x, y], color="0.7", lw=0.8, zorder=1)
    ax.scatter(np.zeros(len(a)), a, s=16, color="#d62728", zorder=3, lw=0)
    ax.scatter(np.ones(len(b)), b, s=16, color=C.C_VIDEO, zorder=3, lw=0)
    ax.set_yscale("log"); ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["as-is", "intrinsics\nfixed"], fontsize=7)
    ax.set_ylabel("torso-length CV [%]", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=.3, axis="y")
    ax.text(.5, .03, f"median {np.median(a):.0f}% $\\rightarrow$ {np.median(b):.0f}%",
            transform=ax.transAxes, ha="center", fontsize=6.6, fontweight="bold")
    ax.set_title("(c) camera-calibration bug", fontsize=8)
    C.save(fig, "fig_H_filter_vs_handcoding.pdf")

    print("\nintrinsics fix (camera-frame, like-for-like):")
    for k, v in intr.items():
        print(f"  {k:28s} CV {v['as_is']['cv']:6.1f}% -> {v['fixed']['cv']:6.1f}%   "
              f"L {v['as_is']['L']:.3f} -> {v['fixed']['L']:.3f} m")
    print(f"  MEDIAN CV {np.median([v['as_is']['cv'] for v in intr.values()]):.1f}% -> "
          f"{np.median([v['fixed']['cv'] for v in intr.values()]):.1f}%   "
          f"L {np.median([v['as_is']['L'] for v in intr.values()]):.3f} -> "
          f"{np.median([v['fixed']['L'] for v in intr.values()]):.3f} m "
          f"(MoCap real dog: 0.408 m)")
    json.dump(dict(per_clip=detail, overall_recall=100 * tot_tp / max(1, tot_hand),
                   total_hand=tot_hand, total_auto=tot_auto, intrinsics=intr,
                   threshold_sweep={str(t): None for t in []}),
              open(os.path.join(C.OUT, "_tmp", "exp4.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    main()
