"""Experiment 1 -- How noisy are keypoint trajectories compared to MoCap?

All windowed metrics are computed on FIXED-LENGTH windows (identical spectral
resolution and leakage for both sources) and compared only over a matched base-
speed band, so the comparison is sensor-vs-sensor rather than behaviour-vs-
behaviour. MoCap source clips contain long stationary stretches and fast running
that the casual video clips do not; without speed matching every windowed metric
measures gait, not noise.
"""
import json, os, itertools
import numpy as np
from scipy.signal import savgol_filter, detrend
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import common as C

W = 16                       # 0.533 s at 30 Hz; identical for both sources
HOP = 8
FC = C.FS_COMMON / 4         # 7.5 Hz -- upper half of the common analysis band

# ------------------------------------------------------------- window metrics
def base_speed(kp, fs):
    L = C.torso_length(kp)
    b = 0.5 * (kp[:, 0] + kp[:, 1])
    v = np.linalg.norm(np.diff(b, axis=0), axis=-1) * fs / L
    return np.concatenate([v[:1], v])

def win_cv_torso(seg):
    d = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=-1)
    return 100.0 * d.std() / np.median(d)

def win_sg_resid(seg, L):
    x = seg / L
    sm = savgol_filter(x, 5, 2, axis=0)
    r = np.linalg.norm(x - sm, axis=-1)[2:-2]
    return 100.0 * float(np.sqrt((r ** 2).mean()))

def win_hfr(seg, L):
    x = (seg / L).reshape(len(seg), -1)
    w = np.hanning(len(x))[:, None]
    P = (np.abs(np.fft.rfft(detrend(x, axis=0) * w, axis=0)) ** 2)[1:]
    f = np.fft.rfftfreq(len(x), 1.0 / C.FS_COMMON)[1:]
    return 100.0 * float(P[f >= FC].sum() / P.sum())

def win_jerk(seg, L):
    dt = 1.0 / C.FS_COMMON
    j = np.diff(seg / L, n=3, axis=0) / dt ** 3
    D = len(seg) * dt
    return float(np.sqrt(0.5 * (np.linalg.norm(j, axis=-1) ** 2).sum() * dt * D ** 5))

KEYS = ["M1_CV_torso_pct", "M2_SG_resid_pct", "M3_HFR_pct", "M4_norm_jerk"]


def foot_plausibility(kp):
    """Geometric plausibility of the reconstruction, independent of jitter.

    A parametric model (AnimalAvatar/SMAL) is smooth by construction, so the
    jitter metrics cannot detect its failure mode. What it does instead is place
    limbs implausibly -- exactly the "overlapping limbs" the paper's Fig. 6 shows
    qualitatively. min_foot_sep is the median smallest pairwise foot separation
    and frac_overlap the fraction of frames with two feet closer than 0.10 L.
    """
    from itertools import combinations
    L = C.torso_length(kp)
    f = kp[:, 2:6]
    d = np.stack([np.linalg.norm(f[:, i] - f[:, j], axis=-1)
                  for i, j in combinations(range(4), 2)], axis=1) / L
    dmin = d.min(axis=1)
    z = f[..., 2]
    return dict(min_foot_sep=float(np.median(dmin)),
                frac_overlap=float((dmin < 0.10).mean()),
                foot_span_L=float((z.max() - z.min()) / L))

def windows(kp, fs, clip):
    """-> list of per-window metric dicts, each tagged with its mean base speed."""
    dec = C.decimate_to_common(kp, fs)
    if len(dec) < W:
        return []
    L = C.torso_length(dec)
    spd = base_speed(dec, C.FS_COMMON)
    out = []
    for s in range(0, len(dec) - W + 1, HOP):
        seg = dec[s:s + W]
        out.append(dict(clip=clip, speed=float(spd[s:s + W].mean()),
                        M1_CV_torso_pct=win_cv_torso(seg),
                        M2_SG_resid_pct=win_sg_resid(seg, L),
                        M3_HFR_pct=win_hfr(seg, L),
                        M4_norm_jerk=win_jerk(seg, L)))
    return out

# ------------------------------------------------------------------ data
def mocap_clips(full_source=True):
    if full_source:
        return [(c, C.load_mocap_keypoints(c)) for c in C.MOCAP_SOURCE_CLIPS]
    return [(n, C.load_mocap_keypoints(f, lo, hi)) for n, f, lo, hi in C.MOCAP_WINDOWS]

def video_clips(names=None, source="depth_cam"):
    out = []
    for c in (names or C.VIDEO_CLIPS_8):
        try:
            out.append((c, C.load_video_keypoints(c, source)))
        except FileNotFoundError:
            pass
    return out

def speed_band(a, b, lo=10, hi=90):
    sa = np.array([w["speed"] for w in a]); sb = np.array([w["speed"] for w in b])
    return (max(np.percentile(sa, lo), np.percentile(sb, lo)),
            min(np.percentile(sa, hi), np.percentile(sb, hi)))

def in_band(ws, band):
    return [w for w in ws if band[0] <= w["speed"] <= band[1]]

def per_clip_median(ws, key):
    out = {}
    for c in sorted({w["clip"] for w in ws}):
        v = [w[key] for w in ws if w["clip"] == c and np.isfinite(w[key])]
        if v:
            out[c] = float(np.median(v))
    return out

# ------------------------------------------------------------------ figure A
def figure_A():
    """Qualitative comparison. Both sources at their native rate (M1 is rate-
    immune and no filtering is applied here, so nothing can be a resampling
    artifact). Coordinates are the dimensionless body frame: rear keypoint at the
    origin, heading along +x, divided by the per-clip torso length L."""
    # Speed-, duration- and behaviour-matched pair: MoCap "right turn0"
    # (1.96 L/s) vs video turn_right (1.98 L/s), both right turns, both trimmed
    # to 1.47 s. Comparing a fast MoCap trot against a slow video walk would
    # confound gait with sensor noise.
    vi = C.load_video_keypoints("turn_right_1771233000")      # 30 Hz
    n_mo = int(round(len(vi) / C.FS_VIDEO * C.FS_MOCAP))
    mo = C.load_mocap_keypoints("dog_walk09", 1000, 1000 + n_mo)   # 60 Hz
    Lm, Lv = C.torso_length(mo), C.torso_length(vi)
    cm, cv = C.canonicalize_global(mo), C.canonicalize_global(vi)   # metres
    tm = np.arange(len(cm)) / C.FS_MOCAP
    tv = np.arange(len(cv)) / C.FS_VIDEO

    fig = plt.figure(figsize=(C.IEEE_2COL, 3.7))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.24,
                          height_ratios=[1.35, 1.0])
    lbl = dict(fontsize=8); tk = dict(labelsize=7)
    SRC = ((cm, C.C_MOCAP), (cv, C.C_VIDEO))

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    for kp, col in SRC:
        for k in range(2, 6):
            ax.plot(*kp[:, k].T, color=col, lw=0.9, alpha=0.95)
        for k in range(2):
            ax.plot(*kp[:, k].T, color=col, lw=1.4, alpha=0.95)
        for t in range(0, len(kp), 3):
            ax.plot(*np.stack([kp[t, 0], kp[t, 1]]).T, color="grey", lw=0.5, alpha=0.55)
    ax.view_init(elev=20, azim=-62); ax.set_box_aspect((2.3, 0.8, 0.8), zoom=1.30)
    ax.set_xlabel("x [m]", labelpad=-6, **lbl)
    ax.set_ylabel("y [m]", labelpad=-6, **lbl)
    ax.set_zlabel("z [m]", labelpad=-6, **lbl)
    ax.locator_params(axis="x", nbins=4); ax.locator_params(axis="y", nbins=3)
    ax.locator_params(axis="z", nbins=3)
    ax.tick_params(**tk, pad=-2); ax.grid(alpha=0.25)
    ax.set_title("(a) global 3D keypoint trajectories", fontsize=8.5, pad=0)

    ax = fig.add_subplot(gs[0, 1])
    dm, dv = C.torso_series(mo) / Lm, C.torso_series(vi) / Lv
    cvm, cvv = 100 * dm.std() / np.median(dm), 100 * dv.std() / np.median(dv)
    ax.axhline(1.0, ls="--", c="k", lw=0.7)
    ax.axhspan(1 - cvm / 100, 1 + cvm / 100, color=C.C_MOCAP, alpha=0.20, lw=0)
    ax.plot(tm, dm, color=C.C_MOCAP, lw=1.3, label=f"MoCap  CV = {cvm:.1f}%")
    ax.plot(tv, dv, color=C.C_VIDEO, lw=1.3, label=f"Video  CV = {cvv:.1f}%")
    ax.set_ylim(0.50, 1.60); ax.grid(True, alpha=0.3)
    ax.set_xlabel("time [s]", **lbl)
    ax.set_ylabel(r"torso length $\ell(t)\,/\,L$", **lbl)
    ax.tick_params(**tk); ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax.set_title("(b) rigid-torso violation", fontsize=8.5)

    for gi, (i, j, nm, ttl) in enumerate(
            [(0, 1, ("x [m]", "y [m]"), "(c) top-down projection"),
             (0, 2, ("x [m]", "z [m]"), "(d) sagittal projection")]):
        ax = fig.add_subplot(gs[1, gi])
        for kp, col in SRC:
            for k in range(2, 6):
                ax.plot(kp[:, k, i], kp[:, k, j], color=col, lw=0.8, alpha=0.9)
            for k in range(2):
                ax.plot(kp[:, k, i], kp[:, k, j], color=col, lw=1.3, alpha=0.95)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_xlabel(nm[0], **lbl); ax.set_ylabel(nm[1], **lbl); ax.tick_params(**tk)
        ax.set_title(ttl, fontsize=8.5)

    h = [plt.Line2D([], [], color=C.C_MOCAP, lw=1.6), plt.Line2D([], [], color=C.C_VIDEO, lw=1.6)]
    fig.legend(h, [f"MoCap (right turn, {len(mo)/C.FS_MOCAP:.2f} s, 60 Hz)",
                   f"Video w. Depth Camera (right turn, {len(vi)/C.FS_VIDEO:.2f} s, 30 Hz)"],
               ncol=2, frameon=False, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 1.06))
    return C.save(fig, "fig_A_keypoint_noise_qualitative.pdf")


# ------------------------------------------------------------------ figure B
def figure_B(mo_w, vi_w, depth_src, plaus):
    fig = plt.figure(figsize=(C.IEEE_2COL, 2.5))
    gs = fig.add_gridspec(1, 5, wspace=0.70)
    titles = {"M1_CV_torso_pct": "M1 torso-length CV\n[%]",
              "M2_SG_resid_pct": "M2 SG residual\n[% of L]",
              "M4_norm_jerk": "M4 normalized jerk\n[1]"}
    for i, key in enumerate(["M1_CV_torso_pct", "M2_SG_resid_pct", "M4_norm_jerk"]):
        ax = fig.add_subplot(gs[i])
        a = np.array(list(per_clip_median(mo_w, key).values()))
        b = np.array(list(per_clip_median(vi_w, key).values()))
        for gi, (v, col) in enumerate(((a, C.C_MOCAP), (b, C.C_VIDEO))):
            ax.scatter(np.full(len(v), gi) + np.random.uniform(-.07, .07, len(v)),
                       v, s=13, color=col, alpha=.85, zorder=3, lw=0)
        ax.boxplot([a, b], positions=[0, 1], widths=.5, showfliers=False,
                   medianprops=dict(color="k", lw=1.1))
        ax.set_title(titles[key], fontsize=7.0)
        ax.text(.5, .97, f"{np.median(b)/np.median(a):.1f}x", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["MoCap", "Video"], fontsize=7)
        ax.tick_params(labelsize=6.5); ax.grid(True, alpha=.3, axis="y")
        if key == "M4_norm_jerk":
            ax.set_yscale("log")

    names = ["MoCap", "Cam.\ndepth", "DA-V2", "Animal\nAvatar"]
    cols = [C.C_MOCAP, C.C_VIDEO, C.C_DEPTHANY, C.C_AVATAR]

    # (d) jitter by depth source
    ax = fig.add_subplot(gs[3])
    for gi, (nm, ws) in enumerate(depth_src):
        v = np.array(list(per_clip_median(ws, "M2_SG_resid_pct").values()))
        if not len(v):
            continue
        ax.scatter(np.full(len(v), gi) + np.random.uniform(-.07, .07, len(v)),
                   v, s=13, color=cols[gi], alpha=.85, zorder=3, lw=0)
        ax.plot([gi - .25, gi + .25], [np.median(v)] * 2, color="k", lw=1.3, zorder=4)
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6.0)
    ax.tick_params(labelsize=6.5); ax.grid(True, alpha=.3, axis="y")
    ax.set_title("M2 jitter by source\n[% of L]", fontsize=7.0)

    # (e) geometric implausibility by depth source -- the metric that catches
    # AnimalAvatar, which is smooth but wrong.
    ax = fig.add_subplot(gs[4])
    for gi, tag in enumerate(["mocap", "depth_cam", "depth_any", "avatar"]):
        v = np.array([m["frac_overlap"] for m in plaus[tag].values()]) * 100
        ax.scatter(np.full(len(v), gi) + np.random.uniform(-.07, .07, len(v)),
                   v, s=13, color=cols[gi], alpha=.85, zorder=3, lw=0)
        ax.plot([gi - .25, gi + .25], [np.median(v)] * 2, color="k", lw=1.3, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=6.0)
    ax.tick_params(labelsize=6.5); ax.grid(True, alpha=.3, axis="y")
    ax.set_title("Overlapping feet\n[% of frames]", fontsize=7.0)
    return C.save(fig, "fig_B_keypoint_noise_quantitative.pdf")


# ------------------------------------------------------------------ main
def main():
    np.random.seed(0)
    mo_all = [w for n, kp in mocap_clips(True) for w in windows(kp, C.FS_MOCAP, n)]
    vi_all = [w for n, kp in video_clips() for w in windows(kp, C.FS_VIDEO, n)]
    band = speed_band(mo_all, vi_all)
    mo_w, vi_w = in_band(mo_all, band), in_band(vi_all, band)
    print(f"speed band [{band[0]:.2f}, {band[1]:.2f}] L/s -> "
          f"MoCap {len(mo_w)}/{len(mo_all)} windows, Video {len(vi_w)}/{len(vi_all)}")

    res = dict(speed_band=list(band),
               n_windows=dict(mocap=len(mo_w), video=len(vi_w)),
               per_clip={k: dict(mocap=per_clip_median(mo_w, k),
                                 video=per_clip_median(vi_w, k)) for k in KEYS},
               stats={})
    for key in KEYS:
        a = np.array(list(per_clip_median(mo_w, key).values()))
        b = np.array(list(per_clip_median(vi_w, key).values()))
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        gt = sum(int(x > y) - int(x < y) for x in b for y in a)
        res["stats"][key] = dict(
            mocap_median=float(np.median(a)), video_median=float(np.median(b)),
            ratio=float(np.median(b) / np.median(a)),
            mocap_range=[float(a.min()), float(a.max())],
            video_range=[float(b.min()), float(b.max())],
            disjoint=bool(a.max() < b.min()), mannwhitney_p=float(p),
            cliffs_delta=float(gt / (len(a) * len(b))), n_mocap=len(a), n_video=len(b))

    Ls = {n: C.torso_length(kp) for n, kp in video_clips()}
    res["video_Lc"] = Ls
    res["video_Lc_cv_pct"] = float(100 * np.std(list(Ls.values())) / np.mean(list(Ls.values())))
    res["mocap_Lbar"] = float(np.mean([C.torso_length(kp) for _, kp in mocap_clips(True)]))
    res["video_Lbar"] = float(np.mean(list(Ls.values())))

    occ = {}
    for n, _ in video_clips():
        m = C.load_occlusion_mask(n)
        if m is None:
            continue
        runs = [len(list(g)) for k, g in itertools.groupby(m.reshape(-1)) if k]
        occ[n] = dict(frac=float(m.mean()), gmax=int(max(runs) if runs else 0))
    res["occlusion"] = occ

    depth_src = [("MoCap", mo_w)]
    for tag in ("depth_cam", "depth_any", "avatar"):
        ws = [w for n, kp in video_clips(C.VIDEO_CLIPS_4, tag)
              for w in windows(kp, C.FS_VIDEO, n)]
        depth_src.append((tag, in_band(ws, band)))
    res["depth_sources"] = {k: {m: per_clip_median(v, m) for m in KEYS} for k, v in depth_src}
    res["depth_source_torso_m"] = {
        tag: {n: C.torso_length(kp) for n, kp in video_clips(C.VIDEO_CLIPS_4, tag)}
        for tag in ("depth_cam", "depth_any", "avatar")}

    plaus = {"mocap": {n: foot_plausibility(C.load_mocap_keypoints(f, lo, hi))
                       for n, f, lo, hi in C.MOCAP_WINDOWS}}
    for tag in ("depth_cam", "depth_any", "avatar"):
        plaus[tag] = {n: foot_plausibility(kp)
                      for n, kp in video_clips(C.VIDEO_CLIPS_4, tag)}
    res["plausibility"] = plaus

    figure_A(); figure_B(mo_w, vi_w, depth_src, plaus)
    json.dump(res, open(os.path.join(C.OUT, "_tmp", "exp1.json"), "w"), indent=1, default=float)

    print()
    for k, v in res["stats"].items():
        print(f"{k:18s} MoCap {v['mocap_median']:8.3f} [{v['mocap_range'][0]:.2f},{v['mocap_range'][1]:.2f}]"
              f"  Video {v['video_median']:8.3f} [{v['video_range'][0]:.2f},{v['video_range'][1]:.2f}]"
              f"  {v['ratio']:5.1f}x disjoint={str(v['disjoint']):5s} p={v['mannwhitney_p']:.4f} d={v['cliffs_delta']:+.2f}")
    print(f"\nL_bar MoCap {res['mocap_Lbar']:.3f} m | Video {res['video_Lbar']:.3f} m"
          f" | video across-clip L_c CV {res['video_Lc_cv_pct']:.1f}%")
    print("depth-source M2 medians:",
          {k: round(float(np.median(list(v['M2_SG_resid_pct'].values()))), 2)
           for k, v in res["depth_sources"].items() if v['M2_SG_resid_pct']})
    print("depth-source mean torso [m]:",
          {k: round(float(np.mean(list(v.values()))), 3) for k, v in res["depth_source_torso_m"].items()})
    print("overlapping-feet % of frames (median):",
          {k: round(float(np.median([m['frac_overlap'] for m in v.values()])) * 100, 1)
           for k, v in plaus.items()})
    print("max overlap rate per source:",
          {k: round(float(np.max([m['frac_overlap'] for m in v.values()])) * 100, 1)
           for k, v in plaus.items()})
    print("foot vertical span [L] (median):",
          {k: round(float(np.median([m['foot_span_L'] for m in v.values()])), 2)
           for k, v in plaus.items()})
    print("occlusion mean frac:", round(float(np.mean([o['frac'] for o in occ.values()])), 4),
          "| max gap:", max(o['gmax'] for o in occ.values()))

if __name__ == "__main__":
    main()
