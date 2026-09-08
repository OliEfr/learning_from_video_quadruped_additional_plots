"""Experiment 1 (author's follow-up): can a filter reduce the keypoint noise while keeping the movement intact?

Proposed two-stage filter, with the stage where it belongs in the pipeline:
  A  depth-domain outlier rejection  -> quadruped_from_video/3d_recon.py, after the depth look-up and before the
     zero-interpolation (this is where Sec. III-A of the paper claims a "0.5 m jump" rule that does not exist in the code).
     Implemented as a Hampel filter on each marker's depth series (window +-2 frames, threshold 3 x 1.4826 x MAD,
     but at least 5 cm); flagged samples are treated like missing depth (interpolated). The paper's 0.5 m rule is
     evaluated as well and its trigger count reported.
  B  zero-phase low-pass on the 3-D keypoints -> postprocess_global_align_plane_approach.py (before retargeting).
     4th-order Butterworth, cutoff FC = 6 Hz, scipy.signal.sosfiltfilt (forward-backward => zero group delay, no
     shift of contact timing). FC is fixed by a physical criterion, NOT by the video/MoCap ratio: the averaged video
     keypoint spectrum becomes white (within 2x of its 10-15 Hz floor) at 6.1 Hz, and 6 Hz keeps the stride
     fundamental (0.7-1.9 Hz in both sources) plus at least its 3rd harmonic and the 0.2-0.3 s swing bump.
The filter is part of the video method; MoCap is the unfiltered baseline dataset and is never filtered in the
comparison (the effect of stage B on MoCap is stored in the JSON for reference only). Video is reported filtered AND
unfiltered, with motion-preservation checks (net displacement, path length, stride frequency, peak paw clearance,
stance fraction, contact-onset timing, base speed).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt

from common import (
    CLIP_SHORT,
    COLORS,
    DATA,
    DOUBLE_W,
    FLAT_VIDEO_CLIPS,
    MOCAP_FPS,
    MOCAP_SEGMENTS,
    NAME_MOCAP,
    NAME_VIDEO,
    NAME_VIDEO_CORR,
    VIDEO_FPS,
    align_clip,
    hip_height,
    load_mocap_segment,
    md_table,
    mocap_keypoints,
    savefig,
)
from exp1_noise import FS, averaged_psd, clip_metrics
from exp1_relift_keypoints import relift

FC = 6.0  # Hz, see module docstring
SOS = butter(4, FC, fs=FS, output="sos")
HAMPEL_HALF, HAMPEL_K, HAMPEL_MIN = 2, 3.0, 0.05
STANCE_Z = 0.02  # m above the ground estimate


# ----------------------------------------------------------------------------------------------
def hampel_depth(m):
    """Stage A on the (T, 6, 3) [u, v, depth] array: depth outliers -> 0 (treated as missing)."""
    m = m.copy()
    T = m.shape[0]
    n_flag = 0
    n_jump = 0
    for k in range(m.shape[1]):
        d = m[:, k, 2].copy()
        valid = d > 0
        dv = np.where(valid, d, np.nan)
        for t in range(T):
            if not valid[t]:
                continue
            w = dv[max(0, t - HAMPEL_HALF) : t + HAMPEL_HALF + 1]
            w = w[~np.isnan(w)]
            if len(w) < 3:
                continue
            med = np.median(w)
            thr = max(HAMPEL_MIN, HAMPEL_K * 1.4826 * np.median(np.abs(w - med)))
            if abs(d[t] - med) > thr:
                m[t, k, 2] = 0.0
                n_flag += 1
        # paper's rule: |d_t - d_{t-1}| > 0.5 m (evaluated on the unfiltered valid samples, for the report only)
        idx = np.where(valid)[0]
        n_jump += int(np.sum(np.abs(np.diff(d[idx])) > 0.5))
    hampel_depth.stats.append(dict(flagged=n_flag, paper_rule_jumps=n_jump, n_valid=int((m[..., 2] > 0).sum() + n_flag)))
    return m


hampel_depth.stats = []


def lowpass(x):
    """Stage B, zero-phase, along axis 1 of (K, T, 3)."""
    x = np.asarray(x, float)
    T = x.shape[1]
    padlen = min(3 * (2 * SOS.shape[0] + 1), T - 2)
    return sosfiltfilt(SOS, x, axis=1, padlen=padlen)


# ----------------------------------------------------------------------------------------------
def build_variants():
    """Return dict source -> list of (clip_unfiltered, clip_filtered[, clip_filtered_B_only])."""
    out = {NAME_MOCAP: [], NAME_VIDEO: [], NAME_VIDEO_CORR: []}
    step = int(round(MOCAP_FPS / FS))
    for seg in MOCAP_SEGMENTS:
        body, feet = mocap_keypoints(load_mocap_segment(seg))
        body, feet = body[:, ::step], feet[:, ::step]
        raw = dict(source=NAME_MOCAP, name=seg[0], fs=FS)
        raw["body"], raw["feet"], _ = align_clip(body, feet)
        ref = dict(raw, stage="B (reference only, NOT part of the comparison)")
        ref["body"], ref["feet"], _ = align_clip(lowpass(body), lowpass(feet))
        out[NAME_MOCAP].append((raw, raw, ref))  # MoCap is the unfiltered baseline; the filter belongs to the video method
    for clip in FLAT_VIDEO_CLIPS:
        for src, mode in [(NAME_VIDEO, "as_code"), (NAME_VIDEO_CORR, "half_res")]:
            r0 = relift(clip, mode)  # unfiltered (== on-disk keypoints for as_code)
            r1 = relift(clip, mode, preproc=hampel_depth)  # stage A
            m0 = r0["markers_ground"][:, 1:]  # drop frame 0 like 3d_recon.py
            m1 = r1["markers_ground"][:, 1:]
            raw = dict(source=src, name=CLIP_SHORT[clip], clip=clip, fs=FS)
            raw["body"], raw["feet"], _ = align_clip(m0[:2], m0[2:])
            ab = dict(raw, stage="A+B")
            ab["body"], ab["feet"], _ = align_clip(lowpass(m1[:2]), lowpass(m1[2:]))
            b = dict(raw, stage="B")
            b["body"], b["feet"], _ = align_clip(lowpass(m0[:2]), lowpass(m0[2:]))
            out[src].append((raw, ab, b))
    return out


# ----------------------------------------------------------------------------------------------
def stride_freq(c):
    vals = []
    for k in range(4):
        z = c["feet"][k, :, 2]
        if len(z) < 12:
            continue
        f, P = periodogram(z - z.mean(), fs=c["fs"], window="hann", nfft=256)
        m = (f >= 0.5) & (f <= 6)
        vals.append(f[m][np.argmax(P[m])])
    return float(np.median(vals)) if vals else np.nan


def contact_onsets(z, fs):
    below = z < STANCE_Z
    return np.where(~below[:-1] & below[1:])[0] / fs


def motion_checks(raw, fil):
    """Motion-preservation numbers for one clip (values for raw and filtered)."""
    out = {}
    for tag, c in [("raw", raw), ("filt", fil)]:
        com = c["body"].mean(axis=0)
        steps = np.linalg.norm(np.diff(com[:, :2], axis=0), axis=1)
        speed = steps * c["fs"]
        ground = np.percentile(c["feet"][..., 2], 5)
        z = c["feet"][..., 2] - ground
        peaks = []
        for k in range(4):
            idx, props = find_peaks(z[k], height=0.02, distance=max(1, int(0.25 * c["fs"])))
            peaks += list(props["peak_heights"])
        out[tag] = dict(
            net_disp=float(np.linalg.norm(com[-1, :2] - com[0, :2])),
            path_len=float(steps.sum()),
            speed_mean=float(speed.mean()),
            speed_p95=float(np.percentile(speed, 95)),
            stride_hz=stride_freq(c),
            peak_clearance=float(np.mean(peaks)) if peaks else np.nan,
            n_swings=len(peaks),
            stance_frac=float((z < STANCE_Z).mean()),
            hip_height=hip_height(c["body"], c["feet"]),
        )
    # contact-onset timing shift (matched nearest events)
    shifts = []
    for k in range(4):
        g0 = np.percentile(raw["feet"][..., 2], 5)
        g1 = np.percentile(fil["feet"][..., 2], 5)
        e0 = contact_onsets(raw["feet"][k, :, 2] - g0, raw["fs"])
        e1 = contact_onsets(fil["feet"][k, :, 2] - g1, fil["fs"])
        for e in e0:
            if len(e1):
                shifts.append(np.min(np.abs(e1 - e)))
    out["onset_shift_mean_s"] = float(np.mean(shifts)) if shifts else np.nan
    out["n_onsets_raw"] = int(sum(len(contact_onsets(raw["feet"][k, :, 2] - np.percentile(raw["feet"][..., 2], 5), raw["fs"])) for k in range(4)))
    out["n_onsets_filt"] = int(sum(len(contact_onsets(fil["feet"][k, :, 2] - np.percentile(fil["feet"][..., 2], 5), fil["fs"])) for k in range(4)))
    return out


def set_summary(clips):
    ms = [clip_metrics(c) for c in clips]
    hh = np.array([m["hip_height"] for m in ms])
    f, P = averaged_psd(clips)
    return dict(
        m1_body_pct=float(np.mean([m["m1_body"] for m in ms] / hh) * 100),
        m1_feet_pct=float(np.mean([m["m1_feet"] for m in ms] / hh) * 100),
        m1_feet_mm=float(np.mean([m["m1_feet"] for m in ms]) * 1000),
        m2_feet_mm=float(np.mean([m["m2_feet"] for m in ms]) * 1000),
        m3_cv_pct=float(np.mean([m["m3_seglen_cv"] for m in ms]) * 100),
        m4_pct=float(np.mean([m["m4_stance"] for m in ms] / hh) * 100),
        m4_mm=float(np.mean([m["m4_stance"] for m in ms]) * 1000),
        psd_hi=float(P[f >= 5].sum() * (f[1] - f[0])),
        psd_lo=float(P[(f > 0) & (f < 5)].sum() * (f[1] - f[0])),
    )


# ----------------------------------------------------------------------------------------------
def fig_filter_effect(V, summ, fname):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 1.9), gridspec_kw=dict(wspace=0.4))
    ax = axes[0]
    for src in [NAME_MOCAP, NAME_VIDEO_CORR, NAME_VIDEO]:
        f, P = averaged_psd([v[0] for v in V[src]])
        ax.semilogy(f[1:], P[1:], color=COLORS[src], lw=1.0, label=f"{src}")
        if src != NAME_MOCAP:
            f, P = averaged_psd([v[1] for v in V[src]])
            ax.semilogy(f[1:], P[1:], color=COLORS[src], lw=1.0, ls="--")
    ax.axvline(FC, color="0.3", lw=0.6, ls=":")
    ax.text(FC + 0.2, ax.get_ylim()[1] * 0.3, f"$f_c$ = {FC:.0f} Hz", fontsize=5.5)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [(hip height)$^2$/Hz]")
    ax.set_title("(a) spectrum\nsolid raw, dashed video filtered", fontsize=7)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(frameon=False, fontsize=5, loc="lower left")
    # (b) walk clip paw z raw vs filtered (corrected re-lift)
    ax = axes[1]
    raw, ab, _ = [v for v in V[NAME_VIDEO_CORR] if v[0]["name"] == "walk"][0]
    t = np.arange(raw["feet"].shape[1]) / FS
    g0, g1 = np.percentile(raw["feet"][..., 2], 5), np.percentile(ab["feet"][..., 2], 5)
    for k, col in enumerate(["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]):
        ax.plot(t, raw["feet"][k, :, 2] - g0, color=col, lw=0.6, alpha=0.45)
        ax.plot(t, ab["feet"][k, :, 2] - g1, color=col, lw=1.1)
    ax.axhline(STANCE_Z, color="0.5", lw=0.5, ls="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("paw height [m]")
    ax.set_title("(b) walk, corrected: paw height\nthin raw, thick filtered (A+B)", fontsize=7)
    ax.grid(alpha=0.3, lw=0.4)
    # (c) metric bars before/after
    ax = axes[2]
    keys = [("m1_feet_pct", "M1 paws\n[% h]"), ("m1_body_pct", "M1 torso\n[% h]"), ("m4_pct", "M4 stance\n[% h]"), ("m3_cv_pct", "M3 seg CV\n[%]")]
    x = np.arange(len(keys))
    w = 0.13
    ax.bar(x - 2.2 * w, [summ[NAME_MOCAP]["raw"][k] for k, _ in keys], w, color=COLORS[NAME_MOCAP], lw=0)
    for i, src in enumerate([NAME_VIDEO, NAME_VIDEO_CORR]):
        ax.bar(x + i * 2.2 * w - w / 2, [summ[src]["raw"][k] for k, _ in keys], w, color=COLORS[src], alpha=0.45, lw=0)
        ax.bar(x + i * 2.2 * w + w / 2, [summ[src]["filt"][k] for k, _ in keys], w, color=COLORS[src], lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in keys], fontsize=5.5)
    ax.set_title("(c) noise metrics\nMoCap raw; video light raw, solid filtered", fontsize=7)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, lw=0.4, which="both")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR]]
    fig.legend(handles, [NAME_MOCAP, NAME_VIDEO + " (pipeline output)", NAME_VIDEO_CORR], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.3))
    savefig(fig, fname)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    V = build_variants()
    summ, checks = {}, {}
    for src in V:
        summ[src] = dict(raw=set_summary([v[0] for v in V[src]]), filt=set_summary([v[1] for v in V[src]]), filt_B_only=set_summary([v[2] for v in V[src]]))
        if src == NAME_MOCAP:
            summ[src] = dict(raw=summ[src]["raw"], filt=summ[src]["raw"], mocap_filtered_reference_only=summ[src]["filt_B_only"])
        checks[src] = {v[0]["name"]: motion_checks(v[0], v[1]) for v in V[src]}
    stageA = dict(flagged=int(sum(s["flagged"] for s in hampel_depth.stats)), paper_rule_jumps=int(sum(s["paper_rule_jumps"] for s in hampel_depth.stats)), n_valid=int(sum(s["n_valid"] for s in hampel_depth.stats)), n_calls=len(hampel_depth.stats))

    ratios = {}
    for k in ["m1_body_pct", "m1_feet_pct", "m1_feet_mm", "m2_feet_mm", "m3_cv_pct", "m4_pct", "m4_mm", "psd_hi"]:
        for s in [NAME_VIDEO, NAME_VIDEO_CORR]:
            ratios[f"{k} {s} unfiltered / MoCap"] = summ[s]["raw"][k] / summ[NAME_MOCAP]["raw"][k]
            ratios[f"{k} {s} filtered / MoCap"] = summ[s]["filt"][k] / summ[NAME_MOCAP]["raw"][k]
        ratios[f"{k} MoCap (if filtered, reference only) / MoCap"] = summ[NAME_MOCAP]["mocap_filtered_reference_only"][k] / summ[NAME_MOCAP]["raw"][k]

    # aggregate motion checks per source (means over clips of the relative change)
    agg = {}
    for src in [NAME_VIDEO, NAME_VIDEO_CORR]:
        rows = list(checks[src].values())
        rel = lambda key: float(np.nanmean([(r["filt"][key] - r["raw"][key]) / r["raw"][key] for r in rows if r["raw"][key]])) * 100  # noqa: E731
        agg[src] = dict(
            net_disp_change_pct=rel("net_disp"),
            path_len_change_pct=rel("path_len"),
            speed_mean_change_pct=rel("speed_mean"),
            speed_p95_change_pct=rel("speed_p95"),
            stride_hz_change_pct=rel("stride_hz"),
            peak_clearance_change_pct=rel("peak_clearance"),
            stance_frac_raw=float(np.mean([r["raw"]["stance_frac"] for r in rows])),
            stance_frac_filt=float(np.mean([r["filt"]["stance_frac"] for r in rows])),
            onset_shift_mean_s=float(np.nanmean([r["onset_shift_mean_s"] for r in rows])),
            n_onsets_raw=int(sum(r["n_onsets_raw"] for r in rows)),
            n_onsets_filt=int(sum(r["n_onsets_filt"] for r in rows)),
            peak_clearance_raw_mm=float(np.nanmean([r["raw"]["peak_clearance"] for r in rows])) * 1000,
            peak_clearance_filt_mm=float(np.nanmean([r["filt"]["peak_clearance"] for r in rows])) * 1000,
        )

    res = dict(settings=dict(fc_hz=FC, butter_order=4, zero_phase=True, hampel_half_window=HAMPEL_HALF, hampel_k=HAMPEL_K, hampel_min_m=HAMPEL_MIN, stance_z_m=STANCE_Z), stageA=stageA, summary=summ, ratios=ratios, motion_checks=checks, motion_checks_aggregate=agg)
    with open(os.path.join(DATA, "exp1_filter.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    # tables
    rows = []
    for src in V:
        tags = ["raw"] if src == NAME_MOCAP else ["raw", "filt_B_only", "filt"]
        for tag in tags:
            s = summ[src][tag]
            rows.append([src, {"raw": "unfiltered (baseline)" if src == NAME_MOCAP else "unfiltered", "filt_B_only": "B only (6 Hz zero-phase)", "filt": "A+B (method)"}[tag], s["m1_body_pct"], s["m1_feet_pct"], s["m1_feet_mm"], s["m2_feet_mm"], s["m3_cv_pct"], s["m4_pct"], s["m4_mm"], s["psd_hi"] * 1e4])
    t1 = md_table(["source", "filter", "M1 torso [% h]", "M1 paws [% h]", "M1 paws [mm]", "M2 paws [mm/fr²]", "M3 seg CV [%]", "M4 stance [% h]", "M4 [mm]", "PSD power >5 Hz [1e-4]"], rows, "{:.2f}")
    rows = []
    for src in agg:
        a = agg[src]
        rows.append([src, a["net_disp_change_pct"], a["path_len_change_pct"], a["speed_mean_change_pct"], a["speed_p95_change_pct"], a["stride_hz_change_pct"], a["peak_clearance_raw_mm"], a["peak_clearance_filt_mm"], a["peak_clearance_change_pct"], a["stance_frac_raw"] * 100, a["stance_frac_filt"] * 100, f"{a['n_onsets_raw']}->{a['n_onsets_filt']}", a["onset_shift_mean_s"] * 1000])
    t2 = md_table(["source", "net displ. [%]", "path length [%]", "mean speed [%]", "p95 speed [%]", "stride freq. [%]", "peak clearance raw [mm]", "filt [mm]", "[%]", "stance frac raw [%]", "filt [%]", "contact onsets", "onset shift [ms]"], rows, "{:.1f}")
    with open(os.path.join(DATA, "exp1_filter_tables.md"), "w") as fh:
        fh.write("## Noise metrics: MoCap baseline (unfiltered) vs video unfiltered / filtered (filter = part of the video method)\n\n" + t1 + "\n\n## Motion preservation (filtered vs unfiltered, mean over clips)\n\n" + t2 + "\n")
    print(t1)
    print(t2)
    print("stage A:", stageA)
    print(json.dumps(ratios, indent=1))

    fig_filter_effect(V, summ, "fig_exp1_filter_effect")

    # trajectory grids with the filtered corrected video rows (2-row and 14-row), same geometry as the corrected grids
    from exp1_trajectory_grid import main as grid_main

    filt_corr = [v[1] for v in V[NAME_VIDEO_CORR]]
    grid_main(NAME_VIDEO_CORR, "two", video_override=filt_corr, suffix="_filtered", footer=f"video: corrected intrinsics + depth Hampel + {FC:.0f} Hz zero-phase low-pass; MoCap unchanged")
    grid_main(NAME_VIDEO_CORR, "all", video_override=filt_corr, suffix="_filtered", footer=f"video: corrected intrinsics + depth Hampel + {FC:.0f} Hz zero-phase low-pass; MoCap unchanged")


if __name__ == "__main__":
    main()
