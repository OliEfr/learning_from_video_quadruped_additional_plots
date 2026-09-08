"""Experiment 1: How noisy are the video keypoint trajectories compared to MoCap?

Sources compared (all in GLOBAL coordinates, one rigid transform per clip, 30 Hz):
  * MoCap  : the 6 Zhang et al. segments used by retarget_motion.py, 60 Hz -> 30 Hz (every 2nd sample)
  * Video  : the 3D keypoints that fed the retargeting (data_fromVision_depth_cam), 8 flat-walking clips, 30 Hz
  * Video (corrected intrinsics): same 2D tracks/depth/poses re-lifted with the 640x360 intrinsics (exp1_relift_keypoints.py)

Metrics (per clip and keypoint), all filter windows/cutoffs identical for every source:
  M1  high-frequency residual: RMS of (x - SavitzkyGolay(x, window=7 samples=0.23 s, poly=3))         [mm, % hip height]
  M2  frame-to-frame jitter: RMS of the 2nd finite difference at 30 Hz                                  [mm]
  M3  rigid-body consistency: std/mean of the distance between the two body keypoints                   [%]
  M4  stance foot-height scatter: robust std (1.4826*MAD) of foot height in the lowest 30 % of frames    [mm, % hip height]
  M5  (video only) fraction of foot samples that are interpolated (occluded track or missing depth)
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from common import (
    CLIP_SHORT,
    COLORS,
    DATA,
    DOUBLE_W,
    FLAT_VIDEO_CLIPS,
    MOCAP_FPS,
    MOCAP_REF_POS_SCALE,
    MOCAP_SEGMENTS,
    NAME_MOCAP,
    NAME_VIDEO,
    NAME_VIDEO_CORR,
    VIDEO_FPS,
    VIDEO_REF_POS_SCALE,
    align_clip,
    hip_height,
    load_mocap_segment,
    load_video_keypoints_as_used,
    md_table,
    mocap_keypoints,
    savefig,
    sg_residual,
)

FS = 30.0
SG_WINDOW, SG_POLY = 7, 3
KP_NAMES = ["body rear", "body front", "foot 1", "foot 2", "foot 3", "foot 4"]
FEET_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
BODY_COLOR = "0.15"


# ----------------------------------------------------------------------------------------------
# data assembly
# ----------------------------------------------------------------------------------------------
def get_clips():
    """Return list of dict(source, name, body(2,T,3), feet(4,T,3), fs, extra) after ONE rigid transform per clip."""
    clips = []
    for seg in MOCAP_SEGMENTS:
        arr = load_mocap_segment(seg)
        body, feet = mocap_keypoints(arr)
        step = int(round(MOCAP_FPS / FS))
        body, feet = body[:, ::step], feet[:, ::step]  # 60 Hz -> 30 Hz so both sources see the same finite-difference operators
        body, feet, tf = align_clip(body, feet)
        clips.append(dict(source=NAME_MOCAP, name=seg[0], body=body, feet=feet, fs=FS, native_fs=MOCAP_FPS, scale=MOCAP_REF_POS_SCALE, interp_frac=0.0))
    for clip in FLAT_VIDEO_CLIPS:
        base, feet = load_video_keypoints_as_used(clip)
        body, feet, tf = align_clip(base, feet)
        rl = np.load(os.path.join(DATA, f"relift_as_code_{clip}.npz"))
        interp = rl["interpolated"][1:, 2:].mean()  # first frame is skipped in 3d_recon.py output
        clips.append(dict(source=NAME_VIDEO, name=CLIP_SHORT[clip], clip=clip, body=body, feet=feet, fs=FS, native_fs=VIDEO_FPS, scale=VIDEO_REF_POS_SCALE, interp_frac=float(interp)))
        rl = np.load(os.path.join(DATA, f"relift_half_res_{clip}.npz"))
        m = rl["markers_ground"][:, 1:]  # drop frame 0 like the pipeline
        body, feet, tf = align_clip(m[:2], m[2:])
        clips.append(dict(source=NAME_VIDEO_CORR, name=CLIP_SHORT[clip], clip=clip, body=body, feet=feet, fs=FS, native_fs=VIDEO_FPS, scale=np.nan, interp_frac=float(interp)))
    return clips


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def clip_metrics(c):
    body, feet = c["body"], c["feet"]
    kps = np.concatenate([body, feet], axis=0)  # (6, T, 3)
    hh = hip_height(body, feet)
    res = np.stack([sg_residual(k, SG_WINDOW, SG_POLY) for k in kps])  # (6, T, 3)
    m1 = np.sqrt((res ** 2).sum(-1).mean(-1))  # RMS of residual norm per keypoint [m]
    acc = np.diff(kps, n=2, axis=1)
    m2 = np.sqrt((acc ** 2).sum(-1).mean(-1))  # [m per frame^2]
    seg = np.linalg.norm(body[1] - body[0], axis=-1)
    m3 = seg.std() / seg.mean()
    ground = np.percentile(feet[..., 2], 5.0)
    m4 = []
    for f in feet:
        zs = f[:, 2] - ground
        smooth = zs - sg_residual(zs[:, None], SG_WINDOW, SG_POLY)[:, 0]
        stance = smooth <= np.percentile(smooth, 30)
        z_st = zs[stance]
        m4.append(1.4826 * np.median(np.abs(z_st - np.median(z_st))))
    m4 = np.array(m4)
    return dict(
        hip_height=hh,
        m1_body=m1[:2].mean(),
        m1_feet=m1[2:].mean(),
        m1_kp=m1,
        m2_body=m2[:2].mean(),
        m2_feet=m2[2:].mean(),
        m3_seglen_cv=m3,
        m4_stance=m4.mean(),
        m4_kp=m4,
        seg_len=seg.mean(),
        n_frames=kps.shape[1],
        duration=kps.shape[1] / c["fs"],
        travel=float(np.linalg.norm(body.mean(0)[-1, :2] - body.mean(0)[0, :2])),
        feet_below_ground_frac=float((feet[..., 2] - ground < -0.02).mean()),
    )


def averaged_psd(clips, nfft=64):
    """Time-weighted average periodogram (Hann window, linear detrend) of every coordinate of every keypoint,
    normalised by hip height^2 so that sources of different size are comparable. Returns f, Pxx."""
    acc, wsum = 0.0, 0.0
    for c in clips:
        kps = np.concatenate([c["body"], c["feet"]], axis=0) / hip_height(c["body"], c["feet"])
        T = kps.shape[1]
        if T < 12:
            continue
        for k in range(6):
            for i in range(3):
                f, P = periodogram(kps[k, :, i], fs=c["fs"], window="hann", detrend="linear", nfft=nfft, scaling="density")
                acc = acc + T * P
                wsum += T
    return f, acc / wsum


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def _plot_traj_panels(axes3d, ax_top, ax_side, c, title):
    body, feet = c["body"], c["feet"]
    T = body.shape[1]
    t = np.arange(T) / c["fs"]
    for i in range(2):
        axes3d.plot(body[i, :, 0], body[i, :, 1], body[i, :, 2], color=BODY_COLOR, lw=0.9, alpha=0.9)
        ax_top.plot(body[i, :, 0], body[i, :, 1], color=BODY_COLOR, lw=0.9)
        ax_side.plot(body[i, :, 0], body[i, :, 2], color=BODY_COLOR, lw=0.9)
    for i in range(4):
        axes3d.plot(feet[i, :, 0], feet[i, :, 1], feet[i, :, 2], color=FEET_COLORS[i], lw=0.8)
        ax_top.plot(feet[i, :, 0], feet[i, :, 1], color=FEET_COLORS[i], lw=0.8)
        ax_side.plot(feet[i, :, 0], feet[i, :, 2], color=FEET_COLORS[i], lw=0.8)
    # body segment (front-rear) at a few instants, to show the animal translating through the scene
    for f in np.linspace(0, T - 1, 5).astype(int):
        seg = body[:, f]
        axes3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=BODY_COLOR, lw=1.4, alpha=0.5)
        ax_top.plot(seg[:, 0], seg[:, 1], color=BODY_COLOR, lw=1.4, alpha=0.5)
        ax_side.plot(seg[:, 0], seg[:, 2], color=BODY_COLOR, lw=1.4, alpha=0.5)
    allp = np.concatenate([body, feet], axis=0).reshape(-1, 3)
    xr = allp[:, 0].min() - 0.05, allp[:, 0].max() + 0.05
    span = max(xr[1] - xr[0], 1.0)
    yc = allp[:, 1].mean()
    zmax = max(allp[:, 2].max() + 0.05, 0.5)
    yspan, zspan = 0.8, 0.6
    axes3d.set_xlim(xr[0], xr[0] + span)
    axes3d.set_ylim(yc - yspan / 2, yc + yspan / 2)
    axes3d.set_zlim(0, zspan)
    axes3d.set_box_aspect((span, yspan, zspan))  # true metric proportions
    axes3d.set_xlabel("x [m]", labelpad=-5)
    axes3d.set_ylabel("y [m]", labelpad=-7)
    axes3d.set_zlabel("z [m]", labelpad=-8)
    axes3d.set_yticks([np.round(yc - 0.3, 1), np.round(yc + 0.3, 1)])
    axes3d.set_zticks([0, 0.3, 0.6])
    axes3d.tick_params(pad=-3, labelsize=5)
    axes3d.view_init(elev=20, azim=-55)
    axes3d.set_title(title, fontsize=7, pad=-2, loc="left")
    for ax, ylab in [(ax_top, "y [m]"), (ax_side, "z [m]")]:
        ax.set_xlim(xr[0], xr[0] + span)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", labelpad=1)
        ax.set_ylabel(ylab, labelpad=1)
        ax.grid(alpha=0.3, lw=0.4)
    ax_top.set_ylim(yc - 0.35, yc + 0.35)
    ax_side.set_ylim(-0.05, zmax)
    ax_side.axhline(0, color="0.6", lw=0.5, ls="--")
    ax_top.set_title(f"top view (x-y), {T} frames, {T / c['fs']:.2f} s", fontsize=6.5, pad=2)
    ax_side.set_title("side view (x-z)", fontsize=6.5, pad=2)


def fig_trajectories(clips, rows, fname):
    fig = plt.figure(figsize=(DOUBLE_W, 1.75 * len(rows)))
    gs = fig.add_gridspec(len(rows), 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.5, hspace=0.25, left=0.02, right=0.99, top=0.95, bottom=0.12)
    for r, (src, name, title) in enumerate(rows):
        c = [c for c in clips if c["source"] == src and c["name"] == name][0]
        ax3 = fig.add_subplot(gs[r, 0], projection="3d")
        axt = fig.add_subplot(gs[r, 1])
        axs = fig.add_subplot(gs[r, 2])
        _plot_traj_panels(ax3, axt, axs, c, title)
    handles = [plt.Line2D([], [], color=BODY_COLOR, lw=1.2, label="body keypoints (rear / front)")]
    handles += [plt.Line2D([], [], color=FEET_COLORS[i], lw=1.0, label=f"foot {i + 1}") for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02 - 0.01 * len(rows)))
    savefig(fig, fname)
    plt.close(fig)


def fig_all_clips(clips, fname, view="side"):
    sel = [c for c in clips if c["source"] in (NAME_MOCAP, NAME_VIDEO)]
    n = len(sel)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_W, 0.95 * nrow), squeeze=False, gridspec_kw=dict(hspace=0.55, wspace=0.25))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    SPAN = 2.4  # identical metric scale for every panel (x extent in m)
    for ax, c in zip(axes.ravel(), sel):
        body, feet = c["body"], c["feet"]
        j = 2 if view == "side" else 1
        for i in range(2):
            ax.plot(body[i, :, 0], body[i, :, j], color=BODY_COLOR, lw=0.8)
        for i in range(4):
            ax.plot(feet[i, :, 0], feet[i, :, j], color=FEET_COLORS[i], lw=0.7)
        allp = np.concatenate([body, feet], axis=0).reshape(-1, 3)
        x0 = allp[:, 0].min() - 0.15
        ax.set_xlim(x0, x0 + SPAN)
        if view == "side":
            ax.set_ylim(-0.08, 0.62)
        else:
            yc = allp[:, 1].mean()
            ax.set_ylim(yc - 0.45, yc + 0.45)
        ax.set_aspect("equal")
        if view == "side":
            ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_title(f"{c['source']}: {c['name']} ({body.shape[1]} fr., {body.shape[1] / c['fs']:.1f} s)", fontsize=6.5, color=COLORS[c["source"]], pad=2)
        ax.tick_params(labelsize=5)
    for ax in axes[-1]:
        ax.set_xlabel("x [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("z [m]" if view == "side" else "y [m]")
    savefig(fig, fname)
    plt.close(fig)


def fig_metrics(clips, metrics, fname):
    sources = [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_W, 1.9), gridspec_kw=dict(wspace=0.45))

    def dot_panel(ax, keys, ylabel, title, norm=True):
        xt, xl = [], []
        x = 0
        for src in sources:
            for key, lab in keys:
                vals = np.array([m[key] / (m["hip_height"] if norm else 1.0) * 100 for c, m in zip(clips, metrics) if c["source"] == src])
                jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.05
                ax.scatter(x + jitter, vals, s=7, color=COLORS[src], alpha=0.85, lw=0, zorder=3)
                ax.hlines(vals.mean(), x - 0.3, x + 0.3, color=COLORS[src], lw=1.4, zorder=4)
                xt.append(x)
                xl.append(lab)
                x += 1
            x += 0.6
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=90, fontsize=5.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.set_ylim(bottom=0)

    dot_panel(axes[0], [("m1_body", "body"), ("m1_feet", "feet")], "residual RMS [% hip height]", "(a) high-frequency\nresidual (SG 7/3)")
    dot_panel(axes[2], [("m3_seglen_cv", "body")], "std / mean [%]", "(c) body-segment\nlength variation", norm=False)
    dot_panel(axes[3], [("m4_stance", "feet")], "robust std [% hip height]", "(d) stance foot-\nheight scatter")
    # PSD
    ax = axes[1]
    for src in sources:
        f, P = averaged_psd([c for c in clips if c["source"] == src])
        ax.semilogy(f[1:], P[1:], color=COLORS[src], lw=1.0, label=src)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [(hip height)$^2$/Hz]")
    ax.set_title("(b) keypoint\npower spectrum", fontsize=7)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.set_xlim(0, 15)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=COLORS[s], label=s, ms=3, lw=1) for s in sources]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.3))
    savefig(fig, fname)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    clips = get_clips()
    metrics = [clip_metrics(c) for c in clips]

    # ---- figures
    fig_trajectories(clips, [(NAME_MOCAP, "pace", "MoCap: pace (dog_walk00, 60 Hz -> 30 Hz)"), (NAME_VIDEO, "walk", "Video: walk (pipeline output, 30 Hz)")], "fig_exp1_trajectories")
    fig_trajectories(
        clips,
        [
            (NAME_MOCAP, "pace", "MoCap: pace (dog_walk00, 60 Hz -> 30 Hz)"),
            (NAME_VIDEO, "walk", "Video: walk (pipeline output, 30 Hz)"),
            (NAME_VIDEO_CORR, "walk", "Video: walk, re-lifted with 640x360 intrinsics"),
        ],
        "fig_exp1_trajectories_with_corrected",
    )
    fig_all_clips(clips, "fig_exp1_all_clips_side", view="side")
    fig_all_clips(clips, "fig_exp1_all_clips_top", view="top")
    fig_metrics(clips, metrics, "fig_exp1_noise_metrics")

    # ---- tables / numbers
    rows = []
    per_clip = []
    for c, m in zip(clips, metrics):
        hh = m["hip_height"]
        rows.append(
            [
                c["source"],
                c["name"],
                m["n_frames"],
                m["duration"],
                hh,
                m["m1_body"] * 1000,
                m["m1_feet"] * 1000,
                m["m1_body"] / hh * 100,
                m["m1_feet"] / hh * 100,
                m["m2_feet"] * 1000,
                m["m3_seglen_cv"] * 100,
                m["m4_stance"] * 1000,
                m["m4_stance"] / hh * 100,
                c["interp_frac"] * 100,
                m["feet_below_ground_frac"] * 100,
                m["travel"],
            ]
        )
        per_clip.append(dict(source=c["source"], name=c["name"], **{k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()}, interp_frac=c["interp_frac"]))
    header = ["source", "clip", "frames@30Hz", "dur [s]", "hip h [m]", "M1 body [mm]", "M1 feet [mm]", "M1 body [%h]", "M1 feet [%h]", "M2 feet [mm/fr²]", "M3 seg-len CV [%]", "M4 stance z [mm]", "M4 [%h]", "M5 interp. feet [%]", "feet <-2cm [%]", "travel [m]"]
    table = md_table(header, rows, "{:.2f}")

    summary = {}
    for src in [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR]:
        idx = [i for i, c in enumerate(clips) if c["source"] == src]
        agg = {}
        for key in ["m1_body", "m1_feet", "m2_body", "m2_feet", "m3_seglen_cv", "m4_stance", "hip_height"]:
            v = np.array([metrics[i][key] for i in idx])
            agg[key] = dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
        for key in ["m1_body", "m1_feet", "m4_stance"]:
            v = np.array([metrics[i][key] / metrics[i]["hip_height"] for i in idx])
            agg[key + "_norm"] = dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
        agg["interp_frac_mean"] = float(np.mean([clips[i]["interp_frac"] for i in idx]))
        agg["n_clips"] = len(idx)
        agg["total_frames"] = int(sum(metrics[i]["n_frames"] for i in idx))
        agg["scale_to_robot"] = None if np.isnan(clips[idx[0]]["scale"]) else float(clips[idx[0]]["scale"])
        summary[src] = agg

    # ratios (fair: same 30 Hz sampling, same filter, normalised by hip height)
    def ratio(key, a=NAME_VIDEO, b=NAME_MOCAP):
        return summary[a][key]["mean"] / summary[b][key]["mean"]

    ratios = {
        "M1 body (norm) video/mocap": ratio("m1_body_norm"),
        "M1 feet (norm) video/mocap": ratio("m1_feet_norm"),
        "M1 body (norm) video_corr/mocap": ratio("m1_body_norm", NAME_VIDEO_CORR),
        "M1 feet (norm) video_corr/mocap": ratio("m1_feet_norm", NAME_VIDEO_CORR),
        "M1 body (mm raw) video/mocap": ratio("m1_body"),
        "M1 feet (mm raw) video/mocap": ratio("m1_feet"),
        "M1 body (mm robot units) video/mocap": summary[NAME_VIDEO]["m1_body"]["mean"] * VIDEO_REF_POS_SCALE / (summary[NAME_MOCAP]["m1_body"]["mean"] * MOCAP_REF_POS_SCALE),
        "M1 feet (mm robot units) video/mocap": summary[NAME_VIDEO]["m1_feet"]["mean"] * VIDEO_REF_POS_SCALE / (summary[NAME_MOCAP]["m1_feet"]["mean"] * MOCAP_REF_POS_SCALE),
        "M3 seg-len CV video/mocap": ratio("m3_seglen_cv"),
        "M3 seg-len CV video_corr/mocap": ratio("m3_seglen_cv", NAME_VIDEO_CORR),
        "M4 stance (norm) video/mocap": ratio("m4_stance_norm"),
        "M4 stance (norm) video_corr/mocap": ratio("m4_stance_norm", NAME_VIDEO_CORR),
    }
    # PSD high-band ratio (>5 Hz)
    psd = {}
    for src in [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_CORR]:
        f, P = averaged_psd([c for c in clips if c["source"] == src])
        psd[src] = dict(f=f.tolist(), P=P.tolist(), high_band_power=float(P[f >= 5].sum() * (f[1] - f[0])), low_band_power=float(P[(f > 0) & (f < 5)].sum() * (f[1] - f[0])))
    ratios["PSD power >5 Hz video/mocap"] = psd[NAME_VIDEO]["high_band_power"] / psd[NAME_MOCAP]["high_band_power"]
    ratios["PSD power >5 Hz video_corr/mocap"] = psd[NAME_VIDEO_CORR]["high_band_power"] / psd[NAME_MOCAP]["high_band_power"]

    # ---- downstream: roughness of the retargeted joint trajectories in the AMP expert files (what the discriminator sees)
    from common import AMP_DIRS, NAME_VIDEO_EXT, load_amp_dir

    amp = {}
    for src in [NAME_MOCAP, NAME_VIDEO, NAME_VIDEO_EXT]:
        num_res, num_acc, num_vel, den, den_acc = 0.0, 0.0, 0.0, 0, 0
        for c in load_amp_dir(AMP_DIRS[src]):
            q = c["frames"][:, 7:19]
            qd = c["frames"][:, 37:49]
            r = sg_residual(q, SG_WINDOW, SG_POLY)
            num_res += (r ** 2).sum()
            acc = np.diff(qd, axis=0) / c["fd"]
            num_acc += (acc ** 2).sum()
            den_acc += acc.size
            num_vel += (qd ** 2).sum()
            den += q.size
        amp[src] = dict(
            joint_pos_sg_residual_rms_deg=float(np.degrees(np.sqrt(num_res / den))),
            joint_acc_rms_rad_s2=float(np.sqrt(num_acc / den_acc)),
            joint_vel_rms_rad_s=float(np.sqrt(num_vel / den)),
        )

    with open(os.path.join(DATA, "exp1_metrics.json"), "w") as fh:
        json.dump(dict(per_clip=per_clip, summary=summary, ratios=ratios, psd=psd, amp_joint_roughness=amp, settings=dict(fs=FS, sg_window=SG_WINDOW, sg_poly=SG_POLY)), fh, indent=2)
    print(json.dumps(amp, indent=1))
    with open(os.path.join(DATA, "exp1_per_clip_table.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(json.dumps(summary, indent=1))
    print(json.dumps(ratios, indent=1))


if __name__ == "__main__":
    main()
