"""Experiment 1, paper figure: how noisy are the video keypoints, and what does the filter do?

Condenses the working figures into one double-column figure with the two panels that carry the
argument:

    (a) paw height over time for one walk clip, raw vs filtered -- the noise you can see
    (b) stance-phase height scatter, high-frequency part: how much a planted paw and a supported
        torso move when they should be holding still

Panel (b) is panel (b) of fig_exp1_stance_scatter, which is the second output of this script and
additionally shows the total (not just high-frequency) scatter.

Every video number here comes from the re-lift with the corrected 640x360 intrinsics
(exp1_relift_keypoints.py); the as-coded pipeline output is not shown, so the legend simply says
"Video".  Panels (b) and (c) are the *unfiltered* keypoints, i.e. the noise as it stands in the data;
the filter appears only in (a), as the thick trace.

Scatter metrics use the ordinary standard deviation, not 1.4826*MAD.  Over these clips the two agree
to within ~1 % on video and ~23 % on MoCap (MoCap is the smaller number, so the plain std is the more
conservative choice for the video/MoCap ratio) -- see fig_exp1_stance_scatter, which is the second
output of this script and extends the stance-height scatter from the paws to the torso.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common import (
    CLIP_SHORT,
    COLORS,
    COL_W,
    DATA,
    DOUBLE_W,
    NAME_MOCAP,
    NAME_VIDEO,
    FLAT_VIDEO_CLIPS,
    NAME_VIDEO_CORR,
    align_clip,
    hip_height,
    md_table,
    savefig,
    sg_residual,
)
from exp1_noise import FS, SG_WINDOW, SG_POLY, clip_metrics, get_clips
from exp1_filter import STANCE_Z, hampel_depth, lowpass, relift

# the two sources that appear in the figure; the corrected re-lift is simply labelled "Video"
SOURCES = [NAME_MOCAP, NAME_VIDEO_CORR]
LABEL = {NAME_MOCAP: "MoCap", NAME_VIDEO_CORR: "Video"}
COLOR = {NAME_MOCAP: COLORS[NAME_MOCAP], NAME_VIDEO_CORR: COLORS[NAME_VIDEO]}  # paper's Video colour
PAW_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

# Fig. 5 of the paper is drawn entirely in viridis; these samples let a figure sit beside it.
# Both series colours are far apart in luminance, so they survive a greyscale print, and the paw
# samples stop at 0.75 because the yellow end of viridis is too light for a thin line on white.
VIRIDIS_SERIES = {NAME_MOCAP: plt.get_cmap("viridis")(0.15), NAME_VIDEO_CORR: plt.get_cmap("viridis")(0.70)}
VIRIDIS_PAWS = [plt.get_cmap("viridis")(v) for v in (0.0, 0.25, 0.5, 0.75)]

TRACE_CLIP = "walk_869488000"
T_MAX = 0.6          # s, the window shown in panel (a)
N_CONTACT = 2        # a support phase = at least this many paws within STANCE_Z of the ground
MIN_SUPPORT = 5      # clips with fewer support frames are dropped from the body-height metric
Y_MAX_HF = 3.0       # % hip height, fixed upper limit of the paper figure's scatter panel


# ----------------------------------------------------------------------------------------------
def walk_traces():
    """Raw and A+B-filtered corrected re-lift of the trace clip -- one clip only, not the whole set."""
    m0 = relift(TRACE_CLIP, "half_res")["markers_ground"][:, 1:]                  # drop frame 0, as 3d_recon.py does
    m1 = relift(TRACE_CLIP, "half_res", preproc=hampel_depth)["markers_ground"][:, 1:]
    raw = m0[2:]                                                                   # (4, T, 3) paws
    fil = lowpass(m1[2:])
    return raw, fil


def filtered_video_clips():
    """The same 8 corrected clips after the proposed filter (depth Hampel + 6 Hz zero-phase low-pass).

    Only the video is filtered: the filter is part of the video method, MoCap is the unfiltered baseline.
    """
    out = []
    for clip in FLAT_VIDEO_CLIPS:
        m = relift(clip, "half_res", preproc=hampel_depth)["markers_ground"][:, 1:]
        body, feet, _ = align_clip(lowpass(m[:2]), lowpass(m[2:]))
        out.append(dict(source=NAME_VIDEO_CORR, name=CLIP_SHORT[clip], clip=clip, body=body, feet=feet, fs=FS))
    return out


def scatter(x, robust=False):
    """Spread of x: the ordinary std, or 1.4826 * median absolute deviation.

    The constant makes the robust version equal the std for Gaussian data while ignoring a few
    outlying samples.  On these clips the two agree to ~1 % on video and ~23 % on MoCap.
    """
    x = np.asarray(x, float)
    if not robust:
        return float(x.std())
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def planted_movement(c):
    """How far a paw moves between two consecutive frames while it is standing on the ground.

    A paw in contact cannot move, so this is measurement error with no signal-processing step in it:
    3-D displacement between frame t and t+1, over the frame pairs in which the paw is within
    STANCE_Z of the ground at both ends.  Median over those pairs and the four paws, in mm.
    """
    feet = c["feet"]
    z = feet[..., 2] - np.percentile(feet[..., 2], 5.0)
    both = (z[:, :-1] < STANCE_Z) & (z[:, 1:] < STANCE_Z)
    if both.sum() < 3:
        return np.nan
    return float(np.median(np.linalg.norm(np.diff(feet, axis=1), axis=-1)[both])) * 1000


def all_metrics(c, robust=False):
    """Every noise metric of the comparison figure, for one clip."""
    m, sc = clip_metrics(c), stance_scatter(c, robust=robust)
    hh = m["hip_height"]
    return dict(
        seglen_cv=m["m3_seglen_cv"] * 100,
        planted_mm=planted_movement(c),
        paw_tot=sc["paw_tot"],
        body_tot=sc["body_tot"],
        m1_body=m["m1_body"] / hh * 100,
        m1_feet=m["m1_feet"] / hh * 100,
        paw_hf=sc["paw_hf"],
        body_hf=sc["body_hf"],
    )


def stance_scatter(c, robust=False):
    """Height scatter during ground contact, for the paws and for the torso.

    paws  : per paw, the frames in the lowest 30 % of its smoothed height (the same stance rule as M4)
    torso : the frames in which at least N_CONTACT paws are within STANCE_Z of the ground
    Both are reported as the plain std of the height, and as the std of the high-frequency part only
    (height minus its local 0.23 s Savitzky-Golay trend), which removes the real vertical motion of
    the gait.  Values are in % of hip height.
    """
    body, feet = c["body"], c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    hh = hip_height(body, feet)

    paw_tot, paw_hf = [], []
    for f in feet:
        z = f[:, 2] - ground
        trend = z - sg_residual(z[:, None], SG_WINDOW, SG_POLY)[:, 0]
        st = trend <= np.percentile(trend, 30)
        paw_tot.append(scatter(z[st], robust))
        paw_hf.append(scatter(sg_residual(z[:, None], SG_WINDOW, SG_POLY)[st, 0], robust))

    n_contact = ((feet[..., 2] - ground) < STANCE_Z).sum(axis=0)
    sup = n_contact >= N_CONTACT
    bz = body[..., 2].mean(axis=0) - ground
    if sup.sum() >= MIN_SUPPORT:
        body_tot = scatter(bz[sup], robust)
        body_hf = scatter(sg_residual(bz[:, None], SG_WINDOW, SG_POLY)[sup, 0], robust)
    else:
        body_tot = body_hf = np.nan  # e.g. a canter has no two-paw support phase in a 15-frame window
    return dict(
        paw_tot=np.mean(paw_tot) / hh * 100,
        paw_hf=np.mean(paw_hf) / hh * 100,
        body_tot=body_tot / hh * 100,
        body_hf=body_hf / hh * 100,
        n_support=int(sup.sum()),
        n_frames=int(len(bz)),
    )


# ----------------------------------------------------------------------------------------------
def dot_column(ax, x, vals, color, alpha=0.85):
    """Per-clip dots plus the set mean, the same idiom as fig_exp1_noise_metrics."""
    vals = np.asarray(vals, float)
    v = vals[~np.isnan(vals)]
    jitter = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.055
    ax.scatter(x + jitter, v, s=7, color=color, alpha=alpha, lw=0, zorder=3)
    ax.hlines(v.mean(), x - 0.3, x + 0.3, color=color, lw=1.4, alpha=alpha, zorder=4)
    return v.mean()


def trace_panel(ax, paws, title):
    """Paw height of the trace clip over the first T_MAX seconds, raw (thin) and A+B filtered (thick)."""
    raw, fil = walk_traces()
    t = np.arange(raw.shape[1]) / FS
    keep = t <= T_MAX
    g0, g1 = np.percentile(raw[..., 2], 5), np.percentile(fil[..., 2], 5)
    for k, col in enumerate(paws):
        ax.plot(t[keep], raw[k, keep, 2] - g0, color=col, lw=0.6, alpha=0.45)
        ax.plot(t[keep], fil[k, keep, 2] - g1, color=col, lw=1.1)
    ax.axhline(STANCE_Z, color="0.5", lw=0.5, ls="--")
    ax.set_xlim(0, T_MAX)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("paw height [m]")
    ax.set_title(title, fontsize=7)
    ax.grid(alpha=0.3, lw=0.4)


def fig_noise(clips, sc, fname, robust=False, viridis=False):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 1.95), gridspec_kw=dict(width_ratios=[1.7, 1.0], wspace=0.3))
    paws = VIRIDIS_PAWS if viridis else PAW_COLORS
    color = VIRIDIS_SERIES if viridis else COLOR

    # ---- (a) paw height, raw vs filtered, first T_MAX seconds
    trace_panel(axes[0], paws, "(a) video paw height, walk clip\nthin raw, thick filtered")

    # ---- (b) stance-phase height scatter, high-frequency part (== fig_exp1_stance_scatter panel b)
    means = scatter_panel(axes[1], clips, sc, "hf", "(b) stance height scatter,\nhigh-frequency part",
                          ymax=Y_MAX_HF, robust=robust, color=color)

    source_legend(fig, -0.22, color=color)
    savefig(fig, fname)
    plt.close(fig)
    return means


def scatter_panel(ax, clips, sc, suffix, title, ymax=None, robust=False, color=None):
    """Per-clip stance-height scatter for the torso and the paws, both sources. Returns column means."""
    color = color or COLOR
    out, xt, xl, x = {}, [], [], 0
    for part in ["body", "paw"]:
        for src in SOURCES:
            vals = [s[f"{part}_{suffix}"] for c, s in zip(clips, sc) if c["source"] == src]
            out[(part, src)] = dot_column(ax, x, vals, color[src])
            xt.append(x)
            xl.append(f"{'torso' if part == 'body' else 'paws'}\n{LABEL[src]}")
            x += 1
        x += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=5.2)
    ax.set_ylabel(("robust std" if robust else "std") + " [% hip height]")
    ax.set_title(title, fontsize=7)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.set_ylim(0, ymax if ymax is not None else ax.get_ylim()[1] * 1.18)  # headroom for the ratio annotation
    for i, part in enumerate(["body", "paw"]):
        r = out[(part, NAME_VIDEO_CORR)] / out[(part, NAME_MOCAP)]
        ax.text(2.6 * i + 0.5, 0.94, f"{r:.1f}x", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
    return out


# full-band metrics on the top row, their high-frequency counterparts underneath
METRIC_PANELS = [
    ("seglen_cv", "std / mean [%]", "(a) torso length\nvariation"),
    ("planted_mm", "movement [mm/frame]", "(b) planted-paw\nmovement"),
    ("paw_tot", "std [% hip height]", "(c) paw stance\nscatter"),
    ("body_tot", "std [% hip height]", "(d) torso stance\nscatter"),
    ("m1_body", "RMS [% hip height]", "(e) HF residual,\ntorso"),
    ("m1_feet", "RMS [% hip height]", "(f) HF residual,\npaws"),
    ("paw_hf", "std [% hip height]", "(g) paw stance\nscatter, HF"),
    ("body_hf", "std [% hip height]", "(h) torso stance\nscatter, HF"),
]
SERIES = ["MoCap", "Video", "Video filtered"]
TICK = {"MoCap": "MoCap", "Video": "Video", "Video filtered": "Video\nfilt."}


SCATTER_KEYS = {"paw_tot", "body_tot", "paw_hf", "body_hf"}


def fig_all_metrics(am, fname, robust=False):
    """Every suggested metric, MoCap vs video before and after the filter.

    Top row is the full band, bottom row the high-frequency part only -- the filter collapses the
    bottom row and barely touches the top one, which is the point.
    """
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_W, 3.5), gridspec_kw=dict(wspace=0.62, hspace=0.95))
    style = {"MoCap": (COLORS[NAME_MOCAP], 0.85), "Video": (COLORS[NAME_VIDEO], 0.40),
             "Video filtered": (COLORS[NAME_VIDEO], 0.95)}
    means = {}
    for ax, (key, ylabel, title) in zip(axes.ravel(), METRIC_PANELS):
        if robust and key in SCATTER_KEYS:
            ylabel = ylabel.replace("std [", "robust std [")
        for x, ser in enumerate(SERIES):
            col, al = style[ser]
            means[(key, ser)] = dot_column(ax, x, [v[key] for v in am[ser]], col, alpha=al)
        ax.set_xticks(range(len(SERIES)))
        ax.set_xticklabels([TICK[s] for s in SERIES], fontsize=5.2)
        ax.set_xlim(-0.6, len(SERIES) - 0.4)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.tick_params(labelsize=5.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.24)  # headroom for the ratio annotations
        for x, ser in enumerate(SERIES[1:], start=1):
            ax.text(x, 0.96, f"{means[(key, ser)] / means[(key, 'MoCap')]:.1f}x",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.8,
                    color=style[ser][0], alpha=max(style[ser][1], 0.7))
    handles = [plt.Line2D([], [], marker="o", ls="-", ms=3, lw=1, color=style[s][0], alpha=style[s][1], label=s) for s in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))
    savefig(fig, fname)
    plt.close(fig)
    return means


def source_legend(fig, y, color=None):
    color = color or COLOR
    handles = [plt.Line2D([], [], marker="o", ls="-", color=color[s], label=LABEL[s], ms=3, lw=1) for s in SOURCES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, y))


def fig_stance(clips, sc, fname):
    """Stance-phase height scatter, paws and torso, total and high-frequency only."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw=dict(wspace=0.55))
    out = {}
    for ax, (suffix, title) in zip(axes, [("tot", "(a) stance height\nscatter"), ("hf", "(b) high-frequency\npart only")]):
        for (part, src), v in scatter_panel(ax, clips, sc, suffix, title).items():
            out[(part, suffix, src)] = v
    source_legend(fig, -0.28)
    savefig(fig, fname)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------------------------
def main():
    clips = get_clips()
    keep = [i for i, c in enumerate(clips) if c["source"] in SOURCES]
    clips = [clips[i] for i in keep]
    metrics = [clip_metrics(c) for c in clips]
    sc = [stance_scatter(c) for c in clips]

    fig_noise(clips, sc, "fig_exp1_noise_paper")

    fig_noise(clips, [stance_scatter(c, robust=True) for c in clips], "fig_exp1_noise_paper_robust", robust=True, viridis=True)

    # same figure with the video AFTER the filter, as a drop-in comparison (panel (a) is identical:
    # it already shows raw and filtered; only panel (b) changes)
    fclips = [c for c in clips if c["source"] == NAME_MOCAP] + filtered_video_clips()
    fsc = [stance_scatter(c) for c in fclips]
    fig_noise(fclips, fsc, "fig_exp1_noise_paper_filtered")
    filt = {f"{part}_{suf}": float(np.nanmean([s[f"{part}_{suf}"] for c, s in zip(fclips, fsc) if c["source"] == NAME_VIDEO_CORR]))
            for part in ["body", "paw"] for suf in ["tot", "hf"]}
    unfilt = {f"{part}_{suf}": float(np.nanmean([s[f"{part}_{suf}"] for c, s in zip(clips, sc) if c["source"] == NAME_VIDEO_CORR]))
              for part in ["body", "paw"] for suf in ["tot", "hf"]}
    mocap = {f"{part}_{suf}": float(np.nanmean([s[f"{part}_{suf}"] for c, s in zip(clips, sc) if c["source"] == NAME_MOCAP]))
             for part in ["body", "paw"] for suf in ["tot", "hf"]}
    filter_effect = {k: dict(mocap=mocap[k], video_unfiltered=unfilt[k], video_filtered=filt[k],
                             ratio_unfiltered=unfilt[k] / mocap[k], ratio_filtered=filt[k] / mocap[k])
                     for k in unfilt}

    # planted-paw movement: the simplest available noise number -- a paw touching the ground should
    # not move at all, so its 3-D displacement between two consecutive frames is pure error
    am = {"MoCap": [all_metrics(c) for c in clips if c["source"] == NAME_MOCAP],
          "Video": [all_metrics(c) for c in clips if c["source"] == NAME_VIDEO_CORR],
          "Video filtered": [all_metrics(c) for c in fclips if c["source"] == NAME_VIDEO_CORR]}
    all_means = fig_all_metrics(am, "fig_exp1_noise_metrics_all")

    # same figure with the original 1.4826*MAD estimator for the four scatter panels; the other four
    # are not standard deviations (a ratio, a median displacement, two RMS values) and are unchanged
    am_rob = {"MoCap": [all_metrics(c, robust=True) for c in clips if c["source"] == NAME_MOCAP],
              "Video": [all_metrics(c, robust=True) for c in clips if c["source"] == NAME_VIDEO_CORR],
              "Video filtered": [all_metrics(c, robust=True) for c in fclips if c["source"] == NAME_VIDEO_CORR]}
    rob_means = fig_all_metrics(am_rob, "fig_exp1_noise_metrics_all_robust", robust=True)
    rob_table = md_table(
        ["metric (robust std)"] + SERIES + ["Video/MoCap", "Video filt./MoCap"],
        [[t.replace("\n", " ").split(") ")[1]] + [rob_means[(k, s)] for s in SERIES]
         + [rob_means[(k, "Video")] / rob_means[(k, "MoCap")], rob_means[(k, "Video filtered")] / rob_means[(k, "MoCap")]]
         for k, _, t in METRIC_PANELS if k in SCATTER_KEYS], "{:.2f}")
    all_table = md_table(
        ["metric"] + SERIES + ["Video/MoCap", "Video filt./MoCap"],
        [[t.replace("\n", " ").split(") ")[1]] + [all_means[(k, s)] for s in SERIES]
         + [all_means[(k, "Video")] / all_means[(k, "MoCap")], all_means[(k, "Video filtered")] / all_means[(k, "MoCap")]]
         for k, _, t in METRIC_PANELS], "{:.2f}")

    planted = {}
    for tag, cs in [("MoCap", [c for c in clips if c["source"] == NAME_MOCAP]),
                    ("Video", [c for c in clips if c["source"] == NAME_VIDEO_CORR]),
                    ("Video filtered", [c for c in fclips if c["source"] == NAME_VIDEO_CORR])]:
        mm, pct = [], []
        for c in cs:
            feet = c["feet"]
            z = feet[..., 2] - np.percentile(feet[..., 2], 5.0)
            both = (z[:, :-1] < STANCE_Z) & (z[:, 1:] < STANCE_Z)  # planted in both frames
            step = np.linalg.norm(np.diff(feet, axis=1), axis=-1)[both]
            mm.append(float(np.median(step)) * 1000)
            pct.append(float(np.median(step)) / hip_height(c["body"], feet) * 100)
        planted[tag] = dict(mm_per_frame=float(np.mean(mm)), pct_hip_height=float(np.mean(pct)))
    for k in ["mm_per_frame", "pct_hip_height"]:
        planted[f"ratio {k} video/mocap"] = planted["Video"][k] / planted["MoCap"][k]
        planted[f"ratio {k} video_filtered/mocap"] = planted["Video filtered"][k] / planted["MoCap"][k]

    stance = fig_stance(clips, sc, "fig_exp1_stance_scatter")
    means = {(k, src): float(np.mean([m[k] * (100.0 if k == "m3_seglen_cv" else 100.0 / m["hip_height"])
                                      for c, m in zip(clips, metrics) if c["source"] == src]))
             for k in ["m3_seglen_cv", "m1_body"] for src in SOURCES}

    rows = []
    for c, m, s in zip(clips, metrics, sc):
        rows.append([
            LABEL[c["source"]], c["name"], m["n_frames"], m["hip_height"],
            m["m3_seglen_cv"] * 100, m["m1_body"] / m["hip_height"] * 100,
            s["paw_tot"], s["paw_hf"], s["body_tot"], s["body_hf"], f"{s['n_support']}/{s['n_frames']}",
        ])
    table = md_table(
        ["source", "clip", "frames@30Hz", "hip h [m]", "M3 seg-len CV [%]", "M1 body [%h]",
         "paw stance std [%h]", "paw HF [%h]", "torso stance std [%h]", "torso HF [%h]", "support frames"],
        rows, "{:.3f}")

    summary = {}
    for src in SOURCES:
        summary[LABEL[src]] = dict(
            m3_seglen_cv_pct=float(means[("m3_seglen_cv", src)]),
            m1_body_pct_h=float(means[("m1_body", src)]),
            paw_stance_std_pct_h=float(stance[("paw", "tot", src)]),
            paw_stance_hf_pct_h=float(stance[("paw", "hf", src)]),
            torso_stance_std_pct_h=float(stance[("body", "tot", src)]),
            torso_stance_hf_pct_h=float(stance[("body", "hf", src)]),
            n_clips=int(sum(c["source"] == src for c in clips)),
            n_clips_with_support=int(sum(1 for c, s in zip(clips, sc) if c["source"] == src and s["n_support"] >= MIN_SUPPORT)),
        )
    ratios = {k: summary["Video"][k] / summary["MoCap"][k] for k in summary["MoCap"] if not k.startswith("n_")}

    with open(os.path.join(DATA, "exp1_noise_paper_figure.json"), "w") as fh:
        json.dump(dict(settings=dict(fs=FS, sg_window=SG_WINDOW, sg_poly=SG_POLY, trace_clip=TRACE_CLIP,
                                     t_max_s=T_MAX, stance_z_m=STANCE_Z, n_contact=N_CONTACT,
                                     min_support_frames=MIN_SUPPORT, video_source="corrected 640x360 intrinsics",
                                     scatter_estimator="plain std"),
                       per_clip=rows, summary=summary, ratios=ratios,
                       filter_effect=filter_effect, planted_paw_movement=planted,
                       all_metrics_robust={f"{k}|{ser}": rob_means[(k, ser)] for k, _, _ in METRIC_PANELS for ser in SERIES},
                       all_metrics={f"{k}|{ser}": all_means[(k, ser)] for k, _, _ in METRIC_PANELS for ser in SERIES}), fh, indent=2)
    with open(os.path.join(DATA, "exp1_noise_paper_table.md"), "w") as fh:
        fh.write(table + "\n\n## All noise metrics, MoCap vs video before and after the filter\n\n" + all_table + "\n\n## The four scatter metrics with the robust (1.4826 MAD) estimator\n\n" + rob_table + "\n")
    print(table)
    print(json.dumps(summary, indent=1))
    print(all_table)
    print()
    print(rob_table)
    print(json.dumps(ratios, indent=1))
    print("filter effect (video, % hip height):", json.dumps(filter_effect, indent=1))
    print("planted-paw movement:", json.dumps(planted, indent=1))


if __name__ == "__main__":
    main()
