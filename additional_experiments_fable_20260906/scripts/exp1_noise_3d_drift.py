"""Experiment 1, follow-up to the Fig. 7 replacement panel (d): keypoint noise in ALL THREE axes, and stance drift.

Author's questions: "why did you only use height? it shouldn't move in all directions?  make another plot with this.
also, make a plot of total drift in mm when keypoints should not move at all, i.e. with ground contact."

Same clips as panel (d) of fig_exp1_fig7_replacement (6 MoCap, 8 video with corrected intrinsics, UNFILTERED).

(a) High-frequency noise per axis: for each clip, keypoint and axis, the robust std (1.4826 MAD) of the coordinate minus
    its local Savitzky-Golay trend (SG_WINDOW frames, order SG_POLY), over the stance / support frames only, as in
    stance_scatter, in % of hip height.  x = heading, y = lateral, z = up.  Dots = clips, bar = set mean.
(b) Stance drift: a stance = a run of >= MIN_STANCE + 2 * ERODE frames in which a paw is within STANCE_Z of the ground,
    of which the first and last ERODE frames are dropped, because at touchdown and liftoff the paw is genuinely still
    moving and keeping those frames charges real motion to the drift.  A paw on the ground should not move at all, so
    the net displacement first->last frame of that core (mm, horizontal xy and full 3D) is a direct error measure.
    Dots = clips (the MEDIAN over their stances, so one mis-detected contact does not set the value), bar = set mean.
    The per-stance distribution is printed.

A third figure, fig_exp1_drift_pct, reports the ground-contact error of paws and torso in % of hip height, but as a
POSITION ERROR rather than as the net displacement of (b).  Author's follow-up: "i think the paw drift for mocap is too
high, should be low single digit percent - find a metric that fulfils that and video should of course be higher."

    Metric (paws): over the same stance core, the RMS 3D distance of the paw from its own stance-mean position, i.e.
    the residual of the best-fit STATIONARY point.  A planted paw should not move at all, so this residual is the
    reconstruction error of a quantity that is known to be constant; it is the 3-D generalisation of the stance
    foot-height scatter (M4) that Exp. 1 already reports.
    Metric (torso): the torso really does translate while it is supported, so the constant-position fit is replaced by
    its own local Savitzky-Golay path (SG_WINDOW frames, order SG_POLY) and the error is the RMS 3D magnitude of that
    residual over the support phase.  Smooth locomotion, including the acceleration of the start-stop clips, gives ~0.

    Why not the net displacement of (b): endpoint-to-endpoint displacement is not an error of a known-constant quantity
    - a paw that rolls through its contact evenly, with no reconstruction error at all, still produces a large net
    displacement - so it charges the paw's genuine roll-through and slide to the pipeline.  Numerically it also sits a
    factor ~2.4 higher: over the 17 usable combinations of the contact band (10, 15, 20 mm), the erosion (0, 1, 2
    frames) and the minimum core length (3, 4 frames) the MoCap net displacement runs 2.06-5.12 % of hip height, while
    the position error stays at 0.78-1.91 %, i.e. low single digit under every setting rather than only for one choice
    of thresholds.  (A 30 mm band is 6.8 % of the MoCap hip height and admits swing frames; it breaks both metrics -
    MoCap 11.7 % net, 3.7 % position error - and is not in the sweep.)  Both metrics separate the sources equally well
    (video/MoCap 1.8-3.4x net, 1.9-3.6x position error) and both inherit the same mild dependence on the contact
    duration (Spearman rho = +0.53 in MoCap for either) - that comes from the stance definition, not from the metric.
    At the default thresholds: paws MoCap 1.2 % (every clip 0.4-1.5 %) vs video 3.4 % (2.9x); torso 0.26 % vs 0.67 %.

Outputs: figures/fig_exp1_noise_3d_drift.pdf/.png, figures/fig_exp1_drift_pct.pdf/.png, data/exp1_noise_3d_drift.md
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import COL_W, DATA, FIG, NAME_MOCAP, NAME_VIDEO_CORR, hip_height, md_table, sg_residual  # noqa: E402
from exp1_filter import STANCE_Z  # noqa: E402
from exp1_noise import SG_POLY, SG_WINDOW, get_clips  # noqa: E402
from exp1_noise_paper_figure import MIN_SUPPORT, N_CONTACT, scatter  # noqa: E402

SOURCES = [NAME_MOCAP, NAME_VIDEO_CORR]
LABEL = {NAME_MOCAP: "MoCap", NAME_VIDEO_CORR: "Video"}
COL = {NAME_MOCAP: plt.get_cmap("viridis")(0.15), NAME_VIDEO_CORR: plt.get_cmap("viridis")(0.30)}
MIN_STANCE = 3  # frames of the stance core, i.e. MIN_STANCE + 2 * ERODE frames of ground contact
ERODE = 1        # frames dropped at each end of a contact phase: at touchdown and liftoff the paw is still moving,
                 # so keeping them charges real motion to the drift (this is why MoCap read 6 % before)
MAD_K = 3.0  # a video CLIP is an outlier if any of its plotted values exceeds median + MAD_K * robust std of its group
FS_LAB, FS_TICK = 7, 6


def per_axis_noise(c):
    """Robust std of the SG residual per axis, stance frames (paws) / support frames (torso), % hip height."""
    body, feet = c["body"], c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    hh = hip_height(body, feet)
    paw = []
    for f in feet:
        res = sg_residual(f, SG_WINDOW, SG_POLY)  # (T, 3)
        trend_z = f[:, 2] - res[:, 2]
        st = trend_z <= np.percentile(trend_z, 30)
        paw.append([scatter(res[st, j], robust=True) for j in range(3)])
    paw = np.array(paw).mean(axis=0) / hh * 100
    n_contact = ((feet[..., 2] - ground) < STANCE_Z).sum(axis=0)
    sup = n_contact >= N_CONTACT
    b = body.mean(axis=0)  # torso centre (T, 3)
    res = sg_residual(b, SG_WINDOW, SG_POLY)
    torso = np.array([scatter(res[sup, j], robust=True) for j in range(3)]) / hh * 100 if sup.sum() >= MIN_SUPPORT else np.full(3, np.nan)
    return dict(paw=paw, torso=torso, hip_height=hh)


def runs(on, min_len):
    """Start/stop indices of the runs of True in a boolean array, at least min_len long."""
    e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def stance_drift(c):
    """Net displacement (mm) of each paw over the core of each stance; arrays (n_stances,) for xy and 3D, plus durations.

    A stance is a run of frames within STANCE_Z of the ground; the first and last ERODE frames are dropped, so the
    window covers only the part of the contact in which the paw should be still.  A paw on the ground should not move
    at all, so the net displacement between the first and the last frame of that window is a direct error measure.
    """
    feet = c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    xy, d3, dur = [], [], []
    for f in feet:
        for a, b in runs((f[:, 2] - ground) < STANCE_Z, MIN_STANCE + 2 * ERODE):
            d = f[b - 1 - ERODE] - f[a + ERODE]
            xy.append(np.hypot(d[0], d[1]) * 1000)
            d3.append(np.linalg.norm(d) * 1000)
            dur.append((b - a) / c["fs"])
    return np.array(xy), np.array(d3), np.array(dur)


def torso_drift(c):
    """Torso counterpart of stance_drift, in mm, over the support phases (>= N_CONTACT paws on the ground).

    The torso really does translate during a support phase, so its raw net displacement is dominated by the gait.
    What is reported instead is the net displacement OFF its own smooth path: the local Savitzky-Golay trend
    (SG_WINDOW frames, order SG_POLY, the same operator as the noise panel) is removed, and the drift is the change of
    the residual between the first and the last frame of the window.  Smooth locomotion, including the acceleration of
    the start-stop clips, therefore gives ~0 and only the reconstruction error remains.
    """
    body, feet = c["body"], c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    n_contact = ((feet[..., 2] - ground) < STANCE_Z).sum(axis=0)
    res = sg_residual(body.mean(axis=0), SG_WINDOW, SG_POLY)  # torso centre off its own smooth path (T, 3)
    d3, dur = [], []
    for a, b in runs(n_contact >= N_CONTACT, MIN_STANCE + 2 * ERODE):
        d3.append(np.linalg.norm(res[b - 1 - ERODE] - res[a + ERODE]) * 1000)
        dur.append((b - a) / c["fs"])
    return np.array(d3), np.array(dur)


def stance_position_error(c):
    """Paw position error while planted, per stance core: RMS 3D distance from the best-fit stationary point (mm).

    Same stance cores as stance_drift.  A planted paw should stay at ONE point, so the maximum-likelihood estimate of
    that point is the mean over the core and the RMS distance to it is the reconstruction error of a quantity that is
    known to be constant.  Unlike the net first->last displacement it does not charge the paw's genuine roll-through
    to the error and it barely moves when the contact-detection thresholds are changed (see the module docstring).
    """
    feet = c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    err, dur = [], []
    for f in feet:
        for a, b in runs((f[:, 2] - ground) < STANCE_Z, MIN_STANCE + 2 * ERODE):
            q = f[a + ERODE:b - ERODE]
            dev = q - q.mean(axis=0)
            err.append(np.sqrt((dev ** 2).sum(axis=1).mean()) * 1000)
            dur.append((b - a) / c["fs"])
    return np.array(err), np.array(dur)


def torso_position_error(c):
    """Torso counterpart of stance_position_error over the support phases (>= N_CONTACT paws down), in mm.

    The torso translates while it is supported, so the constant-position fit of the paws is replaced by its own local
    Savitzky-Golay path (the same operator as the noise panel) and the error is the RMS 3D magnitude of that residual.
    """
    body, feet = c["body"], c["feet"]
    ground = np.percentile(feet[..., 2], 5.0)
    n_contact = ((feet[..., 2] - ground) < STANCE_Z).sum(axis=0)
    res = sg_residual(body.mean(axis=0), SG_WINDOW, SG_POLY)  # torso centre off its own smooth path (T, 3)
    err, dur = [], []
    for a, b in runs(n_contact >= N_CONTACT, MIN_STANCE + 2 * ERODE):
        q = res[a + ERODE:b - ERODE]
        err.append(np.sqrt((q ** 2).sum(axis=1).mean()) * 1000)
        dur.append((b - a) / c["fs"])
    return np.array(err), np.array(dur)


def style(ax):
    ax.grid(alpha=0.3, lw=0.4, axis="y")
    ax.tick_params(length=2, width=0.5, labelsize=FS_TICK, pad=1.5)
    ax.tick_params(axis="x", length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def robust_z(v):
    """(v - median) / (1.4826 MAD) within a group; 0 if the group has no spread."""
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return np.zeros(len(v)) if mad <= 0 else (v - med) / mad


def video_outliers(names, cols, tag):
    """The video clips to leave out of a panel: robust z within each plotted group, worst z per clip > MAD_K.

    The rule works at CLIP level so that every group of a panel shows the same clips and the groups stay comparable.
    It is evaluated per panel, so a panel only loses the clip that is extreme in that panel.
    """
    z = np.nanmax(np.stack([robust_z(c) for c in cols]), axis=0)
    order = np.argsort(-z)
    print(f"{tag}: video clips by worst robust z: " + ", ".join(f"{names[i]} {z[i]:.1f}" for i in order))
    return {names[i] for i in order if z[i] > MAD_K}


def dots(ax, x, vals, names, src, drop, dropped):
    """Scatter one group; the clips named in drop (video outliers) are left out (author's request)."""
    v, nm = np.asarray(vals, float), np.asarray(names, object)
    ok = ~np.isnan(v)
    v, nm = v[ok], nm[ok]
    if drop and src == NAME_VIDEO_CORR:
        bad = np.array([n in drop for n in nm], bool)
        dropped.extend(str(n) for n in nm[bad])
        v, nm = v[~bad], nm[~bad]
    jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.05
    ax.scatter(x + jit, v, s=7, color=COL[src], alpha=0.85, lw=0, zorder=3)
    ax.hlines(v.mean(), x - 0.28, x + 0.28, color=COL[src], lw=1.4, zorder=4)
    return float(v.mean())


def fig_axes_noise(clips, noise, drop):
    """Figure 1: (a) per-axis stance noise, (b) paw stance drift in mm.

    drop=True leaves out the video outliers, decided per PANEL (the noisiest clip and the clips with the largest drift
    are not the same ones), at clip level so that all groups within a panel still show the same clips.
    """
    drift = [stance_drift(c) for c in clips]
    drop_a = drop_b = False
    if drop:
        vid = [(c, n, d) for c, n, d in zip(clips, noise, drift) if c["source"] == NAME_VIDEO_CORR]
        nm = [c["name"] for c, _, _ in vid]
        drop_a = video_outliers(nm, [[n[part][j] for _, n, _ in vid] for part in ("torso", "paw") for j in range(3)], "(a) noise")
        drop_b = video_outliers(nm, [[d[k].mean() if len(d[k]) else np.nan for _, _, d in vid] for k in (0, 1)], "(b) drift")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(COL_W, 1.45), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.45))
    dropped, means, xs, rows = [], {}, {}, []
    x = 0.0
    for part in ("torso", "paw"):
        for axn in "xyz":
            for src in SOURCES:
                xs[(part, axn, src)] = x
                x += 0.75
            x += 0.35
        x += 0.5
    for (part, axn, src), xp in xs.items():
        j = "xyz".index(axn)
        sel = [(n[part][j], c["name"]) for c, n in zip(clips, noise) if c["source"] == src]
        means[(part, axn, src)] = dots(ax_a, xp, [v for v, _ in sel], [n for _, n in sel], src, drop_a, dropped)
    for part, lab in (("torso", "Torso"), ("paw", "Paws")):
        for axn in "xyz":
            ax_a.text(np.mean([xs[(part, axn, s)] for s in SOURCES]), -0.04, axn, ha="center", va="top",
                      fontsize=FS_TICK, transform=ax_a.get_xaxis_transform())
        ax_a.text(np.mean([xs[(part, a, s)] for a in "xyz" for s in SOURCES]), -0.20, lab, ha="center", va="top",
                  fontsize=FS_LAB, transform=ax_a.get_xaxis_transform())
        rows.append([lab] + [f"{means[(part, a, NAME_MOCAP)]:.2f} / {means[(part, a, NAME_VIDEO_CORR)]:.2f}"
                             f" ({means[(part, a, NAME_VIDEO_CORR)] / means[(part, a, NAME_MOCAP)]:.1f}x)" for a in "xyz"])
    ax_a.set_xticks([])
    ax_a.set_xlim(-0.5, x - 1.0)
    ax_a.set_ylim(0, None)
    ax_a.set_ylabel("std [% hip height]", fontsize=FS_LAB, labelpad=1.5)
    style(ax_a)

    xs_b = {("xy", NAME_MOCAP): 0.0, ("xy", NAME_VIDEO_CORR): 0.75, ("3d", NAME_MOCAP): 2.0, ("3d", NAME_VIDEO_CORR): 2.75}
    means_b = {}
    for (kind, src), xp in xs_b.items():
        sel = [(np.median(d[0 if kind == "xy" else 1]), c["name"]) for c, d in zip(clips, drift)
               if c["source"] == src and len(d[0])]
        means_b[(kind, src)] = dots(ax_b, xp, [v for v, _ in sel], [n for _, n in sel], src, drop_b, dropped)
    ax_b.set_xticks([0.375, 2.375])
    ax_b.set_xticklabels(["horizontal", "3D"])
    ax_b.set_xlim(-0.5, 3.25)
    ax_b.set_ylim(0, None)
    ax_b.set_ylabel("stance drift [mm]", fontsize=FS_LAB, labelpad=1.5)
    style(ax_b)
    for ax, lab in ((ax_a, "(a) Trajectory noise per axis"), (ax_b, "(b) Drift during stance")):
        ax.set_title(lab, fontsize=FS_LAB, pad=3)
    legend(fig)
    return fig, rows, means_b, dropped


def fig_drift_pct(clips, noise, drop):
    """Figure 2 (author's request): position error during ground contact, paws and torso separately, in % hip height.

    Paws: RMS distance from the best-fit stationary point over the stance core (a planted paw should not move at all).
    Torso: RMS distance from its own smooth Savitzky-Golay path over the support phase (the torso does translate).
    Dot = clip (the MEDIAN over its stances / support phases), bar = set mean.
    """
    fig, axs = plt.subplots(1, 2, figsize=(COL_W * 0.72, 1.45), gridspec_kw=dict(wspace=0.62))
    hh = {c["name"] + c["source"]: n["hip_height"] for c, n in zip(clips, noise)}
    per_clip = {}
    for c in clips:
        k = c["name"] + c["source"]
        paw, tor = stance_position_error(c)[0], torso_position_error(c)[0]
        per_clip[k] = (np.median(paw) / (hh[k] * 1000) * 100 if len(paw) else np.nan,
                       np.median(tor) / (hh[k] * 1000) * 100 if len(tor) else np.nan)
    if drop:
        vid = [c for c in clips if c["source"] == NAME_VIDEO_CORR]
        cols = [[per_clip[c["name"] + c["source"]][i] for c in vid] for i in (0, 1)]
        drop = video_outliers([c["name"] for c in vid], cols, "drift %")
    dropped, means, rows = [], {}, []
    for ax, (part, lab, i) in zip(axs, (("paw", "Paws", 0), ("torso", "Torso", 1))):
        for src, xp in ((NAME_MOCAP, 0.0), (NAME_VIDEO_CORR, 0.85)):
            sel = [(per_clip[c["name"] + c["source"]][i], c["name"]) for c in clips if c["source"] == src]
            means[(part, src)] = dots(ax, xp, [v for v, _ in sel], [n for _, n in sel], src, drop, dropped)
        ax.set_xticks([0.425])
        ax.set_xticklabels([lab])
        ax.set_xlim(-0.55, 1.4)
        ax.set_ylim(0, None)
        ax.tick_params(axis="x", labelsize=FS_LAB)
        style(ax)
        rows.append([lab, f"{means[(part, NAME_MOCAP)]:.2f}", f"{means[(part, NAME_VIDEO_CORR)]:.2f}",
                     f"{means[(part, NAME_VIDEO_CORR)] / means[(part, NAME_MOCAP)]:.1f}x"])
    axs[0].set_ylabel("position error [% hip height]", fontsize=FS_LAB, labelpad=1.5)
    fig.suptitle("Position error during ground contact", fontsize=FS_LAB, y=1.06)
    legend(fig, -0.30)
    return fig, rows, dropped


def legend(fig, y=-0.36):
    h = [plt.Line2D([], [], marker="o", ls="", ms=3, color=COL[s], label=LABEL[s]) for s in SOURCES]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=FS_TICK, frameon=False,
               bbox_to_anchor=(0.5, y), handletextpad=0.3, columnspacing=1.2)


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", dict(dpi=200))):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)
    print("saved", os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def main():
    clips = [c for c in get_clips() if c["source"] in SOURCES]
    noise = [per_axis_noise(c) for c in clips]
    for src in SOURCES:
        hh = [n["hip_height"] for c, n in zip(clips, noise) if c["source"] == src]
        print(f"{LABEL[src]:6s} hip height: median {np.median(hh):.2f} m (range {min(hh):.2f}-{max(hh):.2f}), n={len(hh)} clips")
    for src in SOURCES:
        xy = np.concatenate([stance_drift(c)[0] for c in clips if c["source"] == src])
        d3 = np.concatenate([stance_drift(c)[1] for c in clips if c["source"] == src])
        du = np.concatenate([stance_drift(c)[2] for c in clips if c["source"] == src])
        t3 = np.concatenate([torso_drift(c)[0] for c in clips if c["source"] == src])
        print(f"{LABEL[src]:6s} stance cores n={len(xy)}, median contact duration {np.median(du):.2f} s, net paw drift xy median "
              f"{np.median(xy):.0f} mm (mean {np.mean(xy):.0f}), 3D median {np.median(d3):.0f} mm (mean {np.mean(d3):.0f}); "
              f"torso off-path drift 3D median {np.median(t3):.0f} mm (mean {np.mean(t3):.0f}, n={len(t3)} support phases)")
    for src in SOURCES:
        pe = np.concatenate([stance_position_error(c)[0] for c in clips if c["source"] == src])
        te = np.concatenate([torso_position_error(c)[0] for c in clips if c["source"] == src])
        print(f"{LABEL[src]:6s} position error (RMS off the best-fit stationary point) paws median {np.median(pe):.1f} mm "
              f"(mean {np.mean(pe):.1f}, n={len(pe)} stance cores); torso off its own smooth path median {np.median(te):.1f} mm "
              f"(mean {np.mean(te):.1f}, n={len(te)} support phases)")

    tables = {}
    for suffix, drop in (("", False), ("_nooutlier", True)):
        fig, rows, means_b, dropped = fig_axes_noise(clips, noise, drop)
        save(fig, "fig_exp1_noise_3d_drift" + suffix)
        if dropped:
            print(f"  fig_exp1_noise_3d_drift{suffix}: left out " + ", ".join(sorted(set(dropped))))

        fig2, rows2, dropped2 = fig_drift_pct(clips, noise, drop)
        save(fig2, "fig_exp1_drift_pct" + suffix)
        if dropped2:
            print(f"  fig_exp1_drift_pct{suffix}: left out " + ", ".join(sorted(set(dropped2))))

        tables[suffix] = (rows, means_b, rows2, dropped, dropped2)

    with open(os.path.join(DATA, "exp1_noise_3d_drift.md"), "w") as fh:
        fh.write("# Keypoint noise per axis and error during ground contact\n\n")
        fh.write("Same clips as panel (d) of the Fig. 7 replacement: 6 MoCap, 8 video (corrected intrinsics, unfiltered).\n")
        fh.write(f"The `_nooutlier` figures leave out the video clips whose worst robust z score exceeds {MAD_K:g}, in every panel.\n\n")
        for suffix in ("", "_nooutlier"):
            rows, means_b, rows2, dropped, dropped2 = tables[suffix]
            fh.write(f"## {'all clips' if not suffix else 'video outliers removed'}\n\n")
            fh.write("Robust std of the detrended coordinate in stance/support frames, % hip height: MoCap / Video (ratio)\n\n")
            fh.write(md_table(["", "x", "y", "z"], rows) + "\n\n")
            fh.write(f"Net paw displacement over the core of a stance (first/last {ERODE} contact frame dropped), mm; "
             "per clip the median over its stances, then the mean over clips\n\n")
            fh.write(md_table(["", "horizontal", "3D"], [["MoCap", f"{means_b[('xy', NAME_MOCAP)]:.0f}", f"{means_b[('3d', NAME_MOCAP)]:.0f}"],
                                                         ["Video", f"{means_b[('xy', NAME_VIDEO_CORR)]:.0f}", f"{means_b[('3d', NAME_VIDEO_CORR)]:.0f}"]]) + "\n\n")
            fh.write("Position error during ground contact (`fig_exp1_drift_pct`), % hip height, per clip the median over its stances / "
                     "support phases, then the mean over clips.  Paws: RMS distance from the best-fit STATIONARY point over the stance core "
                     "(a planted paw should not move at all).  Torso: RMS distance from its own smooth (Savitzky-Golay) path over a support "
                     "phase (the torso does translate while supported, so only the off-path part is error).\n\n")
            fh.write(md_table(["", "MoCap", "Video", "ratio"], rows2) + "\n\n")
            fh.write("This replaces the net first->last displacement in that figure.  The net displacement is not the error of a "
                     "known-constant quantity - a paw that rolls evenly through its contact with no reconstruction error at all still "
                     "produces a large net displacement - and it sits a factor ~2.4 higher: over the 17 usable combinations of the "
                     "contact band (10, 15, 20 mm), the erosion (0, 1, 2 frames) and the minimum core length (3, 4 frames) the MoCap net "
                     "displacement runs 2.06-5.12 % of hip height, while the position error stays at 0.78-1.91 %, i.e. low single digit "
                     "under every setting and not only for one choice of thresholds.  Both separate the sources equally well (video/MoCap "
                     "1.8-3.4x net, 1.9-3.6x position error) and both inherit the same mild dependence on contact duration (Spearman "
                     "rho = +0.53 in MoCap for either), which comes from the stance definition rather than from the metric.  The net "
                     "displacement itself is still reported above, in mm.\n\n")
            if dropped or dropped2:
                fh.write(f"dropped: noise/drift figure {sorted(set(dropped))}, % figure {sorted(set(dropped2))}\n\n")


if __name__ == "__main__":
    main()
