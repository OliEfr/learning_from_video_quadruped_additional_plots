"""Experiment 2 -- What additional part of the target distribution do the video
motions cover?

Fig C  command-space coverage, 3-way, Fig-5 layout
Fig G  target command distribution vs demonstrated velocity distribution
Fig F  coverage over the 24-dim AMP discriminator state
Fig D  terrain / posture coverage (stairs, box, stand-up)
"""
import json, os, itertools
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import common as C

BOX = np.array([[-1.0, 1.0], [-0.3, 0.3], [-1.57, 1.57]])   # vx, vy, wz
NBINS = (10, 6, 10)
TAU = 0.5
AXLBL = ["Target Vel. X [m/s]", "Target Vel. Y [m/s]", "Target Ang. Vel. Z [rad/s]"]

# ------------------------------------------------------- command-space samples
def yaw_from_quat_xyzw(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def body_velocity(F, dt, lowpass=6.0):
    """-> (T-2, 3) body-frame (vx, vy, wz). Recomputed from root pose; the stored
    31:34 block is a backward difference with frame 0 duplicated (F5) and the
    34:37 angular block is all zeros (F4)."""
    p, q = F[:, 0:3].copy(), F[:, 3:7].copy()
    flip = np.cumprod(np.where((q[1:] * q[:-1]).sum(1) < 0, -1.0, 1.0))
    q[1:] *= flip[:, None]
    psi = yaw_from_quat_xyzw(q)
    psi_u = np.unwrap(psi)
    if lowpass and len(p) > 15:
        b, a = butter(4, lowpass / (0.5 / dt), btype="low")
        p = filtfilt(b, a, p, axis=0, padtype="odd", padlen=3 * 5)
        psi_u = filtfilt(b, a, psi_u, padtype="odd", padlen=3 * 5)
    v = np.gradient(p, dt, axis=0)
    wz = np.gradient(psi_u, dt)
    c, s = np.cos(psi), np.sin(psi)
    vx = c * v[:, 0] + s * v[:, 1]
    vy = -s * v[:, 0] + c * v[:, 1]
    return np.stack([vx, vy, wz], 1)[1:-1]      # drop non-generic end samples

def dataset_cmd(folder, lowpass=6.0):
    return np.concatenate([body_velocity(F, dt, lowpass) for _, F, dt in C.load_amp(folder)])

# ------------------------------------------------------------------ coverage
def grid_centres():
    ax = [np.linspace(BOX[i, 0], BOX[i, 1], NBINS[i] + 1) for i in range(3)]
    ce = [0.5 * (a[1:] + a[:-1]) for a in ax]
    G = np.stack(np.meshgrid(*ce, indexing="ij"), -1).reshape(-1, 3)
    cell = np.array([(BOX[i, 1] - BOX[i, 0]) / NBINS[i] for i in range(3)])
    return G, ce, cell

def covered(X, tau=TAU):
    """Cell covered iff a demo lies within tau cell-normalized units."""
    G, ce, cell = grid_centres()
    d = np.linalg.norm((G[:, None, :] - X[None, :, :]) / cell, axis=-1).min(1)
    return d.reshape(NBINS) <= tau, d.reshape(NBINS)

def cov_pct(mask):
    return 100.0 * mask.sum() / mask.size

def bootstrap_new(folders_a, folders_b, n=2000, seed=0):
    """95% CI on New = |B\\A|/N, resampling whole clips (frames autocorrelate)."""
    rng = np.random.default_rng(seed)
    A = [body_velocity(F, dt) for _, F, dt in C.load_amp(folders_a)]
    B = [body_velocity(F, dt) for _, F, dt in C.load_amp(folders_b)]
    out = []
    for _ in range(n):
        a = np.concatenate([A[i] for i in rng.integers(0, len(A), len(A))])
        b = np.concatenate([B[i] for i in rng.integers(0, len(B), len(B))])
        ma, mb = covered(a)[0], covered(b)[0]
        out.append(100.0 * (mb & ~ma).sum() / ma.size)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

# ------------------------------------------------------------------- figure C
def figure_C(sets):
    G, ce, cell = grid_centres()
    masks = {k: covered(v)[0] for k, v in sets.items()}
    dists = {k: covered(v)[1] for k, v in sets.items()}
    rows = list(sets)
    fig = plt.figure(figsize=(C.IEEE_2COL, 5.2))
    gs = fig.add_gridspec(3, 3, wspace=0.52, hspace=0.42)
    planes = [(0, 1), (0, 2)]
    ref = masks[rows[0]]
    cmap4 = ListedColormap(["#EAEAEA", C.C_MOCAP, C.C_VIDEO_EXT, "#5B5B5B"])

    for r, name in enumerate(rows):
        for c_, (i, j) in enumerate(planes):
            ax = fig.add_subplot(gs[r, c_])
            ax_sum = tuple(k for k in range(3) if k not in (i, j))
            D = dists[name].min(axis=ax_sum)
            if i > j:
                D = D.T
            im = ax.pcolormesh(ce[i], ce[j], -D.T, cmap="viridis",
                               vmin=-3.0, vmax=0.0, shading="auto")
            ax.contour(ce[i], ce[j], D.T, levels=[TAU], colors="w", linewidths=0.8)
            ax.scatter(sets[name][:, i], sets[name][:, j], s=1.6, c="w", alpha=0.45, lw=0)
            ax.add_patch(plt.Rectangle((BOX[i, 0], BOX[j, 0]), BOX[i, 1] - BOX[i, 0],
                                       BOX[j, 1] - BOX[j, 0], fill=False, ec="red",
                                       ls="--", lw=0.8))
            ax.set_xlim(*BOX[i]); ax.set_ylim(*BOX[j])
            ax.tick_params(labelsize=6)
            if r == 2:
                ax.set_xlabel(AXLBL[i], fontsize=7)
            ax.set_ylabel(AXLBL[j], fontsize=7)
            if r == 0:
                ax.set_title("distance to nearest demo\n"
                             + [r"$(v_x,\ v_y)$", r"$(v_x,\ \omega_z)$"][c_],
                             fontsize=7.5)
        # set-difference column
        ax = fig.add_subplot(gs[r, 2])
        m = masks[name]
        cat = np.zeros(NBINS, int)
        cat[ref & ~m] = 1
        cat[m & ~ref] = 2
        cat[m & ref] = 3
        if r == 0:
            cat[ref] = 3
        Z = cat.max(axis=2)
        ax.pcolormesh(ce[0], ce[1], Z.T, cmap=cmap4,
                      norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), shading="auto")
        ax.set_xlim(*BOX[0]); ax.set_ylim(*BOX[1]); ax.tick_params(labelsize=6)
        ax.set_ylabel(AXLBL[1], fontsize=7)
        if r == 2:
            ax.set_xlabel(AXLBL[0], fontsize=7)
        if r == 0:
            ax.set_title("coverage vs MoCap", fontsize=7.5)
        new = 100.0 * (m & ~ref).sum() / m.size
        ax.text(0.02, 0.96, f"cov {cov_pct(m):.1f}%" + ("" if r == 0 else f"\nnew +{new:.1f} pp"),
                transform=ax.transAxes, fontsize=6.4, va="top", ha="left",
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.2))
        fig.text(0.005, [0.80, 0.505, 0.21][r], name, fontsize=8, rotation=90,
                 va="center", ha="left", fontweight="bold")

    cb = fig.colorbar(im, ax=fig.axes, shrink=0.35, pad=0.015)
    cb.set_label("-distance to nearest demo [cells]", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    h = [plt.Rectangle((0, 0), 1, 1, fc=cmap4(k)) for k in range(4)]
    fig.legend(h, ["not covered", "MoCap only", "new in video", "covered by both"],
               ncol=4, fontsize=6.8, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    return C.save(fig, "fig_C_command_coverage.pdf")

# ------------------------------------------------------------------- figure G
def figure_G(sets):
    rows = list(sets)
    fig = plt.figure(figsize=(C.IEEE_2COL, 5.4))
    gs = fig.add_gridspec(3, 4, wspace=0.58, hspace=0.45)
    nb = (12, 8)
    for r, name in enumerate(rows):
        X = sets[name]
        for c_, (i, j) in enumerate([(0, 1), (0, 2)]):
            ax = fig.add_subplot(gs[r, c_])
            H, xe, ye = np.histogram2d(X[:, i], X[:, j], bins=nb,
                                       range=[BOX[i], BOX[j]], density=True)
            ax.pcolormesh(xe, ye, H.T, cmap="viridis", shading="auto")
            ax.add_patch(plt.Rectangle((BOX[i, 0], BOX[j, 0]), BOX[i, 1] - BOX[i, 0],
                                       BOX[j, 1] - BOX[j, 0], fill=False, ec="red",
                                       ls="--", lw=0.8))
            ax.tick_params(labelsize=6); ax.set_ylabel(AXLBL[j], fontsize=7)
            if r == 2:
                ax.set_xlabel(AXLBL[i], fontsize=7)
            if r == 0:
                ax.set_title("demonstrated density\n"
                             + [r"$(v_x,\ v_y)$", r"$(v_x,\ \omega_z)$"][c_], fontsize=7.5)
        # log density ratio vs the (uniform) target
        ax = fig.add_subplot(gs[r, 2])
        Hc, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=nb, range=[BOX[0], BOX[1]])
        P = Hc / Hc.sum()
        U = np.full_like(P, 1.0 / P.size)
        # Cells with no demonstration at all are shown as a separate category --
        # otherwise they saturate the diverging scale and hide the actual
        # over/under-representation where data does exist.
        R = np.ma.masked_where(Hc == 0, np.log10(np.where(Hc > 0, P, 1) / U))
        cmR = plt.get_cmap("RdBu_r").copy(); cmR.set_bad("#DDDDDD")
        im2 = ax.pcolormesh(xe, ye, R.T, cmap=cmR,
                            norm=TwoSlopeNorm(0.0, -1.5, 1.5), shading="auto")
        ax.text(.02, .04, f"{100*(Hc==0).mean():.0f}% empty", transform=ax.transAxes,
                fontsize=5.8, va="bottom",
                bbox=dict(fc="w", ec="none", alpha=.75, pad=1.0))
        ax.tick_params(labelsize=6); ax.set_ylabel(AXLBL[1], fontsize=7)
        if r == 2:
            ax.set_xlabel(AXLBL[0], fontsize=7)
        if r == 0:
            ax.set_title(r"$\log_{10}(p_{data}/p_{target})$", fontsize=7.5)
        cb = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.03); cb.ax.tick_params(labelsize=5.5)
        # marginals
        ax = fig.add_subplot(gs[r, 3])
        for k, (lab, col) in enumerate(zip(["$v_x$", "$v_y$", r"$\omega_z$"],
                                           ["#333", "#777", "#bbb"])):
            h, e = np.histogram(X[:, k], bins=14, range=BOX[k], density=True)
            u = 1.0 / (BOX[k, 1] - BOX[k, 0])
            ax.plot((e[:-1] + e[1:]) / 2 / (BOX[k, 1]), h / u, color=col, lw=1.1, label=lab)
        ax.axhline(1.0, ls="--", c="red", lw=0.8)
        ax.set_ylim(0, 4); ax.tick_params(labelsize=6)
        ax.set_ylabel(r"$p_{data}/p_{target}$", fontsize=7)
        if r == 2:
            ax.set_xlabel("command axis / its range", fontsize=7)
        if r == 0:
            ax.set_title("marginals vs uniform", fontsize=7.5)
        ax.legend(fontsize=5.5, frameon=False, ncol=3)
        fig.text(0.005, [0.80, 0.505, 0.21][r], name, fontsize=8, rotation=90,
                 va="center", ha="left", fontweight="bold")
    return C.save(fig, "fig_G_target_vs_data_distribution.pdf")

# ------------------------------------------------------------------- figure F
JOINTS = ["hip", "thigh", "calf"]

def amp_state(folder):
    """-> (N, 4, 3, 2): per sample, per leg, per joint type, (pos, vel)."""
    out = []
    for _, F, dt in C.load_amp(folder):
        jp, jv = F[:, 7:19], F[:, 37:49]
        out.append(np.stack([jp.reshape(-1, 4, 3), jv.reshape(-1, 4, 3)], -1))
    return np.concatenate(out)

def figure_F(state_sets):
    rows = list(state_sets)
    lims = []
    for j in range(3):
        allp = np.concatenate([v[:, :, j, 0].ravel() for v in state_sets.values()])
        allv = np.concatenate([v[:, :, j, 1].ravel() for v in state_sets.values()])
        lims.append(((np.percentile(allp, .5), np.percentile(allp, 99.5)),
                     (np.percentile(allv, .5), np.percentile(allv, 99.5))))
    NB = 14
    occ = {}
    fig = plt.figure(figsize=(C.IEEE_2COL, 5.2))
    gs = fig.add_gridspec(3, 4, wspace=0.40, hspace=0.45)
    cmap4 = ListedColormap(["#EAEAEA", C.C_MOCAP, C.C_VIDEO_EXT, "#5B5B5B"])
    ref_masks = {}
    for r, name in enumerate(rows):
        S = state_sets[name]
        occ[name] = {}
        for j in range(3):
            ax = fig.add_subplot(gs[r, j])
            p = S[:, :, j, 0].ravel(); v = S[:, :, j, 1].ravel()
            H, xe, ye = np.histogram2d(p, v, bins=NB, range=[lims[j][0], lims[j][1]])
            ax.pcolormesh(xe, ye, (H > 0).T * 1.0 + np.log1p(H).T, cmap="viridis",
                          shading="auto")
            occ[name][JOINTS[j]] = 100.0 * (H > 0).sum() / H.size
            m = H > 0
            if r == 0:
                ref_masks[j] = m
            ax.tick_params(labelsize=6)
            ax.set_xlabel(f"{JOINTS[j]} pos [rad]", fontsize=7)
            if j == 0:
                ax.set_ylabel("joint vel [rad/s]", fontsize=7)
            if r == 0:
                ax.set_title(f"{JOINTS[j]} phase portrait", fontsize=7.5)
            ax.text(.02, .97, f"{occ[name][JOINTS[j]]:.0f}%", transform=ax.transAxes,
                    fontsize=6.2, va="top", color="w")
        # set-difference over the pooled joint types
        ax = fig.add_subplot(gs[r, 3])
        j = 2
        p = S[:, :, j, 0].ravel(); v = S[:, :, j, 1].ravel()
        H, xe, ye = np.histogram2d(p, v, bins=NB, range=[lims[j][0], lims[j][1]])
        m = H > 0; ref = ref_masks[2]
        cat = np.zeros_like(H, int)
        cat[ref & ~m] = 1; cat[m & ~ref] = 2; cat[m & ref] = 3
        if r == 0:
            cat[ref] = 3
        ax.pcolormesh(xe, ye, cat.T, cmap=cmap4,
                      norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), shading="auto")
        ax.tick_params(labelsize=6); ax.set_xlabel("calf pos [rad]", fontsize=7)
        if r == 0:
            ax.set_title("calf: coverage vs MoCap", fontsize=7.5)
        else:
            ax.text(.02, .97, f"new +{100.0*(m & ~ref).sum()/m.size:.0f} pp",
                    transform=ax.transAxes, fontsize=6.2, va="top",
                    bbox=dict(fc="w", ec="none", alpha=.75, pad=1.1))
        fig.text(0.005, [0.80, 0.505, 0.21][r], name, fontsize=8, rotation=90,
                 va="center", ha="left", fontweight="bold")
    h = [plt.Rectangle((0, 0), 1, 1, fc=cmap4(k)) for k in range(4)]
    fig.legend(h, ["not covered", "MoCap only", "new in video", "covered by both"],
               ncol=4, fontsize=6.8, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    C.save(fig, "fig_F_state_coverage.pdf")
    return occ

# ------------------------------------------------------------------- figure D
def posture_features(kp, fs):
    """Local-ground-referenced posture features from raw keypoints."""
    L = C.torso_length(kp)
    z = kp[..., 2]
    w = max(3, int(round(0.5 * fs)))
    foot_z = z[:, 2:6]
    ground = np.array([np.percentile(foot_z[max(0, t - w):t + w + 1], 5)
                       for t in range(len(kp))])
    h = (0.5 * (z[:, 0] + z[:, 1]) - ground) / L
    clr = (foot_z.max(1) - ground) / L
    dz = np.linalg.norm(kp[:, 1] - kp[:, 0], axis=-1)
    pitch = np.arcsin(np.clip((z[:, 1] - z[:, 0]) / dz, -1, 1))
    hdot = np.gradient(h, 1.0 / fs)
    return dict(h=h, clearance=clr, pitch=pitch, hdot=hdot)

def figure_D():
    groups = {
        "MoCap": [(n, C.load_mocap_keypoints(f, lo, hi), C.FS_MOCAP)
                  for n, f, lo, hi in C.MOCAP_WINDOWS],
        "Video flat": [(c, C.load_video_keypoints(c), C.FS_VIDEO) for c in C.VIDEO_CLIPS_8],
        "Video box": [(c, C.load_video_keypoints(c), C.FS_VIDEO) for c in C.BOX_CLIPS],
        "Video stairs/stand-up": [(c, C.load_video_keypoints(c), C.FS_VIDEO)
                                  for c in C.STAIRS_CLIPS + C.STANDUP_CLIPS],
    }
    style = {"MoCap": (C.C_MOCAP, "o", 12), "Video flat": (C.C_VIDEO, "o", 12),
             "Video box": (C.C_VIDEO_EXT, "s", 14), "Video stairs/stand-up": ("#d62728", "^", 16)}
    feats = {g: [(n, posture_features(kp, fs)) for n, kp, fs in v] for g, v in groups.items()}

    fig = plt.figure(figsize=(C.IEEE_1COL, 3.3))
    gs = fig.add_gridspec(2, 2, wspace=0.46, hspace=0.58)
    ax = fig.add_subplot(gs[0, 0])
    for g in ["Video stairs/stand-up", "Video box", "Video flat", "MoCap"]:
        rows = feats[g]; col, mk, s = style[g]
        X = np.concatenate([r["h"] for _, r in rows])
        Y = np.concatenate([r["clearance"] for _, r in rows])
        ax.scatter(X, Y, s=s * .35, c=[col], marker=mk, alpha=.55, lw=0, label=g)
    mo = feats["MoCap"]
    hx = np.concatenate([r["h"] for _, r in mo]); hy = np.concatenate([r["clearance"] for _, r in mo])
    try:
        from scipy.spatial import ConvexHull
        pts = np.stack([hx, hy], 1); hull = ConvexHull(pts)
        ax.plot(*np.append(pts[hull.vertices], pts[hull.vertices][:1], 0).T,
                ls="--", c="grey", lw=0.9)
    except Exception:
        pass
    ax.set_xlabel("base height / L", fontsize=7); ax.set_ylabel("max foot clearance / L", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=.3); ax.set_title("(a) posture", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for g in ["Video stairs/stand-up", "Video box", "Video flat", "MoCap"]:
        rows = feats[g]; col, mk, s = style[g]
        ax.scatter(np.concatenate([r["pitch"] for _, r in rows]),
                   np.concatenate([r["hdot"] for _, r in rows]),
                   s=s * .35, c=[col], marker=mk, alpha=.55, lw=0)
    ax.set_xlabel("base pitch [rad]", fontsize=7); ax.set_ylabel(r"$\dot h$ [L/s]", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=.3); ax.set_title("(b) dynamics", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    p95 = None
    for g, rows in feats.items():
        col, mk, s = style[g]
        v = np.sort(np.concatenate([r["clearance"] for _, r in rows]))
        ax.plot(v, np.linspace(0, 1, len(v)), color=col, lw=1.4,
                ls="--" if g == "MoCap" else "-")
        if g == "MoCap":
            p95 = np.percentile(v, 95)
    ax.axvline(p95, ls="--", c="grey", lw=0.9)
    ax.set_xlabel("foot clearance / L", fontsize=7); ax.set_ylabel("ECDF", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=.3); ax.set_title("(c) clearance ECDF", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    bars = []
    for g, rows in feats.items():
        for n, r in rows:
            bars.append((n, float(r["clearance"].max()), style[g][0]))
    bars.sort(key=lambda t: t[1])
    ax.barh(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], height=.75)
    ax.set_yticks([]); ax.set_xlabel("max foot clearance / L", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=.3, axis="x")
    ax.set_title("(d) per clip", fontsize=8)

    h = [plt.Line2D([], [], ls="", marker=style[g][1], color=style[g][0], ms=4) for g in feats]
    fig.legend(h, list(feats), ncol=2, fontsize=6.2, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))
    C.save(fig, "fig_D_terrain_coverage.pdf")
    return {g: dict(max_clearance=[float(r["clearance"].max()) for _, r in rows],
                    clips=[n for n, _ in rows]) for g, rows in feats.items()}

# ------------------------------------------------------------------ main
def main():
    np.random.seed(0)
    names = ["MoCap", "Video", "Video (extended)"]
    folders = [C.SET_MOCAP, C.SET_VIDEO, C.SET_VIDEOEXT]
    sets = {n: dataset_cmd(f) for n, f in zip(names, folders)}
    raw = {n: dataset_cmd(f, lowpass=None) for n, f in zip(names, folders)}

    res = {"n_samples": {k: int(len(v)) for k, v in sets.items()}}
    masks = {k: covered(v)[0] for k, v in sets.items()}
    ref = masks["MoCap"]
    for k in names:
        m = masks[k]
        res[k] = dict(
            coverage_pct=cov_pct(m),
            new_vs_mocap_pp=100.0 * (m & ~ref).sum() / m.size,
            redundant_pp=100.0 * (m & ref).sum() / m.size,
            lost_pp=100.0 * (ref & ~m).sum() / m.size,
            union_pp=100.0 * (m | ref).sum() / m.size,
            in_box_frac=float(np.all((sets[k] >= BOX[:, 0]) & (sets[k] <= BOX[:, 1]), 1).mean()),
            ranges={a: [float(sets[k][:, i].min()), float(sets[k][:, i].max())]
                    for i, a in enumerate(["vx", "vy", "wz"])},
            raw_ranges={a: [float(raw[k][:, i].min()), float(raw[k][:, i].max())]
                        for i, a in enumerate(["vx", "vy", "wz"])},
            coverage_pct_unfiltered=cov_pct(covered(raw[k])[0]),
            wasserstein_to_uniform=[
                float(wasserstein_distance(sets[k][:, i],
                                           np.random.uniform(*BOX[i], 20000)))
                for i in range(3)])
    # tau sensitivity
    res["tau_sensitivity"] = {
        f"{t}": {k: cov_pct(covered(sets[k], t)[0]) for k in names}
        for t in (0.35, 0.5, 0.75, 1.0)}
    res["increment_4to8_pp"] = res["Video (extended)"]["coverage_pct"] - res["Video"]["coverage_pct"]
    res["bootstrap_new_ci_videoext"] = bootstrap_new(C.SET_MOCAP, C.SET_VIDEOEXT)

    # Jensen-Shannon divergence to the uniform target on the 3-D histogram
    for k in names:
        Hc, _ = np.histogramdd(sets[k], bins=NBINS, range=[tuple(b) for b in BOX])
        # JSD on a coarser 5x3x5 grid: with only ~200-360 samples a 600-cell
        # histogram is dominated by the smoothing prior, not the data.
        H, _ = np.histogramdd(sets[k], bins=(5, 3, 5), range=[tuple(b) for b in BOX])
        P = (H + 0.5); P /= P.sum()
        Q = np.full_like(P, 1.0 / P.size)
        M = 0.5 * (P + Q)
        kl = lambda A, B: float(np.sum(A * np.log2(A / B)))
        res[k]["jsd_to_uniform_bits"] = 0.5 * kl(P, M) + 0.5 * kl(Q, M)
        res[k]["frac_target_zero_density"] = float((Hc == 0).mean())

    figure_C(sets); figure_G(sets)
    state_sets = {n: amp_state(f) for n, f in zip(names, folders)}
    res["state_occupancy_pct"] = figure_F(state_sets)
    res["terrain"] = figure_D()

    json.dump(res, open(os.path.join(C.OUT, "_tmp", "exp2.json"), "w"), indent=1, default=float)
    for k in names:
        r = res[k]
        print(f"{k:18s} n={res['n_samples'][k]:4d} cov={r['coverage_pct']:5.2f}%  "
              f"new={r['new_vs_mocap_pp']:5.2f}pp lost={r['lost_pp']:5.2f}pp "
              f"union={r['union_pp']:5.2f}pp inbox={100*r['in_box_frac']:4.1f}% "
              f"JSD={r['jsd_to_uniform_bits']:.3f}b zero={100*r['frac_target_zero_density']:4.1f}%")
    print("4->8 clip increment: +%.2f pp" % res["increment_4to8_pp"])
    print("bootstrap 95%% CI on New (video ext): [%.2f, %.2f] pp" % res["bootstrap_new_ci_videoext"])
    print("tau sensitivity:", json.dumps(res["tau_sensitivity"], default=lambda x: round(x, 2)))
    print("state occupancy %:", json.dumps(res["state_occupancy_pct"], default=lambda x: round(x, 1)))
    for g, d in res["terrain"].items():
        print(f"terrain {g:24s} max clearance/L: {min(d['max_clearance']):.2f}-{max(d['max_clearance']):.2f}")

if __name__ == "__main__":
    main()
