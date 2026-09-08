"""Global motion trajectories of every clip, in world coordinates (metres).

ONE rigid transform per clip (frame-0 rear keypoint to the origin, frame-0
heading to +x). Nothing is re-centred per frame, so what is drawn is the actual
path the animal travelled, including -- for the video clips -- the drift of the
reconstructed camera pose.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import common as C
import exp1_noise as E

def clip_sets():
    mo = [(n, C.load_mocap_keypoints(f, lo, hi), C.FS_MOCAP)
          for n, f, lo, hi in C.MOCAP_WINDOWS]
    vi = [(n, C.load_video_keypoints(n), C.FS_VIDEO) for n in C.VIDEO_CLIPS_8]
    nf = [(n, C.load_video_keypoints(n), C.FS_VIDEO)
          for n in C.STAIRS_CLIPS + C.BOX_CLIPS + C.STANDUP_CLIPS]
    return [("MoCap\n(6 clips)", mo, C.C_MOCAP),
            ("Video flat\n(8 clips)", vi, C.C_VIDEO),
            ("Video stairs/box/\nstand-up (6 clips)", nf, C.C_VIDEO_EXT)]

def main():
    sets = clip_sets()
    fig = plt.figure(figsize=(C.IEEE_2COL, 6.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1.0, 1.0],
                          wspace=0.40, hspace=0.42, left=0.10, bottom=0.10)
    lbl = dict(fontsize=7.5); tk = dict(labelsize=6.5)

    for r, (name, clips, base_col) in enumerate(sets):
        traj = [(n, C.canonicalize_global(kp)) for n, kp, _ in clips]
        shades = plt.cm.viridis(np.linspace(0.15, 0.88, len(traj)))

        ax3d = ax = fig.add_subplot(gs[r, 0], projection="3d")
        for (n, g), col in zip(traj, shades):
            b = 0.5 * (g[:, 0] + g[:, 1])
            ax.plot(*b.T, color=col, lw=1.3)                 # base path
            for k in range(2, 6):
                ax.plot(*g[:, k].T, color=col, lw=0.45, alpha=0.55)
        ax.view_init(elev=22, azim=-60); ax.set_box_aspect((2.0, 1.2, 0.7), zoom=1.22)
        ax.set_xlabel("x [m]", labelpad=-7, **lbl)
        ax.set_ylabel("y [m]", labelpad=-7, **lbl)
        ax.set_zlabel("z [m]", labelpad=-7, **lbl)
        ax.locator_params(nbins=4); ax.tick_params(**tk, pad=-2)
        ax.set_title(f"3D global path", fontsize=7.8, pad=0)

        for c_, (i, j, nm, ttl) in enumerate(
                [(0, 1, ("x [m]", "y [m]"), "top-down (x-y)"),
                 (0, 2, ("x [m]", "z [m]"), "side (x-z)")]):
            ax = fig.add_subplot(gs[r, c_ + 1])
            for (n, g), col in zip(traj, shades):
                b = 0.5 * (g[:, 0] + g[:, 1])
                ax.plot(b[:, i], b[:, j], color=col, lw=1.3)
                for k in range(2, 6):
                    ax.plot(g[:, k, i], g[:, k, j], color=col, lw=0.45, alpha=0.5)
                ax.scatter(b[0, i], b[0, j], s=9, color=col, zorder=5,
                           edgecolors="k", linewidths=0.3)
            ax.set_aspect("equal"); ax.grid(alpha=.3)
            ax.set_xlabel(nm[0], **lbl); ax.set_ylabel(nm[1], **lbl); ax.tick_params(**tk)
            ax.set_title(ttl, fontsize=7.8)

        bb = ax3d.get_position()
        fig.text(0.012, bb.y0 + bb.height / 2, name, fontsize=7.5, rotation=90,
                 va="center", ha="left", fontweight="bold", linespacing=0.95)

    fig.text(0.5, 0.012,
             "One rigid transform per clip (frame-0 rear keypoint to origin, "
             "frame-0 heading to +x); nothing re-centred per frame.\n"
             "Thick = base path, thin = the four feet. Dots mark each clip's start. "
             "Colour distinguishes clips within a row.",
             ha="center", fontsize=6.2, style="italic")
    C.save(fig, "fig_I_global_trajectories.pdf")
    center_paths()

    print(f"{'clip':30s} {'path [m]':>9s} {'net disp [m]':>12s} {'straightness':>12s}")
    for name, clips, _ in sets:
        print(f"-- {name}")
        for n, kp, fs in clips:
            g = C.canonicalize_global(kp)
            b = 0.5 * (g[:, 0] + g[:, 1])
            path = float(np.linalg.norm(np.diff(b, axis=0), axis=-1).sum())
            net = float(np.linalg.norm(b[-1] - b[0]))
            print(f"   {n:27s} {path:9.2f} {net:12.2f} {net/max(path,1e-9):12.2f}")



# ---------------------------------------------------------------------------
def center_paths():
    """Top-down projection of the BASE CENTRE only, with every clip started at
    the origin heading in the same global direction (+x).

    This isolates the *shape* of each demonstrated motion -- straight, left turn,
    right turn, start-stop -- from where in the world it happened, so the motion
    repertoire of the three datasets can be compared directly.
    """
    groups = [
        ("MoCap", [(n, C.load_mocap_keypoints(f, lo, hi), C.FS_MOCAP)
                   for n, f, lo, hi in C.MOCAP_WINDOWS], C.C_MOCAP, None),
        ("Video", [(n, C.load_video_keypoints(n), C.FS_VIDEO)
                   for n in C.VIDEO_CLIPS_4], C.C_VIDEO, None),
        ("Video (extended)", [(n, C.load_video_keypoints(n), C.FS_VIDEO)
                              for n in C.VIDEO_CLIPS_8], C.C_VIDEO,
         set(C.VIDEO_CLIPS_8) - set(C.VIDEO_CLIPS_4)),
        ("Video: stairs / box / stand-up",
         [(n, C.load_video_keypoints(n), C.FS_VIDEO)
          for n in C.STAIRS_CLIPS + C.BOX_CLIPS + C.STANDUP_CLIPS],
         C.C_VIDEO_EXT, None),
    ]
    fig = plt.figure(figsize=(C.IEEE_2COL, 2.75))
    gs = fig.add_gridspec(1, 4, wspace=0.32)
    lim = 0.0
    paths = []
    for name, clips, col, hi_set in groups:
        rows = []
        for n, kp, fs in clips:
            g = C.canonicalize_global(kp)
            b = 0.5 * (g[:, 0] + g[:, 1])[:, :2]
            rows.append((n, b))
            lim = max(lim, np.abs(b).max())
        paths.append((name, rows, col, hi_set))
    # every clip starts at the origin heading +x, so crop to the data instead of
    # a symmetric box -- otherwise most paths occupy a few percent of the panel
    xlo, xhi = -0.35, lim * 1.10
    ylo, yhi = -lim * 1.10, lim * 0.45

    stats = {}
    for a, (name, rows, col, hi_set) in enumerate(paths):
        ax = fig.add_subplot(gs[a])
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        shades = plt.cm.viridis(np.linspace(0.12, 0.86, len(rows)))
        for (n, b), sh in zip(rows, shades):
            new = hi_set is not None and n in hi_set
            cc = "#d62728" if new else sh
            ax.plot(b[:, 0], b[:, 1], color=cc, lw=1.7 if new else 1.2,
                    alpha=0.95, zorder=3 if new else 2)
            ax.scatter(b[-1, 0], b[-1, 1], s=11, marker="o", color=cc, zorder=4, lw=0)
        ax.scatter([0], [0], s=14, marker="s", color="k", zorder=5)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_aspect("equal")
        ax.grid(alpha=.3); ax.tick_params(labelsize=6)
        ax.set_xlabel("x [m]", fontsize=7.5)
        if a == 0:
            ax.set_ylabel("y [m]", fontsize=7.5)
        ax.set_title(f"{name}\n({len(rows)} clips)", fontsize=7.2)
        # net heading change per clip
        hd = []
        for n, b in rows:
            d = b[-1] - b[0]
            hd.append(float(np.degrees(np.arctan2(d[1], d[0]))))
        stats[name] = dict(clips=[n for n, _ in rows], net_heading_deg=hd)
    fig.text(0.5, -0.06,
             "Base centre only, top-down. Every clip is translated to the origin (square) and rotated so "
             "its initial heading points along +x,\nso only the shape of the motion differs. Dots mark each "
             "clip's end; red = clips added by the extended dataset.",
             ha="center", fontsize=6.2, style="italic")
    C.save(fig, "fig_J_center_paths.pdf")
    print("\nnet heading change [deg] (start heading = 0):")
    for k, v in stats.items():
        print(f"  {k}")
        for n, h in zip(v["clips"], v["net_heading_deg"]):
            print(f"     {n:28s} {h:+7.1f}")
    return stats


if __name__ == "__main__":
    main()
