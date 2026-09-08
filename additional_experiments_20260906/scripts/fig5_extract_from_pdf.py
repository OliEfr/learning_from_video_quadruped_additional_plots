"""Recover the numeric cell values of the paper's Fig. 5 from the vector PDF that
`plot_errorOnTargetDistribution.py` produced (plots/combined_heatmap_grid.pdf).

The evaluation YAMLs the original script consumes live on the Nextcloud DVC remote and
are not on this machine, so the figure cannot be regenerated from raw logs.  The shipped
PDF, however, draws every heatmap cell as an individual vector rectangle whose fill is a
verbatim entry of matplotlib's 256-entry viridis LUT.  Inverting the LUT recovers the
value of each cell to within one LUT step, i.e. (vmax-vmin)/256.

Writes one CSV per (dataset row, metric column) to data/fig5_recovered/.
"""
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
import matplotlib.cm as cm

SRC_PDF = "/home/admin_07/project_repos/isaac_lab/IsaacLab/plots/combined_heatmap_grid.pdf"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "fig5_recovered")

# from plot_errorOnTargetDistribution.py::create_plots
COLORBAR_LIMITS = {
    "error_vel_xy": (0.00, 0.1),
    "error_vel_yaw": (0.0, 0.4),
    "mean_mechanical_cot": (0.8, 2.0),
    "agent_expert_distances": (2.0, 4.0),
}
METRIC_ORDER = ["error_vel_xy", "error_vel_yaw", "mean_mechanical_cot", "agent_expert_distances"]
ROW_ORDER = ["mocap_AMP_for_hardware",
             "fromVision_motions_DepthCam",
             "fromVision_motions_DepthCam_extendedWithoutReverse"]


def page_content(path):
    data = open(path, "rb").read()
    for m in re.finditer(rb"stream\r?\n", data):
        s, e = m.end(), data.find(b"endstream", m.end())
        try:
            d = zlib.decompress(data[s:e])
        except zlib.error:
            continue
        if d[:12] == b"/DeviceRGB C":
            return d.decode("latin1")
    raise RuntimeError("page content stream not found")


def parse_filled_rects(content):
    """Mini content-stream walker: yields (clip_rect, (r,g,b), bbox) for every filled path."""
    toks = content.replace("\n", " ").split()
    fill = (0.0, 0.0, 0.0)
    clip = None
    clip_stack = []
    pending_re = None
    pts = []
    out = []
    i = 0
    n = len(toks)

    def num(k):
        return float(toks[k])

    while i < n:
        t = toks[i]
        if t == "rg":
            fill = (num(i - 3), num(i - 2), num(i - 1))
        elif t == "q":
            clip_stack.append(clip)
        elif t == "Q":
            clip = clip_stack.pop() if clip_stack else None
        elif t == "re":
            pending_re = (num(i - 4), num(i - 3), num(i - 2), num(i - 1))
        elif t == "W":
            if pending_re is not None:
                clip = pending_re
        elif t == "m":
            pts = [(num(i - 2), num(i - 1))]
        elif t == "l":
            pts.append((num(i - 2), num(i - 1)))
        elif t in ("f", "F", "B", "b", "B*", "f*"):
            if len(pts) >= 4 and clip is not None:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                out.append((clip, fill, (min(xs), min(ys), max(xs), max(ys))))
            pts = []
        elif t in ("S", "s", "n"):
            pts = []
        i += 1
    return out


def viridis_lut():
    return np.asarray(cm.get_cmap("viridis")(np.arange(256))[:, :3])


def main():
    content = page_content(SRC_PDF)
    rects = parse_filled_rects(content)

    # group cells by their clipping rectangle -> one group per heatmap axes
    groups = {}
    for clip, fill, bbox in rects:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if not (5.0 < w < 40.0 and 5.0 < h < 40.0):
            continue  # not a mesh cell
        groups.setdefault(tuple(np.round(clip, 3)), []).append((fill, bbox))

    panels = {k: v for k, v in groups.items() if len(v) >= 50}
    if len(panels) != 12:
        print(f"WARNING: found {len(panels)} candidate panels (expected 12)", file=sys.stderr)
        for k, v in sorted(panels.items()):
            print("   ", k, len(v), file=sys.stderr)

    # order panels: rows top->bottom (descending y), cols left->right (ascending x)
    keys = sorted(panels, key=lambda k: (-round(k[1], 1), round(k[0], 1)))
    assert len(keys) == 12, len(keys)

    lut = viridis_lut()
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []

    for idx, key in enumerate(keys):
        r, c = divmod(idx, 4)
        metric = METRIC_ORDER[c]
        dataset = ROW_ORDER[r]
        vmin, vmax = COLORBAR_LIMITS[metric]
        cells = panels[key]

        xs = sorted({round(b[0], 2) for _, b in cells})
        ys = sorted({round(b[1], 2) for _, b in cells})
        ncol, nrow = len(xs), len(ys)
        assert ncol * nrow == len(cells), (ncol, nrow, len(cells))

        grid = np.full((nrow, ncol), np.nan)
        sat_lo = sat_hi = 0
        for fill, b in cells:
            ci = xs.index(round(b[0], 2))
            ri = ys.index(round(b[1], 2))          # ri=0 is the BOTTOM row in PDF space
            d = np.sum((lut - np.asarray(fill)) ** 2, axis=1)
            k = int(np.argmin(d))
            assert d[k] < 1e-6, f"colour {fill} not in viridis LUT (d={d[k]:.2e})"
            sat_lo += k == 0
            sat_hi += k == 255
            grid[ri, ci] = vmin + (k + 0.5) / 256.0 * (vmax - vmin)

        grid = grid[::-1]                          # flip to top-row-first, matching the figure

        # seaborn drew index (y) ascending downward, columns (x) ascending rightward
        target_y = np.round(np.linspace(-0.3, 0.3, nrow), 2)
        target_x = np.round(np.linspace(-1.0, 1.0, ncol), 2)
        df = pd.DataFrame(grid, index=pd.Index(target_y, name="target_velocity_y"),
                          columns=pd.Index(target_x, name="target_velocity_x"))
        fn = os.path.join(OUT_DIR, f"{dataset}__{metric}.csv")
        df.to_csv(fn, float_format="%.6f")
        summary.append((dataset, metric, nrow, ncol, float(np.nanmean(grid)),
                        float(np.nanmin(grid)), float(np.nanmax(grid)), sat_lo, sat_hi))

    print(f"grid: {nrow} x {ncol}  (target_vel_y x target_vel_x)")
    print(f"{'dataset':<52}{'metric':<26}{'mean':>9}{'min':>9}{'max':>9}{'satLo':>7}{'satHi':>7}")
    for s in summary:
        print(f"{s[0]:<52}{s[1]:<26}{s[4]:>9.4f}{s[5]:>9.4f}{s[6]:>9.4f}{s[7]:>7d}{s[8]:>7d}")
    print(f"\nWrote {len(summary)} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
