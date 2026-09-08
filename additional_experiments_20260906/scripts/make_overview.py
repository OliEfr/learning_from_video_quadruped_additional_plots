"""Single-page contact sheet of every figure produced in this session."""
import os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import common as C

DPI_RENDER = 150
FIGS = [
    ("fig_A_keypoint_noise_qualitative.pdf",
     "A — Exp 1: global keypoint trajectories, MoCap vs video (speed-matched right turns)"),
    ("fig_I_global_trajectories.pdf",
     "I — Global motion paths of every clip (world coordinates, metres)"),
    ("fig_J_center_paths.pdf",
     "J — Base-centre paths, all clips origin-aligned to a common heading"),
    ("fig_B_keypoint_noise_quantitative.pdf",
     "B — Exp 1: noise metrics + comparison across the four depth sources"),
    ("fig_C_command_coverage.pdf",
     "C — Exp 2: command-space coverage, MoCap / Video / Video (extended)"),
    ("fig_G_target_vs_data_distribution.pdf",
     "G — Exp 2: target command distribution vs demonstrated velocity density"),
    ("fig_F_state_coverage.pdf",
     "F — Exp 2: coverage of the 24-dim AMP discriminator state"),
    ("fig_D_terrain_coverage.pdf",
     "D — Exp 2: terrain / posture coverage (stairs, box, stand-up)"),
    ("fig_E_manual_intervention.pdf",
     "E — Exp 3: per-clip hand-coding in the pipeline"),
    ("fig_H_filter_vs_handcoding.pdf",
     "H — Diagnostics: described filter vs hand-coding, and the intrinsics bug"),
]

def main():
    tmp = os.path.join(C.OUT, "_tmp", "overview")
    os.makedirs(tmp, exist_ok=True)
    items = []
    for fn, cap in FIGS:
        src = os.path.join(C.OUT, fn)
        if not os.path.exists(src):
            print(f"  (missing, skipped) {fn}"); continue
        stem = os.path.join(tmp, fn[:-4])
        subprocess.run(["pdftoppm", "-r", str(DPI_RENDER), "-png", "-singlefile", src, stem],
                       check=True)
        img = mpimg.imread(stem + ".png")
        items.append((img, cap, img.shape[0] / img.shape[1]))

    # two columns, greedily balanced by displayed height
    COLW, LABEL, GAP = 1.0, 0.045, 0.030
    cols = [[], []]
    h = [0.0, 0.0]
    for it in items:
        k = int(np.argmin(h))
        cols[k].append(it)
        h[k] += it[2] * COLW + LABEL + GAP

    page_w = 16.0
    page_h = page_w / 2 * max(h) * 1.02 + 0.5
    fig = plt.figure(figsize=(page_w, page_h))
    fig.patch.set_facecolor("white")

    for ci, col in enumerate(cols):
        x0 = 0.015 + ci * 0.5
        y = 0.975
        for img, cap, ar in col:
            fig.text(x0, y, cap, fontsize=10, fontweight="bold", va="top", ha="left")
            y -= LABEL * (page_w / 2) / page_h * 1.0
            ah = ar * 0.47 * page_w / page_h
            ax = fig.add_axes([x0, y - ah, 0.47, ah])
            ax.imshow(img); ax.axis("off")
            y -= ah + GAP * (page_w / 2) / page_h

    fig.suptitle("Learning Quadruped Locomotion from Casual Videos — additional experiments: "
                 "all figures", fontsize=15, fontweight="bold", y=0.995)
    out = os.path.join(C.OUT, "ALL_FIGURES_overview.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"[saved] {out}  ({len(items)} figures, page {page_w:.0f}x{page_h:.1f} in)")

if __name__ == "__main__":
    main()
