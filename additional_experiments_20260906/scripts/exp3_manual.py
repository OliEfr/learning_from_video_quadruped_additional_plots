"""Experiment 3 -- How much manual intervention does the pipeline require?

Scope: the keypoints are treated as tracked. "Manual intervention" here means the
per-clip HAND-CODING in the pipeline scripts -- the numbers a human had to type
for each individual clip. Counted by parsing the source, so every number is
reproducible and auditable.
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt
import common as C

CFG   = "/home/admin_07/project_repos/motion_imitation/retarget_motion/retarget_config_go2.py"
RETV  = "/home/admin_07/project_repos/motion_imitation/retarget_motion/retarget_motion_fromVision.py"
RECON = "/home/admin_07/project_repos/quadruped_from_video/3d_recon.py"

# how many scalars each hand-set parameter carries
WIDTH = {"SIM_TOE_OFFSET_LOCAL": 12, "SIM_ROOT_OFFSET_LOCAL": 3, "SIM_ROOT_OFFSET": 3}

def strip_comments(t):
    return "\n".join(l for l in t.split("\n") if not l.strip().startswith("#"))

def count_retarget_offsets():
    """Per-clip hand-set scalars in UnitreeGo2ConfigFromVisionDepthCam.__post_init__."""
    src = open(CFG).read()
    chain = src[src.index("class UnitreeGo2ConfigFromVisionDepthCam"):
                src.index("class UnitreeGo2ConfigFromVisionAlignedVideoDepthAnything")]
    clips = re.findall(r'in_sequence_name\s*==\s*"([^"]+)"', chain)
    blocks = re.split(r'if\s+.*in_sequence_name\s*==\s*"[^"]+"[^:]*:', chain)[1:]
    per = {}
    for clip, blk in zip(clips, blocks):
        blk = strip_comments(blk)
        n = 0
        for name in ["SIM_ROOT_OFFSET_LOCAL", "SIM_TOE_OFFSET_LOCAL", "SIM_ROOT_OFFSET",
                     "REF_POS_SCALE", "feet_z_amplification", "RETARGET_METHOD",
                     "start_frame", "end_frame"]:
            k = len(re.findall(r"self\." + name + r"\s*=", blk))
            n += k * WIDTH.get(name, 1)
        per[clip] = per.get(clip, 0) + n
    # MoCap: one global config, no per-clip branches
    mo = src[src.index("class UnitreeGo2ConfigMocap"):
             src.index("class UnitreeGo2ConfigFromVisionDepthCam")]
    mo_n = sum(len(re.findall(r"\b" + n + r"\s*=", strip_comments(mo))) * WIDTH.get(n, 1)
               for n in WIDTH)
    # Dead branches: sequence names that match no directory on disk, so the
    # code can never run. Kept out of the totals but reported as evidence of how
    # fragile per-clip hand-coding is.
    import os as _os
    live, dead = {}, {}
    root = "/home/admin_07/project_repos/motion_imitation/retarget_motion/data_fromVision_depth_cam"
    for k, v in per.items():
        (live if _os.path.isdir(_os.path.join(root, k)) else dead)[k] = v
    return live, mo_n, len(re.findall(r"in_sequence_name", mo)), dead

def count_recon_hardcoding():
    """Hand-typed per-clip special cases in 3d_recon.py."""
    body = open(RECON).read()
    seg = body[body.index("# remove custom points"):body.index("# Interpolate")]
    outlier, cur = {}, None
    for line in seg.split("\n"):
        if line.strip().startswith("#"):
            continue
        m = re.search(r'in_sequence_name\s*==\s*"([^"]+)"', line)
        if m:
            cur = m.group(1)
        elif cur and re.search(r"markers_3d\[.*\]\s*=", line):
            outlier[cur] = outlier.get(cur, 0) + 1
    ground = re.findall(r'in_sequence_name == "([^"]+)":\s*\n\s*ground_mask_first_frame_dir',
                        body)
    slam = re.findall(r'if in_sequence_name == "([^"]+)":\s*\n(?:\s*#[^\n]*\n)*\s*trajectory\[',
                      body)
    return outlier, ground, slam


def count_retarget_fixes():
    """Per-clip corrections hard-coded in retarget_motion_fromVision.py.

    swap:   base marker order was clicked wrong and is swapped back (line ~80)
    freeze: stand-up feet x/y frozen to their frame-10 value (line ~108)
    """
    src = open(RETV).read()
    swap = []
    for line in src.split("\n"):
        if "base_motion[[0, 1]] = base_motion[[1, 0]]" in line:
            break
        if re.search(r'if in_sequence_name ==', line) and " or " in line:
            swap = re.findall(r'"([^"]+)"', line)
    seg = src[src.index('if in_sequence_name == "stand_up_2431270000":'):]
    freeze_n = len([l for l in seg.split("\n")[1:6]
                    if re.match(r"\s*feet_motion\[", l)])
    return swap, {"stand_up_2431270000": freeze_n}


def figure_E(per, mo_n, recon, dur):
    outlier, ground, slam = recon
    STAGES = [("retargeting offsets", "#1f77b4"), ("3D outlier ranges", "#ff7f0e"),
              ("ground-plane frame", "#2ca02c"), ("order / freeze fixes", "#d62728")]
    swap, freeze = count_retarget_fixes()
    clips = sorted(per, key=lambda c: -per[c])
    rows, labels = [], []
    for c in clips:
        rows.append([per[c], outlier.get(c, 0), 1 if c in ground else 0,
                     (1 if c in swap else 0) + freeze.get(c, 0)])
        labels.append(c[:22])
    rows = np.array(rows, float)

    fig = plt.figure(figsize=(C.IEEE_2COL, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(clips) + 2)
    left = np.zeros(len(clips))
    for k, (nm, col) in enumerate(STAGES):
        ax.barh(y[:len(clips)], rows[:, k], left=left, color=col, height=.72, label=nm)
        left += rows[:, k]
    ax.barh([y[-1]], [mo_n], color="#7f7f7f", height=.72, label="MoCap (whole dataset)")
    ax.set_yticks(list(y[:len(clips)]) + [y[-1]])
    ax.set_yticklabels(labels + ["MoCap: ALL 6 clips"], fontsize=5.6)
    ax.set_xlabel("hand-set values [count]", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=.3, axis="x")
    ax.set_title("(a) per-clip hand-coding in the pipeline", fontsize=8)
    ax.legend(fontsize=5.6, frameon=False, loc="center left",
              bbox_to_anchor=(1.005, 0.5))

    ax = fig.add_subplot(gs[1])
    v_tot = rows.sum()
    vals = [v_tot / dur["video"], mo_n / dur["mocap"]]
    ax.bar([0, 1], vals, color=[C.C_VIDEO, C.C_MOCAP], width=.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Video\n(18 clips)", "MoCap\n(6 clips)"], fontsize=7)
    ax.set_ylabel("hand-set values per second\nof demonstration", fontsize=7.5)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=.3, axis="y")
    ax.set_title("(b) normalized cost", fontsize=8)
    C.save(fig, "fig_E_manual_intervention.pdf")
    return rows, v_tot

def main():
    per, mo_n, mo_branches, dead = count_retarget_offsets()
    recon = count_recon_hardcoding()
    outlier, ground, slam = recon
    swap, freeze = count_retarget_fixes()

    # usable demonstration duration per source (paper convention (T-1)*dt)
    dur = {}
    for tag, folder in (("mocap", C.SET_MOCAP),):
        dur[tag] = sum((len(F) - 1) * dt for _, F, dt in C.load_amp(folder))
    vid = 0.0
    for folder in ("fromVision_motions_DepthCam_extendedWithoutReverse",
                   "fromVision_motions_DepthCamStairs",
                   "fromVision_motions_DepthCam_box",
                   "fromVision_motions_DepthCam_standUp_feetZAmpl",
                   "fromVision_motions_DepthCam_obstacle"):
        vid += sum((len(F) - 1) * dt for _, F, dt in C.load_amp(folder))
    dur["video"] = vid

    rows, v_tot = figure_E(per, mo_n, recon, dur)
    res = dict(per_clip_retarget_scalars=per,
               total_retarget_scalars=int(sum(per.values())),
               n_clips=len(per),
               mocap_global_scalars=mo_n, mocap_per_clip_branches=mo_branches,
               dead_branches=dead,
               outlier_statements=outlier, ground_plane_clips=ground, slam_override=slam,
               hip_order_swaps=swap, freeze_fix=freeze,
               total_video_handset=float(v_tot),
               duration_s=dur,
               per_second={"video": v_tot / dur["video"], "mocap": mo_n / dur["mocap"]})
    json.dump(res, open(os.path.join(C.OUT, "_tmp", "exp3.json"), "w"), indent=1, default=float)

    print(f"per-clip retargeting scalars: total {sum(per.values())} over {len(per)} clips "
          f"(median {int(np.median(list(per.values())))}, range "
          f"{min(per.values())}-{max(per.values())})")
    print(f"dead (never-executing) branches: {dead}")
    print(f"MoCap config: {mo_branches} per-clip branches, {mo_n} scalars for ALL 6 clips")
    print(f"3d_recon.py outlier statements: {sum(outlier.values())} over {len(outlier)} clips {list(outlier)}")
    print(f"ground-plane masks hand-picked for {len(ground)} clips: {ground}")
    print(f"SLAM rotation overridden for: {slam}")
    print(f"base marker-order swaps: {len(swap)} clips {swap} | freeze fix: {freeze}")
    print(f"TOTAL hand-set values  video {v_tot:.0f}  vs  MoCap {mo_n}")
    print(f"usable duration  video {dur['video']:.1f}s  mocap {dur['mocap']:.1f}s")
    print(f"per second of demo: video {res['per_second']['video']:.1f}  "
          f"mocap {res['per_second']['mocap']:.1f}  "
          f"ratio {res['per_second']['video']/res['per_second']['mocap']:.1f}x")
    print(f"per clip: video {np.mean(list(per.values())):.1f} retarget scalars/clip "
          f"vs MoCap {mo_n/6:.1f}/clip (one shared config, 0 per-clip branches)")

if __name__ == "__main__":
    main()
