"""Experiment 3: How much manual intervention does the video path need, compared with the MoCap path?

Scope (per the author's instruction): keypoints are treated as tracked. "Manual intervention" = per-clip hand-coding
in the pipeline scripts: hard-coded per-clip constants, special cases and hand-typed frame ranges.

Every number is obtained by parsing the source with `ast` so it can be audited (file:line is stored per finding):
  * quadruped_from_video/3d_recon.py                 (module-level `if in_sequence_name == ...` blocks)
  * motion_imitation/retarget_motion/retarget_config_go2.py (UnitreeGo2ConfigFromVisionDepthCam.__post_init__)
  * motion_imitation/retarget_motion/retarget_motion_fromVision.py (per-clip keypoint fixes)
  * IsaacLab/rsl_rl/.../datasets/rsi_data.py         (per-motion-file reference-state-initialisation transforms)
  * IsaacLab/datasets/*/**_amp.txt                  (per-clip MotionWeight != 1)
  MoCap path: motion_imitation/retarget_motion/retarget_motion.py (`mocap_motions` list: hand-typed frame ranges)

Counting rule for "hand-set numbers": every element of a numeric list/array literal counts (each is a tuned value);
for index-based keypoint fixes (`markers_3d[29:33, :, :] = 0`) the distinct integer literals in the subscripts count
(frame / marker indices); a string switch (method name, file name) counts 1; the clip name itself is not counted.
Frame-range trimming that is set per run without a branch (start_idx_files / end_idx_files in 3d_recon.py) is
counted as 2 numbers per clip for BOTH pipelines, since the MoCap list carries the same two numbers per clip.
"""
import ast
import csv
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common import COL_W, COLORS, DATA, ISAAC, NAME_MOCAP, NAME_VIDEO, RETARGET, VISION, load_amp_dir, md_table, savefig

FILES = {
    "3d_recon": os.path.join(VISION, "3d_recon.py"),
    "retarget_config": os.path.join(RETARGET, "retarget_config_go2.py"),
    "retarget_fromVision": os.path.join(RETARGET, "retarget_motion_fromVision.py"),
    "retarget_mocap": os.path.join(RETARGET, "retarget_motion.py"),
    "rsi_data": os.path.join(ISAAC, "rsl_rl/rsl_rl/rsl_rl/datasets/rsi_data.py"),
}
CLIP_IDS = {
    "walk_869488000": "flat",
    "slow_1313807000": "flat",
    "turn_left_1771233000": "flat",
    "turn_right_1771233000": "flat",
    "slow_turn_1771233000": "flat",
    "left_right_turn_2058226999": "flat",
    "start_stop_1271493000": "flat",
    "start_stop_785558000": "flat",
    "stairs_1_5199540000": "stairs",
    "stairs_2_5299211000": "stairs",
    "stairs_3_5339749000": "stairs",
    "box_1_399682000": "box",
    "box_2_986388000": "box",
    "stand_up_2431270000": "stand-up",
    "obstacle_1_3015061000": "unused",
    "obstacle_2_3126098000": "unused",
    "obstacle_3_3015061000": "unused",
}
PAPER_DATASETS = {  # datasets behind the paper's figures (flat: Fig. 4/5; stairs, box, stand-up: Fig. 4)
    "flat": ["fromVision_motions_DepthCam", "fromVision_motions_DepthCam_extendedWithoutReverse"],
    "stairs": ["fromVision_motions_DepthCamStairs"],
    "box": ["fromVision_motions_DepthCam_box"],
    "stand-up": ["fromVision_motions_DepthCam_standUp_feetZAmpl"],
}
CATEGORIES = [
    "clip trimming (frame range)",
    "input override (mask / track file / camera)",
    "keypoint fix (delete, freeze, reorder)",
    "retargeting: base offset",
    "retargeting: foot offsets",
    "retargeting: scale / foot-lift gain",
    "retargeting: method switch",
    "RL: RSI transform / motion weight",
]
CAT_COLORS = dict(zip(CATEGORIES, ["#7f7f7f", "#8c564b", "#d62728", "#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#9467bd"]))


# ----------------------------------------------------------------------------------------------
def num_literals(node):
    """All numeric literals below `node` (UnaryOp(USub, Constant) counts once)."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            out.append(n.value)
    return out


def str_literals(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def clip_names_in_test(test):
    """Return clip ids compared against `in_sequence_name` in an if-test (handles `or` chains)."""
    names = []
    for n in ast.walk(test):
        if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.Eq):
            left = n.left
            lname = left.attr if isinstance(left, ast.Attribute) else getattr(left, "id", None)
            if lname == "in_sequence_name" and isinstance(n.comparators[0], ast.Constant):
                names.append(n.comparators[0].value)
    return names


def iter_if_chain(node):
    """Yield (test, body) for an if/elif chain."""
    while isinstance(node, ast.If):
        yield node.test, node.body
        node = node.orelse[0] if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) else None
        if node is None:
            break


def subscript_targets_int_literals(stmt):
    """Distinct integer literals inside the subscripts of assignment targets (frame / marker indices)."""
    vals = set()
    for t in stmt.targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Subscript):
                for c in ast.walk(n.slice):
                    if isinstance(c, ast.Constant) and isinstance(c.value, int) and not isinstance(c.value, bool):
                        vals.add(c.value)
                    if isinstance(c, ast.UnaryOp) and isinstance(c.op, ast.USub) and isinstance(c.operand, ast.Constant):
                        vals.add(-c.operand.value)
    return vals


def target_name(stmt):
    t = stmt.targets[0]
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Subscript):
        base = t.value
        while isinstance(base, ast.Subscript):
            base = base.value
        return base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "?")
    return "?"


def classify(fname, stmt, src_lines):
    """Return (category, count, text) for one statement inside a per-clip branch."""
    text = " ".join(src_lines[stmt.lineno - 1 : stmt.end_lineno]).strip()
    text = " ".join(text.split())
    if not isinstance(stmt, ast.Assign):
        return None
    name = target_name(stmt)
    if fname == "3d_recon":
        if name == "ground_mask_first_frame_dir":
            return "input override (mask / track file / camera)", 1, text
        if name == "foot_pos_file":
            return "input override (mask / track file / camera)", 1, text
        if name == "trajectory":
            return "input override (mask / track file / camera)", 1, text
        if name == "markers_3d":
            return "keypoint fix (delete, freeze, reorder)", len(subscript_targets_int_literals(stmt)), text
    if fname == "retarget_config":
        if name in ("start_frame", "end_frame"):
            return "clip trimming (frame range)", len(num_literals(stmt.value)), text
        if name in ("SIM_ROOT_OFFSET_LOCAL", "SIM_ROOT_OFFSET"):
            return "retargeting: base offset", len(num_literals(stmt.value)), text
        if name in ("SIM_TOE_OFFSET_LOCAL", "SIM_TOE_OFFSET"):
            return "retargeting: foot offsets", len(num_literals(stmt.value)), text
        if name in ("REF_POS_SCALE", "REF_Z_POS_SCLAE", "feet_z_amplification"):
            return "retargeting: scale / foot-lift gain", max(1, len(num_literals(stmt.value))), text
        if name == "RETARGET_METHOD":
            return "retargeting: method switch", 1, text
    if fname == "retarget_fromVision":
        if name in ("base_motion", "feet_motion"):
            return "keypoint fix (delete, freeze, reorder)", max(1, len(subscript_targets_int_literals(stmt))), text
    return "other", len(num_literals(stmt)), text


def parse_per_clip_branches(key, tree, src_lines, branch_nodes):
    findings = []
    for node in branch_nodes:
        for test, body in iter_if_chain(node):
            clips = clip_names_in_test(test)
            if not clips:
                continue
            for stmt in body:
                # nested `if depth_type == ...` blocks (3d_recon.py, walk clip) -> descend one level
                stmts = [stmt]
                if isinstance(stmt, ast.If) and not clip_names_in_test(stmt.test):
                    # keep only the branch of the camera-depth path (depth_type == "depth_cam")
                    stmts = [s for t_, b_ in iter_if_chain(stmt) if "depth_cam" in str_literals(t_) for s in b_]
                for s in stmts:
                    c = classify(key, s, src_lines)
                    if c is None:
                        continue
                    cat, cnt, text = c
                    for clip in clips:
                        findings.append(dict(file=os.path.relpath(FILES[key], "/home/admin_07/project_repos"), line=s.lineno, clip=clip, category=cat, count=int(cnt), statement=text))
    return findings


def top_level_ifs(body):
    return [n for n in body if isinstance(n, ast.If)]


def main():
    findings = []
    dead_branches = []

    # --- 3d_recon.py: module-level if chains
    src = open(FILES["3d_recon"]).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    findings += parse_per_clip_branches("3d_recon", tree, lines, top_level_ifs(tree.body))

    # --- retarget_config_go2.py: UnitreeGo2ConfigFromVisionDepthCam.__post_init__
    src = open(FILES["retarget_config"]).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnitreeGo2ConfigFromVisionDepthCam"]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"]:
            chain_nodes = top_level_ifs(fn.body)
            findings += parse_per_clip_branches("retarget_config", tree, lines, chain_nodes)
            for node in chain_nodes:
                for test, body in iter_if_chain(node):
                    for clip in clip_names_in_test(test):
                        if clip not in CLIP_IDS and not any(clip.startswith(k.split("_")[0]) and clip in k for k in CLIP_IDS):
                            dead_branches.append(dict(file="motion_imitation/retarget_motion/retarget_config_go2.py", line=test.lineno, clip=clip))

    # --- retarget_motion_fromVision.py
    src = open(FILES["retarget_fromVision"]).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    findings += parse_per_clip_branches("retarget_fromVision", tree, lines, top_level_ifs(tree.body))

    # --- rsi_data.py: per-motion-file transforms (7 numbers each)
    src = open(FILES["rsi_data"]).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and target_name(node) == "rsi_params" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                path = k.value
                clip = next((c for c in CLIP_IDS if c in path), None)
                if clip is None or not isinstance(v, ast.Dict):
                    continue
                if path.split("/")[1] not in {ds for dsets in PAPER_DATASETS.values() for ds in dsets}:
                    continue  # e.g. datasets/fromVision_motions_DepthCam_obstacle is not behind any paper figure
                n = len(num_literals(v))
                if n:
                    findings.append(dict(file="isaac_lab/IsaacLab/rsl_rl/rsl_rl/rsl_rl/datasets/rsi_data.py", line=k.lineno, clip=clip, category="RL: RSI transform / motion weight", count=n, statement=f"rsi_params[{os.path.basename(path)}] (dataset {path.split('/')[1]})", dataset=path.split("/")[1]))

    # --- MotionWeight != 1 in the paper's datasets
    for scen, dsets in PAPER_DATASETS.items():
        for ds in dsets:
            for c in load_amp_dir(os.path.join(ISAAC, "datasets", ds)):
                if c["weight"] != 1.0:
                    clip = next((k for k in CLIP_IDS if k in c["name"]), c["name"])
                    findings.append(dict(file=f"isaac_lab/IsaacLab/datasets/{ds}/{os.path.basename(c['path'])}", line=0, clip=clip, category="RL: RSI transform / motion weight", count=1, statement=f"MotionWeight = {c['weight']}", dataset=ds))

    # --- clip trimming in 3d_recon.py (start_idx_files / end_idx_files are set per run; 2 numbers per clip)
    for clip in CLIP_IDS:
        findings.append(dict(file="quadruped_from_video/3d_recon.py", line=13, clip=clip, category="clip trimming (frame range)", count=2, statement="start_idx_files / end_idx_files (set by hand per clip; e.g. '# 254 for maila, 24 for walk')"))

    # --- MoCap path: retarget_motion.py mocap_motions list
    src = open(FILES["retarget_mocap"]).read()
    tree = ast.parse(src)
    mocap = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and target_name(node) == "mocap_motions":
            for elt in node.value.elts:
                name = elt.elts[0].value
                nums = num_literals(elt)
                mocap.append(dict(file="motion_imitation/retarget_motion/retarget_motion.py", line=elt.lineno, clip=name, category="clip trimming (frame range)", count=len(nums), statement=ast.unparse(elt)))
    # shared (set once per pipeline, not per clip) constants -- reported separately
    shared = {
        "MoCap path": dict(
            description="retarget_motion.py: FRAME_DURATION, REF_COORD_ROT, REF_ROOT_ROT; UnitreeGo2ConfigMocap: SIM_ROOT_OFFSET(3), SIM_TOE_OFFSET_LOCAL(12); REF_POS_SCALE",
            count=1 + 1 + 1 + 3 + 12 + 1,
        ),
        "Video path": dict(
            description="retarget_motion_fromVision.py: FRAME_DURATION; UnitreeGo2ConfigFromVisionDepthCam defaults: SIM_ROOT_OFFSET(3), SIM_ROOT_OFFSET_LOCAL(3), SIM_TOE_OFFSET_LOCAL(12), REF_POS_SCALE(1, inherited), feet_z_amplification default(1); 3d_recon.py plane RANSAC (distance 0.01, n=3, iters 1000)",
            count=1 + 3 + 3 + 12 + 1 + 1 + 3,
        ),
    }

    # ----------------------------------------------------------------------------------------------
    # aggregate
    per_clip = {}
    for f in findings:
        d = per_clip.setdefault(f["clip"], {c: 0 for c in CATEGORIES} | {"other": 0, "n_statements": 0, "scenario": CLIP_IDS.get(f["clip"], "unused")})
        d[f["category"]] = d.get(f["category"], 0) + f["count"]
        d["n_statements"] += 1
    for d in per_clip.values():
        d["total"] = sum(d[c] for c in CATEGORIES) + d["other"]
    mocap_per_clip = {m["clip"]: m["count"] for m in mocap}

    # write findings CSV (auditable)
    with open(os.path.join(DATA, "exp3_findings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "line", "clip", "category", "count", "statement", "dataset"])
        w.writeheader()
        for f in findings + mocap:
            w.writerow({k: f.get(k, "") for k in w.fieldnames})

    paper_clips = [c for c in CLIP_IDS if CLIP_IDS[c] != "unused"]
    flat = [c for c in CLIP_IDS if CLIP_IDS[c] == "flat"]
    summary = dict(
        per_clip=per_clip,
        mocap_per_clip=mocap_per_clip,
        shared_constants=shared,
        dead_branches=dead_branches,
        totals=dict(
            video_flat_total=sum(per_clip[c]["total"] for c in flat),
            video_flat_mean=float(np.mean([per_clip[c]["total"] for c in flat])),
            video_flat_median=float(np.median([per_clip[c]["total"] for c in flat])),
            video_flat_min=int(min(per_clip[c]["total"] for c in flat)),
            video_flat_max=int(max(per_clip[c]["total"] for c in flat)),
            video_paper_total=sum(per_clip[c]["total"] for c in paper_clips),
            video_paper_mean=float(np.mean([per_clip[c]["total"] for c in paper_clips])),
            video_paper_median=float(np.median([per_clip[c]["total"] for c in paper_clips])),
            video_paper_statements=sum(per_clip[c]["n_statements"] for c in paper_clips),
            mocap_total=sum(mocap_per_clip.values()),
            mocap_per_clip=float(np.mean(list(mocap_per_clip.values()))),
            n_video_paper_clips=len(paper_clips),
            n_mocap_clips=len(mocap_per_clip),
            category_totals_flat={c: sum(per_clip[k][c] for k in flat) for c in CATEGORIES},
            category_totals_paper={c: sum(per_clip[k][c] for k in paper_clips) for c in CATEGORIES},
            clips_with_retargeting_offsets_flat=sum(1 for c in flat if per_clip[c]["retargeting: foot offsets"] > 0),
        ),
    )
    with open(os.path.join(DATA, "exp3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ----------------------------------------------------------------------------------------------
    # figure: stacked horizontal bars per clip
    order = flat + [c for c in paper_clips if c not in flat]
    labels = [c.rsplit("_", 1)[0].replace("_", " ") for c in order]
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    y = np.arange(len(order) + 1 + len(mocap_per_clip))
    for i, clip in enumerate(order):
        left = 0
        for cat in CATEGORIES:
            v = per_clip[clip][cat]
            if v:
                ax.barh(y[i], v, left=left, color=CAT_COLORS[cat], height=0.75, lw=0)
                left += v
        ax.text(left + 0.5, y[i], str(int(left)), va="center", fontsize=5.5)
    # scenario separators
    scen = [CLIP_IDS[c] for c in order]
    for i in range(1, len(order)):
        if scen[i] != scen[i - 1]:
            ax.axhline(y[i] - 0.5, color="0.7", lw=0.5, ls=":")
    ax.axhline(y[len(order)] - 0.5, color="0.3", lw=0.6)
    for j, (name, cnt) in enumerate(mocap_per_clip.items()):
        yi = y[len(order) + 1 + j]
        ax.barh(yi, cnt, color=CAT_COLORS["clip trimming (frame range)"], height=0.75, lw=0)
        ax.text(cnt + 0.5, yi, str(cnt), va="center", fontsize=5.5)
    ax.set_yticks(list(y[: len(order)]) + list(y[len(order) + 1 :]))
    ax.set_yticklabels(labels + [f"{n} (dog_{'walk' if 'turn' in n or n in ('pace', 'trot') else 'run'}*)" for n in mocap_per_clip], fontsize=5.5)
    for t, c in zip(ax.get_yticklabels()[: len(order)], order):
        t.set_color(COLORS[NAME_VIDEO])
    for t in ax.get_yticklabels()[len(order) :]:
        t.set_color(COLORS[NAME_MOCAP])
    ax.invert_yaxis()
    ax.set_xlabel("hand-set per-clip numbers / switches in the pipeline scripts")
    ax.text(ax.get_xlim()[1] * 0.99, y[len(order) // 2], "Video clips", ha="right", va="center", color=COLORS[NAME_VIDEO], fontsize=6.5, fontweight="bold")
    ax.text(ax.get_xlim()[1] * 0.99, y[len(order) + 1 + len(mocap_per_clip) // 2], "MoCap clips", ha="right", va="center", color=COLORS[NAME_MOCAP], fontsize=6.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, lw=0.4)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in CATEGORIES]
    ax.legend(handles, CATEGORIES, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, frameon=False, fontsize=5.2, handlelength=1.0, columnspacing=0.8)
    savefig(fig, "fig_exp3_manual_intervention")
    plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    rows = []
    for c in order:
        d = per_clip[c]
        rows.append([c, d["scenario"]] + [d[k] for k in CATEGORIES] + [d["total"], d["n_statements"]])
    table = md_table(["clip", "scenario"] + [c.replace("retargeting: ", "rt: ") for c in CATEGORIES] + ["total", "statements"], rows)
    with open(os.path.join(DATA, "exp3_per_clip_table.md"), "w") as fh:
        fh.write(table + "\n")
    print(table)
    print(json.dumps(summary["totals"], indent=1))
    print("mocap per clip:", mocap_per_clip)
    print("dead branches:", dead_branches)
    print("shared:", json.dumps(shared, indent=1))


if __name__ == "__main__":
    main()
