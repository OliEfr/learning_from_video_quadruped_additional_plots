#!/usr/bin/env bash
# Regenerates every figure, table and number of the three additional experiments from scratch.
# All source repositories are read-only; outputs go to ../figures and ../data. Runtime: ~1 min on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniconda3/envs/isaaclab/bin/python}

echo "== Exp 1a: re-lift 2D tracks to 3D (as-coded and corrected intrinsics) + validation against on-disk keypoints"
$PY exp1_relift_keypoints.py
echo "== Exp 1b: keypoint noise vs MoCap (figures + data/exp1_metrics.json)"
$PY exp1_noise.py > ../data/exp1_stdout.txt
echo "== Exp 1c: trajectory grids, top + side view: 2-row (MoCap trot2 / Video walk) and 14-row, each as-used and with corrected intrinsics"
$PY exp1_trajectory_grid.py > ../data/exp1_grid_stdout.txt
echo "== Exp 1c': walk clip through the pipeline stages (corrected re-lift / post-processed / retargeted Go2) vs MoCap trot2"
$PY exp1_trajectory_grid_stages.py > ../data/exp1_stages_stdout.txt
echo "== Exp 1e: filter proposal (depth Hampel + 6 Hz zero-phase low-pass), before/after metrics, motion checks, filtered grids"
$PY exp1_filter.py > ../data/exp1_filter_stdout.txt
echo "== Exp 1 (paper figure): paw height raw/filtered + torso-length + body residual, corrected intrinsics only"
$PY exp1_noise_paper_figure.py > ../data/exp1_noise_paper_stdout.txt
echo "== Exp 1 (Fig. 7 replacement): trajectory grid + paw-height trace + stance scatter, single column"
$PY exp1_fig7_replacement.py > ../data/exp1_fig7_replacement_stdout.txt
echo "== Exp 1d: top-view overlay of torso-centre trajectories, Video vs MoCap (author's follow-up)"
$PY exp1_com_topview.py > ../data/exp1_com_stdout.txt
echo "== Exp 2: coverage of the target distribution (figures + data/exp2_coverage.json)"
$PY exp2_coverage.py > ../data/exp2_stdout.txt
echo "== Exp 2 (author's follow-up): expert-data coverage, 2 base-velocity columns, linear + log colour"
$PY exp2_state_coverage_2col.py > ../data/exp2_2col_stdout.txt
echo "== Fig. 5 replacement: expert data content (a) + policy performance (b) in one double-column figure"
$PY fig5_replacement_combined.py > ../data/fig5_combined_stdout.txt
echo "== Exp 3: manual intervention parsed from source (figure + data/exp3_findings.csv, exp3_summary.json)"
$PY exp3_manual_intervention.py > ../data/exp3_stdout.txt
echo "== Sanity checks (data/checks_report.md)"
$PY checks.py > ../data/checks_stdout.txt
grep -c "\[FAIL\]" ../data/checks_report.md && echo "There are FAILED checks, see data/checks_report.md" || echo "All checks passed."
echo "done: figures in ../figures, numbers in ../data"
