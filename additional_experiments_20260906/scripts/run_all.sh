#!/usr/bin/env bash
# Regenerates every figure and every number. All source repos are read-only.
set -e
PY=~/miniconda3/envs/isaaclab/bin/python
cd "$(dirname "$0")"
echo "=== Experiment 1: keypoint noise vs MoCap ==="   && $PY exp1_noise.py
echo; echo "=== Experiment 2: target-distribution coverage ===" && $PY exp2_coverage.py
echo; echo "=== Experiment 3: manual intervention ==="  && $PY exp3_manual.py
echo; echo "=== Diagnostics: described filter + intrinsics ===" && $PY exp4_filter.py
echo; echo "=== Global motion trajectories ===" && $PY exp5_global_traj.py
echo; echo "=== Fig. 5 recovery + reproduction ===" && $PY fig5_extract_from_pdf.py && $PY fig5_reproduce.py
echo; echo "All figures written to $(cd .. && pwd)"
