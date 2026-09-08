# Additional experiments — results

All numbers below are produced by `scripts/run_all.sh` and printed by the scripts
themselves. Every source repository was read only; nothing outside this folder was
modified or deleted.

**Figures**

| file | supports |
|---|---|
| `fig_A_keypoint_noise_qualitative.pdf` | Exp 1 — 3D keypoints + 2D projections + torso-length trace |
| `fig_B_keypoint_noise_quantitative.pdf` | Exp 1 — noise metrics, and the depth-source comparison for Fig. 6 |
| `fig_C_command_coverage.pdf` | Exp 2 — command-space coverage, MoCap / Video / Video (extended) |
| `fig_G_target_vs_data_distribution.pdf` | Exp 2 — target command distribution vs demonstrated velocity density |
| `fig_F_state_coverage.pdf` | Exp 2 — coverage of the 24-dim AMP discriminator state |
| `fig_D_terrain_coverage.pdf` | Exp 2 — terrain/posture coverage (stairs, box, stand-up) |
| `fig_E_manual_intervention.pdf` | Exp 3 — per-clip hand-coding in the pipeline |
| `fig_H_filter_vs_handcoding.pdf` | pipeline diagnostics (see `paper_notes.md`) |
| `fig_I_global_trajectories.pdf` | global motion paths of every clip, world coordinates in metres |
| `fig_J_center_paths.pdf` | top-down base-centre paths, every clip started at the origin in the same heading |

---

## Experiment 1 — How noisy are keypoint trajectories compared to MoCap?

**Method.** The Zhang et al. MoCap skeleton contains an exact analogue of all six
tracked keypoints (4 toes + pelvis ≈ rear hip + neck ≈ front hip), so the comparison
is like-for-like. All metrics are computed on fixed 16-sample (0.533 s) windows after
anti-aliased rate matching to a common 30 Hz, normalized by the per-clip torso length
L, and compared only over a **matched base-speed band of 0.73–2.87 L/s**
(224 MoCap windows, 26 video windows). Speed matching is essential: the MoCap
source clips contain long stationary stretches and fast running that the casual video
clips do not, and without it every windowed metric measures gait rather than sensor noise.

| metric | MoCap median (range) | Video median (range) | ratio | disjoint | p | Cliff's δ |
|---|---|---|---|---|---|---|
| M1 torso-length CV [%] | 0.70 (0.57–1.23) | 15.17 (1.73–21.79) | **21.6×** | yes | 0.0025 | +1.00 |
| M2 SG residual [% of L] | 0.409 (0.26–0.52) | 2.728 (1.67–3.55) | **6.7×** | yes | 0.0025 | +1.00 |
| M3 HF energy ≥7.5 Hz [%] | 0.283 (0.27–0.48) | 3.471 (1.91–13.41) | **12.3×** | yes | 0.0025 | +1.00 |
| M4 normalized jerk [1] | 199 (138.25–256.89) | 1130 (690.32–1489.02) | **5.7×** | yes | 0.0025 | +1.00 |

All four are **fully disjoint** across clips (every video clip is worse than every
MoCap clip), δ = +1.00, p = 0.0025 — the smallest value attainable at n = 5 vs 8, so
report the disjointness rather than the p-value.

Supporting numbers:

- Mean torso length: MoCap **0.408 m** vs video **0.237 m**.
- **Across-clip scale spread of the video reconstruction: 19.6%** (0.162–0.299 m
  for the *same dog*). Same animal, same rig, different clip — this is pure scale error
  and needs no biological null.
- **Interpolated foot samples — two distinct causes, and the second dominates.**
  Only **7.2%** of foot keypoints are untracked (the 2D track is exactly `(0,0)`), but a
  further large fraction are tracked yet land on a pixel with **no valid depth reading**,
  and those are interpolated too. Counting both: **24.8% of foot samples on average are
  interpolated rather than measured**, per-clip 10.0–48.2% (worst: `walk_869488000` 48.2%,
  `start_stop_785558000` 43.5%, `slow_1313807000` 30.7%). Longest untracked gap: 5 frames.
  This materially strengthens the caveat below: metrics M2/M3/M4 are computed partly over
  linearly interpolated spans, and interpolation is a low-pass operation, so the measured
  video-vs-MoCap ratios are **lower bounds** by a wider margin than the untracked-only
  figure would suggest. (Credit: an independent second analysis flagged that counting only
  untracked keypoints understates this; verified here.)

**Coordinates in Figs. A and I.** Both use GLOBAL world coordinates in metres. A
single rigid transform is applied per clip — frame-0 rear keypoint to the origin,
frame-0 heading to +x — and nothing is re-centred per frame, so the animal translates
through the scene and the real trajectory is visible. The per-clip transform is needed
only because clips were captured at different places in the world (the MoCap clips start
1.7–6.7 m apart and travel along y, while the video clips are already origin-referenced
and travel along x). Fig. I additionally reports path length, net displacement and
straightness per clip; note the stairs/box/stand-up side view rises to +0.5 m in z,
terrain the MoCap set never covers.

**Motion repertoire (Fig. J).** Top-down projection of the base centre only, with every
clip translated to the origin and rotated so its initial heading points along +x — so
only the *shape* of the motion differs. Net heading change per clip (start heading = 0°):

| dataset | net heading change of each clip |
|---|---|
| MoCap (6) | −3.6, +1.9, −4.0, −3.1 (four essentially straight), then −91.5 and −145.1 |
| Video (4) | −13.4, +18.9, −8.1, and −114.3 |
| Video extended (8) | the above plus **+47.3, −118.9, −73.4, −23.0** |
| stairs / box / stand-up (6) | −7.5, −13.7, −23.7, −174.9, −162.2, +87.8 |

MoCap is bimodal — four near-straight clips and two large turns, with nothing in
between. The extended video set fills the intermediate band (−119° to +47°, including
+47° and −73°), which is the same gap the command-space coverage figure shows. Caveat:
clip durations differ (MoCap `right turn0` is 2.5 s vs 0.9–2.3 s for the video clips), so
compare path *shape*, not path length.

**Clip selection for Fig. A (not cherry-picked).** The qualitative pair was chosen by
an objective speed match, not by noise level: MoCap `right turn0` (1.96 L/s) vs video
`turn_right` (1.98 L/s), both right turns, both trimmed to 1.47 s. That clip's torso CV
(33.4%) is *above* the video median of 15.2% — the full per-clip distribution for every
clip is in `fig_B`. Comparing a fast MoCap trot against a slow video walk would confound
gait with sensor noise, which is why the pair is matched on speed, duration and behaviour.

**Clips used, per category (all of them, no subsetting).**

| category | clips | source |
|---|---|---|
| MoCap (coverage) | all 6 | `mocap_AMP_for_hardware/` |
| Video (coverage) | all 4 | `fromVision_motions_DepthCam/` |
| Video extended (coverage) | all 8 | `fromVision_motions_DepthCam_extendedWithoutReverse/` |
| MoCap (noise baseline) | 5 full source clips (`dog_walk00/03/09`, `dog_run00/04`) | `data/dog_*_joint_pos.txt` |
| Video (noise) | all 8 extended clips | `data_fromVision_depth_cam/` |

Two deliberate exceptions: the terrain figure uses the **2 real box clips**, not the 5
files in `..._box/` (three of those are repeated `walk` clips, not box motions); and the
`obstacle_*` clips are excluded from all coverage numbers because they belong to no
dataset used in the paper — they *are* included in the hand-coding count, since they
were hand-tuned.

**Sentence for the paper.** *Under identical scale normalization, a common 0–15 Hz
analysis band and a matched base-speed range, video-derived keypoints violate the
rigid-torso constraint 22× more than MoCap (0.70% vs 15.2% length CV) and carry
6.7× more energy that a 0.17 s quadratic cannot explain (0.41% vs 2.7% of torso
length); the two distributions do not overlap on any clip. Because occlusion gaps are
linearly interpolated — a low-pass operation — these are lower bounds.*

### Depth-source comparison (quantifies Fig. 6)

Speed-matched M2 jitter and a geometric-plausibility measure over the four sources:

| source | M2 jitter [% of L] | frames with two feet <0.10 L apart | foot vertical span [L] | mean torso [m] |
|---|---|---|---|---|
| MoCap | 0.26–0.52 | 0.0% | 0.33 | 0.408 |
| Camera depth | 1.67–3.55 | 0.0% (max 4.4%) | 0.38 | 0.276 |
| DepthAnythingV2 | 2.99–7.12 | 0.0% (max 2.9%) | 0.38 | 0.267 |
| AnimalAvatar | 0.71–0.93 | **14.1% (max 29.6%)** | **1.34** | **0.795** |

**Important nuance, and it strengthens the paper.** AnimalAvatar has the *lowest*
jitter of the three video sources — a parametric SMAL model is smooth by construction —
yet it is the least physically plausible: up to **29.6% of frames place two feet closer
than 0.10 torso lengths** (overlapping limbs, exactly what Fig. 6 shows), the feet span
**1.34 L vertically** vs 0.33–0.38 L for every other source, and it reconstructs a
**0.79 m torso where camera depth gives 0.28 m on the same clips**. So the correct claim is
not "AnimalAvatar is noisier" but **"AnimalAvatar is smooth but geometrically wrong"**,
which is precisely why it yields a high cost of transport. MoCap and camera depth never
place two feet that close; DepthAnythingV2 is the jitteriest source.

---

## Experiment 2 — What additional part of the target distribution do the video motions cover?

Command box (verified in `velocity_env_cfg.py`, `heading_command=False`, so ω_z really
is sampled uniformly): v_x ∈ [−1, 1], v_y ∈ [−0.3, 0.3], ω_z ∈ [−1.57, 1.57], plus a 2%
standing atom. Coverage = fraction of a 10×6×10 grid whose centre lies within τ = 0.5
cell-normalized units of a demonstration.

| dataset | n samples | coverage | new vs MoCap | lost vs MoCap | union | inside box |
|---|---|---|---|---|---|---|
| MoCap | 318 | 5.8% | — | — | — | 64% |
| Video | 177 | 5.3% | **+4.5 pp** | 5.0 pp | 10.3% | 63% |
| Video (extended) | 358 | **10.0%** | **+8.5 pp** | 4.3 pp | **14.3%** | 63% |

- **Extending 4 → 8 clips adds +4.7 pp**, nearly doubling coverage (5.3% → 10.0%).
  This is the direct quantitative backing for the paper's claim that cheap extra clips help.
- Bootstrap 95% CI on "new vs MoCap" for the extended set (resampling whole clips):
  **[3.2, 8.8] pp** — excludes zero.
- τ sensitivity (absolute values move, the ordering and ~2× ratio do not):
  τ=0.35 → 2.3/3.0/5.7%, τ=0.5 → 5.8/5.3/10.0%,
  τ=0.75 → 10.0/10.7/19.3%, τ=1.0 → 13.8/18.0/28.0% (MoCap/Video/Video ext).
- **Honest framing: the sources are complementary, not one-sided.** Video (extended)
  adds 8.5 pp but *loses* 4.3 pp that MoCap covers (the fast-gait region: MoCap
  reaches v_x = 2.36 m/s). The union is the claim.
- Only ~64% of frames fall inside the command box at all, for every source.

### Density mismatch (Fig. G)

Even where covered, the demonstrations are far from the uniform command distribution:
**91% (MoCap), 92% (Video) and 86% (Video extended)** of the command box contains
no demonstration at all. Jensen–Shannon divergence to the uniform target (5×3×5 grid):
MoCap 0.420 bits, Video 0.313, Video extended 0.334. Per-axis Wasserstein distance to
uniform (v_x, v_y, ω_z): MoCap 0.72, 0.08, 0.34; Video ext 0.40, 0.07, 0.67.
MoCap's mass sits at **positive v_x only**, which is exactly the backward/yaw region
where Figs. 5 and 7 of the paper report the tracking failure.

### AMP state-space coverage (Fig. F)

Occupancy of the (joint position, joint velocity) phase portrait, pooled over the four
legs — the discriminator's own coordinates:

| dataset | hip | thigh | calf |
|---|---|---|---|
| MoCap | 67% | 78% | 51% |
| Video | 64% | 70% | 56% |
| Video (extended) | **78%** | **81%** | **68%** |

Video (extended) covers more of the discriminator's state space than MoCap on all three
joint types. (Caveat: the velocity axis is affected by the MoCap timebase issue below;
the joint-position axis is not.)

### Terrain / posture coverage (Fig. D)

Maximum foot clearance in torso lengths:

| group | max clearance / L |
|---|---|
| MoCap (6 clips) | 0.21–0.66 |
| Video flat (8 clips) | 0.23–0.65 |
| Video box | 0.80–0.92 |
| Video stairs + stand-up | **1.16–2.97** |

This is the cleanest coverage result and needs no threshold tuning. **Video flat
(0.23–0.65) matches MoCap (0.21–0.66) almost exactly** — which rules out
"video simply inflates clearance" — while box, stairs and stand-up reach **up to
3.0 L, i.e. 4.5× beyond anything in the MoCap dataset**, whose own maximum is a
canter flight phase rather than a step-up.

---

## Experiment 3 — How much manual intervention is required?

Scope per your direction: keypoints are treated as tracked; this counts the **per-clip
hand-coding in the pipeline scripts**, parsed directly from source so it is auditable.

| site | hand-coded content | amount |
|---|---|---|
| `retarget_config_go2.py` (video) | per-clip `SIM_ROOT_OFFSET_LOCAL`, `SIM_TOE_OFFSET_LOCAL`, `REF_POS_SCALE`, `feet_z_amplification`, `RETARGET_METHOD`, `start/end_frame` | **296 scalars over 17 clips** (median 17, range 16–21) |
| `retarget_config_go2.py` (MoCap) | one global config, **no per-clip branches** | **15 scalars for all 6 clips** |
| `3d_recon.py:279-298` | hand-typed 3D outlier frame ranges | 9 statements / 6 clips |
| `3d_recon.py:17-27` | hand-picked ground-plane frame | 5 clips |
| `3d_recon.py:108-112` | SLAM rotation overridden to identity | 1 clip |
| `retarget_motion_fromVision.py:80` | base marker order swapped back | 3 clips |
| `retarget_motion_fromVision.py:108` | stand-up feet x/y frozen after frame 10 | 4 statements |

**Headline: 317 hand-set values for 32 s of video vs 15 for the entire MoCap dataset**
— **9.9 vs 2.2 per second of usable demonstration (4.5×)**, or
**17.4 retargeting scalars per clip vs 2.5** (MoCap needs none per clip — one shared config).

Evidence of how fragile this is: one branch (`obstacle_1_301506103500`, 6 scalars) has a
13-digit sequence name that matches no directory and **can never execute**.

**Recommended framing.** Separate *acquisition* cost from *processing* cost. Recording
is genuinely cheap — handheld camera, outdoors, seconds of footage, no studio, no
markers, no animal training. Processing currently is not, and `paper_notes.md` gives a
concrete path to reducing it.

---

# Appendix — Reproduction of the paper's Fig. 5

**Original generator (found).**
`project_repos/isaac_lab/IsaacLab/plot_errorOnTargetDistribution.py` → `main()`,
which writes `plots/combined_heatmap_grid.pdf`. Row order, column order, colour limits and
all styling in the paper's Fig. 5 come from that file; the row/column labels come from
`plot_DEFINITIONS.py`.

- rows = `2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_*`,
  `2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_*`,
  `2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_*`
- columns = `error_vel_xy`, `error_vel_yaw`, `mean_mechanical_cot`, `agent_expert_distances`
- eval dir = `TargetXYDistributionEvaluation`, grid = **21 (v_x) × 7 (v_y)** = 147 commands per panel
- fixed colour limits: `error_vel_xy` (0, 0.1), `error_vel_yaw` (0, 0.4),
  `mean_mechanical_cot` (0.8, 2.0), `agent_expert_distances` (2.0, 4.0)

**The raw inputs are not on this machine.** The script reads one YAML per command from
`logs/rsl_rl/unitree_go2_AMPflat/<experiment>_SEED_*/TargetXYDistributionEvaluation/*.yaml`.
`logs/` is a nested DVC project whose remote is
`webdavs://nextcloud.in.tum.de/.../IsaacLabRunsBackupDVC`; the local cache holds one training
run (15 objects, 27 MB) and no evaluation YAMLs, and no `.dvc` pointer files remain, so a
`dvc pull` has nothing to resolve. Re-running the sweep would need IsaacLab + the trained
policies.

**How it was reproduced anyway.** Every heatmap cell in the shipped
`plots/combined_heatmap_grid.pdf` is an individual vector rectangle whose fill is a verbatim
entry of matplotlib's 256-entry viridis LUT. Inverting the LUT recovers each cell value to
within one LUT step, i.e. `(vmax − vmin)/256` (±0.0002 m/s for `error_vel_xy`).
Verified that seaborn's `center=` argument does not resample the colormap here, so the
mapping is exactly `idx = floor(256·(v − vmin)/(vmax − vmin))`.

- `scripts/fig5_extract_from_pdf.py` → `data/fig5_recovered/*.csv` (12 files, 21×7 each)
- `scripts/fig5_reproduce.py` → `fig5_reproduced.pdf` / `.png`

**Verification.**
1. Re-render vs. the original PDF at 100 dpi: **max abs pixel difference 0.0** over
   822 × 2293 px — the reproduction is pixel-identical.
2. The same extractor run on the paper PDF itself (page 6) returns the same values to
   **4.4e-7**, confirming `plots/combined_heatmap_grid.pdf` is the figure in the paper and
   that these CSVs are Fig. 5's real numbers.

**Recovered values** (mean over the 21×7 command grid; `[n]` = cells that saturate at a
colour-bar limit, whose true value is only bounded):

| dataset | Track. Err. Vel. [m/s] | Track. Err. Yaw [rad] | CoT [1] | Imitation score ↓ |
|---|---|---|---|---|
| MoCap (AMP) | 0.0657 [6] | 0.3374 [72] | 1.3215 [12] | 2.9283 [2] |
| Video w. Depth Camera (AMP) | 0.0594 [5] | 0.2133 [1] | 1.3307 [26] | 3.1656 [16] |
| Video w. Depth Camera (extended) (AMP) | **0.0496** [0] | **0.1119** [0] | **1.1820** [26] | 3.0059 [23] |

Change vs. MoCap over the grid:

| | Track. Err. Vel. | Track. Err. Yaw | CoT | Imitation score |
|---|---|---|---|---|
| Video | −9.5 % | −36.8 % | +0.7 % | +8.1 % |
| Video (extended) | **−24.4 %** | **−66.9 %** | **−10.6 %** | +2.6 % |

These bracket the abstract's "**23 %** velocity tracking / **13 %** cost of transport" but do
not equal it, for two reasons worth knowing before quoting them: (i) the MoCap row saturates
at the colour-bar ceiling in 6 velocity-error and 12 CoT cells, so its true grid mean is
*higher* and the true reductions are *larger* than the table shows; (ii) the abstract's
numbers are aggregate metrics, not grid means over this figure. **Use the abstract's numbers
in the paper; use these only to talk about the figure.**

Directional detail that supports the text's claim about backward walking:

| dataset | yaw err. `v_x < 0` | yaw err. `v_x > 0` |
|---|---|---|
| MoCap (AMP) | 0.3739 | 0.3080 |
| Video (AMP) | 0.1918 | 0.2430 |
| Video (extended) (AMP) | 0.0989 | 0.1284 |

MoCap is the only row whose yaw error is *worse* going backward (1.21× vs forward); both
video rows invert that ratio (0.79× / 0.77×). Note that 72 of MoCap's 147 yaw cells sit at
the 0.4 rad ceiling, so the MoCap deficit is understated here.
