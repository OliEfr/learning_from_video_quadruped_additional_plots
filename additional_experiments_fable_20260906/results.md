# Additional experiments for "Learning Quadruped Locomotion from Casual Videos" (ICRA)

All numbers below are produced by `scripts/run_all.sh` (≈10 s, numpy/scipy/matplotlib only, all repositories read-only).
Figures are vector PDFs in `figures/` (PNG previews next to them); per-clip tables and JSON dumps are in `data/`;
`data/checks_report.md` lists the independent sanity checks (all PASS).

Datasets used (identified from the paper's 6.0 s / 12.2 s and the run names in `plot_errorOnTargetDistribution.py`):

| paper name | folder under `IsaacLab/datasets/` | clips | frames | duration |
|---|---|---|---|---|
| MoCap (AMP) | `mocap_AMP_for_hardware` | 6 (pace, trot, trot2, canter, right turn0, left turn0) | 330 | 6.80 s at FrameDuration 0.021 s |
| Video w. Depth Camera (AMP) | `fromVision_motions_DepthCam` | 4 (walk, slow, turn L, turn R) | 185 | 6.03 s at 0.03334 s |
| Video w. Depth Camera (extended) (AMP) | `fromVision_motions_DepthCam_extendedWithoutReverse` | 8 (+ slow turn, L-R turn, 2 start-stop) | 374 | 12.20 s at 0.03334 s |

Colours in every figure follow `IsaacLab/plot_DEFINITIONS.py` (tab10 index 5 = MoCap, 1 = Video, 6 = Video (extended)).

---------------------------------------------------------------------------------------------------

## Experiment 1 - How noisy are the keypoint trajectories compared to MoCap?

### Figures
* `fig_exp1_trajectories.pdf` (7.16 in, 2 rows x [3D | top view x-y | side view x-z]): MoCap *pace* vs. Video *walk*, six keypoints in **global coordinates** - the animal translates 0.9 m (MoCap) and 1.6 m (Video) through the scene. One rigid transform per clip (origin = body centre of frame 0, ground = 5th percentile of foot heights, yaw such that the mean heading is +x); nothing is re-centred per frame. Needed because MoCap studio frame and SLAM/camera frame are unrelated.
* `fig_exp1_trajectories_with_corrected.pdf`: same with a third row showing the walk clip re-lifted with the correct 640x360 intrinsics (see notes.md, item 1).
* `fig_exp1_all_clips_side.pdf`, `fig_exp1_all_clips_top.pdf`: every clip of both sets (6 MoCap, 8 Video) at one common metric scale (no cherry-picking).
* `fig_exp1_trajectory_grid_all_clips.pdf` (author's follow-up, revision 3; single-column): two rows only - MoCap *trot2* and Video *walk* (author: "keep only the mocap trot2 and video walk") - two columns (left top view x-y, right side view x-z), global coordinates with one rigid transform per clip, no annotations. Panels have the size and equal metric aspect of the panels in `fig_exp1_all_clips_*` (x span 2.4 m, y/z span 0.7 m). Torso keypoints black, paws in four green shades sampled from the viridis colormap of Fig. 5, row labels in the Fig. 4/5 method colours. Per-clip numbers: `data/exp1_per_clip_table.md`. (The 14-row version and the `_sameFigSize` variant of revision 2 are superseded; the `_sameFigSize` PDF on disk is no longer regenerated.)
* `fig_exp1_trajectory_grid_all_clips_corrected.pdf` (author's follow-up, "corrected version of grid_all_clips"): identical layout, but the Video row (*walk*) is re-lifted from the same 2D tracks, depth maps and camera poses with the correct 640x360 intrinsics (`exp1_relift_keypoints.py`, mode `half_res`); the MoCap row is unchanged. Because the corrected reconstruction is about twice as large (hip height 0.48 m instead of 0.25 m), the shared y/z span is widened from 0.7 m to 0.9 m for **both** rows (MoCap and video rows always share identical ranges within a figure); x span stays 2.4 m. Hold it next to `fig_exp1_trajectory_grid_all_clips.pdf`: the corrected dog has the same body height as the MoCap dog and a longer path (1.95 m vs 1.62 m in 0.9 s).
* `fig_exp1_trajectory_grid_walk_stages.pdf` (author's follow-up): the corrected 2-row grid extended to four rows that follow the *walk* clip through the pipeline - (1) MoCap trot2 markers; (2) Video walk re-lifted with the corrected intrinsics (analysis only); (3) Video walk **post-processed keypoints as fed to the retargeting** (`data_fromVision_depth_cam/walk_869488000`, i.e. the pipeline output used in the paper); (4) Video walk **retargeted onto the Go2**: robot root position (black) and paw positions obtained by forward kinematics of the 12 joint angles in the paper's expert file `datasets/fromVision_motions_DepthCam/walk_869488000_amp.txt` (Go2 geometry from `assets/go2/go2.urdf`; joint columns in URDF leg order FL, FR, RL, RR - verified by matching FK paws to the saved MoCap toe targets, 13 cm residual vs 19-28 cm for the other candidate orders). Same panel geometry, colours and shared axis ranges (y/z span 0.9 m) as the corrected grid. Reading across rows: the post-processed dog (row 3) is half the height of the MoCap dog and of the corrected reconstruction (rows 1-2); the retargeting (row 4) restores a robot-sized body (root 0.28 m above the lowest paws after SIM_ROOT_OFFSET, REF_POS_SCALE = 1.0) but inherits the compressed lateral geometry - the IK drives the FL/RL hips to -0.42/-0.34 rad on average and the FK paws span y = -0.27..+0.07 m relative to the root (paws pulled to one side), which the per-clip toe offsets were tuned to compensate. The retargeted root travels 1.34 m end-to-end vs 1.62 m for the post-processed torso centre on disk: the expert file was produced from an earlier reconstruction/trimming of the same clip (see notes.md, item 8).
* `fig_exp1_trajectory_grid_all_clips_14rows.pdf` and `fig_exp1_trajectory_grid_all_clips_corrected_14rows.pdf`: the same two variants with the full clip set (6 MoCap segments, then all 8 flat-walking video clips; single-column, 6.9 in tall), in case "grid_all_clips" was meant literally.
* `fig_exp1_com_topview_overlay.pdf` (author's follow-up; 3.5 in x 2.0 in): top view of the **torso centre only** (mean of the two torso keypoints, no paws). Left: Video clips L-R turn, turn R, turn L, walk overlaid; right: all six MoCap segments overlaid. One rigid transform per clip: origin = torso centre of frame 0, yaw such that the torso heading averaged over the first 0.2 s points to +x, i.e. every movement starts in positive x direction (o = start, > = end). Colours from the Fig. 5 viridis colormap: green shades for the video clips, blue shades for MoCap. Both panels share the same axes (x -0.3..1.9 m, y +-0.7 m, equal aspect). It visualises what Exp. 2 quantifies: MoCap = straight forward locomotion at 0.7-2.1 m/s plus one wide right arc and one in-place left pivot; video = slow/medium forward walking with tighter left and right turns (mean 1.7 rad/s) and the fast straight *walk* clip.
* `fig_exp1_noise_metrics.pdf` (7.16 in x 1.9 in, 4 panels): per-clip values (dots) and set means (bars) of the four noise metrics for MoCap, Video (pipeline output) and Video (corrected intrinsics).

### What was controlled for
* **Sampling rate**: MoCap is 60 Hz (frames/seconds in `data/dog_clips_info.txt` = 59.7), video is 30 Hz. MoCap is sub-sampled to 30 Hz before any finite difference or smoothing so both sources see identical operators. Left at 60 Hz the feet ratio would read 8.8x instead of 2.9x (`checks_report.md`).
* **Subject size**: metrics are reported in mm and normalised by hip height (median body-centre height above ground): MoCap dog 0.44 m, video dog 0.26 m in the pipeline output (0.52 m after the intrinsics correction). Also given in "robot units" (x REF_POS_SCALE of the retargeting: 0.825 MoCap, 1.0 video).
* **Behaviour**: only the flat-walking clips are compared (walk/trot/pace/turns/start-stop on both sides). Metrics that need no ground truth were chosen: high-frequency residual, rigid-segment consistency, stance foot-height scatter, power spectrum.
* **Clip selection**: all 8 flat-walking video clips and all 6 MoCap segments exactly as used by `retarget_motion.py`; per-clip values in `data/exp1_per_clip_table.md`.

### Metrics (30 Hz, identical settings for every source)
* **M1** high-frequency residual: RMS of x - SavitzkyGolay(x; 7 frames = 0.23 s, poly 3) per keypoint.
* **M2** frame-to-frame jitter: RMS of the second finite difference.
* **M3** rigid-body consistency: std/mean of the distance between the two body keypoints (rear hip-front hip; pelvis-neck for MoCap).
* **M4** stance foot-height scatter: robust std (1.4826 MAD) of the foot height in the lowest 30 % of frames of each foot.
* **M5** (video only) fraction of foot samples that are interpolated (occluded track or zero depth at the tracked pixel).

### Headline numbers (set means; per clip in `data/exp1_per_clip_table.md`)

| metric | MoCap (6 clips, 167 fr.) | Video, pipeline output (8 clips, 366 fr.) | ratio | Video, corrected intrinsics | ratio |
|---|---|---|---|---|---|
| M1 body [mm] / [% hip h.] | 1.9 / 0.42 | 4.6 / 1.81 | 2.4x / **4.3x** | 4.6 / 0.91 | 2.5x / 2.1x |
| M1 feet [mm] / [% hip h.] | 5.9 / 1.35 | 9.7 / 3.81 | 1.6x / **2.8x** | 12.8 / 2.51 | 2.1x / 1.9x |
| M1 feet in robot units [mm] | 4.9 (x0.825) | 9.7 (x1.0) | **2.0x** | - | - |
| M2 feet [mm / frame^2] | 27 | 31 | 1.1x | 40 | 1.5x |
| M3 body-segment length CV [%] | 1.2 (median 0.7) | 21.0 (median 19.4; turning clips 27-38, straight clips 6-12) | **17x** | 7.8 (median 6.0) | 6.5x |
| M4 stance foot height [mm] / [% hip h.] | 3.1 / 0.71 | 7.6 / 2.77 | 2.4x / **3.9x** | 11.6 / 2.04 | 3.7x / 2.9x |
| power above 5 Hz (PSD, normalised) | 1.1e-4 | 5.4e-4 | **4.7x** | 2.0e-4 | 1.8x |
| M5 interpolated foot samples | 0 % | 27 % (per clip 10-49 %; zero depth alone 3-48 %, mean 18 %) | - | same | - |
| foot samples > 2 cm below ground | 0 % | 0-1.1 % | - | 0-1.9 % | - |

Sensitivity (`checks_report.md`): M1-feet ratio 3.9x / 2.9x / 2.1x for SG windows 5 / 7 / 9 frames; 3.6x with a 4th-order zero-phase Butterworth at 5 Hz instead of SG; plain sub-sampling vs. anti-aliased decimation of MoCap changes M1 by <3 %.

Downstream (what the AMP discriminator actually sees, joint pos + joint vel, native FrameDuration):

| | MoCap | Video | Video (extended) |
|---|---|---|---|
| SG residual RMS of joint angles [deg] | 0.25 | 2.16 (8.5x) | 1.74 (6.8x) |
| RMS joint acceleration from stored joint velocities [rad/s^2] | 44 | 103 (2.4x) | 83 (1.9x) |
| RMS joint velocity [rad/s] | 2.48 | 3.06 | 2.57 |

### Filter proposal (author's follow-up): the filter is part of the video method; MoCap is the unfiltered baseline
Script `scripts/exp1_filter.py`; tables `data/exp1_filter_tables.md`, numbers `data/exp1_filter.json`; figures `fig_exp1_filter_effect.pdf`, `fig_exp1_trajectory_grid_all_clips_corrected_filtered.pdf` (+ `_14rows`). Unfiltered figures/numbers are kept.

*Where it belongs.* **Stage A** (depth-domain outlier rejection) goes into `3d_recon.py` between the depth look-up and the zero-interpolation - the place where Sec. III-A describes a "discard depth jumps > 0.5 m" rule. Implemented as a Hampel filter on each marker's depth series (+-2 frames, 3 x 1.4826 x MAD, at least 5 cm); flagged samples are interpolated like missing depth. **Stage B** (zero-phase 4th-order Butterworth low-pass, `sosfiltfilt`, no group delay) goes into `postprocess_global_align_plane_approach.py`, on the ground-aligned 3-D keypoints before retargeting; it replaces the causal 3-frame moving average of the paper text (a causal average delays contacts by one frame).

*Cutoff.* f_c = 6 Hz, fixed by a physical criterion: the averaged video keypoint spectrum is white (within 2x of its 10-15 Hz floor) above 6.1 Hz; stride fundamentals are 0.7-1.5 Hz (video) and 1.4-1.9 Hz (MoCap), so 6 Hz keeps the fundamental plus >= 3 harmonics and the 0.2-0.3 s swing bump.

*Noise metrics (30 Hz, same operators for every row; MoCap = baseline dataset as used, unfiltered):*

| source | filter | M1 torso [% h] | M1 paws [% h] | M1 paws [mm] | M2 paws [mm/fr²] | M3 seg CV [%] | M4 stance [% h] | M4 stance [mm] | PSD >5 Hz [1e-4] |
|---|---|---|---|---|---|---|---|---|---|
| MoCap (baseline) | none | 0.42 | 1.35 | 5.9 | 27.4 | 1.21 | 0.71 | 3.2 | 1.14 |
| Video (pipeline output) | none | 1.84 | 3.67 | 9.3 | 29.9 | 21.1 | 2.49 | 6.9 | 4.71 |
| Video (pipeline output) | B only | 0.37 | 1.01 | 2.6 | 14.1 | 20.9 | 2.62 | 7.2 | 0.81 |
| Video (pipeline output) | **A+B (method)** | **0.37** | **0.99** | **2.5** | **13.7** | **20.9** | **2.61** | **7.2** | **0.79** |
| Video (corrected intrinsics) | none | 0.91 | 2.51 | 12.8 | 40.4 | 7.8 | 2.04 | 11.6 | 2.04 |
| Video (corrected intrinsics) | A+B (method) | 0.20 | 0.67 | 3.4 | 18.2 | 7.7 | 2.02 | 11.7 | 0.34 |

Video (method, pipeline output) vs MoCap: M1 paws 0.73x (mm: 0.42x), M1 torso 0.87x, M2 0.50x, PSD >5 Hz 0.69x - i.e. *below* MoCap; M4 stance scatter 3.7x in % hip height, **2.3x in mm (7.2 vs 3.2 mm)**; M3 torso-segment CV 17x. Stage A is nearly inert on these clips (14 of 3730 depth samples flagged, the 0.5 m rule fires twice); its value is robustness, not the metrics.

*Motion preservation (video, filtered vs unfiltered, mean over the 8 clips):* net displacement 0 %, stride frequency 0 %, stance fraction 44.0 -> 44.1 %, peak paw clearance 63 -> 62 mm (-1 %; corrected re-lift -6 %); path length -5.6 % and 95th-percentile speed -10 % are the removed noise excursions. Zero-phase: no timing delay; the raw paw heights cross the 2 cm stance threshold 85 times vs 73 after filtering (spurious double crossings removed), which is what the 36 ms "onset shift" of matched events reflects.

*Which metric/figure to report in the paper (recommendation).* Report **M4, the stance paw-height scatter in millimetres**: "During stance the reconstructed paw height scatters by 7.2 mm (robust std) with our pipeline versus 3.2 mm in the MoCap data." It is physically meaningful (ground-contact consistency, directly relevant to foot clearance), it is independent of the filter cutoff (the same 7 mm with or without filtering), it is worse than MoCap by a plausible factor (2.3x) and it holds for the corrected intrinsics as well (11.7 mm, i.e. 3.7x - quote the pipeline value, that is the data the policies were trained on). Pair it with `fig_exp1_trajectory_grid_all_clips_corrected_filtered.pdf` (or the unfiltered corrected grid) and `fig_exp1_filter_effect.pdf` (a) as the visual.

Do **not** headline M1, M2 or the >5 Hz power after filtering: with the filter the video values fall below the unfiltered MoCap values (paw residual 2.5 vs 5.9 mm), and a reviewer will immediately ask whether MoCap was filtered the same way - it was not, and if it were its residual would drop by 44 % (`exp1_filter.json`, reference entry), which turns the comparison into a comparison of filters. If a residual number is wanted, quote the *unfiltered* pipeline value (paw residual 9.3 mm vs 5.9 mm, 1.6x; 3.7 % vs 1.35 % of hip height, 2.7x) and say that the method's low-pass removes the white part of it. Avoid M3 (torso-segment CV 21 % vs 1.2 %) as a headline number; it is the honest weak spot (heading noise during turns) and belongs in the limitations sentence.

Consistency note: the expert files behind the paper's results were produced *without* stages A/B; if the filter is described as part of the method, either regenerate the expert data with it or state that the filter was added after the reported experiments.

*Paste-ready method text.* `paper_postprocessing.tex` holds a drop-in replacement (full and compact variant) for the "Post-Processing." paragraph of Sec. III-A - method description only, no numbers. The quantitative effect of the filter and the Fig. 7 caption are still to be written; the numbers are in the tables above.

### Paste-ready sentences
* "At a common 30 Hz sampling rate and after normalising by the animals' hip height, the high-frequency residual (above ~5 Hz) of the video keypoints is 2.8x that of MoCap for the paws (3.8 % vs. 1.4 % of hip height, 9.7 mm vs. 5.9 mm) and 4.3x for the torso keypoints (1.8 % vs. 0.4 %); in the units of the retargeted robot this corresponds to 2.0x (paws) and 3.0x (torso)."
* "The power spectrum of the video keypoints shows a flat noise floor above ~6 Hz that lies more than an order of magnitude above MoCap; integrated above 5 Hz the video keypoints carry 4.7x the MoCap power."
* "During stance the reconstructed paw height scatters by 7.6 mm (robust std) in the video data versus 3.1 mm in MoCap."
* "The distance between the two torso keypoints, which should be constant, varies by 21 % (CV) in the video clips - 27-38 % in the turning clips - compared with 1.2 % in MoCap; this directly perturbs the reconstructed base heading."
* "On average 27 % of all paw samples (10-49 % per clip) are linearly interpolated because the paw is occluded or the time-of-flight depth is invalid at the tracked pixel; MoCap has no gaps."
* "After retargeting, the joint trajectories of the video expert set are 6.8-8.5x rougher than the MoCap set (SG residual 1.7-2.2 deg vs. 0.25 deg) and have 1.9-2.4x higher joint accelerations; despite this, the policies trained on them match or exceed MoCap in task tracking (Fig. 4), i.e. AMP is tolerant to this noise level."
* Caveat for the paper: "Absolute video noise figures are lower bounds for the lateral directions because the on-disk keypoints were back-projected with 1280x720 intrinsics onto 640x360 pixels (see notes.md); re-lifting with the correct intrinsics gives 1.9x (paws) / 2.1x (torso) relative to MoCap."

---------------------------------------------------------------------------------------------------

## Experiment 2 - Which additional part of the target distribution do the video motions cover?

### Setup (verified in the source)
* Command distribution (`velocity_env_cfg.py`, `CommandsCfg`): uniform vx in [-1, 1] m/s, vy in [-0.3, 0.3] m/s, yaw rate in [-1.57, 1.57] rad/s, `heading_command=False`, `rel_standing_envs=0.02`.
* Evaluation grids of Fig. 5 (`targetDistributions.sh`): vx x vy at yaw rate 0 (21 x 7 cells, step 0.1) and vx x yaw rate at vy = 0 (21 x 21).
* Expert velocities: base-frame vx, vy and yaw rate from the AMP files (root position/quaternion), Savitzky-Golay smoothed (7 frames, poly 3) before central differencing; the stored linear velocity (= diff(root_pos)/FrameDuration) and the raw yaw-rate are reported as "raw". The angular-velocity columns of all expert files are zero. Frames are weighted like the AMP loader samples them (MotionWeight / clip length).
* Coverage rule: an evaluation cell is covered if at least one expert frame lies within +-0.05 (half the grid step) of the cell centre in both plotted components and within +-0.25 rad/s (vx-vy grid) / +-0.1 m/s (vx-yaw grid) in the third. 3-D box coverage: fraction of uniformly sampled commands with an expert frame within (0.1 m/s, 0.1 m/s, 0.2 rad/s).
* The AMP discriminator observes joint positions and joint velocities only (`AMPLoader` default `amp_data`); base velocity enters training through RSI and through the gait speed encoded in the joints.

### Figures
* `fig_exp2_target_coverage.pdf` (three-way comparison, rows = MoCap / Video / Video (extended), columns = the two Fig.-5 grids): colour = normalised distance from each target command to the nearest expert frame, white dot = cell covered.
* `fig_exp2_state_coverage.pdf` (Fig.-5 layout, rows = dataset): AMP-weighted state heatmaps for vx-vy, vx-yaw rate (red dashed box = command range), front-leg and rear-leg thigh-calf joint space (red + = Go2 default pose).
* `fig_exp2_velocity_distributions.pdf` (Fig.-5 layout): target distribution (grey, uniform + 2 % standing spike) vs. the AMP-weighted distribution of vx, vy, yaw rate actually present in each dataset.

### Headline numbers (`data/exp2_coverage.json`, `data/exp2_summary_table.md`)

| | MoCap | Video | Video (extended) |
|---|---|---|---|
| covered vx-vy cells (yaw 0), of 147 | 14 (10 %) | 22 (15 %) | 24 (16 %) |
| covered vx-yaw cells (vy 0), of 441 | 56 (13 %) | 48 (11 %) | 83 (19 %) |
| 3-D command-box coverage | 13.2 % | 13.9 % | **25.0 %** |
| vx p5-p95 [m/s] (mean) | [-0.09, 2.22] (0.93) | [0.05, 1.68] (0.60) | [-0.04, 1.67] (0.53) |
| vy p5-p95 [m/s] | [-0.27, 0.11] | [-0.29, 0.10] | [-0.29, 0.09] |
| yaw rate p5-p95 [rad/s] | [-0.94, 2.43] | [-2.82, 3.49] | [-3.09, 2.31] |
| expert mass with speed < 0.1 m/s and yaw rate < 0.2 rad/s ("standing") | 0.0 % | 2.2 % | 4.0 % |
| expert mass with yaw rate > 0.5 rad/s | 33 % | 42 % | 54 % |
| expert mass with vx > 1 m/s (outside range) | 45 % | 25 % | 22 % |
| expert mass outside the command box | 56 % | 49 % | 46 % |
| covered cells with vx < 0 (vx-vy / vx-yaw) | 0 / 0 | 2 / 2 | 2 / 7 |
| covered vx-vy cells with 0 <= vx <= 0.5 (of 42) | 3 | 17 | 18 |
| covered vx-yaw cells with yaw rate <= -0.5 / >= +0.5 | 17 / 10 | 12 / 4 | 31 / 8 |

Set relations (cells): vx-vy grid: Video (ext) covers 22 cells MoCap does not, MoCap covers 12 cells Video (ext) does not (all at vx 0.4-1.0), union of all three = 36/147; Video (ext) adds only 2 vx-vy cells over Video. vx-yaw grid: Video (ext) covers 64 cells MoCap does not and 35 more than Video; MoCap covers 37 cells Video (ext) does not (fast forward walking with small yaw).

Per clip (smoothed, `data/exp2_coverage.json["per_clip"]`): *slow* alone covers 18 vx-vy cells (vx 0-0.37 m/s, vy +-0.15); *walk* is the only clip above 0.8 m/s and lies at vx 1.2-1.8 m/s (mean 1.54), i.e. mostly **outside** the +-1 m/s command range; *turn L / turn R / slow turn* have net heading changes of +143 / -142 / -142 deg in 1.45 s (mean 1.70 / -1.73 / -1.73 rad/s, above the 1.57 rad/s command limit) with a smoothed-rate p5-p95 of [0.05, 4.44] / [-3.32, -0.48] / [-3.83, -0.49] rad/s; *start-stop 1* has vx in [-0.11, 0.16] m/s (mean 0.05) and *L-R turn* swings between -3.9 and +5.6 rad/s with a net change of only -18 deg. MoCap: pace 0.71 m/s, trot/trot2 1.15 m/s, canter 2.08 m/s (all straight), right turn0 (-128 deg in 3.1 s, vx 0.49), left turn0 (pivot at 0.02 m/s, +1.84 rad/s).

Robustness (`exp2_coverage.json["sensitivity"]`): with raw instead of smoothed velocities the noisy video sets gain spurious cells (Video 32/55, Video (ext) 37/89 vs. MoCap 14/60); halving/doubling the third-component slice gives (12, 37) / (17, 69) for MoCap, (17, 33) / (22, 53) for Video and (17, 58) / (25, 96) for Video (ext) - the ordering MoCap < Video < Video (ext) on the yaw grid and on the 3-D box is stable, the vx-vy ordering MoCap < Video ~ Video (ext) is stable.

### Paste-ready sentences
* "Over the 3-D command box the expert data cover 13 % (MoCap), 14 % (Video) and 25 % (Video (extended)) of the target commands within one evaluation-grid step; on the vx-yaw-rate evaluation grid of Fig. 5 the extended video set covers 83 of 441 cells versus 56 (MoCap) and 48 (Video), and it adds 64 cells that MoCap does not cover, predominantly at yaw rates beyond 0.5 rad/s (31 cells with yaw rate <= -0.5 rad/s vs. 17 for MoCap)."
* "The MoCap set contains no standing frames and 45 % of its (AMP-weighted) samples are faster than the 1 m/s command limit (trot 1.15 m/s, canter 2.1 m/s at the replay rate); the video sets instead concentrate on 0-0.7 m/s (17-18 of the 42 vx-vy cells with 0 <= vx <= 0.5 m/s are covered, versus 3 for MoCap) and include 2 % (Video) and 4 % (extended) near-standing samples."
* "The additional clips of the extended set contribute almost exclusively turning and slow/standing behaviour: they add 35 covered cells on the vx-yaw grid but only 2 on the vx-vy grid."
* "Neither source contains backward locomotion: MoCap covers no cell with vx < 0, and the only negative-vx cells covered by the video sets are at vx = -0.1/-0.2 m/s during pivoting turns (2 of 63 negative-vx cells on the vx-vy grid, 7 of 210 on the vx-yaw grid); this is consistent with the poor tracking of negative vx targets in Fig. 5." (supports the Fig. 5 caption)
* "Within the +-1 m/s command range MoCap covers vx 0.4-1.0 m/s more densely than the video sets (12 vx-vy cells covered by MoCap but not by Video (extended)); the video sets' only fast clip (walk) runs at 1.2-1.8 m/s."
* Caveat: "Base velocities in the video expert files are affected by the intrinsics scaling issue (notes.md, item 1): lateral image-plane motion is underestimated by up to 2x, so the 'true' dog speeds are higher than the dataset values; the dataset values are nevertheless what the policies were trained to imitate."

---------------------------------------------------------------------------------------------------

## Experiment 3 - How much manual intervention is required?

Scope as instructed: keypoints are treated as tracked; "manual intervention" = per-clip hand-coding in the pipeline scripts, obtained by AST parsing (every finding with file:line and statement in `data/exp3_findings.csv`; rules in the docstring of `scripts/exp3_manual_intervention.py`). The MoCap path is `retarget_motion.py` + `UnitreeGo2ConfigMocap`; the video path is `3d_recon.py` -> `postprocess_global_align_plane_approach.py` -> `retarget_config_go2.py`/`retarget_motion_fromVision.py` -> IsaacLab (`rsi_data.py`, MotionWeight in the expert files).

### Figure
`fig_exp3_manual_intervention.pdf` (3.5 in): stacked bars of hand-set per-clip numbers/switches by category for the 14 video clips behind the paper's figures (8 flat, 3 stairs, 2 box, 1 stand-up) and the 6 MoCap clips.

### Headline numbers (`data/exp3_summary.json`, `data/exp3_per_clip_table.md`)

| | MoCap path | Video path, flat walking (8 clips) | Video path, all 14 paper clips |
|---|---|---|---|
| hand-set per-clip numbers/switches, total | 12 | 169 | 349 |
| per clip: mean / median / min-max | 2.0 / 2 / 2-2 | 21.1 / 20 / 18-29 | 24.9 / 22 / 18-54 |
| per-clip code statements (branches) | 1 list entry | 4-8 | 4-13 |
| clips with their own foot-offset table (12 numbers) | 0 of 6 | 8 of 8 | 14 of 14 |
| clips with their own base offset (3) and foot-lift gain (1) | 0 | 8 of 8 | 14 of 14 |
| clips with hand-typed keypoint deletions/fixes | 0 | 3 of 8 (turn L: 3 idx, slow turn: 6, L-R turn: 11) | 6 of 14 (box 1: 5, box 2: 2, stand-up: 19) |
| clips with input overrides (ground mask, alternative track file) | 0 | 1 (walk: 2) | 4 |
| shared constants set once per pipeline | 19 | 24 | 24 |

Category totals for the flat set (169): foot offsets 96 (57 %), base offsets 24, frame ranges 17, keypoint fixes 20, scale / foot-lift gain 8 (gains 2x-7x per clip), input overrides 2, RL-side (MotionWeight 2 for walk, 0.5 for the walk copy in the box set) 2. Stand-up needs 54 numbers (feet frozen after frame 10, extra SIM_TOE_OFFSET table, base offsets), box 28-32 (incl. 7 RSI-transform numbers each), stairs 22 each.

### Paste-ready sentences
* "Once the six keypoints are available, the video path still requires per-clip hand-tuning in the pipeline scripts: on average 21 hand-set numbers or switches per flat-walking clip (range 18-29; 349 for the 14 clips behind our results), of which 57 % are per-clip foot-offset tables for the kinematic retargeting and 14 % are per-clip base offsets; three of the eight flat clips additionally needed hand-typed frame ranges in which depth values were discarded. The MoCap path needs two numbers per clip (the start and end frame of the segment) and no special cases."
* "Per clip, the video path therefore involves roughly ten times as many hand-set constants as the MoCap path; most of them compensate embodiment and reconstruction offsets in the retargeting (Sec. III-A, 'local base offsets and feet offsets for each trajectory') rather than the vision stage itself."
* Honest addition recommended (see notes.md, item 2): "The paw keypoints of the clips used here were annotated per frame in a notebook because TAP tracking of paws was not reliable; only the two torso keypoints were tracked automatically."


---------------------------------------------------------------------------------------------------

## Experiment 4 - Does the AMP mechanism explain why coverage matters? (author's follow-up)

Script `scripts/exp4_amp_evidence.py`; numbers `data/exp4_amp_evidence.json`, tables `data/exp4_tables.md`, console `data/exp4_stdout.txt`.
Fig. 5 shows *that* the expert sets cover the command box poorly and *that* the policies track poorly in some cells. This experiment
inserts the AMP mechanism between the two, using only what the implementation in the IsaacLab repo does:

* **What the discriminator sees.** `get_amp_observations()` (`manager_based_rl_env.py:255`) returns joint positions + joint velocities (24-D);
  `AMPLoader` default `amp_data = ["JOINT_POS", "JOINT_VEL"]`. The commanded base velocity never enters the discriminator - it only sees the
  gait that produces it. A command without expert support therefore forces (q, qdot) transitions that are outside D, the style reward
  `r_style = 2 * max(0, 1 - 0.25 (D-1)^2)` (`amp_discriminator.py`, `amp_reward_coef = 2.0`) drops, and the task reward
  (`60 exp(-|v_cmd - v|^2 / 0.22^2) + 20 exp(-(w_cmd - w)^2 / 0.22^2)`, `parameters.set_velocity_rewards_amp`) and the style reward pull in
  different directions; they are mixed 0.5/0.5 (`amp_task_reward_lerp = 0.5`). RSI additionally starts every episode in an expert state.
* **The paper already measures this per target cell.** `play.py:469` accumulates the Euclidean nearest-neighbour distance between the agent's
  AMP observation and the 10x-interpolated expert set at every step and divides by the episode length (`agent_expert_distances`,
  title "Imitation score" in `plot_DEFINITIONS.py`). It is the fourth heatmap column of the shipped Fig. 5 PDF and was recovered from it
  (`data/fig5_recovered/*__agent_expert_distances.csv`, colour-bar range 2..4, so values are clipped: 1-10 % of cells at each end, see JSON).

### Figures
* `fig5_combined_data_and_performance_4col.pdf`: the combined Fig. 5 with the agent-expert distance as fourth "Trained Policies" column
  (`fig5_replacement_combined.py`, `METRICS_4COL`; the 3-column file is unchanged).
* `fig_exp4_agent_expert_distance_grid.pdf` (single column, 3 rows): the same heatmaps with the Exp.-2 covered cells overlaid as white dots.
  The dark (well-imitated) region coincides with the covered cells for both video sets; for MoCap the minimum sits at the standing command
  (see caveat below).
* `fig_exp4_coverage_vs_performance.pdf` (double column, 2 x 3 panels; MoCap and Video (extended) only, author's request - the Video set
  stays in the tables): one marker per evaluation cell (147 per dataset). Every y quantity is "higher = worse". Top row: command distance ->
  agent-expert distance; agent-expert distance -> tracking error vel.; agent-expert distance -> tracking error yaw. Bottom row: command distance ->
  combined tracking error; agent-expert distance -> combined tracking error; agent-expert distance -> cost of transport. Combined tracking error =
  mean of the two Fig. 5 errors, each divided by its colour-bar maximum (0.1 m/s, 0.4 rad), i.e. 0..1 with equal weight. Spearman rho per set in
  the legend, over both sets in the title; dotted lines = colour-bar clip limits of the recovered data.
* `fig_exp4_joint_speed_vs_base_speed.pdf` (single column; MoCap and Video (extended) only): per expert clip, mean base speed vs. RMS joint speed ||qdot|| (the discriminator's
  dominant input), marker size = AMP sampling mass; faint = per frame; grey = command range; star = Go2 default standing pose (qdot = 0).

* `fig_exp4_key_result.pdf` (single column, one panel; the paper-ready condensation of the 2 x 3 figure): x = distance of the target command
  to the closest expert frame (unitless: each component divided by half the command range, 1.0 m/s, 0.3 m/s, 1.57 rad/s; 0 = the dog did
  exactly this, 1 = off by a full half-range), y = combined tracking error (vel. and yaw, each divided by its Fig.-5 colour-bar maximum, then
  averaged; higher = worse), marker colour = "Expert Imitation" = minus the agent-expert distance in AMP observation space (same numbers as
  Fig. 5's imitation column with the sign flipped so that higher = better, colour bar -4 .. -2; yellow = the policy's (q, qdot) stay close to the
  expert data, purple = far away), marker shape = dataset, straight line = least-squares fit per dataset. Legend: Spearman rank correlation rho of the plotted points
  (MoCap 0.61, Video (extended) 0.31; Pearson r 0.53 / 0.25 is stored alongside in the JSON). Supporting numbers for the caption:
  covered -> uncovered cell means MoCap 0.67 -> 0.76, Video (extended) 0.34 -> 0.40 (both p < 1e-3; covered = the cell contains at least one
  expert frame, Exp. 2 rule). Reads: (i) error rises with the coverage gap for both sets, (ii) the extended video set is lower everywhere and
  flatter, (iii) the far-from-data cells are the purple ones, i.e. the policy leaves the expert distribution exactly where the command is
  uncovered. Caveat: the Video (extended) x-range is shorter because its data cover more of the box.
* `fig_exp4_key_result_ms.pdf` (variant with a physical x unit): x = planar distance in m/s between the target (vx, vy) and the closest expert
  frame whose yaw rate is within +-0.25 rad/s of the grid's yaw rate 0 (the Exp.-2 slice). No normalisation, so vy differences count the
  same as vx differences although the vy command range is 3.3x narrower. rho = 0.44 / 0.25 (r 0.39 / 0.22). Same picture, slightly weaker fits because a
  0.3 m/s lateral gap is "far" for the task but only 0.3 on this axis. Use this one if the unitless axis is hard to explain; use the
  normalised one if the yaw-rate dimension or the different command ranges matter to the argument.
* `fig_exp4_key_result_aed.pdf`, `fig_exp4_key_result_aed_ms.pdf` (companions, same layout): y = agent-expert distance, colour = combined tracking
  error; legend rho of the plotted points (normalised / m/s x: MoCap 0.24 / 0.14, Video (extended) 0.78 / 0.78; Pearson r in the JSON). Covered -> uncovered means: MoCap 2.88 -> 2.93 (n.s.),
  Video (extended) 2.24 -> 3.16 (p < 1e-9). Reads: for the video policy the coverage gap drives the robot out of the expert distribution
  almost linearly; the MoCap policy sits at 2.5-3.5 everywhere. Dotted lines = colour-bar clip limits of the recovered Fig. 5 data; the fits
  are computed on the clipped values and drawn only inside the axes. All fits (r, slope, intercept) are in `exp4_amp_evidence.json` under
  `cells.key_result_fits`.
* `fig_exp4_amp_state_projection.pdf` (double column, 3 panels; `scripts/exp4_state_projection.py`, numbers `data/exp4_state_projection.json`):
  2-D PCA projection of the state the discriminator consumes. Every dimension is standardised by the pooled expert mean/std (as the AMP
  Normalizer does), PCA is fitted on the pooled expert frames of MoCap + Video (extended). Panels: (q, qdot) 24-D, q only, qdot only. Dots =
  expert frames, outline = convex hull (the region the discriminator has ever seen as expert), star = Go2 default standing pose with qdot = 0.
  **Policy side:** no rollouts are on this machine, so the policy overlay is empty. Drop the eval joint logs into `data/policy_rollouts/`
  (`play.py` with `record_episode_jpos`, files `x_*_y_*_yaw_*.th`, shape (T, num_envs, 12) in Isaac Lab joint order, or any (T, 12) .npy /
  .npz with q, qd) and re-run the script: they are projected with the same standardisation and PCA and drawn with their own hull, without
  re-training or re-evaluating anything.
  Numbers (2-D, first two PCs): 99 % of MoCap frames lie inside the Video (extended) hull in the (q, qdot) projection but only 55 % of
  Video (extended) frames lie inside the MoCap hull; the Video (extended) hull is 2.5x larger (60 vs 24). In q-only 94 % / 46 %, in qdot-only
  100 % / 91 % - the extra coverage of the extended video set is mostly in joint *positions* (postures: standing, start-stop, pivots), while
  the joint-velocity ranges of the two sets largely coincide. In the 2-D picture the standing pose falls inside both hulls for (q, qdot) and qdot-only
  (qdot = 0 is the centre of the velocity distribution once projected; a convex hull over-approximates), and on the edge of the MoCap hull
  in q-only - the full-dimensional nearest-neighbour distances (1.91 MoCap vs 1.15 Video (extended)) are the statement to quote. The first two PCs carry only 21 % (24-D) / 37 % (q) / 32 % (qdot) of the variance, so this is a coarse
  picture; the nearest-neighbour numbers above are the full-dimensional statement.

### Headline numbers (vx-vy grid at yaw rate 0; Spearman rho, all p < 0.05 unless marked)

| | MoCap | Video | Video (extended) | pooled (441 cells) |
|---|---|---|---|---|
| rho(command distance to expert data, agent-expert distance) | 0.24 | 0.59 | **0.78** | 0.49 |
| rho(command distance, tracking error vel.) | 0.25 | 0.62 | 0.38 | 0.43 |
| rho(command distance, tracking error yaw) | **0.71** | 0.18 | 0.16 | 0.37 |
| rho(agent-expert distance, tracking error vel.) | 0.58 | 0.14 (n.s.) | 0.50 | 0.30 |
| rho(agent-expert distance, tracking error yaw) | 0.01 (n.s.) | -0.28 | 0.21 | -0.11 |
| rho(command distance, combined tracking error) | 0.61 | 0.44 | 0.31 | 0.44 |
| rho(agent-expert distance, combined tracking error) | 0.38 | -0.13 (n.s.) | 0.40 | 0.04 (n.s.) |
| rho(command distance, cost of transport) | 0.30 | -0.01 (n.s.) | -0.02 (n.s.) | 0.15 |
| rho(agent-expert distance, cost of transport) | -0.20 | 0.35 | -0.01 (n.s.) | 0.12 |
| agent-expert distance, covered / uncovered cells | 2.88 / 2.93 (n.s.) | 2.54 / 3.28 | 2.24 / 3.16 | 2.50 / 3.12 |
| tracking error vel. [m/s], covered / uncovered | 0.06 / 0.07 (n.s.) | 0.04 / 0.06 | 0.04 / 0.05 | 0.048 / 0.060 |
| tracking error yaw [rad], covered / uncovered | 0.28 / 0.34 | 0.20 / 0.22 (n.s.) | 0.10 / 0.11 | 0.18 / 0.23 |

(covered = cell contains an expert frame, Exp. 2 rule; one-sided Mann-Whitney U, uncovered > covered. Orientation check: reading the Fig. 5
CSVs un-flipped changes rho(command distance, agent-expert distance) to 0.37 / 0.50 / 0.81 - same conclusion.)

*Scale of the agent-expert distance, measured on the expert data themselves (24-D, same 10x interpolation as `play.py`):*

| | MoCap | Video | Video (extended) |
|---|---|---|---|
| leave-one-clip-out NN distance, mean over clips (range) | 5.0 (3.7 pace .. 10.1 canter) | 6.1 (5.0 .. 9.5 walk) | 4.4 (2.4 start-stop 1 .. 9.1 walk) |
| nearest other set (mean) | -> Video (ext) 4.9 | -> MoCap 6.3 | -> MoCap 5.5 |
| Go2 default standing pose (qdot = 0) -> nearest expert frame | 1.91 | 2.40 | **1.15** |
| smallest expert \|\|qdot\|\| / AMP-weighted 5th percentile [rad/s] | 1.8 / 3.4 | 2.4 / 3.6 | 0.9 / 2.6 |
| share of the squared NN distance carried by the 12 joint velocities | 95 % | 96 % | 94 % |

So an agent-expert distance of ~2 means "as close as a gait cycle of the set is to the rest of the set", ~3 is at the level of the closest
clip-to-clip distances, and 4+ (clipped) is "as far as a different clip or a different dataset". The metric is essentially a
joint-*velocity* distance (94-96 %), and the standing robot is trivially close to the slowest expert frame (1.9 for MoCap although MoCap has
no standing frames) - see caveat.

*What the discriminator sees per clip and how AMP samples it (`data/exp4_tables.md`, third table):* the AMP loader draws a clip with
probability MotionWeight / sum and a time uniformly within it, so the per-frame density is MotionWeight / frames. In MoCap the fastest clip,
canter (2.09 m/s, outside the command range, RMS ||qdot|| 14.1 rad/s), has 5.2x the per-frame density of right turn0 and 17 % of the expert
mass; in Video (extended) walk (1.55 m/s, outside the range, 13.9 rad/s, MotionWeight 2) has 5.4x the density of L-R turn and 22 % of the mass.
Both sets put their highest sampling density on the one clip that lies outside the command range. Base speed and ||qdot|| are monotone in
MoCap (rho 0.68) but only weakly in the video sets (0.22 / 0.27) because the retargeted video joint velocities carry the reconstruction noise
of Exp. 1 (RMS joint acceleration 1.9-2.4x MoCap).

### Reading
* Which metrics respond to coverage: velocity tracking does (both sets), the combined error does (0.61 MoCap, 0.31 Video (ext)), yaw only
  for MoCap (0.71) - the yaw error of the Video (extended) policy is small and flat (0.10 mean) so there is little left to correlate. Cost of
  transport does **not** follow coverage or the agent-expert distance in any consistent direction (rho between -0.20 and 0.35, signs differ
  between sets): CoT is governed by the commanded speed itself (Fig. 5 shows the CoT ridge at vx ~ 0 for every dataset, where the
  denominator |v| is small), not by how far the policy is from the expert data. Do not claim an energy-efficiency effect of coverage.
* For the video sets the chain holds cell by cell: where the target command is far from any expert frame, the policy's (q, qdot) are far from
  the expert set (rho 0.59 / 0.78), and where they are far the velocity tracking is worse (rho 0.50 for Video (extended)). The additional
  clips move the well-imitated region onto the low-speed / standing part of the box (agent-expert distance in covered cells 2.24 vs 3.16;
  standing pose 1.15 vs 2.40 from the expert set) - this is the mechanism behind the paper's "start-stop motion improves tracking and
  imitation for a standing command".
* For MoCap the coverage link shows in the yaw error (rho 0.71: the yaw error is worst exactly where the command is far from the data,
  i.e. negative vx), not in the agent-expert distance, whose MoCap minimum sits at the standing command although MoCap has no standing
  frames (rho(agent-expert distance, yaw error) = 0.01; 52 % of MoCap yaw cells are clipped at 0.4 rad, so the rank statistics there are weak).
* Sampling: 45 % (MoCap) / 22 % (Video ext) of the discriminator's expert samples come from clips faster than the 1 m/s command limit, and
  those clips have the highest per-frame density. Down-weighting them (MotionWeight) is a zero-cost lever the paper does not use.

### Imitation vs cost of transport (author's follow-up, 2026-09-10): does imitating noisy keypoints still give efficient gaits?
Figure `fig_exp4_imitation_vs_cot.pdf`: CoT vs agent-expert distance per command cell, colour = commanded speed, cells with |v| >= 0.3 m/s only (122 per set; the |v| -> 0 ridge, where CoT diverges and is clipped at 2.0, is left out). Raw rho(agent-expert distance, CoT) is uninformative (-0.20 / 0.35 / -0.01) because CoT is dominated by the commanded speed (rho(speed, CoT) = -0.75 / -0.50 / -0.54). Conditioning on speed:

| | MoCap | Video | Video (extended) |
|---|---|---|---|
| partial Spearman rho(agent-expert distance, CoT | speed), 147 cells | 0.30 | 0.50 | 0.60 |
| Spearman rho, cells with |v| >= 0.3 | 0.07 | 0.55 | 0.36 |
| mean CoT, better-imitated half vs worse half (|v| >= 0.3) | 1.21 vs 1.23 | 1.05 vs 1.40 | 1.02 vs 1.16 |

Reading: for the video sets, the cells in which the policy stays closest to the (noisy) expert data are the cells with the LOWEST cost of transport - imitating the video keypoints more closely makes the gait more efficient, not less. Together with the expert-side roughness (retargeted video joint trajectories 6.8-8.5x rougher than MoCap: SG residual 2.2 / 1.7 deg vs 0.25 deg; joint accelerations 103 / 83 vs 44 rad/s^2, `exp1_metrics.json`) and the policy-level CoT (Fig. 4: 1.59 / 1.31 vs 1.51 MoCap), this is the evidence that the discriminator learns the gait, not the noise. Correlational (same three policies), colour-map values.

### Caveats
* The Fig. 5 values are recovered from the PDF colour map (quantised, clipped at the colour-bar limits). If the evaluation YAMLs
  (`logs/rsl_rl/unitree_go2_AMPflat/*/eval_*/*.yaml`, key `agent_expert_distances`) are still on the training machine, re-plot from them;
  the 4-column script only needs the CSVs replaced.
* The agent-expert distance is a nearest-neighbour distance dominated by joint velocities; a slow or standing robot scores well against any
  set that contains slow frames. Call it "agent-expert distance" rather than "imitation score" in the paper, or normalise per dimension.
* Cell-wise correlations are across 147 cells of the same three policies (seed-averaged); they show consistency of the mechanism, not
  causality. A clean causal test would be an ablation that adds the base velocity to the discriminator input, or re-trains with the
  fast clips down-weighted - both need the simulator.

### Paste-ready sentences
* "AMP's discriminator observes joint positions and velocities only; the commanded velocity enters it solely through the gait that
  produces it. Consequently, for target commands that lie outside the expert data, the policy has to produce state transitions the
  discriminator has never seen as expert samples, and the style reward competes with the task reward."
* "This is visible in the evaluation: the per-episode nearest-neighbour distance between the agent's and the expert's (q, qdot) grows with
  the distance of the target command to the nearest expert frame (Spearman rho = 0.59 and 0.78 for the two video sets, 0.49 pooled over
  all 441 cells), is 25 % larger in uncovered than in covered cells (3.12 vs 2.50), and correlates with the velocity tracking error
  (rho = 0.50 for Video (extended))."
* "The additional start-stop clips bring the expert data within 1.15 of the standing pose in AMP observation space (MoCap: 1.91, Video: 2.40),
  and the smallest expert joint speed drops from 1.8-2.4 rad/s to 0.9 rad/s; the standing command is the cell with the largest improvement."
* "For MoCap the yaw tracking error is largest exactly where the target command is far from the expert data (rho = 0.71 over the
  vx-vy grid), consistent with the missing backward-walking and slow-turning demonstrations."
* "The AMP loader samples clips by MotionWeight and time uniformly within a clip, so short fast clips dominate the expert samples per frame:
  45 % (MoCap) and 22 % (Video (extended)) of the expert samples are faster than the 1 m/s command limit."
