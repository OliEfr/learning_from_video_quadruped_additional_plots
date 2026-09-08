# Sanity checks

## Datasets

- [PASS] MoCap: 6 clips, 6.80 s: found 6 clips, 6.804 s (paper: n/a s)
- [PASS] MoCap: angular-velocity columns 34:37 are all zero: so the yaw rate has to be derived from the root quaternion
- [PASS] MoCap: foot-position columns 19:31 and 49:61 are all zero: 
- [PASS] MoCap: stored linear velocity == diff(root_pos)/FrameDuration: max abs err 4.47e-04 m/s (rounding to 5 decimals in the files)
- [INFO] MoCap: FrameDuration: [0.021]  (MoCap raw capture is 60 Hz -> 0.021 s plays the dog 0.79x slower; video 30 fps -> 0.03334 s is real time)
- [INFO] MoCap: MotionWeight != 1: none
- [PASS] Video: 4 clips, 6.03 s: found 4 clips, 6.035 s (paper: 6.0 s)
- [PASS] Video: angular-velocity columns 34:37 are all zero: so the yaw rate has to be derived from the root quaternion
- [PASS] Video: foot-position columns 19:31 and 49:61 are all zero: 
- [PASS] Video: stored linear velocity == diff(root_pos)/FrameDuration: max abs err 2.87e-04 m/s (rounding to 5 decimals in the files)
- [PASS] Video: quaternion x-component (pybullet xyzw) is zero -> no roll reconstructed: consistent with Sec. III-A of the paper
- [INFO] Video: FrameDuration: [0.03334]  (MoCap raw capture is 60 Hz -> 0.021 s plays the dog 0.79x slower; video 30 fps -> 0.03334 s is real time)
- [INFO] Video: MotionWeight != 1: none
- [PASS] Video (extended): 8 clips, 12.20 s: found 8 clips, 12.202 s (paper: 12.2 s)
- [PASS] Video (extended): angular-velocity columns 34:37 are all zero: so the yaw rate has to be derived from the root quaternion
- [PASS] Video (extended): foot-position columns 19:31 and 49:61 are all zero: 
- [PASS] Video (extended): stored linear velocity == diff(root_pos)/FrameDuration: max abs err 2.92e-04 m/s (rounding to 5 decimals in the files)
- [PASS] Video (extended): quaternion x-component (pybullet xyzw) is zero -> no roll reconstructed: consistent with Sec. III-A of the paper
- [INFO] Video (extended): FrameDuration: [0.03334]  (MoCap raw capture is 60 Hz -> 0.021 s plays the dog 0.79x slower; video 30 fps -> 0.03334 s is real time)
- [INFO] Video (extended): MotionWeight != 1: {'walk_869488000': 2.0}

## Experiment 1

- [PASS] re-lift ('as_code' intrinsics) reproduces the on-disk keypoints: max over clips of median |inter-marker distance diff| = 0.000 mm; frame offset [1] (3d_recon.py skips frame 0)
- [INFO] clips with boundary differences (different trimming -> different extrapolation of missing values): {'slow_1313807000': 19.3, 'turn_right_1771233000': 207.9, 'start_stop_785558000': 55.7}
- [PASS] half-res intrinsics roughly double the reconstructed hip height (ratio 1.8-3.0; >2 because the ground-plane fit changes as well): hip height as_code 0.21 m (range 0.16-0.24) vs half_res 0.47 m (0.38-0.52); ratio 2.09-2.88
- [INFO] single-frame dog height from pixel extent x depth: hip->paw 92 px at 1.51 m -> 0.19 m with the 1280x720 focal length, 0.37 m with the 640x360 focal length (a medium-sized dog, cf. the RGB frame)
- [INFO] fraction of foot samples with zero (invalid) depth at the tracked pixel, per clip: walk:47%, slow:25%, turn:3%, turn:3%, slow:7%, left:20%, start:16%, start:20%; mean 18%
- [PASS] camera.yaml intrinsics are for 1280x720 while tracks/depth are 640x360: cx=634.5, cy=370.8 (image centre of 1280x720); tracks max x=604<640, max y=357<360; 3d_recon.py applies FX,CX unchanged to the half-res pixels
- [INFO] M1 feet ratio video/mocap with SG window 5: 3.93
- [INFO] M1 feet ratio video/mocap with SG window 7: 2.91
- [INFO] M1 feet ratio video/mocap with SG window 9: 2.13
- [INFO] M1 feet ratio if MoCap were left at 60 Hz (unfair) vs matched 30 Hz: 8.76 vs 2.91
- [PASS] MoCap 60->30 Hz: plain subsampling vs anti-aliased decimation give similar M1: dog_walk09 full file: 0.893 % vs 0.917 % hip height (anti-aliasing removes some genuine >15 Hz content)
- [INFO] M1 feet ratio video/mocap with a 4th-order zero-phase Butterworth (5 Hz) instead of SG (clips >= 20 frames): 3.60
- [INFO] hip heights used for normalisation: MoCap 0.438 m, Video (pipeline output) 0.257 m

## Experiment 2

- [PASS] MoCap: AMP sampling weights sum to 1: 1.000000
- [PASS] MoCap: KD-tree coverage count == loop count: xy 14 vs 14, xyaw 56 vs 56
- [INFO] MoCap/canter: net yaw change: +4 deg over 0.59 s = mean rate +0.11 rad/s; smoothed-rate p5..p95 [-0.11, 0.44] rad/s; std(raw - smoothed rate) 0.04 rad/s
- [INFO] MoCap/left turn0: net yaw change: +100 deg over 0.95 s = mean rate +1.84 rad/s; smoothed-rate p5..p95 [0.73, 2.76] rad/s; std(raw - smoothed rate) 0.02 rad/s
- [INFO] MoCap/pace: net yaw change: -3 deg over 0.80 s = mean rate -0.06 rad/s; smoothed-rate p5..p95 [-0.67, 0.50] rad/s; std(raw - smoothed rate) 0.02 rad/s
- [INFO] MoCap/right turn0: net yaw change: -128 deg over 3.13 s = mean rate -0.71 rad/s; smoothed-rate p5..p95 [-1.38, -0.03] rad/s; std(raw - smoothed rate) 0.02 rad/s
- [INFO] MoCap/trot2: net yaw change: -2 deg over 0.67 s = mean rate -0.06 rad/s; smoothed-rate p5..p95 [-0.35, 0.11] rad/s; std(raw - smoothed rate) 0.03 rad/s
- [INFO] MoCap/trot: net yaw change: -3 deg over 0.67 s = mean rate -0.08 rad/s; smoothed-rate p5..p95 [-0.48, 0.29] rad/s; std(raw - smoothed rate) 0.03 rad/s
- [PASS] Video: AMP sampling weights sum to 1: 1.000000
- [PASS] Video: KD-tree coverage count == loop count: xy 22 vs 22, xyaw 48 vs 48
- [INFO] Video/slow_1313807000: net yaw change: +5 deg over 2.27 s = mean rate +0.04 rad/s; smoothed-rate p5..p95 [-0.16, 0.27] rad/s; std(raw - smoothed rate) 0.19 rad/s
- [INFO] Video/turn_left_1771233000: net yaw change: +143 deg over 1.47 s = mean rate +1.70 rad/s; smoothed-rate p5..p95 [0.05, 4.44] rad/s; std(raw - smoothed rate) 0.24 rad/s
- [INFO] Video/turn_right_1771233000: net yaw change: -142 deg over 1.43 s = mean rate -1.73 rad/s; smoothed-rate p5..p95 [-3.32, -0.48] rad/s; std(raw - smoothed rate) 0.77 rad/s
- [INFO] Video/walk_869488000: net yaw change: -3 deg over 0.87 s = mean rate -0.05 rad/s; smoothed-rate p5..p95 [-0.93, 0.29] rad/s; std(raw - smoothed rate) 0.33 rad/s
- [PASS] Video (extended): AMP sampling weights sum to 1: 1.000000
- [PASS] Video (extended): KD-tree coverage count == loop count: xy 24 vs 24, xyaw 83 vs 83
- [INFO] Video (extended)/left_right_turn_2058226999: net yaw change: -18 deg over 2.40 s = mean rate -0.13 rad/s; smoothed-rate p5..p95 [-3.92, 5.63] rad/s; std(raw - smoothed rate) 0.57 rad/s
- [INFO] Video (extended)/slow_1313807000: net yaw change: +5 deg over 2.27 s = mean rate +0.04 rad/s; smoothed-rate p5..p95 [-0.16, 0.27] rad/s; std(raw - smoothed rate) 0.19 rad/s
- [INFO] Video (extended)/slow_turn_1771233000: net yaw change: -142 deg over 1.43 s = mean rate -1.73 rad/s; smoothed-rate p5..p95 [-3.83, -0.49] rad/s; std(raw - smoothed rate) 0.81 rad/s
- [INFO] Video (extended)/start_stop_1271493000: net yaw change: -21 deg over 0.87 s = mean rate -0.42 rad/s; smoothed-rate p5..p95 [-0.95, 0.27] rad/s; std(raw - smoothed rate) 0.22 rad/s
- [INFO] Video (extended)/start_stop_785558000: net yaw change: +29 deg over 1.47 s = mean rate +0.34 rad/s; smoothed-rate p5..p95 [-0.88, 1.30] rad/s; std(raw - smoothed rate) 0.82 rad/s
- [INFO] Video (extended)/turn_left_1771233000: net yaw change: +143 deg over 1.47 s = mean rate +1.70 rad/s; smoothed-rate p5..p95 [0.05, 4.44] rad/s; std(raw - smoothed rate) 0.24 rad/s
- [INFO] Video (extended)/turn_right_1771233000: net yaw change: -142 deg over 1.43 s = mean rate -1.73 rad/s; smoothed-rate p5..p95 [-3.32, -0.48] rad/s; std(raw - smoothed rate) 0.77 rad/s
- [INFO] Video (extended)/walk_869488000: net yaw change: -3 deg over 0.87 s = mean rate -0.05 rad/s; smoothed-rate p5..p95 [-0.93, 0.29] rad/s; std(raw - smoothed rate) 0.33 rad/s
- [INFO] command ranges / heading: vx (-1.0, 1.0), vy (-0.3, 0.3), wz (-1.57, 1.57), heading_command=False (velocity_env_cfg.py lines 95-117)

## Experiment 3

- [PASS] every Exp-3 finding points at a source line in a branch mentioning its clip: 127 findings, 0 unresolved
- [PASS] regex count of `in_sequence_name == "..."` comparisons == AST count: 35 vs 35 (3d_recon.py + retarget_config_go2.py + retarget_motion_fromVision.py)
- [INFO] per-clip totals (flat-walking video clips): {'walk_869488000': 22, 'turn_left_1771233000': 21, 'slow_turn_1771233000': 24, 'left_right_turn_2058226999': 29, 'turn_right_1771233000': 18, 'start_stop_1271493000': 18, 'start_stop_785558000': 18, 'slow_1313807000': 19}
- [INFO] MoCap per clip: {'pace': 2, 'trot': 2, 'trot2': 2, 'canter': 2, 'right turn0': 2, 'left turn0': 2}
- [INFO] dead branch (clip id typo, never matches): [{'file': 'motion_imitation/retarget_motion/retarget_config_go2.py', 'line': 219, 'clip': 'obstacle_1_301506103500'}]
- [PASS] motion_loader.py currently asserts FrameDuration == 0.06 for 'vision' files: the paper's flat-walking video datasets have FrameDuration 0.03334 -> loading them with the current loader fails (reproducibility note)
