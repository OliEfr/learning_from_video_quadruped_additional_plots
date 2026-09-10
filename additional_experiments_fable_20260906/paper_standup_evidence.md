# Stand-up clip (stand_up_2431270000): evidence for the keypoint-reconstruction difficulty

Computed with `scripts/exp1_relift_keypoints.relift(clip, "as_code")` (tracks + depth at the tracked pixel + the per-clip hacks of `3d_recon.py`), 2026-09-09. Paw order RR, RL, FR, FL. "Interpolated" = no valid depth at the tracked pixel OR zeroed by hand in `3d_recon.py`, then linearly interpolated.

| clip | frames | paw samples w/o depth at pixel | zeroed by hand | interpolated total | per paw RR RL FR FL | torso missing |
|---|---|---|---|---|---|---|
| stand-up | 87 | 31.9 % | 11.5 % | **43.4 %** | 0 / 6 / **83** / **85** % | 0 % |
| box 1 | 27 | 42.6 % | 7.4 % | 50.0 % | 44 / 44 / 52 / 59 | 0 % |
| box 2 | 44 | 28.4 % | 0 | 28.4 % | 34 / 27 / 34 / 18 | 0 % |
| stairs 1-3 | 51-70 | 29.5-39.7 % | 0 | 29.5-39.7 % | 23-45 | 3-12 % |
| flat, 8 clips (mean) | 28-74 | 27 % | 0 | 27 % | 10-49 % per clip | ~0 % |

Hand-coded fixes for this clip only (`data/exp3_findings.csv`):
* `3d_recon.py:294`  `markers_3d[10:-10, -2:, 2] = 0.0` - depth of BOTH FRONT paws discarded for frames 10..77 (67 of 87 frames), i.e. the front-paw depth was judged unusable for the raised phase and replaced by interpolation between frame 9 and frame 78.
* `retarget_motion_fromVision.py:109-112` - front paw x and y frozen at their frame-10 value for the rest of the clip; only the height keeps moving. With `end_frame = 28`, 18 of the 28 retargeted frames (64 %) use frozen front-paw positions.
* 54 hand-set numbers in total vs 18-29 for the flat clips (Exp. 3), the most of any clip.

Reading: the difficulty is NOT the far side of the dog - the rear paws (one of them on the far side) are almost fully reconstructed (0 / 6 % interpolated). It is the RAISED FRONT PAWS: small, fast, held against the person and the wall, where the time-of-flight depth is invalid or wrong in 83-85 % of the frames (RGB frame 000202 shows the pose). Torso depth is complete.

Suggested sentence (replaces "the limbs facing away from the camera are occluded most of the time due to the chosen camera angle"):
> Stand-up is the most challenging scenario for our method. The motion is dynamic, and the raised front paws are small, fast, and held against the person and the background, so that the depth measurement at the tracked pixel is invalid or unreliable in 83-85 % of the frames (vs. 27 % of the paw samples for flat walking); their positions have to be interpolated over most of the clip. Together with the dynamic movement, which transfers comparatively poorly between embodiments, this makes the scenario difficult to learn.
