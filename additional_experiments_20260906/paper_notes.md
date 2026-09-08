# Notes for the camera-ready

Issues found while building the additional experiments. Each is verified against the
code/data and each has a concrete fix. None of them changes the paper's conclusions;
two of them make the paper *stronger* if acted on.

---

## 1. Camera intrinsics are mis-assigned — this is the big one

`quadruped_from_video/camera_info/camera.yaml` stores

```
intrinsics: [748.0999, 634.5353, 747.5494, 370.8257]
```

The values pattern-match `[fx, cx, fy, cy]` for the **full 1280×720** frame
(cx ≈ 634 ≈ 1280/2, cy ≈ 371 ≈ 720/2). But `3d_recon.py:210` reads them as

```python
FX, FY, CX, CY = camera_info["intrinsics"]     # -> CX = 747.55, FY = 634.54
```

and then applies them, unscaled, to keypoints that are in the **640×360** half-size
frame (`markers` comes straight from `tracks/`, never rescaled). Two compounding errors:

1. **Mis-ordered unpack** — `CX` receives *fy*'s value and `FY` receives *cx*'s value.
2. **No half-size scaling** — all four should be halved for the 640×360 images.

Correct for 640×360: `fx = 374.05, fy = 373.77, cx = 317.27, cy = 185.41`.
Currently used: `FX = 748.10, FY = 634.54, CX = 747.55, CY = 370.83`.

**Why it matters.** With `CX = 747.55` on a 640-pixel-wide image, `(x − CX)` is negative
for *every* pixel, so the inverse projection

```python
markers_3d[..., 0] = (markers_3d[..., 0] - CX) * markers_3d[..., 2] / FX
```

makes `x_cam ≈ −z`: the horizontal coordinate collapses onto the depth axis, and depth
noise leaks directly into x. The x and y scales are also wrong by different factors
(0.50 vs 0.59), giving an ~18% anisotropic distortion on top of the offset.

**Measured effect** (like-for-like camera-frame test, `fig_H` panel (c), 6 clips):

| | median torso-length CV | median torso length |
|---|---|---|
| as-is | **28.7%** | 0.220 m |
| intrinsics fixed | **10.2%** | 0.337 m |

So roughly **two-thirds of the measured keypoint noise is a fixable calibration bug**,
and the corrected torso length moves from 0.220 m toward the real dog's 0.408 m (MoCap).
This is good news for the paper: the reconstruction noise is not fundamental to the
approach. *Caveat:* this test is in camera frame without SLAM/ground alignment, so the
absolute CVs are not the shipped pipeline's — the relative change is the result. Re-running
the full pipeline with corrected intrinsics would be needed before quoting new dataset numbers.

---

## 2. Section III describes post-processing that does not exist — and could not work

The paper states two filtering steps: discard depth differing by more than 0.5 m from
the previous value, and a 3-frame moving average. Neither exists in any script or in the
git history of `quadruped_from_video`, `motion_imitation`, `co-tracker` or
`shape-of-motion`. What exists instead is linear interpolation of zero-valued keypoints
plus a block of **hand-typed per-clip frame ranges** (`3d_recon.py:279-298`), e.g.

```python
markers_3d[29:33, :, :] = 0.0   # There is some noise here in the depth so we remove it and let interpolate
```

I implemented the described filter and measured how much of the hand-typed deletion it
recovers: **2.0% (2 of 99 samples)**. It is not a threshold-tuning problem —

- typical frame-to-frame depth jumps are **0.005–0.008 m (median), ≤0.11 m (p99)**, so a
  0.5 m threshold sits ~50× above p99 and essentially never fires;
- lowering it does not help: at 0.05 m recall is still only 27% while flagging 513
  non-hand samples;
- the hand-deleted samples have depth statistics **indistinguishable from the rest**
  (median 1.60–2.05 m vs 1.46–2.06 m).

**Conclusion:** the hand-deleted samples are not depth *spikes*. They are samples where
the tracked pixel landed on a different surface (ground, the other leg, the body) and
read a smooth, entirely plausible depth belonging to the wrong object. A temporal jump
filter cannot detect that by construction. `fig_H` panel (a) shows this directly — the
hand-deleted ranges sit in visibly smooth stretches of the depth trace.

**Options for the camera-ready**, in order of preference:

1. Fix the intrinsics (item 1) and re-check how many of these deletions are still
   needed — several may be artifacts of the projection bug rather than of the depth.
2. Replace the description with a **geometric** plausibility criterion, which *can*
   catch this failure: reject a foot whose distance to the body exceeds a plausible limb
   reach, or whose separation from another foot collapses below ~0.10 L. The same
   criterion already cleanly separates AnimalAvatar from the depth-based sources
   (0% vs up to 29.6% of frames), so it is known to work on this data.
3. Failing both, describe the actual procedure honestly as manual outlier removal.

---

## 3. MoCap AMP timebase is mislabelled (1.26× time dilation)

`retarget_motion.py:42` stamps `FRAME_DURATION = 0.021` into the MoCap AMP files, with
the comment *"original data seems to have 0.01667 though"*. The raw capture rate is
~60 fps, confirmed from `data/dog_clips_info.txt` (`dog_walk00` 1.53 s / 90 frames =
58.8 fps; `dog_run04` 8.15 s / 487 frames = 59.8 fps).

Consequences: MoCap motions are replayed **1.26× slower** than recorded, and all MoCap
joint/root velocities in the AMP files are **under-estimated by ~21%**. Since
`motion_loader.py` interpolates in continuous time using `FrameDuration`, the dilation is
real inside the RL system, not just a label.

**Practical impact on these experiments:** the naive "video joint velocities are ~3×
MoCap's" comparison is partly an artifact, so I did not use it. Experiment 1 uses true
rates (60/30 Hz) throughout; Experiment 2 deliberately uses the stamped rate, because
that is the distribution the discriminator actually sees.

---

## 4. Twelve of 21 sequences have no SLAM camera poses

`camera_recon/<clip>/traj_est.npy` is missing for `box_1`, `box_2`, `left_right_turn`,
`obstacle_2`, `obstacle_3`, `stairs_1/2/3`, `stand_up`, `start_stop_785558000`,
`start_stop_1271493000`. `3d_recon.py` falls back to assuming a **static camera** with
only a printed warning. That covers *all* of the stairs, box, stand-up and start-stop
clips — i.e. every non-flat scenario in the paper. Section III presents SLAM camera-pose
estimation as part of the method, so this deserves a sentence.

---

## 5. Smaller bugs

- **`int(feet_z_amplification)`** truncates: all three stairs clips are configured with
  `1.2` but silently receive **k = 1** (no foot-z amplification). `box_2` (1.5) and
  `obstacle_3` (1.2, commented out) are the only ones where the value survives.
- **Dead config branch**: `obstacle_1_301506103500` (13 digits) matches no directory and
  can never execute; there is also a duplicated, empty `obstacle_3_3015061000` branch.
- **`stairs_1` `SIM_TOE_OFFSET_LOCAL[3] = [0.5, 0.07, 0.01]`** — a 50 cm x-offset on the
  hind-left toe where every sibling entry is ±0.05. Near-certainly a typo for 0.05.
- `dog_motion_info` lists `right turn0` as `dog_walk09[1085:1124]`, but
  `retarget_motion.py:62` actually uses `[1000:1150]`. The code is authoritative (the
  150-frame AMP file matches it); the note file is stale.

---

## 6. Claim that needs rewording

Section IV says collecting extra data by video is *"comparatively easy"*, while the
pipeline needs **317 hand-set values for ~32 s of video vs 15 for the entire MoCap
dataset**. Suggested split, which is both honest and supportive: **recording** is cheap
(handheld camera, outdoors, seconds of footage, no studio, no markers, no animal
training); **processing** currently requires per-clip tuning, and items 1–2 above are a
concrete path to reducing it.
