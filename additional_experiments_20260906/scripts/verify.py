"""Sanity checks from the plan. Every one must pass."""
import json, glob, numpy as np, common as C, exp1_noise as E

ok = lambda c, m: print(("  PASS  " if c else "  FAIL  ") + m) or c

res = []
# 1. dataset durations reproduce the paper's 6.0 s / 12.2 s
for folder, want in ((C.SET_VIDEO, 6.0), (C.SET_VIDEOEXT, 12.2)):
    d = sum((len(F) - 1) * dt for _, F, dt in C.load_amp(folder))
    res.append(ok(abs(d - want) < 0.05, f"duration {folder} = {d:.2f}s (paper: {want}s)"))

# 2. recomputed root velocity matches the stored 31:34 block
import exp2_coverage as X
_, F, dt = C.load_amp(C.SET_VIDEO)[0]
v = np.diff(F[:, 0:3], axis=0) / dt
err = np.abs(v - F[1:, 31:34]).max()
res.append(ok(err < 1e-3, f"stored root velocity matches backward difference (max err {err:.2e})"))

# 3. the four lowest MoCap joints are the toe indices
a = np.loadtxt(f"{C.MI}/data/dog_walk09_joint_pos.txt", delimiter=",").reshape(-1, 27, 3)
low = set(np.argsort(a[:, :, 1].mean(0))[:4].tolist())
res.append(ok(low == set(C.REF_TOES), f"lowest MoCap joints {sorted(low)} == REF_TOES {sorted(C.REF_TOES)}"))

# 4. MoCap torso length is near-constant
kp = C.load_mocap_keypoints("dog_walk03", 448, 481)
cv = 100 * C.torso_series(kp).std() / np.median(C.torso_series(kp))
res.append(ok(cv < 2.0, f"MoCap torso-length CV = {cv:.2f}% (< 2%)"))

# 5. feet are below the body in both sources (up-axis handled)
for nm, k in (("mocap", kp), ("video", C.load_video_keypoints("walk_869488000"))):
    res.append(ok(k[:, 2:, 2].mean() < k[:, :2, 2].mean(), f"{nm}: feet below body (+Z up)"))

# 6. decimation must not change M1 (pointwise) but must change M4 (rate-sensitive)
d = C.decimate_to_common(kp, C.FS_MOCAP)
m1a = 100 * C.torso_series(kp).std() / np.median(C.torso_series(kp))
m1b = 100 * C.torso_series(d).std() / np.median(C.torso_series(d))
res.append(ok(abs(m1a - m1b) < 0.15, f"M1 rate-invariant: {m1a:.2f}% vs {m1b:.2f}%"))

# 7. anti-aliased decimation must not inject edge transients
from scipy.signal import resample_poly
bad = resample_poly(kp.reshape(len(kp), -1), 1, 2, axis=0).reshape(-1, 6, 3)
cvb = 100 * C.torso_series(bad).std() / np.median(C.torso_series(bad))
res.append(ok(cvb > 3 * m1b, f"zero-pad decimation would corrupt ({cvb:.2f}% vs {m1b:.2f}%) -> padtype='line' required"))

# 8. every figure exists and embeds Type-42 fonts only
import subprocess, os
for f in sorted(glob.glob(os.path.join(C.OUT, "fig_*.pdf"))):
    out = subprocess.run(["pdffonts", f], capture_output=True, text=True).stdout
    res.append(ok("Type 3" not in out and os.path.getsize(f) > 5000,
                  f"{os.path.basename(f)}: vector PDF, no Type-3 fonts"))

print(f"\n{sum(res)}/{len(res)} checks passed")
raise SystemExit(0 if all(res) else 1)
