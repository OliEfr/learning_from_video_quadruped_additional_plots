"""Experiment 1 (support): re-lift the 2D keypoint tracks to 3D with numpy.

Two variants are produced for every flat-walking clip:
  * "as_code"  : exact re-implementation of quadruped_from_video/3d_recon.py, i.e. the 1280x720 intrinsics
                 of camera_info/camera.yaml applied to the 640x360 pixel tracks (this is what the paper used).
                 Used only to VALIDATE the re-implementation against the on-disk keypoints.
  * "half_res" : identical, but with the intrinsics scaled to the 640x360 image the tracks live in
                 (fx/2, fy/2, cx/2, cy/2). This is the physically correct back-projection.

Nothing heavy is run: only per-pixel depth look-ups, one RANSAC plane fit per clip and small matrix ops.
Outputs: data/relift_<clip>.npz and data/relift_validation.json
"""
import json
import os

import numpy as np
import yaml
from scipy import interpolate

from common import DATA, FLAT_VIDEO_CLIPS, VISION, load_video_keypoints_as_used

CAM = yaml.safe_load(open(os.path.join(VISION, "camera_info/camera.yaml")))
FX, FY, CX, CY = CAM["intrinsics"]  # for 1280x720
H, W = 360, 640  # depth / track resolution

# per-clip hacks copied from 3d_recon.py (lines 17-27, 57-63, 110-113, 280-296); frame indices relative to clip start
GROUND_MASK = {
    "walk_869488000": "in/walk_869488000/000111_color_873179000_1280x720.png",
    "stairs_1_5199540000": "in/stairs_1_5199540000/000133_color_5203962000_1280x720.png",
    "stairs_2_5299211000": "in/stairs_2_5299211000/000053_color_5300972000_1280x720.png",
    "stairs_3_5339749000": "in/stairs_3_5339749000/000085_color_5342575000_1280x720.png",
    "obstacle_3_3015061000": "in/obstacle_3_3015061000/000173_color_3020815000_1280x720.png",
}


def apply_marker_hacks(clip, m):
    """m: (T, 6, 3) pixel x, pixel y, depth. Replicates the 'remove custom points' block of 3d_recon.py."""
    if clip == "turn_left_1771233000":
        m[0:4, 0:2, 2] = 0.0
    if clip == "slow_turn_1771233000":
        m[29:33, :, :] = 0.0
        m[40:42, -1, :] = 0.0
    if clip == "left_right_turn_2058226999":
        m[22:28, 0, :] = 0.0
        m[15:21, -1, :] = 0.0
        m[33:39, -1, :] = 0.0
    if clip == "obstacle_2_3126098000":
        m[21:23, 5, :] = 0.0
    if clip == "stand_up_2431270000":
        m[10:-10, -2:, 2] = 0.0
    if clip == "box_1_399682000":
        m[20:, -2:, 2] = 0.0
    return m


def load_tracks(clip):
    d = os.path.join(VISION, "tracks", clip)
    foot_file = "foot_pos_for_depth_cam.npy" if clip == "walk_869488000" else "foot_pos.npy"
    feet = np.load(os.path.join(d, foot_file))  # (T, 4, 2)  order RR, RL, FR, FL
    hips = np.load(os.path.join(d, "hip_pos.npy"))[0]  # (T, 2, 2)  order rear hip, front hip
    T = min(len(feet), len(hips))
    return np.concatenate([hips[:T], feet[:T]], axis=1).astype(int)  # (T, 6, 2)


def load_depths(clip, T):
    d = os.path.join(VISION, "in", clip, "depth_cam")
    files = sorted(f for f in os.listdir(d) if f.endswith(".raw"))[:T]
    return np.stack([np.fromfile(os.path.join(d, f), dtype=np.uint16).reshape(H, W).astype(np.float32) / 1000.0 for f in files])


def load_camera(clip, T):
    p = os.path.join(VISION, "camera_recon", clip, "traj_est.npy")
    if not os.path.exists(p):  # 3d_recon.py falls back to a static camera
        return np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (T, 1)), False
    return np.load(p)[:T], True


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cam_to_world(p_cam, pose):
    R = quat_xyzw_to_R(pose[3:7])
    return p_cam @ R.T + pose[:3]


def intrinsics(mode):
    if mode == "as_code":
        return FX, FY, CX, CY
    if mode == "half_res":
        return FX / 2, FY / 2, CX / 2, CY / 2
    raise ValueError(mode)


def backproject(u, v, z, K):
    fx, fy, cx, cy = K
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def ransac_plane(P, n_iter=400, thr=0.01, seed=0):
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(n_iter):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        inl = np.abs(P @ n + d) < thr
        k = inl.sum()
        if k > best_n:
            best_n, best = k, inl
    Q = P[best]
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[-1]
    return np.r_[n, -n @ c], best_n / len(P)


def relift(clip, mode, preproc=None):
    """preproc: optional callable m -> m applied to the (T, 6, 3) [u, v, depth] array after the per-clip hacks and
    before the zero-interpolation (this is where 3d_recon.py would host the depth-outlier rule of Sec. III-A).
    Returns dict with markers_world (6, T, 3), plane (4,), ground-plane frame markers (6, T, 3), bookkeeping."""
    tracks = load_tracks(clip)  # (T, 6, 2)
    T = len(tracks)
    D = load_depths(clip, T)
    poses, cam_ok = load_camera(clip, T)
    K = intrinsics(mode)

    m = np.zeros((T, 6, 3))
    missing_track = np.zeros((T, 6), bool)
    missing_depth = np.zeros((T, 6), bool)
    for f in range(T):
        for k in range(6):
            x, y = tracks[f, k]
            if x == 0 and y == 0:
                missing_track[f, k] = True
                continue
            z = D[f, y, x]
            m[f, k] = (x, y, z)
            if z == 0.0:
                missing_depth[f, k] = True
    m = apply_marker_hacks(clip, m)
    if preproc is not None:
        m = preproc(m)
    hacked = (m == 0).all(-1) & ~missing_track  # entries zeroed by the hand-written hacks
    # column-wise interpolation of zero entries (as in 3d_recon.py; extrapolation at the clip borders)
    interpolated = np.zeros((T, 6), bool)
    for k in range(6):
        for i in range(3):
            col = m[:, k, i]
            nz = col != 0.0
            interpolated[:, k] |= ~nz
            if nz.sum() < 2:
                raise RuntimeError(f"{clip}: marker {k} has <2 valid samples")
            fn = interpolate.interp1d(np.where(nz)[0], col[nz], bounds_error=False, fill_value="extrapolate")
            col[~nz] = fn(np.where(~nz)[0])
            m[:, k, i] = col
    p_cam = backproject(m[..., 0], m[..., 1], m[..., 2], K)  # (T, 6, 3)
    p_world = np.stack([cam_to_world(p_cam[f], poses[f]) for f in range(T)])  # (T, 6, 3)

    # ground plane from the first depth frame (optionally masked), in world coordinates
    depth0 = D[0].copy()
    if clip in GROUND_MASK:
        from PIL import Image

        mask = np.array(Image.open(os.path.join(VISION, GROUND_MASK[clip])).convert("L")) > 0
        depth0 = np.where(mask, depth0, 0.0)
    vv, uu = np.mgrid[0:H:2, 0:W:2]
    z0 = depth0[::2, ::2]
    ok = (z0 > 0) & (z0 < 10.0)
    P0 = cam_to_world(backproject(uu[ok].astype(float), vv[ok].astype(float), z0[ok], K), poses[0])
    plane, inlier_frac = ransac_plane(P0)
    # orient the normal so that the body is above the feet (positive = up)
    n = plane[:3]
    sd = lambda p: (p @ n + plane[3]) / np.linalg.norm(n)  # noqa: E731
    if sd(p_world[:, :2].reshape(-1, 3)).mean() < sd(p_world[:, 2:].reshape(-1, 3)).mean():
        plane = -plane
        n = plane[:3]
    # express markers in a ground-aligned frame: z = signed distance to the plane
    z_axis = n / np.linalg.norm(n)
    tmp = np.array([1.0, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1.0, 0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    Rg = np.stack([x_axis, y_axis, z_axis])  # rows
    foot_point = -plane[3] * z_axis / np.linalg.norm(n)  # a point on the plane
    p_ground = (p_world - foot_point) @ Rg.T  # (T, 6, 3): z is height above the fitted ground
    return dict(
        clip=clip,
        mode=mode,
        markers_world=p_world.transpose(1, 0, 2),
        markers_ground=p_ground.transpose(1, 0, 2),
        plane=plane,
        plane_inlier_frac=inlier_frac,
        camera_poses_used=cam_ok,
        missing_track=missing_track,
        missing_depth=missing_depth,
        hacked=hacked,
        interpolated=interpolated,
        n_frames=T,
    )


def pairwise_dists(markers):
    """(6, T, 3) -> (T, 15) inter-marker distances (invariant to the rigid post-processing transform)."""
    iu = np.triu_indices(6, 1)
    d = np.linalg.norm(markers[:, None] - markers[None, :], axis=-1)  # (6, 6, T)
    return d[iu].T


def validate_against_disk(clip, res):
    """Compare the 'as_code' re-lift with the keypoints that fed the retargeting (rigid-invariant distances)."""
    base, feet = load_video_keypoints_as_used(clip)
    disk = np.concatenate([base, feet], axis=0)  # (6, Td, 3)
    d_disk = pairwise_dists(disk)
    d_mine = pairwise_dists(res["markers_world"])
    best = None
    for off in range(0, max(1, len(d_mine) - len(d_disk) + 1)):
        seg = d_mine[off : off + len(d_disk)]
        if len(seg) != len(d_disk):
            continue
        err = np.abs(seg - d_disk)
        med = float(np.median(err))
        if best is None or med < best["median_abs_diff_m"]:
            best = dict(frame_offset=off, median_abs_diff_m=med, p95_abs_diff_m=float(np.percentile(err, 95)), max_abs_diff_m=float(err.max()))
    best["n_frames_disk"] = int(disk.shape[1])
    best["n_frames_relift"] = int(res["n_frames"])
    return best


def main():
    validation = {}
    for clip in FLAT_VIDEO_CLIPS:
        for mode in ["as_code", "half_res"]:
            res = relift(clip, mode)
            np.savez(
                os.path.join(DATA, f"relift_{mode}_{clip}.npz"),
                markers_world=res["markers_world"],
                markers_ground=res["markers_ground"],
                plane=res["plane"],
                missing_track=res["missing_track"],
                missing_depth=res["missing_depth"],
                hacked=res["hacked"],
                interpolated=res["interpolated"],
                camera_poses_used=res["camera_poses_used"],
                plane_inlier_frac=res["plane_inlier_frac"],
            )
            if mode == "as_code":
                v = validate_against_disk(clip, res)
                v["camera_poses_used"] = bool(res["camera_poses_used"])
                v["plane_inlier_frac"] = float(res["plane_inlier_frac"])
                v["frac_feet_missing_depth"] = float(res["missing_depth"][:, 2:].mean())
                v["frac_feet_missing_track"] = float(res["missing_track"][:, 2:].mean())
                v["frac_feet_interpolated_any"] = float(res["interpolated"][:, 2:].mean())
                v["frac_base_interpolated_any"] = float(res["interpolated"][:, :2].mean())
                validation[clip] = v
                print(f"{clip:28s} offset={v['frame_offset']} median|dd|={v['median_abs_diff_m']*1000:.2f} mm  p95={v['p95_abs_diff_m']*1000:.2f} mm  max={v['max_abs_diff_m']*1000:.1f} mm  cam={v['camera_poses_used']}  feet depth missing={v['frac_feet_missing_depth']:.2f}")
            else:
                hh_code = np.median(np.load(os.path.join(DATA, f"relift_as_code_{clip}.npz"))["markers_ground"][:2, :, 2].mean(0))
                hh_half = np.median(res["markers_ground"][:2, :, 2].mean(0))
                validation[clip]["hip_height_as_code_m"] = float(hh_code)
                validation[clip]["hip_height_half_res_m"] = float(hh_half)
                print(f"    hip height above fitted ground: as_code {hh_code:.3f} m  |  half_res {hh_half:.3f} m")
    with open(os.path.join(DATA, "relift_validation.json"), "w") as fh:
        json.dump(validation, fh, indent=2)


if __name__ == "__main__":
    main()
