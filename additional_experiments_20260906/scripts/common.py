"""Shared loaders, normalization and style for the additional experiments.

READ-ONLY with respect to every source repository. Nothing outside the output
folder is written or modified.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
MI   = "/home/admin_07/project_repos/motion_imitation/retarget_motion"
DS   = "/home/admin_07/project_repos/isaac_lab/IsaacLab/datasets"
QFV  = "/home/admin_07/project_repos/quadruped_from_video"
OUT  = os.path.expanduser("~/Downloads/additional_experiments_20260906")

# ---------------------------------------------------------------- style
# Matches isaac_lab/IsaacLab/plot_*.py: Type-42 fonts, DejaVu Sans, vector PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["font.family"]  = "DejaVu Sans"
matplotlib.rcParams["axes.titleweight"] = "bold"

_TAB10 = plt.cm.get_cmap("tab10", 10)
# Fixed indices copied from plot_DEFINITIONS.get_color_for_experiment_name()
COLOR = {
    "Manual Trajectory": _TAB10(0),
    "Video w. Depth Camera": _TAB10(1),
    "Complex Reward": _TAB10(2),
    "Animal Avatar": _TAB10(4),
    "MoCap": _TAB10(5),
    "Video w. Depth Camera (extended)": _TAB10(6),
    "Video w. DepthAnythingV2": _TAB10(7),
    "Simple Reward": _TAB10(8),
}
# short aliases used across the scripts
C_MOCAP = COLOR["MoCap"]
C_VIDEO = COLOR["Video w. Depth Camera"]
C_VIDEO_EXT = COLOR["Video w. Depth Camera (extended)"]
C_DEPTHANY = COLOR["Video w. DepthAnythingV2"]
C_AVATAR = COLOR["Animal Avatar"]

IEEE_1COL, IEEE_2COL = 3.5, 7.16

# ---------------------------------------------------------------- constants
# retarget_motion.py:47-50  (Zhang et al. dog skeleton -- NOT the SMAL indices
# in retarget_motion_AnimalAvatar.py)
REF_PELVIS, REF_NECK = 0, 3
REF_TOES = [10, 19, 15, 23]           # FR, RR, FL, RL
# quadruped_from_video feet.npy order is [HR, HL, VR, VL] = [RR, RL, FR, FL];
# permutation into the MoCap order [FR, RR, FL, RL]:
VIDEO_TO_MOCAP_FEET = [2, 0, 3, 1]

FS_VIDEO, FS_MOCAP = 30.0, 60.0       # true capture rates (see F3)
FS_COMMON = 30.0

# retarget_motion.py:53-64 -- the six windows the MoCap AMP dataset is built from
MOCAP_WINDOWS = [
    ("pace",        "dog_walk00", 162, 201),
    ("trot",        "dog_walk03", 448, 481),
    ("trot2",       "dog_run04",  630, 663),
    ("canter",      "dog_run00",  430, 459),
    ("right turn0", "dog_walk09", 1000, 1150),
    ("left turn0",  "dog_walk09", 2404, 2450),
]
MOCAP_SOURCE_CLIPS = ["dog_walk00", "dog_walk03", "dog_run04", "dog_run00", "dog_walk09"]

# F7: dataset identity, verified by reproducing the paper's 6.0 s / 12.2 s
SET_MOCAP    = "mocap_AMP_for_hardware"
SET_VIDEO    = "fromVision_motions_DepthCam"
SET_VIDEOEXT = "fromVision_motions_DepthCam_extendedWithoutReverse"

VIDEO_CLIPS_4 = ["slow_1313807000", "turn_left_1771233000",
                 "turn_right_1771233000", "walk_869488000"]
VIDEO_CLIPS_8 = VIDEO_CLIPS_4 + ["left_right_turn_2058226999", "slow_turn_1771233000",
                                 "start_stop_1271493000", "start_stop_785558000"]
STAIRS_CLIPS = ["stairs_1_5199540000", "stairs_2_5299211000", "stairs_3_5339749000"]
BOX_CLIPS    = ["box_1_399682000", "box_2_986388000"]
STANDUP_CLIPS = ["stand_up_2431270000"]

# ---------------------------------------------------------------- loaders
def load_mocap_keypoints(clip, lo=None, hi=None):
    """-> (T, 6, 3) in +Z-up metres: [rear(pelvis), front(neck), FR, RR, FL, RL]."""
    a = np.loadtxt(f"{MI}/data/{clip}_joint_pos.txt", delimiter=",")
    a = a.reshape(len(a), 27, 3)
    if lo is not None:
        a = a[lo:hi]
    kp = np.stack([a[:, REF_PELVIS], a[:, REF_NECK]] + [a[:, j] for j in REF_TOES], axis=1)
    # F2: MoCap raw is +Y up (REF_COORD_ROT = rot_x(+90deg)); map (x,y,z)->(x,-z,y)
    return np.stack([kp[..., 0], -kp[..., 2], kp[..., 1]], axis=-1)


def load_video_keypoints(clip, source="depth_cam"):
    """-> (T, 6, 3) in +Z-up metres, same keypoint order as load_mocap_keypoints."""
    d = {"depth_cam": f"{MI}/data_fromVision_depth_cam/{clip}",
         "depth_any": f"{MI}/data_fromVision_aligned_video_depth_anything/{clip}",
         "avatar":    f"{MI}/data_AnimalAvatar/experiments_maila_{clip}"}[source]
    base = np.load(f"{d}/base.npy")   # (2, T, 3) = [rear hip, front hip]
    feet = np.load(f"{d}/feet.npy")   # (4, T, 3) = [HR, HL, VR, VL]
    feet = feet[VIDEO_TO_MOCAP_FEET]
    return np.concatenate([base, feet], axis=0).transpose(1, 0, 2)


def load_amp(folder):
    """-> list of (name, Frames (T,61), dt). Frames layout per motion_loader.py."""
    out = []
    for f in sorted(glob.glob(os.path.join(DS, folder, "*.txt"))):
        j = json.load(open(f))
        out.append((os.path.basename(f)[:-4], np.asarray(j["Frames"], float),
                    float(j["FrameDuration"])))
    return out


def load_occlusion_mask(clip):
    """F6: (T,4) bool, True where the foot keypoint was NOT visible ((0,0) in the
    2D tracks). Returns None when the clip has no track file."""
    cands = [f"{QFV}/tracks/{clip}/foot_pos.npy",
             # 3d_recon.py:56-66 uses the depth-friendly re-annotation for this clip
             f"{QFV}/tracks/{clip}/foot_pos_for_depth_cam.npy"]
    p = next((c for c in cands if os.path.exists(c)), None)
    if p is None:
        return None
    a = np.load(p)                     # (T, 4, 2) int
    return (a == 0).all(axis=-1)

# ---------------------------------------------------------------- normalization
def torso_length(kp):
    """L_c: median |front - rear|. Median, not mean -- vision has heavy tails."""
    return float(np.median(np.linalg.norm(kp[:, 1] - kp[:, 0], axis=-1)))


def torso_series(kp):
    return np.linalg.norm(kp[:, 1] - kp[:, 0], axis=-1)


def decimate_to_common(kp, fs):
    """Anti-aliased rate match onto FS_COMMON. Naive slicing would alias real
    15-30 Hz MoCap content into the band and inflate the MoCap baseline."""
    if abs(fs - FS_COMMON) < 1e-9:
        return kp
    from scipy.signal import resample_poly
    assert abs(fs / FS_COMMON - 2.0) < 1e-9, f"unexpected rate {fs}"
    T, K, D = kp.shape
    # padtype='line' is essential: the default zero padding injects large edge
    # transients that inflate short-clip metrics (trot CV 0.61% -> 8.58%).
    return resample_poly(kp.reshape(T, K * D), 1, 2, axis=0,
                         padtype="line").reshape(-1, K, D)


def canonicalize(kp, L=None):
    """Dimensionless body frame: rear keypoint at origin, heading along +x,
    coordinates divided by L_c. Required before overlaying the two sources."""
    if L is None:
        L = torso_length(kp)
    rear, front = kp[:, 0], kp[:, 1]
    psi = np.arctan2(front[:, 1] - rear[:, 1], front[:, 0] - rear[:, 0])
    q = kp - rear[:, None, :]
    c, s = np.cos(-psi), np.sin(-psi)
    x = c[:, None] * q[..., 0] - s[:, None] * q[..., 1]
    y = s[:, None] * q[..., 0] + c[:, None] * q[..., 1]
    return np.stack([x, y, q[..., 2]], axis=-1) / L


def canonicalize_global(kp, L=None, normalize=False):
    """GLOBAL trajectory in metres: ONE rigid transform for the whole clip, so the
    animal actually translates through the scene.

    Frame-0 rear keypoint to the origin and frame-0 heading to +x -- a single
    per-clip transform, never per frame. Both are needed to overlay clips that
    were captured at different places in the world (MoCap clips start 1.7-6.7 m
    apart and travel along y; the video clips are already origin-referenced and
    travel along x). Set normalize=True to additionally divide by the torso
    length; the default keeps real metres.

    Contrast with canonicalize(), which re-centres EVERY frame and so gives a
    treadmill view of limb configuration with the trajectory removed.
    """
    if L is None:
        L = torso_length(kp) if normalize else 1.0
    if not normalize:
        L = 1.0
    rear0, front0 = kp[0, 0], kp[0, 1]
    psi0 = np.arctan2(front0[1] - rear0[1], front0[0] - rear0[0])
    q = kp - rear0[None, None, :]
    c, s = np.cos(-psi0), np.sin(-psi0)
    x = c * q[..., 0] - s * q[..., 1]
    y = s * q[..., 0] + c * q[..., 1]
    return np.stack([x, y, q[..., 2]], axis=-1) / L


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")
    return p
