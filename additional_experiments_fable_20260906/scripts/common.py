"""Shared paths, style, loaders and geometry helpers for the three additional experiments.

Everything here is read-only with respect to the source repositories.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

# --------------------------------------------------------------------------------------
# Paths (all repositories are treated as read-only)
# --------------------------------------------------------------------------------------
OUT = "/home/admin_07/Downloads/additional_experiments_fable_20260906"
FIG = os.path.join(OUT, "figures")
DATA = os.path.join(OUT, "data")
VISION = "/home/admin_07/project_repos/quadruped_from_video"
RETARGET = "/home/admin_07/project_repos/motion_imitation/retarget_motion"
ISAAC = "/home/admin_07/project_repos/isaac_lab/IsaacLab"
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

# --------------------------------------------------------------------------------------
# Paper style (matches IsaacLab/plot_*.py: Type-42 fonts, DejaVu Sans, bbox_inches="tight")
# --------------------------------------------------------------------------------------
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 0.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
)
COL_W = 3.5  # IEEE single column [in]
DOUBLE_W = 7.16  # IEEE double column [in]


def savefig(fig, name, png=True):
    """Save vector PDF (+ PNG preview) into FIG with the paper's savefig options."""
    pdf = os.path.join(FIG, name + ".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    if png:
        fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    print("saved", pdf)


# --------------------------------------------------------------------------------------
# Colours: reuse the per-method assignment of IsaacLab/plot_DEFINITIONS.py
# (tab10 index 5 = MoCap, 1 = Video w. Depth Camera, 6 = Video w. Depth Camera (extended))
# --------------------------------------------------------------------------------------
NAME_MOCAP = "MoCap"
NAME_VIDEO = "Video"
NAME_VIDEO_EXT = "Video (extended)"
NAME_VIDEO_CORR = "Video (corrected intrinsics)"


def _paper_colors():
    sys.path.insert(0, ISAAC)
    try:
        import plot_DEFINITIONS as PD  # noqa: E402

        names = PD.ExperimentNames
        return {
            NAME_MOCAP: PD.get_color_for_experiment_name(names.mocap),
            NAME_VIDEO: PD.get_color_for_experiment_name(names.video_depth_cam),
            NAME_VIDEO_EXT: PD.get_color_for_experiment_name(names.video_depth_cam_extendedWithoutReverse),
        }
    except Exception:  # plot_DEFINITIONS uses plt.cm.get_cmap, removed in matplotlib>=3.9
        cmap = plt.get_cmap("tab10", 10)
        return {NAME_MOCAP: cmap(5), NAME_VIDEO: cmap(1), NAME_VIDEO_EXT: cmap(6)}


COLORS = _paper_colors()
COLORS[NAME_VIDEO_CORR] = plt.get_cmap("tab10", 10)(0)  # blue, not used by any paper method except "Manual Trajectory"
PAPER_LABEL = {  # legend labels exactly as in the paper's Fig. 4/5
    NAME_MOCAP: "MoCap (AMP)",
    NAME_VIDEO: "Video w. Depth Camera (AMP)",
    NAME_VIDEO_EXT: "Video w. Depth Camera (extended) (AMP)",
}

# --------------------------------------------------------------------------------------
# Dataset definitions
# --------------------------------------------------------------------------------------
AMP_DIRS = {
    NAME_MOCAP: os.path.join(ISAAC, "datasets/mocap_AMP_for_hardware"),  # 6.80 s
    NAME_VIDEO: os.path.join(ISAAC, "datasets/fromVision_motions_DepthCam"),  # 6.03 s  (paper: 6.0 s)
    NAME_VIDEO_EXT: os.path.join(ISAAC, "datasets/fromVision_motions_DepthCam_extendedWithoutReverse"),  # 12.20 s (paper: 12.2 s)
}

# flat-walking video clips (the 8 clips of the extended dataset; the first four form the base dataset)
VIDEO_BASE_CLIPS = ["walk_869488000", "slow_1313807000", "turn_left_1771233000", "turn_right_1771233000"]
VIDEO_EXT_ONLY_CLIPS = ["slow_turn_1771233000", "left_right_turn_2058226999", "start_stop_1271493000", "start_stop_785558000"]
FLAT_VIDEO_CLIPS = VIDEO_BASE_CLIPS + VIDEO_EXT_ONLY_CLIPS
CLIP_SHORT = {
    "walk_869488000": "walk",
    "slow_1313807000": "slow",
    "turn_left_1771233000": "turn L",
    "turn_right_1771233000": "turn R",
    "slow_turn_1771233000": "slow turn",
    "left_right_turn_2058226999": "L-R turn",
    "start_stop_1271493000": "start-stop 1",
    "start_stop_785558000": "start-stop 2",
}

# MoCap segments exactly as used by retarget_motion.py (name, raw file, start, end)
MOCAP_SEGMENTS = [
    ("pace", "dog_walk00", 162, 201),
    ("trot", "dog_walk03", 448, 481),
    ("trot2", "dog_run04", 630, 663),
    ("canter", "dog_run00", 430, 459),
    ("right turn0", "dog_walk09", 1000, 1150),
    ("left turn0", "dog_walk09", 2404, 2450),
]
MOCAP_FPS = 60.0  # dog_clips_info.txt: frames/seconds = 59.7 +- 0.3 for 24 clips
VIDEO_FPS = 30.0  # RGB/depth timestamps are 33.3 ms apart
MOCAP_FRAME_DURATION_USED = 0.021  # retarget_motion.py (AMP_for_hardware value); true 1/60 = 0.01667
VIDEO_FRAME_DURATION_USED = 0.03334
MOCAP_REF_POS_SCALE = 0.825  # retarget_config_go2.py UnitreeGo2Config.REF_POS_SCALE (mocap path)
VIDEO_REF_POS_SCALE = 1.0  # flat-walking video clips keep the default scale

# Zhang et al. joint indices used by retarget_motion.py
MOCAP_PELVIS, MOCAP_NECK = 0, 3
MOCAP_TOES = [10, 19, 15, 23]  # paired with SIM order FR, RR, FL, RL in retarget_motion.py
MOCAP_HIPS = [6, 16, 11, 20]

# Command / target distribution of the flat-walking RL task (velocity_env_cfg.py CommandsCfg)
CMD_RANGES = {"vx": (-1.0, 1.0), "vy": (-0.3, 0.3), "wz": (-1.57, 1.57)}
HEADING_COMMAND = False  # heading_command=False -> ang_vel_z is sampled directly
REL_STANDING_ENVS = 0.02
EVAL_GRID = {  # targetDistributions.sh
    "xy": (np.round(np.arange(-1.0, 1.0001, 0.1), 2), np.round(np.arange(-0.3, 0.3001, 0.1), 2)),
    "xyaw": (np.round(np.arange(-1.0, 1.0001, 0.1), 2), np.round(np.arange(-1.0, 1.0001, 0.1), 2)),
}


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def load_amp_dir(path):
    """Load all AMP expert files (JSON with .txt extension) of a dataset folder."""
    out = []
    for f in sorted(glob.glob(os.path.join(path, "*.txt"))):
        with open(f) as fh:
            j = json.load(fh)
        out.append(
            dict(
                name=os.path.basename(f).replace("_amp.txt", ""),
                frames=np.asarray(j["Frames"], dtype=float),
                fd=float(j["FrameDuration"]),
                weight=float(j.get("MotionWeight", 1.0)),
                path=f,
            )
        )
    return out


def load_video_keypoints_as_used(clip):
    """3D keypoints that fed the retargeting: (base (2,T,3) [rear hip, front hip], feet (4,T,3) [RR, RL, FR, FL])."""
    d = os.path.join(RETARGET, "data_fromVision_depth_cam", clip)
    return np.load(os.path.join(d, "base.npy")), np.load(os.path.join(d, "feet.npy"))


def load_mocap_raw(fname):
    """Raw Zhang et al. joint positions (T, 27, 3), converted to z-up (proper rotation +90 deg about x)."""
    a = np.loadtxt(os.path.join(RETARGET, "data", f"{fname}_joint_pos.txt"), delimiter=",").reshape(-1, 27, 3)
    return np.stack([a[..., 0], -a[..., 2], a[..., 1]], axis=-1)


def load_mocap_segment(seg):
    name, fname, s, e = seg
    return load_mocap_raw(fname)[s:e]


def mocap_keypoints(arr):
    """Return the 6 keypoints comparable to the video pipeline: body (2,T,3) = [pelvis, neck], feet (4,T,3)."""
    body = np.stack([arr[:, MOCAP_PELVIS], arr[:, MOCAP_NECK]], axis=0)
    feet = np.stack([arr[:, t] for t in MOCAP_TOES], axis=0)
    return body, feet


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def yaw_from_quat_xyzw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def align_clip(body, feet, ground_percentile=5.0):
    """ONE rigid transform per clip (translation + yaw rotation), applied to all frames identically.

    * origin: body centre of the FIRST frame (x, y)
    * yaw: mean heading (front - rear body keypoint, averaged over the clip) rotated onto +x
    * z: ground level := `ground_percentile`-th percentile of all foot heights (same rule for every source)
    The animal keeps translating through the scene; nothing is re-centred per frame.
    """
    body = np.asarray(body, float).copy()
    feet = np.asarray(feet, float).copy()
    heading = (body[1] - body[0])[:, :2].mean(axis=0)
    yaw = np.arctan2(heading[1], heading[0])
    R = rot_z(-yaw)
    origin = body[:, 0, :].mean(axis=0).copy()
    origin[2] = np.percentile(feet[..., 2], ground_percentile)
    body = (body - origin) @ R.T
    feet = (feet - origin) @ R.T
    return body, feet, dict(yaw=yaw, origin=origin)


def sg_residual(x, window=7, poly=3):
    """Residual of a trajectory (T, d) w.r.t. its Savitzky-Golay smoothing along axis 0."""
    x = np.asarray(x, float)
    w = min(window, len(x) - (1 - len(x) % 2))  # odd and <= len
    if w <= poly:
        return np.zeros_like(x)
    return x - savgol_filter(x, w, poly, axis=0, mode="interp")


def hip_height(body, feet, ground_percentile=5.0):
    """Median body-centre height above the estimated ground level."""
    ground = np.percentile(feet[..., 2], ground_percentile)
    return float(np.median(body[..., 2].mean(axis=0)) - ground)


def md_table(header, rows, floatfmt="{:.3f}"):
    """Small markdown table helper."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)
