# Keypoint noise per axis and error during ground contact

Same clips as panel (d) of the Fig. 7 replacement: 6 MoCap, 8 video (corrected intrinsics, unfiltered).
The `_nooutlier` figures leave out the video clips whose worst robust z score exceeds 3, in every panel.

## all clips

Robust std of the detrended coordinate in stance/support frames, % hip height: MoCap / Video (ratio)

|  | x | y | z |
|---|---|---|---|
| Torso | 0.17 / 0.39 (2.3x) | 0.12 / 0.39 (3.3x) | 0.19 / 0.31 (1.6x) |
| Paws | 0.55 / 0.86 (1.6x) | 0.40 / 0.79 (2.0x) | 0.33 / 0.71 (2.2x) |

Net paw displacement over the core of a stance (first/last 1 contact frame dropped), mm; per clip the median over its stances, then the mean over clips

|  | horizontal | 3D |
|---|---|---|
| MoCap | 11 | 13 |
| Video | 43 | 44 |

Position error during ground contact (`fig_exp1_drift_pct`), % hip height, per clip the median over its stances / support phases, then the mean over clips.  Paws: RMS distance from the best-fit STATIONARY point over the stance core (a planted paw should not move at all).  Torso: RMS distance from its own smooth (Savitzky-Golay) path over a support phase (the torso does translate while supported, so only the off-path part is error).

|  | MoCap | Video | ratio |
|---|---|---|---|
| Paws | 1.20 | 3.43 | 2.9x |
| Torso | 0.26 | 0.67 | 2.6x |

This replaces the net first->last displacement in that figure.  The net displacement is not the error of a known-constant quantity - a paw that rolls evenly through its contact with no reconstruction error at all still produces a large net displacement - and it sits a factor ~2.4 higher: over the 17 usable combinations of the contact band (10, 15, 20 mm), the erosion (0, 1, 2 frames) and the minimum core length (3, 4 frames) the MoCap net displacement runs 2.06-5.12 % of hip height, while the position error stays at 0.78-1.91 %, i.e. low single digit under every setting and not only for one choice of thresholds.  Both separate the sources equally well (video/MoCap 1.8-3.4x net, 1.9-3.6x position error) and both inherit the same mild dependence on contact duration (Spearman rho = +0.53 in MoCap for either), which comes from the stance definition rather than from the metric.  The net displacement itself is still reported above, in mm.

## video outliers removed

Robust std of the detrended coordinate in stance/support frames, % hip height: MoCap / Video (ratio)

|  | x | y | z |
|---|---|---|---|
| Torso | 0.17 / 0.29 (1.7x) | 0.12 / 0.37 (3.1x) | 0.19 / 0.28 (1.5x) |
| Paws | 0.55 / 0.67 (1.2x) | 0.40 / 0.71 (1.8x) | 0.33 / 0.63 (1.9x) |

Net paw displacement over the core of a stance (first/last 1 contact frame dropped), mm; per clip the median over its stances, then the mean over clips

|  | horizontal | 3D |
|---|---|---|
| MoCap | 11 | 13 |
| Video | 19 | 21 |

Position error during ground contact (`fig_exp1_drift_pct`), % hip height, per clip the median over its stances / support phases, then the mean over clips.  Paws: RMS distance from the best-fit STATIONARY point over the stance core (a planted paw should not move at all).  Torso: RMS distance from its own smooth (Savitzky-Golay) path over a support phase (the torso does translate while supported, so only the off-path part is error).

|  | MoCap | Video | ratio |
|---|---|---|---|
| Paws | 1.20 | 2.12 | 1.8x |
| Torso | 0.26 | 0.61 | 2.3x |

This replaces the net first->last displacement in that figure.  The net displacement is not the error of a known-constant quantity - a paw that rolls evenly through its contact with no reconstruction error at all still produces a large net displacement - and it sits a factor ~2.4 higher: over the 17 usable combinations of the contact band (10, 15, 20 mm), the erosion (0, 1, 2 frames) and the minimum core length (3, 4 frames) the MoCap net displacement runs 2.06-5.12 % of hip height, while the position error stays at 0.78-1.91 %, i.e. low single digit under every setting and not only for one choice of thresholds.  Both separate the sources equally well (video/MoCap 1.8-3.4x net, 1.9-3.6x position error) and both inherit the same mild dependence on contact duration (Spearman rho = +0.53 in MoCap for either), which comes from the stance definition rather than from the metric.  The net displacement itself is still reported above, in mm.

dropped: noise/drift figure ['slow', 'turn L', 'walk'], % figure ['slow', 'turn L', 'walk']

