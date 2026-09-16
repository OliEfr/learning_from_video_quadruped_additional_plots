# Exp. 5: evaluation metrics (DefaultEvalConfig, target command distribution, as Fig. 4)

| set | seeds | vel. error [m/s] (mean [min, max]) | yaw error [rad/s] | CoT | d vel vs Video(ext) | d yaw | d CoT |
|---|---|---|---|---|---|---|---|
| MoCap (paper) | 3 (paper logs) | 0.0624 [0.0609, 0.0633] | 0.6450 [0.5899, 0.6825] | 1.51 [1.48, 1.55] |  |  |  |
| Video (paper) | 3 (paper logs) | 0.0565 [0.0540, 0.0579] | 0.2130 [0.1896, 0.2278] | 1.59 [1.52, 1.69] |  |  |  |
| Video (extended) (paper) | 3 (paper logs) | 0.0481 [0.0471, 0.0497] | 0.1292 [0.1238, 0.1356] | 1.31 [1.25, 1.43] |  |  |  |
| E2 Video (added clips only) | 3 (2,3,4) | 0.1953 [0.1818, 0.2136] | 0.131 [0.119, 0.137] | 1.93 [1.85, 2.07] | +306 % | +1 % | +47 % |
| E1b Video (extended, half, coverage-matched) | 3 (1,2,3) | 0.0535 [0.0519, 0.0544] | 0.131 [0.125, 0.134] | 1.15 [1.09, 1.22] | +11 % | +1 % | -12 % |
| E1 Video (extended, half) | 3 (1,2,3) | 0.0575 [0.0560, 0.0594] | 0.181 [0.136, 0.229] | 1.35 [1.20, 1.54] | +20 % | +40 % | +3 % |
| E3 Video (extended minus turning) | 3 (1,2,3) | 0.0523 [0.0496, 0.0544] | 0.145 [0.133, 0.158] | 1.34 [1.28, 1.39] | +9 % | +12 % | +3 % |
| E0 Video (extended), re-run seed | 1 (1) | 0.0471 [0.0471, 0.0471] | 0.124 [0.124, 0.124] | 1.26 [1.26, 1.26] | -2 % | -4 % | -4 % |

Excluded from the means (diverged training runs):
- fromVision_motions_DepthCam_extAddedOnly_SEED_1: vel 0.5441, yaw 0.691, CoT 19.94, 23760 episodes; diverged at iter ~9.7k (critic blow-up), collapsed policy
