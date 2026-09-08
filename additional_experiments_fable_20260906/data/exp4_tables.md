## Cell-wise link (7 x 21 vx-vy grid at yaw rate 0), Spearman rho and covered/uncovered means

| dataset | rho(cmd dist, agent-expert dist) | rho(cmd dist, err vel) | rho(cmd dist, err yaw) | rho(agent-expert dist, err vel) | rho(agent-expert dist, err yaw) | agent-expert dist covered | uncovered | err vel covered | uncovered | err yaw covered | uncovered |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MoCap | 0.24 | 0.25 | 0.71 | 0.58 | 0.01 | 2.88 | 2.93 | 0.06 | 0.07 | 0.28 | 0.34 |
| Video | 0.59 | 0.62 | 0.18 | 0.14 | -0.28 | 2.54 | 3.28 | 0.04 | 0.06 | 0.20 | 0.22 |
| Video (ext.) | 0.78 | 0.38 | 0.16 | 0.50 | 0.21 | 2.24 | 3.16 | 0.04 | 0.05 | 0.10 | 0.11 |
| pooled (441 cells) | 0.49 | 0.43 | 0.37 | 0.30 | -0.11 | 2.50 | 3.12 | 0.05 | 0.06 | 0.18 | 0.23 |

## Expert-set calibration of the agent-expert distance (leave-one-clip-out NN distance, 24-D AMP space)

| dataset | clip | frames | LOO NN dist mean | median | max |
|---|---|---|---|---|---|
| MoCap | canter | 29 | 10.05 | 10.06 | 14.97 |
| MoCap | left turn0 | 46 | 4.38 | 4.11 | 7.20 |
| MoCap | pace | 39 | 3.67 | 3.69 | 5.15 |
| MoCap | right turn0 | 150 | 4.38 | 4.24 | 8.49 |
| MoCap | trot2 | 33 | 3.74 | 3.79 | 4.83 |
| MoCap | trot | 33 | 4.04 | 3.79 | 7.81 |
| Video | slow | 69 | 4.95 | 3.74 | 22.92 |
| Video | turn L | 45 | 5.08 | 4.87 | 11.06 |
| Video | turn R | 44 | 4.99 | 4.24 | 31.16 |
| Video | walk | 27 | 9.50 | 9.46 | 13.22 |
| Video (ext.) | L-R turn | 73 | 4.37 | 4.15 | 9.16 |
| Video (ext.) | slow | 69 | 4.63 | 3.25 | 22.92 |
| Video (ext.) | slow turn | 44 | 3.15 | 3.01 | 8.01 |
| Video (ext.) | start-stop 1 | 27 | 2.43 | 2.38 | 4.03 |
| Video (ext.) | start-stop 2 | 45 | 3.10 | 3.07 | 5.46 |
| Video (ext.) | turn L | 45 | 4.51 | 3.98 | 9.61 |
| Video (ext.) | turn R | 44 | 4.03 | 3.15 | 31.16 |
| Video (ext.) | walk | 27 | 9.08 | 9.29 | 11.73 |

## What the discriminator sees per clip, and how AMP samples it

| dataset | clip | frames | MotionWeight | AMP mass [%] | per-frame density (rel. to min) | mean speed [m/s] | mean |yaw rate| [rad/s] | RMS |qdot| [rad/s] |
|---|---|---|---|---|---|---|---|---|
| MoCap | canter | 29 | 1.00 | 16.67 | 5.17 | 2.09 | 0.16 | 14.08 |
| MoCap | left turn0 | 46 | 1.00 | 16.67 | 3.26 | 0.13 | 1.83 | 5.12 |
| MoCap | pace | 39 | 1.00 | 16.67 | 3.85 | 0.71 | 0.24 | 7.96 |
| MoCap | right turn0 | 150 | 1.00 | 16.67 | 1.00 | 0.51 | 0.71 | 6.82 |
| MoCap | trot2 | 33 | 1.00 | 16.67 | 4.55 | 1.15 | 0.12 | 10.73 |
| MoCap | trot | 33 | 1.00 | 16.67 | 4.55 | 1.17 | 0.20 | 11.14 |
| Video | slow | 69 | 1.00 | 25.00 | 1.00 | 0.22 | 0.11 | 10.64 |
| Video | turn L | 45 | 1.00 | 25.00 | 1.53 | 0.42 | 1.70 | 8.67 |
| Video | turn R | 44 | 1.00 | 25.00 | 1.57 | 0.33 | 1.71 | 9.92 |
| Video | walk | 27 | 1.00 | 25.00 | 2.56 | 1.55 | 0.28 | 13.91 |
| Video (ext.) | L-R turn | 73 | 1.00 | 11.11 | 1.00 | 0.33 | 2.12 | 8.05 |
| Video (ext.) | slow | 69 | 1.00 | 11.11 | 1.06 | 0.22 | 0.11 | 10.64 |
| Video (ext.) | slow turn | 44 | 1.00 | 11.11 | 1.66 | 0.34 | 1.74 | 7.15 |
| Video (ext.) | start-stop 1 | 27 | 1.00 | 11.11 | 2.70 | 0.12 | 0.51 | 3.96 |
| Video (ext.) | start-stop 2 | 45 | 1.00 | 11.11 | 1.62 | 0.29 | 0.69 | 5.60 |
| Video (ext.) | turn L | 45 | 1.00 | 11.11 | 1.62 | 0.42 | 1.70 | 8.67 |
| Video (ext.) | turn R | 44 | 1.00 | 11.11 | 1.66 | 0.33 | 1.71 | 9.92 |
| Video (ext.) | walk | 27 | 2.00 | 22.22 | 5.41 | 1.55 | 0.28 | 13.91 |
