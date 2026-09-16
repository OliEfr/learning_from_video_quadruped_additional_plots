# Exp. 5: E1 grid evaluation (TargetXYDistribution, 7 x 21 vx-vy cells at yaw rate 0)

E1 seeds with complete grids: [1, 2, 3]. Paper rows use the Fig.-5 values recovered from the figure (Exp. 4); E1 rows are exact yaml values (seed mean). Command distance = normalised distance of the cell's command to the nearest expert frame of the set the policy was trained on.

| set | covered cells /147 | rho(cmd dist, err vel) | rho(cmd dist, err yaw) | rho(cmd dist, comb. err) | rho(cmd dist, agent-expert dist) | slope comb. err vs cmd dist | intercept | err vel covered | uncovered | err yaw covered | uncovered | agent-expert dist covered | uncovered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 Video (ext., half) | 24 | 0.207 | 0.679 | 0.603 | 0.819 | 0.362 | 0.353 | 0.051 | 0.063 | 0.109 | 0.171 | 2.185 | 3.111 |
| MoCap (paper, recovered) | 14 | 0.247 | 0.709 | 0.610 | 0.241 | 0.218 | 0.612 | 0.064 | 0.066 | 0.280 | 0.344 | 2.884 | 2.933 |
| Video (paper, recovered) | 22 | 0.619 | 0.177 | 0.442 | 0.592 | 0.238 | 0.456 | 0.044 | 0.062 | 0.201 | 0.215 | 2.538 | 3.276 |
| Video (ext.) (paper, recovered) | 24 | 0.375 | 0.162 | 0.313 | 0.777 | 0.078 | 0.356 | 0.042 | 0.051 | 0.102 | 0.114 | 2.238 | 3.156 |

Per E1 seed:

| E1 seed | rho(cmd, err vel) | rho(cmd, err yaw) | rho(cmd, comb) | slope | grid mean err vel | grid mean err yaw | grid mean CoT |
|---|---|---|---|---|---|---|---|
| 1 | 0.100 | 0.743 | 0.601 | 0.488 | 0.059 | 0.203 | 1.623 |
| 2 | 0.348 | 0.290 | 0.408 | 0.211 | 0.061 | 0.118 | 1.284 |
| 3 | 0.172 | 0.537 | 0.570 | 0.386 | 0.062 | 0.161 | 1.304 |
