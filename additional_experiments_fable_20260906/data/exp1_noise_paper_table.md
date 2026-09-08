| source | clip | frames@30Hz | hip h [m] | M3 seg-len CV [%] | M1 body [%h] | paw stance std [%h] | paw HF [%h] | torso stance std [%h] | torso HF [%h] | support frames |
|---|---|---|---|---|---|---|---|---|---|---|
| MoCap | pace | 20 | 0.435 | 0.556 | 0.245 | 0.463 | 0.194 | 1.167 | 0.132 | 17/20 |
| MoCap | trot | 17 | 0.438 | 0.606 | 0.576 | 0.754 | 0.215 | 2.066 | 0.384 | 8/17 |
| MoCap | trot2 | 17 | 0.426 | 0.805 | 0.503 | 0.594 | 0.251 | 2.021 | 0.245 | 10/17 |
| MoCap | canter | 15 | 0.457 | 0.585 | 0.534 | 2.543 | 1.496 | nan | nan | 2/15 |
| MoCap | right turn0 | 75 | 0.430 | 2.050 | 0.315 | 0.601 | 0.247 | 1.611 | 0.132 | 60/75 |
| MoCap | left turn0 | 23 | 0.443 | 2.650 | 0.374 | 0.310 | 0.108 | 1.677 | 0.214 | 23/23 |
| Video | walk | 27 | 0.483 | 6.026 | 1.339 | 1.783 | 1.441 | 2.497 | 0.640 | 13/27 |
| Video | slow | 69 | 0.500 | 3.636 | 0.852 | 1.509 | 0.738 | 1.434 | 0.321 | 39/69 |
| Video | turn L | 45 | 0.495 | 14.315 | 0.828 | 1.053 | 0.805 | 0.984 | 0.207 | 33/45 |
| Video | turn R | 44 | 0.498 | 10.652 | 1.039 | 1.119 | 0.527 | 1.236 | 0.280 | 21/44 |
| Video | slow turn | 44 | 0.487 | 14.910 | 1.024 | 0.922 | 0.472 | 0.963 | 0.270 | 35/44 |
| Video | L-R turn | 73 | 0.498 | 6.027 | 0.872 | 1.414 | 0.993 | 1.354 | 0.356 | 34/73 |
| Video | start-stop 1 | 27 | 0.653 | 2.241 | 0.604 | 6.596 | 0.762 | nan | nan | 3/27 |
| Video | start-stop 2 | 45 | 0.513 | 4.621 | 0.702 | 0.605 | 0.671 | 3.991 | 0.227 | 45/45 |

## All noise metrics, MoCap vs video before and after the filter

| metric | MoCap | Video | Video filtered | Video/MoCap | Video filt./MoCap |
|---|---|---|---|---|---|
| torso length variation | 1.21 | 7.80 | 7.69 | 6.46 | 6.36 |
| planted-paw movement | 4.65 | 11.97 | 10.10 | 2.58 | 2.17 |
| paw stance scatter | 0.88 | 1.87 | 1.71 | 2.14 | 1.95 |
| torso stance scatter | 1.71 | 1.78 | 1.65 | 1.04 | 0.96 |
| HF residual, torso | 0.42 | 0.91 | 0.20 | 2.14 | 0.47 |
| HF residual, paws | 1.35 | 2.51 | 0.67 | 1.86 | 0.50 |
| paw stance scatter, HF | 0.42 | 0.80 | 0.29 | 1.91 | 0.69 |
| torso stance scatter, HF | 0.22 | 0.33 | 0.10 | 1.48 | 0.46 |

## The four scatter metrics with the robust (1.4826 MAD) estimator

| metric (robust std) | MoCap | Video | Video filtered | Video/MoCap | Video filt./MoCap |
|---|---|---|---|---|---|
| paw stance scatter | 0.71 | 2.04 | 2.02 | 2.86 | 2.84 |
| torso stance scatter | 1.49 | 1.68 | 1.51 | 1.13 | 1.01 |
| paw stance scatter, HF | 0.33 | 0.71 | 0.27 | 2.18 | 0.82 |
| torso stance scatter, HF | 0.19 | 0.31 | 0.12 | 1.63 | 0.60 |
