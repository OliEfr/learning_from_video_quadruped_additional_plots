## Noise metrics: MoCap baseline (unfiltered) vs video unfiltered / filtered (filter = part of the video method)

| source | filter | M1 torso [% h] | M1 paws [% h] | M1 paws [mm] | M2 paws [mm/fr²] | M3 seg CV [%] | M4 stance [% h] | M4 [mm] | PSD power >5 Hz [1e-4] |
|---|---|---|---|---|---|---|---|---|---|
| MoCap | unfiltered (baseline) | 0.42 | 1.35 | 5.94 | 27.36 | 1.21 | 0.71 | 3.15 | 1.14 |
| Video | unfiltered | 1.84 | 3.67 | 9.31 | 29.87 | 21.05 | 2.49 | 6.85 | 4.71 |
| Video | B only (6 Hz zero-phase) | 0.37 | 1.01 | 2.55 | 14.08 | 20.88 | 2.62 | 7.20 | 0.81 |
| Video | A+B (method) | 0.37 | 0.99 | 2.50 | 13.74 | 20.88 | 2.61 | 7.18 | 0.79 |
| Video (corrected intrinsics) | unfiltered | 0.91 | 2.51 | 12.76 | 40.42 | 7.80 | 2.04 | 11.64 | 2.04 |
| Video (corrected intrinsics) | B only (6 Hz zero-phase) | 0.20 | 0.68 | 3.41 | 18.40 | 7.69 | 2.03 | 11.68 | 0.35 |
| Video (corrected intrinsics) | A+B (method) | 0.20 | 0.67 | 3.39 | 18.23 | 7.69 | 2.02 | 11.67 | 0.34 |

## Motion preservation (filtered vs unfiltered, mean over clips)

| source | net displ. [%] | path length [%] | mean speed [%] | p95 speed [%] | stride freq. [%] | peak clearance raw [mm] | filt [mm] | [%] | stance frac raw [%] | filt [%] | contact onsets | onset shift [ms] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Video | -0.0 | -5.6 | -5.6 | -10.0 | 0.0 | 63.1 | 61.9 | -1.3 | 44.0 | 44.1 | 85->73 | 36.3 |
| Video (corrected intrinsics) | -0.0 | -4.9 | -4.9 | -9.1 | 0.0 | 102.0 | 96.3 | -6.1 | 44.4 | 42.4 | 92->72 | 60.8 |
