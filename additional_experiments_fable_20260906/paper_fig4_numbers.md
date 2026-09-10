# Numbers behind Fig. 4 (flat walking unless stated), for the caption sentences

Source: `~/Desktop/selected_metrics_raw_values.md` (2026-09-06): seed means read back from the vector geometry of `IsaacLab/plots/selected_metrics_2.pdf`, validated to 5 digits against the two panels whose `metrics.yaml` exist (stairs, stand-up video). The flat-walking `metrics.yaml` are not on this machine. Relative change = (A - B) / B; negative = A better (all three metrics are lower-is-better).

## A. Expert demonstrations vs engineered rewards (flat walking)

| A vs B | vel. error | yaw error | CoT |
|---|---|---|---|
| VideoExt vs Complex | -26 % (0.0481 vs 0.0652) | +121 % (0.129 vs 0.059) | -22 % (1.31 vs 1.69) |
| VideoExt vs Simple | -1 % (0.0481 vs 0.0487) | +83 % (0.129 vs 0.071) | -78 % (1.31 vs 6.08) |
| Video vs Complex | -13 % (0.0565 vs 0.0652) | +264 % (0.213 vs 0.059) | -6 % (1.59 vs 1.69) |
| Video vs Simple | +16 % (0.0565 vs 0.0487) | +201 % (0.213 vs 0.071) | -74 % (1.59 vs 6.08) |
| MoCap vs Complex | -4 % (0.0624 vs 0.0652) | +1003 % (0.645 vs 0.059) | -10 % (1.51 vs 1.69) |
| MoCap vs Simple | +28 % (0.0624 vs 0.0487) | +812 % (0.645 vs 0.071) | -75 % (1.51 vs 6.08) |
| Manual vs Complex | +1 % (0.0658 vs 0.0652) | +164 % (0.155 vs 0.059) | +11 % (1.88 vs 1.69) |
| Manual vs Simple | +35 % (0.0658 vs 0.0487) | +119 % (0.155 vs 0.071) | -69 % (1.88 vs 6.08) |

Other scenarios, Video (AMP) vs engineered rewards:

| scenario | vel. error vs Complex | vel. error vs Simple | CoT vs Complex | CoT vs Simple |
|---|---|---|---|---|
| Stand-Up | n/a | n/a | +32 % (2.45 vs 1.85) | -56 % |
| Box | +42 % | +32 % | -28 % (1.34 vs 1.85) | -89 % |
| Stairs | +25 % | +43 % | -20 % (1.41 vs 1.78) | -86 % |

## B. Video (extended) vs MoCap and vs Video (flat walking)

| A vs B | vel. error | yaw error | CoT |
|---|---|---|---|
| VideoExt vs MoCap | -23 % (0.0481 vs 0.0624) | -80 % (0.129 vs 0.645) | -13 % (1.31 vs 1.51) |
| VideoExt vs Video | -15 % (0.0481 vs 0.0565) | -39 % (0.129 vs 0.213) | -17 % (1.31 vs 1.59) |
| Video vs MoCap | -9 % (0.0565 vs 0.0624) | -67 % (0.213 vs 0.645) | +5 % (1.59 vs 1.51) |

Fig. 5 caption states -24 % / -67 % / -11 % for the same comparison: those are means over the Fig. 5 evaluation grid (uniform over the vx-yaw cells), Fig. 4 evaluates on the target command distribution. Both are correct for their figure; do not mix them in one caption.

Coverage of the 3-D command box (results.md Exp. 2): MoCap 13.2 %, Video 13.9 %, Video (extended) 25.0 %; vx-yaw evaluation grid: 56 / 48 / 83 of 441 cells.

## C. Caption sentences with numbers (Fig. 4) - final choice: Video (extended) vs Complex Reward only; sentence 2 vel + CoT only

> Learning from expert demonstrations beats the engineered complex reward in velocity tracking (-26 %) and cost of transport (-22 %), showing that efficient priors are difficult to replicate manually.

> Our extended video dataset improves upon MoCap for flat walking (-23 % velocity tracking error, -13 % cost of transport) due to better coverage of the command range by the expert data.

Abstract check: "-23 % velocity tracking error, -13 % cost of transport over MoCap" = Fig. 4 Video (extended) vs MoCap, consistent. "reduce ... by -23 %" is a double negative -> "reduce by 23 %". Against the complex reward the extended set is -26 % / -22 % (vel / CoT) but +121 % in yaw error, which is what "competitive" covers.
