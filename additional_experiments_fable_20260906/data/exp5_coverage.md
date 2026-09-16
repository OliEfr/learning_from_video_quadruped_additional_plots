# Exp. 5 ablation sets: command coverage (Exp.-2 rule and tolerances)

| set | clips | frames | dur [s] | vx-vy cells | vx-wz cells | box cov [%] | standing [%] | turning [%] | vx>1 [%] | note |
|---|---|---|---|---|---|---|---|---|---|---|
| Video (paper) | 4 | 185 | 6.03 | 22/147 | 48/441 | 13.9 | 2.2 | 42.0 | 25.0 |  |
| Video (extended) (paper) | 8 | 374 | 12.20 | 24/147 | 83/441 | 25.0 | 4.0 | 54.0 | 22.2 |  |
| E1 extHalf | 8 | 186 | 5.93 | 24/147 | 71/441 | 20.5 | 4.1 | 44.4 | 20.6 | windows: left_right_turn_2058226999 [14:50) of 73, slow_1313807000 [23:57) of 69, slow_turn_1771233000 [15:37) of 44, start_stop_1271493000 [13:27) of 27, start_stop_785558000 [22:44) of 45, turn_left_1771233000 [2:24) of 45, turn_right_1771233000 [22:44) of 44, walk_869488000 [3:17) of 27 |
| (E1 alt: first half of every clip, not used) | 8 | 186 | 5.93 | 18/147 | 49/441 | 17.5 | 3.0 | 52.3 | 22.2 |  |
| E1b extHalfCov | 16 | 186 | 5.67 | 25/147 | 78/441 | 24.9 | 4.6 | 46.4 | 20.6 | two windows per clip, weight halved: left_right_turn_2058226999 [17:35) + [48:66) of 73, slow_1313807000 [16:33) + [44:61) of 69, slow_turn_1771233000 [20:31) + [32:43) of 44, start_stop_1271493000 [0:7) + [12:19) of 27, start_stop_785558000 [0:11) + [16:27) of 45, turn_left_1771233000 [2:13) + [14:25) of 45, turn_right_1771233000 [20:31) + [32:43) of 44, walk_869488000 [3:10) + [14:21) of 27 |
| (every 2nd frame of every clip, not used) | 8 | 190 | 12.14 | 20/147 | 56/441 | 18.8 | 4.0 | 49.9 | 22.2 |  |
| E2 extAddedOnly | 4 | 189 | 6.17 | 8/147 | 48/441 | 19.1 | 6.9 | 75.9 | 0.0 |  |
| E3 extNoTurn | 6 | 257 | 8.37 | 24/147 | 64/441 | 21.3 | 5.2 | 43.0 | 28.6 |  |
| E4 extNoStartStop | 6 | 302 | 9.87 | 22/147 | 70/441 | 18.6 | 1.2 | 52.5 | 28.6 |  |
