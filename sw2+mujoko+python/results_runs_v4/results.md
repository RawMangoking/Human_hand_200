# Results (success rate, mean ± sd over seeds)

| task | algo | training | actions | seeds | nominal | in-distribution | UNSEEN |
|---|---|---|---|:-:|:-:|:-:|:-:|
| door | ars | fixed | design | 1 | 100 ± 0 % | 90 ± 0 % | 89 ± 0 % |
| door | ars | random | design | 1 | 100 ± 0 % | 90 ± 0 % | 88 ± 0 % |
| handle | ars | fixed | design | 1 | 50 ± 0 % | 48 ± 0 % | 42 ± 0 % |
| handle | ars | random | design | 1 | 70 ± 0 % | 71 ± 0 % | 60 ± 0 % |
| reach | ars | fixed | design | 1 | 100 ± 0 % | 97 ± 0 % | 84 ± 0 % |
| reach | ars | random | design | 1 | 100 ± 0 % | 100 ± 0 % | 100 ± 0 % |

# Chained three-policy runs (reach -> handle -> door)

| run | door set | doors | reach | handle | door | END-TO-END | time (s) | recoveries | grasped (handle stage) | arm on door (door stage) |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| chain_ars_fixed_s0 | in_dist | 10 | 97 % | 24 % | 17 % | **17 %** | 9.6 | 0.7 | 24 % | 14 % |
| chain_ars_fixed_s0 | nominal | 1 | 100 % | 0 % | 0 % | **0 %** | nan | 1.0 | 24 % | - |
| chain_ars_fixed_s0 | unseen | 9 | 94 % | 21 % | 13 % | **13 %** | 9.6 | 0.7 | 22 % | 13 % |
| chain_ars_random_s0 | in_dist | 10 | 100 % | 82 % | 70 % | **70 %** | 7.7 | 0.2 | 60 % | 10 % |
| chain_ars_random_s0 | nominal | 1 | 100 % | 70 % | 70 % | **70 %** | 6.7 | 0.3 | 55 % | 0 % |
| chain_ars_random_s0 | unseen | 9 | 99 % | 72 % | 62 % | **62 %** | 7.0 | 0.3 | 54 % | 10 % |

# Research questions

## Q1 - fixed vs domain-randomized training (three chained policies, end-to-end success)

| training | seeds | nominal | in-distribution | **unseen** |
|---|:-:|:-:|:-:|:-:|
| fixed | 1 | 0 ± 0 % | 17 ± 0 % | 13 ± 0 % |
| random | 1 | 70 ± 0 % | 70 ± 0 % | 62 ± 0 % |

Unseen doors, randomized minus fixed: **+49 percentage points**, 95 % bootstrap interval [+32, +64] over the 9 doors -> randomized training generalizes BETTER to unseen doors.

| unseen door | fixed | randomized |
|---|:-:|:-:|
| deep_latch | 10 % | 70 % |
| handle_high | 10 % | 60 % |
| handle_low | 0 % | 0 % |
| hard_combo | 0 % | 40 % |
| heavy_slippery | 10 % | 100 % |
| shifted_side | 20 % | 80 % |
| stiff_handle | 0 % | 70 % |
| stiff_hinge | 60 % | 80 % |
| turned_door | 10 % | 60 % |

## Q2a - action space of the hand (handle stage, randomized training, trained FROM SCRATCH - no cloning, so the learning itself is compared): design = index + thumb independent, middle/ring/little coupled (18 actions); synergy = finger curls (12); full = every joint (24)

(no handle runs for both action modes yet)

## Q2b - three chained policies vs one policy for the whole sequence (randomized training; both cloned from the same hand-written controller with the same budget)

(needs the chained runs and the 'full' baseline)
