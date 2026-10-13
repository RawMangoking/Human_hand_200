# Results (success rate, mean ± sd over seeds)

| task | algo | training | actions | seeds | nominal | in-distribution | UNSEEN |
|---|---|---|---|:-:|:-:|:-:|:-:|
| door | ars | fixed | design/scratch | 3 | 67 ± 47 % | 47 ± 34 % | 50 ± 35 % |
| door | ars | random | design/scratch | 3 | 100 ± 0 % | 90 ± 0 % | 88 ± 1 % |
| handle | ars | fixed | design/scratch | 3 | 97 ± 5 % | 97 ± 1 % | 95 ± 2 % |
| handle | ars | random | design/scratch | 3 | 97 ± 5 % | 97 ± 4 % | 93 ± 5 % |
| reach | ars | fixed | design/scratch | 3 | 90 ± 8 % | 71 ± 5 % | 63 ± 10 % |
| reach | ars | random | design/scratch | 3 | 93 ± 9 % | 79 ± 10 % | 76 ± 5 % |

# Chained three-policy runs (reach -> handle -> door)

| run | door set | doors | reach | handle | door | END-TO-END | time (s) | recoveries |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| chain_ars_fixed_s0 | in_dist | 10 | 77 % | 73 % | 34 % | **34 %** | 6.0 | 0.0 |
| chain_ars_fixed_s0 | nominal | 1 | 100 % | 100 % | 70 % | **70 %** | 5.9 | 0.0 |
| chain_ars_fixed_s0 | unseen | 9 | 68 % | 60 % | 37 % | **37 %** | 6.5 | 0.1 |
| chain_ars_fixed_s1 | in_dist | 10 | 80 % | 70 % | 43 % | **43 %** | 8.4 | 0.1 |
| chain_ars_fixed_s1 | nominal | 1 | 100 % | 90 % | 80 % | **80 %** | 6.8 | 0.1 |
| chain_ars_fixed_s1 | unseen | 9 | 69 % | 60 % | 41 % | **41 %** | 8.1 | 0.1 |
| chain_ars_fixed_s2 | in_dist | 10 | 77 % | 75 % | 42 % | **42 %** | 7.4 | 0.0 |
| chain_ars_fixed_s2 | nominal | 1 | 100 % | 100 % | 80 % | **80 %** | 5.7 | 0.0 |
| chain_ars_fixed_s2 | unseen | 9 | 60 % | 57 % | 44 % | **44 %** | 6.7 | 0.0 |
| chain_ars_random_s0 | in_dist | 10 | 78 % | 69 % | 51 % | **51 %** | 5.6 | 0.1 |
| chain_ars_random_s0 | nominal | 1 | 100 % | 90 % | 70 % | **70 %** | 4.5 | 0.1 |
| chain_ars_random_s0 | unseen | 9 | 71 % | 54 % | 36 % | **36 %** | 6.6 | 0.2 |
| chain_ars_random_s1 | in_dist | 10 | 93 % | 73 % | 29 % | **29 %** | 7.2 | 0.2 |
| chain_ars_random_s1 | nominal | 1 | 100 % | 80 % | 60 % | **60 %** | 7.4 | 0.2 |
| chain_ars_random_s1 | unseen | 9 | 91 % | 66 % | 33 % | **33 %** | 8.8 | 0.3 |
| chain_ars_random_s2 | in_dist | 10 | 73 % | 62 % | 39 % | **39 %** | 7.7 | 0.1 |
| chain_ars_random_s2 | nominal | 1 | 100 % | 100 % | 100 % | **100 %** | 6.0 | 0.0 |
| chain_ars_random_s2 | unseen | 9 | 68 % | 56 % | 42 % | **42 %** | 7.1 | 0.1 |

# Research questions

## Q1 - fixed vs domain-randomized training (three chained policies, end-to-end success)

| training | seeds | nominal | in-distribution | **unseen** |
|---|:-:|:-:|:-:|:-:|
| fixed | 3 | 77 ± 5 % | 40 ± 4 % | 41 ± 3 % |
| random | 3 | 77 ± 17 % | 40 ± 9 % | 37 ± 4 % |

Unseen doors, randomized minus fixed: **-4 percentage points**, 95 % bootstrap interval [-12, +6] over the 9 doors -> no clear difference (the interval includes 0).

| unseen door | fixed | randomized |
|---|:-:|:-:|
| deep_latch | 77 % | 63 % |
| handle_high | 30 % | 57 % |
| handle_low | 0 % | 0 % |
| hard_combo | 13 % | 13 % |
| heavy_slippery | 77 % | 77 % |
| shifted_side | 27 % | 13 % |
| stiff_handle | 77 % | 77 % |
| stiff_hinge | 30 % | 20 % |
| turned_door | 37 % | 13 % |

## Q2a - action space of the hand (handle stage, randomized training, trained FROM SCRATCH - no cloning, so the learning itself is compared): design = index + thumb independent, middle/ring/little coupled (18 actions); synergy = finger curls (12); full = every joint (24)

| actions | seeds | unseen success | env steps to 80 % eval success |
|---|:-:|:-:|:-:|
| design | 3 | 93 % | 170 k |

## Q2b - three chained policies vs one policy for the whole sequence (randomized training; both cloned from the same hand-written controller with the same budget)

(needs the chained runs and the 'full' baseline)
