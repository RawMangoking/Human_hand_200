# Results (success rate, mean ± sd over seeds)

| task | algo | training | actions | seeds | nominal | in-distribution | UNSEEN |
|---|---|---|---|:-:|:-:|:-:|:-:|
| door | ars | fixed | synergy | 3 | 100 ± 0 % | 90 ± 0 % | 89 ± 0 % |
| door | ars | random | synergy | 3 | 100 ± 0 % | 90 ± 0 % | 88 ± 2 % |
| full | ars | random | synergy | 3 | 70 ± 29 % | 66 ± 23 % | 56 ± 21 % |
| handle | ars | fixed | synergy | 3 | 80 ± 8 % | 78 ± 5 % | 69 ± 7 % |
| handle | ars | random | full/scratch | 3 | 90 ± 14 % | 92 ± 11 % | 91 ± 12 % |
| handle | ars | random | synergy | 3 | 67 ± 17 % | 67 ± 12 % | 67 ± 16 % |
| handle | ars | random | synergy/scratch | 3 | 90 ± 8 % | 89 ± 6 % | 85 ± 10 % |
| reach | ars | fixed | synergy | 3 | 100 ± 0 % | 100 ± 0 % | 94 ± 5 % |
| reach | ars | random | synergy | 3 | 100 ± 0 % | 99 ± 2 % | 90 ± 3 % |

# Chained three-policy runs (reach -> handle -> door)

| run | door set | doors | reach | handle | door | END-TO-END | time (s) | recoveries |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| chain_ars_fixed_s0 | in_dist | 10 | 100 % | 76 % | 69 % | **69 %** | 8.6 | 0.2 |
| chain_ars_fixed_s0 | nominal | 1 | 100 % | 90 % | 90 % | **90 %** | 9.5 | 0.0 |
| chain_ars_fixed_s0 | unseen | 9 | 100 % | 82 % | 71 % | **71 %** | 8.4 | 0.1 |
| chain_ars_fixed_s1 | in_dist | 10 | 99 % | 79 % | 63 % | **63 %** | 8.7 | 0.2 |
| chain_ars_fixed_s1 | nominal | 1 | 100 % | 70 % | 70 % | **70 %** | 7.7 | 0.3 |
| chain_ars_fixed_s1 | unseen | 9 | 93 % | 61 % | 49 % | **49 %** | 8.2 | 0.4 |
| chain_ars_fixed_s2 | in_dist | 10 | 100 % | 81 % | 72 % | **72 %** | 9.2 | 0.1 |
| chain_ars_fixed_s2 | nominal | 1 | 100 % | 90 % | 80 % | **80 %** | 8.4 | 0.1 |
| chain_ars_fixed_s2 | unseen | 9 | 96 % | 66 % | 58 % | **58 %** | 9.1 | 0.2 |
| chain_ars_random_s0 | in_dist | 10 | 91 % | 50 % | 47 % | **47 %** | 11.4 | 0.1 |
| chain_ars_random_s0 | nominal | 1 | 100 % | 50 % | 50 % | **50 %** | 10.4 | 0.1 |
| chain_ars_random_s0 | unseen | 9 | 87 % | 49 % | 44 % | **44 %** | 10.1 | 0.1 |
| chain_ars_random_s1 | in_dist | 10 | 100 % | 74 % | 68 % | **68 %** | 8.9 | 0.2 |
| chain_ars_random_s1 | nominal | 1 | 100 % | 50 % | 50 % | **50 %** | 7.1 | 0.5 |
| chain_ars_random_s1 | unseen | 9 | 90 % | 59 % | 53 % | **53 %** | 7.6 | 0.3 |
| chain_ars_random_s2 | in_dist | 10 | 100 % | 81 % | 73 % | **73 %** | 10.4 | 0.1 |
| chain_ars_random_s2 | nominal | 1 | 100 % | 60 % | 60 % | **60 %** | 9.3 | 0.4 |
| chain_ars_random_s2 | unseen | 9 | 90 % | 67 % | 61 % | **61 %** | 9.9 | 0.2 |

# Research questions

## Q1 - fixed vs domain-randomized training (three chained policies, end-to-end success)

| training | seeds | nominal | in-distribution | **unseen** |
|---|:-:|:-:|:-:|:-:|
| fixed | 3 | 80 ± 8 % | 68 ± 4 % | 59 ± 9 % |
| random | 3 | 53 ± 5 % | 63 ± 11 % | 53 ± 7 % |

Unseen doors, randomized minus fixed: **-6 percentage points**, 95 % bootstrap interval [-14, +2] over the 9 doors -> no clear difference (the interval includes 0).

| unseen door | fixed | randomized |
|---|:-:|:-:|
| deep_latch | 67 % | 47 % |
| handle_high | 73 % | 80 % |
| handle_low | 0 % | 0 % |
| hard_combo | 43 % | 33 % |
| heavy_slippery | 67 % | 70 % |
| shifted_side | 50 % | 47 % |
| stiff_handle | 80 % | 53 % |
| stiff_hinge | 60 % | 73 % |
| turned_door | 93 % | 73 % |

## Q2a - reduced (synergy) actions vs every joint (handle stage, randomized training, trained FROM SCRATCH - no cloning, so the learning itself is compared)

| actions | seeds | unseen success | env steps to 80 % eval success |
|---|:-:|:-:|:-:|
| synergy | 3 | 85 % | 88 k |
| full | 3 | 91 % | 112 k |

## Q2b - three chained policies vs one policy for the whole sequence (randomized training; both cloned from the same hand-written controller with the same budget)

| approach | nominal | in-distribution | unseen |
|---|:-:|:-:|:-:|
| three chained policies | 53 % | 63 % | 53 % |
| one policy (task full) | 70 % | 66 % | 56 % |
