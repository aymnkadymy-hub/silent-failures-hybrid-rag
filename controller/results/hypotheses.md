| Bed | Status | Hyp. | Policy | Evidence | Verdict |
|---|---|---|---|---|---|
| T1 | exploratory | H1 | G | max Hit@20 loss vs RRF in healthy/shift = 0.06 pts | supported |
| T1 | exploratory | H1 | CIFC_r20 | max Hit@20 loss vs RRF in healthy/shift = 0.00 pts | supported |
| T1 | exploratory | H2/H14 | G | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T1 | exploratory | H2/H14 | CIFC_r20 | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T1 | exploratory | H2/H14 | P3_deadarm | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T1 | exploratory | H3 | G | D_unk: 22 vs P3 55, P0 21; D_minilm: 22 vs P3 55, P0 20; D_rand1: 22 vs P3 54, P0 2; D_offset: 23 vs P3 55, P0 21 | 0/4 faults |
| T1 | exploratory | H3 | CIFC_r20 | D_unk: 11 vs P3 55, P0 21; D_minilm: 11 vs P3 55, P0 20; D_rand1: 10 vs P3 54, P0 2; D_offset: 11 vs P3 55, P0 21 | 0/4 faults |
| T1 | exploratory | H4 | G | L_empty: 17 vs P0 2; L_shard: 45 vs P0 6; L_cache: 17 vs P0 23; Q_strip: 93 vs P0 2; S_shift: 93 vs P0 2 | 0/5 faults |
| T1 | exploratory | H4 | CIFC_r20 | L_empty: 8 vs P0 2; L_shard: 36 vs P0 6; L_cache: 10 vs P0 23; Q_strip: 31 vs P0 2; S_shift: 25 vs P0 2 | 0/5 faults |
| T1 | exploratory | H6 | G | share of seeds recovered within 600: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 1.00, Q_strip 1.00, S_shift 1.00 | 6/6 faults |
| T1 | exploratory | H8 | G_vgate | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 1.00, Q_strip 1.00, S_shift 1.00 | 6/6 faults |
| T1 | exploratory | H8-CIFC | CIFC_r20 | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 1.00, Q_strip 1.00, S_shift 1.00 | 6/6 faults |
| T1 | exploratory | H9 | G_vgate | Hit@20 policy/P0/G: D_unk 70.2/69.1/69.0; D_minilm 70.2/69.2/69.0; D_offset 70.2/69.1/69.0; L_shard 72.0/75.0/71.2; Q_strip 43.0/46.0/36.9; S_shift 43.0/46.0/36.9 | 3/6 faults |
| T1 | exploratory | H11 | G_kat | Hit@20 policy/P0/G: D_unk 70.2/69.1/69.0; D_minilm 70.2/69.2/69.0; D_offset 70.2/69.1/69.0; L_shard 72.0/75.0/71.2; Q_strip 43.0/46.0/36.9; S_shift 43.0/46.0/36.9 | 3/6 faults |
| T1 | exploratory | H10 | G_kat | detect within 20 queries in >= 90% of seeds | 6/6 faults |
| T1 | exploratory | H12 | CIFC_r20 | seen-family accuracy mean 1.00 (min 1.00); unseen correct-or-UNK {'D_ann': 1.0, 'D_corrupt': 0.1, 'D_adv': 1.0, 'L_cache': 1.0} | supported |
| T1 | exploratory | H12 | CIFC_noint | seen-family accuracy mean 0.75 (min 0.00); unseen correct-or-UNK {'D_ann': 0.14, 'D_corrupt': None, 'D_adv': 1.0, 'L_cache': 1.0} | not supported |
| T1 | exploratory | H13 | CIFC_r20 vs G_kat | D_unk exp 0/0 cost 349/1135; D_minilm exp 0/0 cost 352/1135; D_offset exp 0/0 cost 350/1135; L_shard exp 0/0 cost 348/1135; Q_strip exp 0/0 cost 661/1285; S_shift exp 0/0 cost 1161/1285 | 6/6 faults |
| T1 | exploratory | H16 | CIFC_r20 vs CIFC_noint | D_offset acc 1.0/1.0 cost 350/865 exp 0/40; D_unk acc 1.0/1.0 cost 349/417 exp 0/0; D_ann acc 1.0/0.04 cost 774/340 exp 62/909; Q_strip acc 1.0/1.0 cost 661/643 exp 0/0; L_shard acc 1.0/1.0 cost 348/329 exp 0/0 | 0/5 faults |
| T1 | exploratory | H17 | CIFC_grade | Hit@20 graded/CIFC/P0: D_unk 70.2/70.2/69.1; D_minilm 70.2/70.2/69.2; D_rand1 70.2/70.2/71.1; D_offset 70.2/70.2/69.1; D_ann 73.8/70.2/73.9; D_corrupt 75.4/75.3/75.5; D_adv 69.7/69.7/46.0; second clause 2/2 | supported |
| T1 | exploratory | H18 | CIFC_gradec | Hit@20 graded/CIFC/P0: D_ann 73.8/70.2/73.9; D_corrupt 75.4/75.3/75.5; second clause 0/2 | not supported |
| T2 | confirmatory | H1 | G | max Hit@20 loss vs RRF in healthy/shift = 1.67 pts | not supported |
| T2 | confirmatory | H1 | CIFC_r20 | max Hit@20 loss vs RRF in healthy/shift = 0.00 pts | supported |
| T2 | confirmatory | H2/H14 | G | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T2 | confirmatory | H2/H14 | CIFC_r20 | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T2 | confirmatory | H2/H14 | P3_deadarm | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T2 | confirmatory | H3 | G | D_unk: 17 vs P3 93, P0 46; D_minilm: 10 vs P3 84, P0 -9; D_rand: 16 vs P3 89, P0 -9; D_offset: 39 vs P3 112, P0 -21 | 0/4 faults |
| T2 | confirmatory | H3 | CIFC_r20 | D_unk: 5 vs P3 93, P0 46; D_minilm: -2 vs P3 84, P0 -9; D_rand: 4 vs P3 89, P0 -9; D_offset: 29 vs P3 112, P0 -21 | 2/4 faults |
| T2 | confirmatory | H4 | G | L_empty: 2 vs P0 -9; L_shard: 9 vs P0 -9; L_cache: 9 vs P0 28; L_nofold: 4 vs P0 -15; Q_strip: 25 vs P0 286; S_shift: 173 vs P0 16 | 1/6 faults |
| T2 | confirmatory | H4 | CIFC_r20 | L_empty: -8 vs P0 -9; L_shard: -1 vs P0 -9; L_cache: -1 vs P0 28; L_nofold: -7 vs P0 -15; Q_strip: 15 vs P0 286; S_shift: 154 vs P0 16 | 5/6 faults |
| T2 | confirmatory | H6 | G | share of seeds recovered within 600: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 0.00, L_nofold 0.00, Q_strip 1.00, S_shift 1.00 | 5/7 faults |
| T2 | confirmatory | H8 | G_vgate | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 0.00, L_nofold 0.00, Q_strip 1.00, S_shift 1.00 | 5/7 faults |
| T2 | confirmatory | H8-CIFC | CIFC_r20 | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 0.00, L_nofold 0.00, Q_strip 1.00, S_shift 1.00 | 5/7 faults |
| T2 | confirmatory | H9 | G_vgate | Hit@20 policy/P0/G: D_unk 63.3/59.9/62.8; D_minilm 63.3/64.8/62.9; D_offset 63.3/69.1/63.1; L_shard 73.7/75.3/73.5; L_nofold 73.7/75.2/73.4; Q_strip 70.9/45.1/71.2; S_shift 42.9/58.0/42.3 | 1/7 faults |
| T2 | confirmatory | H11 | G_kat | Hit@20 policy/P0/G: D_unk 63.3/59.9/62.8; D_minilm 63.3/64.8/62.9; D_offset 63.3/69.1/63.1; L_shard 73.6/75.3/73.5; L_nofold 73.5/75.2/73.4; Q_strip 71.5/45.1/71.2; S_shift 42.6/58.0/42.3 | 2/7 faults |
| T2 | confirmatory | H10 | G_kat | detect within 20 queries in >= 90% of seeds | 7/7 faults |
| T2 | confirmatory | H12 | CIFC_r20 | seen-family accuracy mean 1.00 (min 1.00); unseen correct-or-UNK {'D_ann': 1.0, 'D_corrupt': 0.99, 'D_adv': 1.0, 'L_cache': 1.0} | supported |
| T2 | confirmatory | H12 | CIFC_noint | seen-family accuracy mean 0.90 (min 0.00); unseen correct-or-UNK {'D_ann': 0.0, 'D_corrupt': 0.0, 'D_adv': 0.0, 'L_cache': 1.0} | not supported |
| T2 | confirmatory | H13 | CIFC_r20 vs G_kat | D_unk exp 0/0 cost 325/1163; D_minilm exp 0/0 cost 322/1163; D_offset exp 0/0 cost 324/1163; L_shard exp 0/0 cost 380/1926; L_nofold exp 0/0 cost 410/2083; Q_strip exp 0/0 cost 636/1313; S_shift exp 0/0 cost 1132/1313 | 7/7 faults |
| T2 | confirmatory | H16 | CIFC_r20 vs CIFC_noint | D_offset acc 1.0/1.0 cost 324/908 exp 0/0; D_unk acc 1.0/1.0 cost 325/463 exp 0/0; D_ann acc 1.0/0.0 cost 338/332 exp 168/901; Q_strip acc 1.0/1.0 cost 636/680 exp 0/0; L_shard acc 1.0/1.0 cost 380/428 exp 0/0 | 1/5 faults |
| T2 | confirmatory | H17 | CIFC_grade | Hit@20 graded/CIFC/P0: D_unk 61.6/64.1/59.9; D_minilm 64.9/64.1/64.8; D_rand 64.5/64.1/65.4; D_offset 68.8/64.1/69.1; D_noise 69.3/64.4/69.8; D_ann 69.3/65.0/69.7; D_corrupt 72.9/69.0/73.4; D_adv 61.0/63.8/45.1; second clause 2/2 | not supported |
| T2 | exploratory | H18 | CIFC_gradec | Hit@20 graded/CIFC/P0: D_unk 64.1/64.1/59.9; D_minilm 64.9/64.1/64.8; D_rand 65.4/64.1/65.4; D_offset 68.8/64.1/69.1; D_noise 69.3/64.4/69.8; D_ann 69.3/65.0/69.7; D_corrupt 72.9/69.0/73.4; D_adv 61.0/63.8/45.1; second clause 0/3 | not supported |
| T3a | confirmatory | H1 | G | max Hit@20 loss vs RRF in healthy/shift = 0.17 pts | supported |
| T3a | confirmatory | H1 | CIFC_r20 | max Hit@20 loss vs RRF in healthy/shift = 0.00 pts | supported |
| T3a | confirmatory | H2/H14 | G | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3a | confirmatory | H2/H14 | CIFC_r20 | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3a | confirmatory | H2/H14 | P3_deadarm | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3a | confirmatory | H3 | G | D_unk: 9 vs P3 28, P0 6; D_minilm: 13 vs P3 2, P0 2; D_rand: 9 vs P3 28, P0 6; D_offset: 10 vs P3 29, P0 20 | 0/4 faults |
| T3a | confirmatory | H3 | CIFC_r20 | D_unk: 9 vs P3 28, P0 6; D_minilm: 18 vs P3 2, P0 2; D_rand: 8 vs P3 28, P0 6; D_offset: 9 vs P3 29, P0 20 | 0/4 faults |
| T3a | confirmatory | H4 | G | L_empty: 9 vs P0 6; L_shard: 22 vs P0 16; L_cache: 10 vs P0 20; L_nofold: 16 vs P0 14; Q_strip: 36 vs P0 353; S_shift: 43 vs P0 6 | 1/6 faults |
| T3a | confirmatory | H4 | CIFC_r20 | L_empty: 7 vs P0 6; L_shard: 19 vs P0 16; L_cache: 7 vs P0 20; L_nofold: 18 vs P0 14; Q_strip: 29 vs P0 353; S_shift: 21 vs P0 6 | 1/6 faults |
| T3a | confirmatory | H6 | G | share of seeds recovered within 600: D_unk 1.00, D_minilm 0.00, D_offset 1.00, L_shard 1.00, L_nofold 0.00, Q_strip 1.00, S_shift 1.00 | 5/7 faults |
| T3a | confirmatory | H8 | G_vgate | share of seeds recovered within 200: D_unk 1.00, D_minilm 0.00, D_offset 1.00, L_shard 1.00, L_nofold 0.00, Q_strip 1.00, S_shift 1.00 | 5/7 faults |
| T3a | confirmatory | H8-CIFC | CIFC_r20 | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 1.00, L_nofold 1.00, Q_strip 1.00, S_shift 1.00 | 7/7 faults |
| T3a | confirmatory | H9 | G_vgate | Hit@20 policy/P0/G: D_unk 87.8/88.1/87.8; D_minilm 88.6/89.7/88.6; D_offset 87.8/86.7/87.7; L_shard 88.9/89.2/88.6; L_nofold 89.3/89.6/89.3; Q_strip 87.4/55.7/87.3; S_shift 52.7/54.9/51.2 | 2/7 faults |
| T3a | confirmatory | H11 | G_kat | Hit@20 policy/P0/G: D_unk 87.9/88.1/87.8; D_minilm 87.9/89.7/88.6; D_offset 87.9/86.7/87.7; L_shard 88.9/89.2/88.6; L_nofold 88.9/89.6/89.3; Q_strip 88.1/55.7/87.3; S_shift 52.7/54.9/51.2 | 2/7 faults |
| T3a | confirmatory | H10 | G_kat | detect within 20 queries in >= 90% of seeds | 7/7 faults |
| T3a | confirmatory | H12 | CIFC_r20 | seen-family accuracy mean 1.00 (min 1.00); unseen correct-or-UNK {'D_ann': 0.98, 'D_corrupt': 1.0, 'D_adv': 1.0, 'L_cache': 1.0} | supported |
| T3a | confirmatory | H12 | CIFC_noint | seen-family accuracy mean 0.90 (min 0.00); unseen correct-or-UNK {'D_ann': 0.92, 'D_corrupt': 0.02, 'D_adv': 1.0, 'L_cache': 1.0} | not supported |
| T3a | confirmatory | H13 | CIFC_r20 vs G_kat | D_unk exp 0/0 cost 419/1105; D_minilm exp 32/0 cost 284/1105; D_offset exp 0/0 cost 322/1105; L_shard exp 0/0 cost 746/1135; L_nofold exp 142/0 cost 274/1105; Q_strip exp 0/0 cost 633/1255; S_shift exp 0/0 cost 1131/1255 | 5/7 faults |
| T3a | confirmatory | H16 | CIFC_r20 vs CIFC_noint | D_offset acc 1.0/1.0 cost 322/839 exp 0/0; D_unk acc 1.0/1.0 cost 419/387 exp 0/0; D_ann acc 0.98/0.0 cost 752/774 exp 137/151; Q_strip acc 1.0/1.0 cost 633/613 exp 0/0; L_shard acc 1.0/1.0 cost 746/679 exp 0/0 | 1/5 faults |
| T3a | confirmatory | H17 | CIFC_grade | Hit@20 graded/CIFC/P0: D_unk 87.9/87.9/88.1; D_minilm 89.6/88.0/89.7; D_rand 87.9/87.9/88.1; D_offset 87.9/87.9/86.7; D_noise 89.6/88.1/89.8; D_ann 88.4/87.9/88.5; D_corrupt 89.3/88.0/89.4; D_adv 87.2/87.2/54.3; second clause 2/2 | supported |
| T3a | confirmatory | H18 | CIFC_gradec | Hit@20 graded/CIFC/P0: D_unk 88.1/87.9/88.1; D_minilm 89.6/88.0/89.7; D_rand 88.1/87.9/88.1; D_offset 87.9/87.9/86.7; D_noise 89.6/88.1/89.8; D_ann 88.8/87.9/88.5; D_corrupt 89.3/88.0/89.4; D_adv 87.2/87.2/54.3; second clause 1/3 | not supported |
| T3b | confirmatory | H1 | G | max Hit@20 loss vs RRF in healthy/shift = 4.07 pts | not supported |
| T3b | confirmatory | H1 | CIFC_r20 | max Hit@20 loss vs RRF in healthy/shift = 0.00 pts | supported |
| T3b | confirmatory | H2/H14 | G | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3b | confirmatory | H2/H14 | CIFC_r20 | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3b | confirmatory | H2/H14 | P3_deadarm | share of healthy/shift streams with a false fallback = 0.00 | supported |
| T3b | confirmatory | H3 | G | D_unk: 3 vs P3 16, P0 1; D_minilm: 7 vs P3 3, P0 3; D_rand: 3 vs P3 16, P0 1; D_offset: 7 vs P3 -1, P0 -1 | 0/4 faults |
| T3b | confirmatory | H3 | CIFC_r20 | D_unk: 3 vs P3 16, P0 1; D_minilm: 11 vs P3 3, P0 3; D_rand: 3 vs P3 16, P0 1; D_offset: 14 vs P3 -1, P0 -1 | 0/4 faults |
| T3b | confirmatory | H4 | G | L_empty: 9 vs P0 1; L_shard: 4 vs P0 3; L_cache: 17 vs P0 5; L_nofold: 53 vs P0 25; Q_strip: 26 vs P0 300; S_shift: 1 vs P0 -1 | 1/6 faults |
| T3b | confirmatory | H4 | CIFC_r20 | L_empty: 7 vs P0 1; L_shard: 49 vs P0 3; L_cache: 16 vs P0 5; L_nofold: 52 vs P0 25; Q_strip: 20 vs P0 300; S_shift: 75 vs P0 -1 | 1/6 faults |
| T3b | confirmatory | H6 | G | share of seeds recovered within 600: D_unk 1.00, D_minilm 0.00, D_offset 0.00, L_shard 0.00, L_nofold 1.00, Q_strip 1.00, S_shift 0.00 | 3/7 faults |
| T3b | confirmatory | H8 | G_vgate | share of seeds recovered within 200: D_unk 1.00, D_minilm 0.00, D_offset 0.00, L_shard 0.00, L_nofold 1.00, Q_strip 1.00, S_shift 0.00 | 3/7 faults |
| T3b | confirmatory | H8-CIFC | CIFC_r20 | share of seeds recovered within 200: D_unk 1.00, D_minilm 1.00, D_offset 1.00, L_shard 1.00, L_nofold 1.00, Q_strip 1.00, S_shift 1.00 | 7/7 faults |
| T3b | confirmatory | H9 | G_vgate | Hit@20 policy/P0/G: D_unk 73.7/73.9/73.7; D_minilm 74.2/74.6/74.1; D_offset 74.9/75.3/74.6; L_shard 74.4/74.9/74.8; L_nofold 70.4/73.1/70.3; Q_strip 73.0/45.6/73.0; S_shift 70.8/71.1/71.0 | 0/7 faults |
| T3b | confirmatory | H11 | G_kat | Hit@20 policy/P0/G: D_unk 73.7/73.9/73.7; D_minilm 73.7/74.6/74.1; D_offset 73.7/75.3/74.6; L_shard 70.4/74.9/74.8; L_nofold 70.4/73.1/70.3; Q_strip 73.6/45.6/73.0; S_shift 55.4/71.1/71.0 | 1/7 faults |
| T3b | confirmatory | H10 | G_kat | detect within 20 queries in >= 90% of seeds | 7/7 faults |
| T3b | confirmatory | H12 | CIFC_r20 | seen-family accuracy mean 1.00 (min 1.00); unseen correct-or-UNK {'D_ann': 0.98, 'D_corrupt': 0.06, 'D_adv': 1.0, 'L_cache': 1.0} | supported |
| T3b | confirmatory | H12 | CIFC_noint | seen-family accuracy mean 0.90 (min 0.00); unseen correct-or-UNK {'D_ann': 0.96, 'D_corrupt': 0.9, 'D_adv': 0.52, 'L_cache': 0.98} | not supported |
| T3b | confirmatory | H13 | CIFC_r20 vs G_kat | D_unk exp 0/0 cost 319/1105; D_minilm exp 37/0 cost 281/1105; D_offset exp 66/0 cost 280/1105; L_shard exp 40/0 cost 319/1105; L_nofold exp 0/0 cost 718/1465; Q_strip exp 0/0 cost 601/1225; S_shift exp 60/0 cost 429/1345 | 3/7 faults |
| T3b | confirmatory | H16 | CIFC_r20 vs CIFC_noint | D_offset acc 1.0/1.0 cost 280/459 exp 66/141; D_unk acc 1.0/1.0 cost 319/393 exp 0/0; D_ann acc 0.98/0.96 cost 275/448 exp 581/418; Q_strip acc 1.0/1.0 cost 601/583 exp 0/0; L_shard acc 1.0/1.0 cost 319/295 exp 40/40 | 1/5 faults |
| T3b | confirmatory | H17 | CIFC_grade | Hit@20 graded/CIFC/P0: D_unk 73.9/73.7/73.9; D_minilm 74.5/73.8/74.6; D_rand 73.9/73.7/73.9; D_offset 75.2/73.8/75.3; D_noise 74.7/73.8/74.9; D_ann 75.0/74.5/75.1; D_corrupt 75.6/75.5/75.6; D_adv 57.5/72.9/55.4; second clause 2/2 | not supported |
| T3b | confirmatory | H18 | CIFC_gradec | Hit@20 graded/CIFC/P0: D_unk 73.9/73.7/73.9; D_minilm 74.5/73.8/74.6; D_rand 73.9/73.7/73.9; D_offset 75.2/73.8/75.3; D_noise 74.7/73.8/74.9; D_ann 75.0/74.5/75.1; D_corrupt 75.6/75.5/75.6; D_adv 57.5/72.9/55.4; second clause 0/3 | not supported |