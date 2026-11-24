# I01 / X2: predictions, registered before any run (2026-10-28 14:51)

**Data.** Qur'an QA 2023 Task A (Malhas, Mansour and Elsayed, ArabicNLP 2023): passage retrieval over the Thematic Qur'an Passage Collection.
- **Source.** 1,266 passages; 251 questions in Modern Standard Arabic (train, dev and test pooled, no training involved), with the official relevance judgments.
- **Exclusions.** The 37 questions with no answer are excluded, which leaves 214.
- **Licence.** CC BY-NC-ND 4.0. Local, non-commercial research use only; nothing is redistributed, and the dataset is cited.

**Arms.**
- **BM25:** this study's own implementation (k1 = 1.2, b = 0.75, the P1 Arabic normaliser). It is not the deployed service's BM25.
- **Dense:** the same encoders and settings as X1 (arctic-embed-l-v2.0: CLS, "query: ", 512 tokens; MiniLM: mean pooling, 128 tokens).
- **Conditions:** on (arctic), minilm, unk (all unknown tokens), rand (3 seeds), off (BM25 only).

**Fusion.** RRF (k = 60) of the dense and BM25 top-N lists, for two list lengths:
- **N = 48:** a random arm is expected to share about 48 x 48 / 1,266 = 1.8 passages with BM25's list;
- **N = 200:** about 200 x 200 / 1,266 = 31.6.

**Positive controls, run first.**
- **C1.** Our BM25 recall on the dev questions is close to the organisers' BM25 run (runs/bigIR_BM25.tsv), at R@10 and R@20.
- **C2.** The working dense arm reaches a dense R@10 well above BM25's, as Salman et al. found dense stronger than BM25 on this collection.

**Predictions.**
1. **The halving rule holds at N = 48.** With a dead arm (minilm, unk, rand), fused R@k equals BM25 R@(k/2) for at least 95% of questions, at k = 10 and 20.
2. **It degrades at N = 200.** Per-question agreement is lower than at N = 48, because a random top-200 list shares many passages with BM25's.
3. **Gap decomposition.** The visible gap between working and dead arm equals the working arm's net gain over BM25 plus the halving cost (BM25 R@k minus BM25 R@k/2). BM25 is weak here, so the net gain should be large and the halving cost small: the opposite regime from the I01 books bank.
4. **Per-arm check.** A labelled check of 20 questions at the dense stage (R@10, exact McNemar, one-sided in direction) detects each dead arm with probability of at least 0.95.

**What is reported anyway.** Detection probability against bank size, limited to the sizes this set allows (30, 60, 120 and 200), and the dense-BM25 top-48 overlap per condition.


## Deviation (recorded 2026-10-28 after the run)
The analysed set has 213 questions, not 214: question 504 has no relevance judgments at all (neither answers nor a no-answer mark).
