# I01 / X2: the dead-arm study replicated on a public Arabic collection (2026-10-28)

**Data.** Qur'an QA 2023 Task A (Malhas, Mansour and Elsayed, ArabicNLP 2023).
- **Collection:** 1,266 passages, the Thematic Qur'an Passage Collection.
- **Questions:** 213 answer-bearing questions in Modern Standard Arabic. Of 251, 37 have no answer by design and one (504) has no relevance judgments.
- **Licence:** CC BY-NC-ND 4.0. Used locally for non-commercial research, not redistributed; checksums in `~/research/rag-papers-2026-10-27/external/quran_qa_2023/`.

**Method.** Predictions were registered first (`PREDICTIONS.md`); the run is `x2_run.py`, with the GPU thermal guard (76 s).
- Encoders and settings as in X1.
- BM25 is this study's own implementation, not the deployed service's.
- RRF (k = 60) at list lengths 48 and 200.

**Positive controls.**
- **C1.** On the dev questions, our BM25 matches the organisers' BM25 run exactly at R@10 (9 of 21, 42.9%). At R@20 it is higher, 57.1%; the organisers' run apparently lists only 10 passages per question.
- **C2.** The working dense arm reaches R@10 68.1%, against 41.3% for BM25.

**Results** (`results/x2_results.json`; recall in %):

| Condition | Dense R@10 | Fused R@10 (N = 48) | Fused R@20 (N = 48) | Fused R@20 (N = 200) |
|---|---|---|---|---|
| Working (arctic) | 68.1 | 64.3 | 76.1 | 70.9 |
| MiniLM | 25.8 | 40.8 | 52.6 | 44.1 |
| Collapsed tokenizer | 0.5 | 31.9 | 40.8 | 36.6 |
| Random vectors (3 seeds) | 5.2-7.0 | 33.3-36.6 | 43.2-45.1 | 39.0-42.7 |
| BM25 alone | n/a | 41.3 | 49.8 | 49.8 |

BM25 alone: R@5 31.9.

**Predictions.**
1. **Halving at N = 48 (agreement of at least 95%):**
   - collapsed tokenizer: 100% at k = 10 and 99.5% at k = 20; **holds**;
   - random vectors: 94.8% and 94.4%; **narrowly fails**. With 1,266 passages, a random arm hits some gold passages by chance (dense R@10 5-7%) and shares about 2 passages with BM25's list.
   - MiniLM: 83.6% and 85.9%; **fails**. MiniLM was registered as a dead arm, but after the run it turned out to reach a dense R@10 of 25.8% on this collection, against 1.8% on the main study's classical commentaries. That explanation was found after the run, not registered.
2. **Degradation at N = 200: holds** for every dead arm. Agreement drops to 89.7-94.4% for the collapsed tokenizer and 77.9-87.3% for random vectors.
3. **Gap decomposition (N = 48):**
   - collapsed tokenizer: observed 32.4 points at R@10 against predicted 32.4, and 35.2 against 34.7 at R@20; **holds**;
   - random vectors: 31.0 against 32.4 and 32.9 against 34.7, within 2 points;
   - MiniLM: 23.5 observed against 34.7 predicted at R@20; **fails** (same post-run explanation).
   - The regime is the reverse of the books bank, as predicted: a large net gain of the working arm (+23 to +26 points) and a smaller halving cost (8-9 points).
4. **A 20-question labelled check at the dense stage:**
   - collapsed tokenizer 100%, random vectors 99%: **holds**;
   - MiniLM 74%, and 94% at 30 questions; **fails** for MiniLM (same post-run explanation as above).

**Detection at the fused stage (R@20).** Because the working arm's gain is large here, faults show with fewer questions than on the main study's generated bank:
- collapsed tokenizer: 81% at 20 questions, 99% at 30;
- random vectors: 65% at 20, 94% at 30;
- MiniLM: 60% at 30, 95% at 60.

**Overlap monitor (dense and BM25 top-48, mean per query).** Working arm 8.6; collapsed tokenizer 0.8; random 1.8; MiniLM 7.2. The monitor separates the dead arms but not a weak live one.

**What this adds to I01.**
- On a second, public, Modern Standard Arabic collection where the dense arm dominates:
  - the mechanism held for the collapsed tokenizer, with the gap predicted from BM25 ranks plus the working arm's gain;
  - it narrowly missed the registered 95% for random vectors (94.4-94.8%);
  - it failed for MiniLM. Found after the run: MiniLM is not dead on this collection, so whether an arm is "dead" depends on the corpus.
- The limits of the rule are now measured: long lists on a small collection, and arms that are weak rather than dead.
