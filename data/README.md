# Data

Three data sets are used. None is redistributed here; this folder holds their sources and SHA-256 hashes, so that a copy can be checked against the one used in the paper.

## T1: the frozen classical-Arabic retrieval package

A read-only snapshot of the retrieval package of a deployed question-answering service over classical Arabic commentaries (Section 3.2 of the paper): 134,837 passages with their stored vectors, and 2,069 questions with their gold passages and the live BM25 top-200 lists.

| File | Contents |
|---|---|
| `corpus.jsonl` | 134,837 passages |
| `questions.json` | 2,069 questions (N5 bank, books bank and scholarly questions) with gold passages and BM25 lists |
| `arctic-l2.nodiac.f16.npy` | 1,024-dimensional passage vectors of the working arm (Arctic-Embed-L 2.0, diacritics removed), float16 |
| `minilm.plain.f16.npy` | 384-dimensional passage vectors of the near-dead encoder (multilingual MiniLM), float16 |

Hashes: `SHA256SUMS_T1.txt`. Size: about 520 MB.

**Availability.** The passages come from printed editions under copyright and from a deployed service, so the package is not published. It is available from the corresponding author on reasonable request, for research use. Place it in `data/t1/` or set `T1_DIR`.

The query vectors, dense lists and per-question results of the fault-injection study, which are derived from T1, are included in `dead_arm/results/`, so the analyses of Sections 7.1 to 7.5 can be checked without the package.

## Qur'an QA 2023 (T2)

Task A of the Qur'an QA 2023 shared task (Malhas et al., 2023), version 1.2, from https://gitlab.com/bigirqu/quran-qa-2023 (licence CC BY-NC-ND 4.0). Place the files in `data/quran_qa_2023/` with the layout listed in `SHA256SUMS_QURAN_QA_2023.txt`, or set `QURAN_QA_DIR`.

## BEIR: SciFact, NFCorpus and ArguAna (T3 and T4)

Public collections of the BEIR benchmark (Thakur et al., 2021). Place each in `data/beir/<name>/` (`corpus`, `queries`, `qrels`) with the layout listed in `SHA256SUMS_BEIR.txt`, or set `BEIR_DIR`.

## INT8 reranker runs

The batch-arrangement runs of the deployed INT8 reranker come from a separate study and are not included; `controller/results/v2/int8_probe_noise.json` and `rprobe_tolerance.json` hold the values used in the paper (Table S8).
