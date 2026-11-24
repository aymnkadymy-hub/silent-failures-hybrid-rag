# Silent Failures in Hybrid Retrieval-Augmented Generation: code, results and data

Code, stored results and data descriptions for the paper

> Khudaier, A. H., & Yousef, A. K. *Silent Failures in Hybrid Retrieval-Augmented Generation: Cost, Detection, Localization and Containment.* Submitted to *Advances in Distributed Computing and Artificial Intelligence Journal* (ADCAIJ).

Department of Artificial Intelligence Engineering, Al-Safwa University, Karbala, Iraq.

This repository is public so that readers, editors and reviewers can check the results of the paper. All rights reserved.

## What is here

| Folder | Contents |
|---|---|
| `dead_arm/X1_fault_injection/` | Fault injection into the dense arm on the frozen Arabic package T1: encoding, dense lists, reranking, analysis (`x1_steps.py`, `x1_analyze.py`, `x1_extra.py`, `run_steps.sh`) |
| `dead_arm/X2_quran_qa_replication/` | The replication on Qur'an QA 2023, with the predictions written before the run (`PREDICTIONS.md`, `x2_run.py`) |
| `dead_arm/linking/` | Weighted RRF, the protected merge and the worst-case band on T1 (`extra.py`) |
| `dead_arm/results/` | Stored outputs of X1, X2 and the linking runs, including the stored dense lists and query vectors of X1 |
| `controller/src/` | The controller (CIFC, its query-scoped variant CIFC-Q, the baselines), the testbed builders and the replay harness |
| `controller/results/` | Stored outputs of every controller run: per-scenario aggregates and per-stream rows for T1 to T4, certificates, separability, registered hypotheses, frozen constants |
| `controller/run_all.sh` | The order in which the controller study was run |
| `figures/` | Scripts that draw Figures 1 to 4 and Figure S1 from the stored results |
| `verification/` | Independent recomputations: Proposition 1 and the merge guarantee by simulation, the Clopper-Pearson bounds, the ArguAna damage table, the per-question halving check |
| `data/` | Hashes and sources of the data that are not redistributed here (see `data/README.md`) |

## Where each result of the paper comes from

| In the paper | Stored result | Produced by |
|---|---|---|
| Proposition 1 and the protected-merge bound (Section 4) | simulation output of `verification/v2_math.py` | `verification/v2_math.py` |
| Table 4, hit rate at each stage on T1 | `dead_arm/results/x1_summary.txt`, `per_question.json` | `dead_arm/X1_fault_injection/x1_analyze.py` |
| Figure 2, halving rule; Section 7.2 | `dead_arm/results/x1_extra.json` (`halving`), `x2_results.json` | `x1_extra.py`, `x2_run.py`; drawn by `figures/make_dead_arm_figs.py` |
| Figure 3, detection probability; Section 7.3 | `dead_arm/results/x1_detect.json`, `x1_extra.json` (`labelled`) | `x1_analyze.py`, `x1_extra.py` |
| Table S9, label-free overlap monitor | `dead_arm/results/x1_extra.json` (`monitor`), `x1_probes.json` | `x1_extra.py` |
| Table 5, Qur'an QA 2023 replication | `dead_arm/results/x2_results.json`, `x2_per_question.json` | `dead_arm/X2_quran_qa_replication/x2_run.py` |
| Table S10 and Section 7.5, permanent protection | `dead_arm/results/extra.json` | `dead_arm/linking/extra.py` |
| Table 6, controller on Qur'an QA (T2) | `controller/results/tb_quran/test_pool_*.json`, `test_rerank_*.json` | `controller/src/tb_run.py` |
| Table 7, localization, exposure and cost | `controller/results/tb_quran/`, `distinguish_*.json` | `tb_run.py`, `distinguish.py` |
| Figure S1, cost against exposure | `controller/results/tb_quran/` | `figures/make_result_figs.py fig3` |
| Figure 4 and Section 7.8, protected-merge certificate | `controller/results/t1_certify.json`, `tb_*/certify.json` | `t1_run.py`, `tb_run.py`; drawn by `make_result_figs.py fig2`; recomputed by `verification/v3_clopper.py` |
| Table S1, SciFact and NFCorpus | `controller/results/tb_scifact/`, `tb_nfcorpus/` | `tb_run.py` |
| Table S2, registered hypotheses | `controller/results/hypotheses.json`, `hypotheses.md`, `v2/hypotheses_v2.json` | `hypotheses.py`, `hypotheses_v2.py` |
| Table 8 and Tables S3 to S7, second phase | `controller/results/v2/` | `tb_run.py` with `POLSET=v2`; Table 8 recomputed by `verification/v1_arguana_damage.py` |
| Table S8 and Note S1, probes under the INT8 reranker | `controller/results/v2/rprobe_tolerance.json`, `int8_probe_noise.json` | `rprobe_tolerance.py`, `int8_probe_noise.py` |
| Probe noise on a GPU (Section 7.10) | `controller/results/tb_*/probe_noise.json` | `probe_noise.py` |
| Constants fixed before the confirmatory runs | `controller/results/frozen_params.json` | |

## Reproducing

### Environment

GPU steps (encoding, reranking, testbed builds) were run with Python 3.12, PyTorch 2.14 (CUDA 13), Transformers 5.17 and Tokenizers 0.23. Replays and analysis were run with Python 3.14, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. See `requirements.txt`.

Models, from the Hugging Face Hub (the exact revisions used are given; set the environment variables to a local copy if needed):

| Variable | Model | Revision |
|---|---|---|
| `ARCTIC_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` |
| `MINILM_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| `MMARCO_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` |

### Data

Place the data as described in `data/README.md`. Every path can be changed with an environment variable: `T1_DIR`, `QURAN_QA_DIR`, `BEIR_DIR`, `RUNS_DIR`, `X1_RESULTS_DIR`, `BAKEOFF_DIR`, `LINK_RESULTS_DIR`, `INT8_RUNS_DIR`.

### Without any data: checks from the stored results

These run from the repository alone:

```bash
python3 verification/v2_math.py              # Proposition 1 and the merge guarantee, by simulation
python3 verification/v3_clopper.py           # the certificate bounds, recomputed
python3 verification/v1_arguana_damage.py    # the ArguAna damage table, recomputed from per-stream rows
(cd controller/src && python3 hypotheses.py) # the registered-hypothesis table, recomputed
(cd figures && python3 make_result_figs.py fig2 && python3 make_result_figs.py fig3 && python3 fig1_architecture.py \
   && python3 make_dead_arm_figs.py ../dead_arm/results ../dead_arm/results .)
```

The figure scripts reproduce Figures 1 to 4 and Figure S1 of the paper pixel for pixel when Times New Roman is installed.

### With the data: the full runs

```bash
bash dead_arm/X1_fault_injection/run_steps.sh        # needs T1
python3 dead_arm/X2_quran_qa_replication/x2_run.py    # needs Qur'an QA 2023
python3 dead_arm/linking/extra.py                     # needs T1
bash controller/run_all.sh                            # needs T1, Qur'an QA 2023 and BEIR
```

Replays are seeded; the number of streams per scenario is set by `SEEDS` (100 on Qur'an QA, 50 on the other testbeds, as in the paper).
