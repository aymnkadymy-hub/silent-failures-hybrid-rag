#!/usr/bin/env bash
# Order in which the controller study was run. Each line can also be run by hand.
# Inputs: public testbeds in $QURAN_QA_DIR and $BEIR_DIR, the T1 package in $T1_DIR (see data/README.md).
# Outputs: $RUNS_DIR (default: controller/results).
set -eu
cd "$(dirname "$0")/src"
# T1, frozen classical-Arabic package (pool level, exploratory)
python3 t1_precompute.py
./t1_chunks.sh pool
# T2 Qur'an QA 2023 and T3 SciFact, NFCorpus: build (GPU), precompute, replay
for tb in quran scifact nfcorpus; do
  python3 -I tb_build.py $tb
  python3 tb_run.py $tb precompute
done
SEEDS=100 ./tb_chunks.sh quran pool
SEEDS=100 ./tb_chunks.sh quran rerank S1_healthy S2_shift_split S3_D_unk S3_D_minilm S3_L_shard S3_L_nofold S3_Q_strip S3_R_random S3_R_inverted S3_R_constant S3_R_trunc
SEEDS=50 ./tb_chunks.sh scifact pool
SEEDS=50 ./tb_chunks.sh nfcorpus pool
# probe noise on a GPU, and the INT8 reranker probes
python3 -I probe_noise.py
python3 rprobe_tolerance.py
# second phase: T4 ArguAna (confirmatory), timing, then T2/T3 again (exploratory)
python3 -I tb_build.py arguana
python3 -I timing_bench.py
python3 tb_run.py arguana precompute
export POLSET=v2 RUNVER=v2 SEEDS=50
./tb_chunks.sh arguana pool
for tb in quran scifact nfcorpus; do ./tb_chunks.sh $tb pool; done
# analysis
for tb in t1 quran scifact nfcorpus; do python3 distinguish.py $tb; done
python3 hypotheses.py
python3 hypotheses_v2.py
python3 analysis.py
