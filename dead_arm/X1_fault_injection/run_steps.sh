#!/usr/bin/env bash
# X1 fault injection, step by step (each step is its own process).
# Needs the T1 package in $T1_DIR (default: ../../data/t1); outputs go to ../results.
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
T1=${T1_DIR:-../../data/t1}
for step in enc_arctic enc_minilm dense rerank table; do
  echo "$(date -Is) step $step"
  OMP_NUM_THREADS=2 "$PY" -I x1_steps.py "$step" "$T1" ../results
done
"$PY" -I x1_analyze.py
"$PY" -I x1_extra.py
