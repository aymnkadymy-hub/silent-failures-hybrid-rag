#!/usr/bin/env bash
# One scenario per process on a public testbed. Usage: tb_chunks.sh NAME pool|rerank [SCEN1 SCEN2 ...]
cd "$(dirname "$0")"
NAME=$1; MODE=$2; shift 2
LIST="$*"
if [ -z "$LIST" ]; then
  LIST=$(python3 -c "import tb_run; print(' '.join(k for k in tb_run.scenarios('$NAME') if '$MODE'=='rerank' or 'R_' not in k))")
fi
EXTRA=""; [ "$MODE" = rerank ] && EXTRA=rerank
for s in $LIST; do
  OMP_NUM_THREADS=1 SEEDS=${SEEDS:-100} SCEN=$s TAG=$s python3 tb_run.py $NAME test $EXTRA
done
