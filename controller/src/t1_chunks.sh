#!/usr/bin/env bash
# T1 (frozen classical-Arabic package) test: one scenario per process. Usage: t1_chunks.sh pool|rerank
cd "$(dirname "$0")"
MODE=${1:-pool}
LIST=$(python3 - "$MODE" <<'PYEOF'
import json, os, sys
import t1_run
T1 = os.environ.get("T1_DIR", os.path.join(os.getcwd(), "../../data/t1"))
q = json.load(open(os.path.join(T1, "questions.json"), encoding="utf-8"))
print(" ".join(k for k in t1_run.scenarios(q) if sys.argv[1] == "rerank" or "R_" not in k))
PYEOF
)
EXTRA=""; [ "$MODE" = rerank ] && EXTRA=rerank
for s in $LIST; do
  OMP_NUM_THREADS=1 SEEDS=${SEEDS:-100} SCEN=$s TAG=$s python3 t1_run.py test $EXTRA
done
