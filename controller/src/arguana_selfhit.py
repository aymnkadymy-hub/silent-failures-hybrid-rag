"""A17: ArguAna's corpus holds each query's own argument under the query's id (1,298 of 1,406 queries). BEIR's
evaluation drops a retrieved passage whose id equals the query id (ignore_identical_ids). We apply the same rule at
retrieval time: the query's own passage is removed from every stored lexical and dense list (top 200 kept, so the
top 48 used downstream stays full). Writes build.npz in place after saving build_with_selfhits.npz."""
import os
import json, os, shutil, numpy as np
D = os.path.join(os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results")), "tb_arguana")
C = os.path.join(os.environ.get("BEIR_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/beir")), "arguana/corpus.jsonl")
src = os.path.join(D, "build.npz"); bak = os.path.join(D, "build_with_selfhits.npz")
if not os.path.exists(bak):
    shutil.copy2(src, bak)
z = dict(np.load(bak))
meta = json.load(open(os.path.join(D, "meta.json")))
pidx = {json.loads(l)["_id"]: i for i, l in enumerate(open(C, encoding="utf-8"))}
own = np.array([pidx.get(q["id"], -2) for q in meta["qs"]])
removed = {}
for k in list(z):
    if not (k.startswith("L_") or k.endswith("_ids")):
        continue
    A = z[k]; S = z.get(k[:-4] + "_sc") if k.endswith("_ids") else None
    hit = A == own[:, None]
    removed[k] = int(hit.any(1).sum())
    newA = np.full_like(A, -1); newS = None if S is None else np.full_like(S, -1.0)
    for n in range(len(A)):
        keep = ~hit[n]
        a = A[n][keep]; newA[n, :len(a)] = a
        if S is not None:
            s = S[n][keep]; newS[n, :len(s)] = s
    z[k] = newA
    if S is not None:
        z[k[:-4] + "_sc"] = newS
np.savez_compressed(src, **z)
json.dump(removed, open(os.path.join(D, "selfhit_removed.json"), "w"), indent=1)
print({k: v for k, v in removed.items() if k in ("L_ok", "D_ok_ids", "D_unk_ids", "D_minilm_ids")})
