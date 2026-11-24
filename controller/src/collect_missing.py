"""List (question, passage) pairs that the T1 tables need reranker scores for but the stored I01 cache lacks."""
import json, os, sys
import t1_data as T, t1_precompute as P
SETS = {("n5", "dev"), ("n5", "heldout"), ("books_v1", "dev")}
t = T.T1()
need = set()
for wn, w in P.WORLDS.items():
    if w.get("S") or w.get("R"):
        continue                      # same pairs as healthy
    for n, q in enumerate(t.qs):
        if not q["rel"] or (q["set"], q["split"]) not in SETS:
            continue
        L = t.L(n, w); Draw, dsc = t.D_raw(n, w); D = [i for i, x in zip(Draw, dsc) if x >= T.FLOOR]
        heads = [P.served(t, n, w, a, L, D) for a in P.LIST_ACTIONS] + [t.L(n, {})]
        for h in heads:
            for i in h:
                k = f"{n}:{i}"
                if k not in t.rr:
                    need.add(k)
print(len(need))
json.dump(sorted(need), open(os.path.join(T.RUNS, "rerank_need.json"), "w"))
