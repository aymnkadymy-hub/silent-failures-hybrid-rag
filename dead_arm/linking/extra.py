import os
"""Linking experiments for the paper (pool level, N5 dev and held-out, bake pool = top-200 lists, dense-first ties).
1) weighted RRF: BM25 weight w, working and dead arms.
2) protected merge (share beta) on top of plain RRF under the working arm and the three dead arms.
3) worst-case band share: first gold lexical rank in (ceil(beta k), k].
Reads the frozen package and the stored dense lists of the fault-injection study; writes dead_arm/results/extra.json."""
import json, math, os, sys
import numpy as np
COP = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1"))
RES = os.environ.get("X1_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/results"))
qs = json.load(open(f"{COP}/questions.json", encoding="utf-8"))
P = lambda x: json.loads(x) if isinstance(x, str) else x
SPL = {s: [i for i, q in enumerate(qs) if P(q['rel']) and q['set'] == 'n5' and q['split'] == s] for s in ('dev', 'heldout')}
print({k: len(v) for k, v in SPL.items()}, sorted({(q['set'], q['split']) for q in qs}))
CONDS = sys.argv[1:] or ['on', 'minilm', 'unk', 'rand1']
D = {}
for c in CONDS:
    z = np.load(f"{RES}/dense_{c}.npz"); D[c] = z['ids']; print(c, z.files, D[c].shape)
K = 60

def fuse(dl, L, w=1.0):
    acc = {}
    for r, p in enumerate(dl):
        p = int(p); acc[p] = acc.get(p, 0) + 1 / (K + 1 + r)
    for r, p in enumerate(L):
        acc[p] = acc.get(p, 0) + w / (K + 1 + r)
    return sorted(acc, key=lambda p: -acc[p])  # stable: dense first on ties

def protected(L, B, beta, N):
    out, used, bi = [], set(), 0
    for i in range(1, N + 1):
        need = math.ceil(beta * i); lead = 0
        while lead < len(L) and L[lead] in used: lead += 1
        if lead < need and lead < len(L): p = L[lead]
        else:
            while bi < len(B) and B[bi] in used: bi += 1
            if bi >= len(B): break
            p = B[bi]
        out.append(p); used.add(p)
    return out

hit = lambda lst, rel, k: any(p in rel for p in lst[:k])
out = {}
for sp, idx in SPL.items():
    n = len(idx); o = out[sp] = {'n': n}
    o['bm25_20'] = sum(hit(P(qs[i]['bm25']), set(P(qs[i]['rel'])), 20) for i in idx)
    o['bm25_10'] = sum(hit(P(qs[i]['bm25']), set(P(qs[i]['rel'])), 10) for i in idx)
    o['bm25_15'] = sum(hit(P(qs[i]['bm25']), set(P(qs[i]['rel'])), 15) for i in idx)
    for c in CONDS:
        for w in (1.0, 1.2, 1.5, 2.0):
            o[f'w|{c}|{w}'] = sum(hit(fuse(D[c][i], P(qs[i]['bm25']), w), set(P(qs[i]['rel'])), 20) for i in idx)
        for beta in (0.5, 0.75, 1.0):
            h = 0
            for i in idx:
                L = P(qs[i]['bm25']); F = fuse(D[c][i], L)
                h += hit(protected(L, F, beta, 48), set(P(qs[i]['rel'])), 20)
            o[f'p|{c}|{beta}'] = h
    for beta in (0.5, 0.625, 0.75, 0.875, 1.0):
        b = 0
        for i in idx:
            L = P(qs[i]['bm25']); rel = set(P(qs[i]['rel']))
            r = next((k for k, p in enumerate(L, 1) if p in rel), None)
            b += r is not None and math.ceil(beta * 20) < r <= 20
        o[f'band|{beta}'] = b
json.dump(out, open(os.path.join(os.environ.get("LINK_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/results")), "extra.json"), "w"), indent=1)
for sp, o in out.items():
    print(sp, {k: round(100 * v / o['n'], 1) for k, v in o.items() if k != 'n'})
