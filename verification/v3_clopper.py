"""Clopper-Pearson upper bounds of certify.json recomputed with an independent implementation (exact binomial tail,
bisection; no scipy), and the smallest certified beta per (testbed, risk level)."""
import os
import json, math, os, glob
def tail_le(x, n, p):  # P[Bin(n,p) <= x]
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(x + 1))
def ucb(x, n, delta=0.05):
    """All testbeds use delta/5 = 0.01 per beta (Bonferroni over the five beta values, as in t1_run.certify)."""
    if x >= n: return 1.0
    lo, hi = x / n, 1.0
    for _ in range(100):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if tail_le(x, n, m) > delta else (lo, m)
    return hi
R = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller/results"))
fs = [f'{R}/t1_certify.json'] + sorted(glob.glob(f'{R}/tb_*/certify.json'))
worst = 0
for f in fs:
    d = json.load(open(f)); tb = f.split('/')[-2] if 'tb_' in f else 't1'
    dl = 0.05 / 5
    for bank, bv in d['banks'].items():
        for b, v in bv.items():
            u = ucb(v['loss'], v['n'], dl); worst = max(worst, abs(u - v['ucb']))
        cert = {e: min([float(b) for b, v in bv.items() if ucb(v['loss'], v['n'], dl) <= e], default=None) for e in (0.01, 0.02, 0.05)}
        print(f'{tb:10} {bank:5} n={list(bv.values())[0]["n"]:5}  smallest beta certified at eps 1%/2%/5%: {cert}')
print(f'max |recomputed - stored| upper bound: {worst:.2e}')
