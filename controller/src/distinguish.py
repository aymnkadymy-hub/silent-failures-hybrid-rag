"""Distinguishability map of fault causes (report Section 10.x; paper Table).

For every pair of causes (ci, cj) we take the largest total-variation distance, over the available tests, between the
outcome distributions P(Y_a | ci) and P(Y_a | cj). With no test above a small margin, no procedure that uses only
these tests can separate the pair better than the prior allows. Computed twice: with the ordinary tests only
(output-level probes and labelled canaries) and with the interventional tests added (t_srch, t_lexraw, t_col).
Likelihoods are the CIFC model's, estimated on calibration questions from seen fault families.
Usage: distinguish.py t1|quran|scifact|nfcorpus   ->  runs/distinguish_<tb>.json
"""
import json
import math
import os
import sys

import numpy as np

import cafi as CF
import governor as G
import t1_run as R

RUNS = R.RUNS
INTERV = {"t_srch", "t_lexraw", "t_col"}


def tv_canary(p, q, n=30):
    return 0.5 * sum(abs(math.comb(n, k) * (p ** k * (1 - p) ** (n - k) - q ** k * (1 - q) ** (n - k))) for k in range(n + 1))


def main():
    tbn = sys.argv[1]
    if tbn == "t1":
        qs, tab, qcol = R.load()
        bk, can = R.banks(qs)
    else:
        import tb_run
        qs, tab, qcol = tb_run.load(tbn)
        bk, can = tb_run.banks(tbn, qs)
    use_r = "kat|R" in tab["healthy"] and tab["healthy"]["kat|ok_r"].mean() > 0.5 and tbn in ("t1", "quran")
    env = R.build_env(qs, tab, qcol, can, use_r)
    env.collapse = lambda ws, ps: qcol.pair(ws, ps)
    acts = ["N", "Doff", "Loff"]
    M = CF.Model(tab, bk["dev"], can, env.p0, lambda f: R.collapse_rate(env, tab, f, can), use_r, acts)
    causes = [c for c in CF.CAUSES if c not in ("UNK",)]
    tests = M.tests()
    out = {"tests": tests, "causes": causes, "with": {}, "without": {}, "best_test": {}}
    for i, a in enumerate(causes):
        for b in causes[i + 1:]:
            best = {True: (0.0, None), False: (0.0, None)}
            cheap = {True: 0.0, False: 0.0}
            for tn in tests:
                if tn in CF.PROBE_TESTS:
                    d = abs(M.p_flag(a, tn) - M.p_flag(b, tn))
                else:
                    d = tv_canary(M.can[(a, tn)], M.can[(b, tn)])
                for allow in (True, False):
                    if not allow and tn in INTERV:
                        continue
                    if d > best[allow][0]:
                        best[allow] = (d, tn)
                    if tn in CF.PROBE_TESTS:
                        cheap[allow] = max(cheap[allow], d)
            k = f"{a}|{b}"
            out["with"][k] = round(best[True][0], 3)
            out["without"][k] = round(best[False][0], 3)
            out["best_test"][k] = [best[True][1], best[False][1]]
            out.setdefault("cheap_with", {})[k] = round(cheap[True], 3)
            out.setdefault("cheap_without", {})[k] = round(cheap[False], 3)
    only_interv = [k for k in out["with"] if out["with"][k] >= 0.5 and out["without"][k] < 0.5]
    out["separable_only_with_interventions"] = only_interv
    out["inseparable_even_with"] = [k for k in out["with"] if out["with"][k] < 0.5]
    out["cheap_only_with_interventions"] = [k for k in out["with"] if out["cheap_with"][k] >= 0.5 and out["cheap_without"][k] < 0.5]
    out["cheap_inseparable"] = [k for k in out["with"] if out["cheap_with"][k] < 0.5]
    print(tbn, "cheap probes separate only with interventions:", out["cheap_only_with_interventions"])
    print(tbn, "cheap probes cannot separate even with interventions:", out["cheap_inseparable"])
    json.dump(out, open(os.path.join(RUNS, f"distinguish_{tbn}.json"), "w"), indent=1)
    print(tbn, "only with interventions:", only_interv)
    print(tbn, "inseparable (<0.5 TV) even with:", out["inseparable_even_with"])
    print(tbn, "likelihoods:", {f"{c}|{t}": round(M.lik[(c, t)], 2) for c in causes for t in tests
                                if t in CF.PROBE_TESTS and (c, t) in M.lik and M.lik[(c, t)] > 0.05})


if __name__ == "__main__":
    main()
