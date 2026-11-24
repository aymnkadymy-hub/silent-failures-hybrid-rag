#!/usr/bin/env python3
"""I01 / X1 analysis (CPU only): stage recall per dense-arm condition, paired tests against the working arm,
detection probability of each fault by evaluation-bank size, and label-free / labelled per-arm health probes.

Inputs: results/per_question.json, results/probe_material.json, results/Q_*.npy (from x1_steps.py).
Outputs: results/x1_metrics.json, results/x1_detect.json, results/x1_probes.json, results/x1_replication.json,
results/x1_summary.txt
Questions without a gold chunk are excluded. Bootstrap and subsampling resample questions (seed 20261028).
Banks: n5 dev cells (fusha/iraqi x named/open), books_v1 (198), rgold (36 dev), n5 held-out (496, reported separately)."""
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEED = 20261028
CONDS = ["on", "minilm", "unk", "rand1", "rand2", "rand3", "off"]


def bank_of(r):
    if r["set"] == "n5" and r["split"] == "dev":
        return f"n5dev_{r['lang']}_{'named' if r['named'] else 'open'}"
    if r["set"] == "n5" and r["split"] == "heldout":
        return "n5heldout"
    return f"{r['set']}_{r['split']}"


def hit(rank, k):
    if rank == "NA":                         # reranking not run for this condition/bank (scope reduced for heat)
        return None
    return int(rank is not None and rank > 0 and rank <= k)


def mcnemar_p(b, c):
    """exact two-sided binomial test on discordant pairs b (only first correct) and c (only second correct)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


BAKE = os.environ.get("BAKEOFF_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/results/bakeoff"))
CELL = {"n5dev_all": "all", "n5dev_fusha_named": "fusha_named", "n5dev_fusha_open": "fusha_open",
        "n5dev_iraqi_named": "iraqi_named", "n5dev_iraqi_open": "iraqi_open", "books_v1_dev": "set:books_v1",
        "rgold_dev": "set:rgold"}


def replication(metrics):
    """Our recomputation against the cells the round-3 bake-off stored (same package, same pool definition 'bake')."""
    res = json.load(open(os.path.join(BAKE, "results.json")))
    rr = json.load(open(os.path.join(BAKE, "rerank_results.json")))
    out = {}
    for bank, cell in CELL.items():
        for cond, enc, var in (("on", "arctic-l2", "nodiac/raw"), ("minilm", "minilm", "plain/raw")):
            c = res[enc][var]["cells"][cell]
            m = metrics.get(f"{cond}|bake|{bank}", {})
            out[f"{cond}|{bank}"] = {"n_stored": c["n"], "n_ours": m.get("n"),
                                     "dense_R10": [c["dense"]["R@10"], m.get("dense_R10", [None])[0]],
                                     "fused_R20": [c["fused"]["R@20"], m.get("pool_R20", [None])[0]],
                                     "fused_R48": [c["fused"]["R@48"], m.get("pool_R48", [None])[0]]}
        c = res["arctic-l2"]["nodiac/raw"]["cells"][cell]
        m = metrics.get(f"off|bake|{bank}", {})
        out[f"off|{bank}"] = {"bm25_R20": [c["bm25"]["R@20"], m.get("pool_R20", [None])[0]],
                              "bm25_R48": [c["bm25"]["R@48"], m.get("pool_R48", [None])[0]]}
        c = rr["mmarco"]["cells"][cell]
        m = metrics.get(f"on|bake|{bank}", {})
        out[f"on_rerank|{bank}"] = {"rerank_R1": [c["R@1"], m.get("rerank_R1", [None])[0]],
                                    "rerank_R8": [c["R@8"], m.get("rerank_R8", [None])[0]]}
    worst = max(abs(a - b) for v in out.values() for k, ab in v.items() if isinstance(ab, list)
                for a, b in [ab] if a is not None and b is not None)
    return {"cells": out, "max_abs_diff": round(worst, 4),
            "note": "pairs are [stored bake-off value, this study]; stored values are rounded to 3 decimals"}


def prod_threshold_probe():
    """How much of each dense arm survives the live server's cosine floor (0.30 on the top-48),
    and how often the resulting pool head is identical to the BM25-only head."""
    sys.path.insert(0, HERE)
    import x1_steps as X
    copies = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1"))
    qs, P = X.pools(copies, RES)
    keep = [n for n, q in enumerate(qs) if q["rel"]]
    out = {}
    for c in X.CONDS:
        sc = np.load(os.path.join(RES, f"dense_{c}.npz"))["scores"][:, :48].astype(np.float32)
        cnt = (sc >= 0.30).sum(axis=1)[keep]
        out[c] = {"dense_candidates_ge_0.30_median": float(np.median(cnt)), "share_zero": round(float((cnt == 0).mean()), 4),
                  "share_all_48": round(float((cnt == 48).mean()), 4)}
        for variant in ("bake", "prod"):
            same = [P[(c, variant)][n] == P[("off", variant)][n] for n in keep]
            out[c][f"head_identical_to_bm25_only_{variant}"] = round(float(np.mean(same)), 4)
    return out


def main():
    rows = json.load(open(os.path.join(RES, "per_question.json")))
    by = defaultdict(dict)                                       # (cond, pool) -> qid -> row
    for r in rows:
        by[(r["cond"], r["pool"])][r["qid"]] = r
    qmeta = {r["qid"]: r for r in rows if r["cond"] == "off" and r["pool"] == "bake"}
    banks = defaultdict(list)
    for q, r in qmeta.items():
        banks[bank_of(r)].append(q)
        if r["set"] == "n5" and r["split"] == "dev":
            banks["n5dev_all"].append(q)
    metrics_def = {"dense_R10": ("dense_gold_rank", 10), "pool_R20": ("pool_gold_rank", 20), "pool_R48": ("pool_gold_rank", 48),
                   "rerank_R1": ("rerank_gold_rank", 1), "rerank_R8": ("rerank_gold_rank", 8)}
    nrng = np.random.default_rng(SEED)
    metrics = {}
    for (cond, pool), d in by.items():
        for bank, qids in banks.items():
            out = {"n": len(qids)}
            for mname, (field, k) in metrics_def.items():
                if mname == "dense_R10" and cond == "off":
                    continue
                hv = [hit(d[q][field], k) for q in qids]
                if any(x is None for x in hv):
                    continue
                v = np.array(hv, dtype=np.float64)
                idx = nrng.integers(0, len(v), size=(1000, len(v)))
                bs = np.sort(v[idx].mean(axis=1))
                out[mname] = [round(float(v.mean()), 4), round(float(bs[25]), 4), round(float(bs[974]), 4)]
            metrics[f"{cond}|{pool}|{bank}"] = out
    json.dump(metrics, open(os.path.join(RES, "x1_metrics.json"), "w"), indent=1)
    # paired tests vs the working arm, and detection probability by bank size (subsampling without replacement)
    detect = {}
    sizes = [30, 60, 120, 200, 500]
    main_banks = ["n5dev_all", "n5dev_fusha_named", "n5dev_fusha_open", "n5dev_iraqi_named", "n5dev_iraqi_open",
                  "books_v1_dev", "rgold_dev", "n5heldout"]
    for pool in ("bake", "prod"):
        ref = by[("on", pool)]
        for cond in CONDS[1:]:
            d = by[(cond, pool)]
            for bank in main_banks:
                qids = banks.get(bank) or []
                if not qids:
                    continue
                for mname in ("pool_R20", "pool_R48", "rerank_R8"):
                    field, k = metrics_def[mname]
                    ha = [hit(ref[q][field], k) for q in qids]
                    hb = [hit(d[q][field], k) for q in qids]
                    if any(x is None for x in ha + hb):
                        continue
                    a = np.array(ha, dtype=bool)
                    b = np.array(hb, dtype=bool)
                    only_on = int((a & ~b).sum())
                    only_fault = int((b & ~a).sum())
                    rec = {"n": len(qids), "on": int(a.sum()), "fault": int(b.sum()), "only_on": only_on,
                           "only_fault": only_fault, "p_exact_mcnemar": mcnemar_p(only_on, only_fault)}
                    det, det_any = {}, {}
                    for n in sizes:
                        if n > len(qids):
                            continue
                        hits = hits_any = 0
                        for _ in range(300):
                            sel = nrng.choice(len(qids), size=n, replace=False)
                            b_ = int((a[sel] & ~b[sel]).sum())
                            c_ = int((b[sel] & ~a[sel]).sum())
                            p_ = mcnemar_p(b_, c_)
                            hits += (p_ < 0.05 and b_ > c_)
                            hits_any += (p_ < 0.05)
                        det[str(n)] = round(hits / 300, 3)
                        det_any[str(n)] = round(hits_any / 300, 3)
                    rec["detect_prob_by_n"] = det                 # fault significantly worse than the working arm
                    rec["detect_any_direction_by_n"] = det_any    # any significant difference (a fault can also look better)
                    dd = (b.astype(np.int8) - a.astype(np.int8)).astype(np.float64)
                    bs = np.sort(dd[nrng.integers(0, len(dd), size=(1000, len(dd)))].mean(axis=1))
                    rec["diff_fault_minus_on"] = [round(float(dd.mean()), 4), round(float(bs[25]), 4), round(float(bs[974]), 4)]
                    detect[f"{cond}|{pool}|{bank}|{mname}"] = rec
    json.dump(detect, open(os.path.join(RES, "x1_detect.json"), "w"), indent=1)
    # health probes
    pm = json.load(open(os.path.join(RES, "probe_material.json")))
    qs_order = [r["id"] for r in json.load(open(os.path.join(os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1")), "questions.json")))]
    probes = {}
    canary_idx = random.Random(7).sample(range(len(qs_order)), 50)
    Q = {"on": np.load(os.path.join(RES, "Q_arctic.npy")), "unk": np.load(os.path.join(RES, "Q_unk.npy")),
         "minilm": np.load(os.path.join(RES, "Q_minilm.npy"))}
    for s in (1, 2, 3):
        r = np.random.default_rng(1000 + s).standard_normal((len(qs_order), 1024)).astype(np.float32)
        Q[f"rand{s}"] = r / np.linalg.norm(r, axis=1, keepdims=True)
    for cond in CONDS[:-1]:
        C = Q[cond][canary_idx]
        S = C @ C.T
        off = S[~np.eye(len(C), dtype=bool)]
        ov = pm["overlap48"][cond]
        top1 = pm["top1_id"][cond]
        from collections import Counter
        mc = Counter(top1).most_common(1)[0][1]
        frozen = [q for q in canary_idx]
        dr = {r["qid"]: r["dense_gold_rank"] for r in rows if r["cond"] == cond and r["pool"] == "bake"}
        labelled = [hit(dr.get(qs_order[i]), 10) for i in frozen if qs_order[i] in dr]
        probes[cond] = {"canary_pairwise_cos_mean": round(float(off.mean()), 4), "canary_pairwise_cos_max": round(float(off.max()), 4),
                        "dense_bm25_overlap48_median": float(np.median(ov)), "dense_bm25_overlap48_mean": round(float(np.mean(ov)), 3),
                        "share_queries_with_most_common_top1": round(mc / len(top1), 4),
                        "score_rank1_median": round(float(np.median(pm["score_rank1"][cond])), 4),
                        "probe_dense_R10_on_50_frozen": round(sum(labelled) / max(1, len(labelled)), 3)}
    probes["_prod_threshold"] = prod_threshold_probe()
    json.dump(probes, open(os.path.join(RES, "x1_probes.json"), "w"), indent=1)
    rep = replication(metrics)
    json.dump(rep, open(os.path.join(RES, "x1_replication.json"), "w"), indent=1)
    with open(os.path.join(RES, "x1_summary.txt"), "w") as f:
        for pool in ("bake", "prod"):
            for bank in ("n5dev_all", "n5dev_fusha_named", "n5dev_fusha_open", "n5dev_iraqi_named", "n5dev_iraqi_open",
                         "books_v1_dev", "rgold_dev", "n5heldout"):
                line = [f"{pool:4} {bank:20}"]
                for cond in CONDS:
                    m = metrics.get(f"{cond}|{pool}|{bank}")
                    if m:
                        rr = f"{m['rerank_R8'][0]:.3f}" if "rerank_R8" in m else "n/a"
                        line.append(f"{cond}: pool20 {m['pool_R20'][0]:.3f} pool48 {m['pool_R48'][0]:.3f} rr8 {rr}")
                f.write(" | ".join(line) + "\n")
        f.write("\nprobes " + json.dumps(probes) + "\n")
        f.write("\nreplication max_abs_diff " + str(rep["max_abs_diff"]) + "\n")
    print(open(os.path.join(RES, "x1_summary.txt")).read())


if __name__ == "__main__":
    main()
