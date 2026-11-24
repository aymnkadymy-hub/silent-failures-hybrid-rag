#!/usr/bin/env python3
"""I01 / X1 follow-up analyses (CPU only, no model):

1. dead_vs_off   Is a dead dense arm worse than no dense arm? Paired exact McNemar of each dead condition against
                 BM25-only, with a paired bootstrap interval of the difference (questions resampled, seed 20261028).
2. monitor       A label-free health monitor: mean overlap between the dense top-48 and the BM25 top-48 over a batch of
                 k live queries. The alert threshold is the 1st percentile of the working arm's batch means on the
                 N5 development questions (calibration); false alarms and detections are measured on held-out
                 questions (N5 held-out, and the books bank as a shift in question style). No gold labels are used.
3. labelled      How many labelled questions a per-arm check needs: detection probability of each fault at the dense
                 stage (R@10), McNemar p < 0.05 in the right direction, 300 random subsets per size.
4. halving       Mechanism check. With equal-weight RRF and a dense list that shares nothing with BM25, the two lists
                 interleave, so a fused head of depth k should keep BM25's top k/2. Prediction: fused R@20 with a dead
                 arm = BM25 R@10, fused R@48 = BM25 R@24. Compared per question (share of questions where they agree).

Inputs: results/per_question.json, results/probe_material.json, copies/i01/questions.json.
Output: results/x1_extra.json
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COPIES = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1"))
SEED = 20261028
DEAD = ["minilm", "unk", "rand1"]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def hit(rank, k):
    if rank == "NA":
        return None
    return int(rank is not None and 0 < rank <= k)


def main():
    rng = np.random.default_rng(SEED)
    rows = json.load(open(os.path.join(RES, "per_question.json")))
    by = {}
    for r in rows:
        by.setdefault((r["cond"], r["pool"]), {})[r["qid"]] = r
    meta = by[("off", "bake")]
    banks = {"n5dev_all": [q for q, r in meta.items() if r["set"] == "n5" and r["split"] == "dev"],
             "n5heldout": [q for q, r in meta.items() if r["set"] == "n5" and r["split"] == "heldout"],
             "books_v1_dev": [q for q, r in meta.items() if r["set"] == "books_v1"],
             "rgold_dev": [q for q, r in meta.items() if r["set"] == "rgold"]}
    out = {"dead_vs_off": {}, "monitor": {}, "labelled": {}}

    # 1. dead arm against no arm
    fields = {"pool_R20": ("pool_gold_rank", 20), "pool_R48": ("pool_gold_rank", 48), "rerank_R8": ("rerank_gold_rank", 8)}
    for pool in ("bake", "prod"):
        for cond in DEAD:
            for bank, qids in banks.items():
                for m, (f, k) in fields.items():
                    a = [hit(by[("off", pool)][q][f], k) for q in qids]
                    b = [hit(by[(cond, pool)][q][f], k) for q in qids]
                    if None in a or None in b:
                        continue
                    a, b = np.array(a, bool), np.array(b, bool)
                    only_off, only_dead = int((a & ~b).sum()), int((b & ~a).sum())
                    d = b.astype(np.float64) - a.astype(np.float64)
                    bs = np.sort(d[rng.integers(0, len(d), size=(1000, len(d)))].mean(axis=1))
                    out["dead_vs_off"][f"{cond}|{pool}|{bank}|{m}"] = {
                        "n": len(qids), "off": int(a.sum()), "dead": int(b.sum()), "only_off": only_off,
                        "only_dead": only_dead, "p_exact_mcnemar": mcnemar_p(only_off, only_dead),
                        "diff_dead_minus_off": [round(float(d.mean()), 4), round(float(bs[25]), 4), round(float(bs[974]), 4)]}

    # 2. label-free monitor (all questions of a split count, with or without gold: no labels are used)
    qs = json.load(open(os.path.join(COPIES, "questions.json"), encoding="utf-8"))
    pm = json.load(open(os.path.join(RES, "probe_material.json")))
    ov = {c: np.array(v, dtype=np.float64) for c, v in pm["overlap48"].items()}
    idx = {"calib_n5dev": [n for n, q in enumerate(qs) if q["set"] == "n5" and q["split"] == "dev"],
           "test_n5heldout": [n for n, q in enumerate(qs) if q["set"] == "n5" and q["split"] == "heldout"],
           "test_books": [n for n, q in enumerate(qs) if q["set"] == "books_v1"]}
    out["monitor"]["per_query_overlap"] = {
        c: {s: {"mean": round(float(ov[c][ix].mean()), 3), "median": float(np.median(ov[c][ix])),
                "share_zero": round(float((ov[c][ix] == 0).mean()), 4)} for s, ix in idx.items()}
        for c in ["on", "minilm", "unk", "rand1"]}
    trials = 10000
    for k in (1, 3, 5, 10, 20, 50):
        cal = ov["on"][idx["calib_n5dev"]]
        cal_means = cal[rng.integers(0, len(cal), size=(trials, k))].mean(axis=1)
        tau = float(np.quantile(cal_means, 0.01))
        rec = {"threshold_mean_overlap_below": round(tau, 3)}
        for split in ("test_n5heldout", "test_books"):
            ix = np.array(idx[split])
            draw = rng.integers(0, len(ix), size=(trials, k))
            for c in ["on", "minilm", "unk", "rand1"]:
                means = ov[c][ix][draw].mean(axis=1)
                rec[f"{split}|{c}|alert_rate"] = round(float((means < tau).mean()), 4)
        out["monitor"][f"k={k}"] = rec

    # 3. labelled per-arm check at the dense stage
    for cond in DEAD:
        for bank in ("n5dev_all", "books_v1_dev"):
            qids = banks[bank]
            a = np.array([hit(by[("on", "bake")][q]["dense_gold_rank"], 10) for q in qids], bool)
            b = np.array([hit(by[(cond, "bake")][q]["dense_gold_rank"], 10) for q in qids], bool)
            det = {}
            for n in (10, 20, 30, 50, 100):
                if n > len(qids):
                    continue
                hits = 0
                for _ in range(300):
                    sel = rng.choice(len(qids), size=n, replace=False)
                    b_, c_ = int((a[sel] & ~b[sel]).sum()), int((b[sel] & ~a[sel]).sum())
                    hits += (mcnemar_p(b_, c_) < 0.05 and b_ > c_)
                det[str(n)] = round(hits / 300, 3)
            out["labelled"][f"{cond}|{bank}|dense_R10"] = {"on": round(float(a.mean()), 4), "fault": round(float(b.mean()), 4),
                                                           "detect_prob_by_n": det}
    # 4. halving: fused recall with a dead arm against BM25 recall at half the depth
    out["halving"] = {}
    for pool in ("bake", "prod"):
        for cond in DEAD + ["rand2", "rand3"]:
            for bank, qids in banks.items():
                for k, half in ((20, 10), (48, 24)):
                    obs = np.array([hit(by[(cond, pool)][q]["pool_gold_rank"], k) for q in qids], bool)
                    pred = np.array([hit(by[(cond, pool)][q]["bm25_gold_rank"], half) for q in qids], bool)
                    bm_full = np.array([hit(by[(cond, pool)][q]["bm25_gold_rank"], k) for q in qids], bool)
                    out["halving"][f"{cond}|{pool}|{bank}|R@{k}"] = {
                        "n": len(qids), "fused_dead": round(float(obs.mean()), 4), "bm25_at_half_depth": round(float(pred.mean()), 4),
                        "bm25_at_full_depth": round(float(bm_full.mean()), 4), "per_question_agreement": round(float((obs == pred).mean()), 4)}
    json.dump(out, open(os.path.join(RES, "x1_extra.json"), "w"), indent=1)
    print(json.dumps(out["monitor"], indent=0)[:3000])
    for k, v in out["dead_vs_off"].items():
        if "|n5dev_all|" in k or "|books_v1_dev|" in k or "|n5heldout|" in k:
            print(k, v["off"], v["dead"], "p=%.2g" % v["p_exact_mcnemar"], v["diff_dead_minus_off"])
    for k, v in out["labelled"].items():
        print(k, v)
    for k, v in out["halving"].items():
        if "|bake|" in k or "|prod|n5dev_all|" in k:
            print(k, v)


if __name__ == "__main__":
    main()
