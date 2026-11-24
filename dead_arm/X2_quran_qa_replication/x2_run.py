#!/usr/bin/env python3
"""I01 / X2: dead-arm fault injection replicated on a public Arabic collection (Qur'an QA 2023 Task A).

See PREDICTIONS.md (registered before running). One process: BM25 (this study's own implementation), dense encoders
from X1 (arctic-embed-l-v2.0 and MiniLM, same settings and helpers), RRF k=60 at list lengths 48 and 200, halving check,
gap decomposition, detection curves and the dense/BM25 overlap probe.
GPU use is tiny (1,266 passages, 251 questions) but still goes through the per-batch thermal guard of X1.
Usage: x2_run.py   (outputs results/x2_results.json and results/x2_per_question.json; numbers only)
"""
import collections
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X1 = os.path.join(HERE, "..", "X1_fault_injection")
P1 = os.path.join(HERE, "..", "..", "P1", "E1.ABC_routing_retrieval")
sys.path.insert(0, X1)
sys.path.insert(0, P1)
import x1_steps as X  # noqa: E402  (puts the accelerate overlay on the path before transformers is imported)
from e1_common import norm_words  # noqa: E402

DATA = os.environ.get("QURAN_QA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/quran_qa_2023"))
SEED = 20261028
K1, B = 1.2, 0.75


def load():
    passages = []
    for line in open(os.path.join(DATA, "Thematic_QPC", "QQA23_TaskA_QPC_v1.1.tsv"), encoding="utf-8"):
        pid, text = line.rstrip("\n").split("\t", 1)
        passages.append((pid, text))
    qs = []
    for split in ("train", "dev", "test"):
        for line in open(os.path.join(DATA, f"QQA23_TaskA_ayatec_v1.2_{split}.tsv"), encoding="utf-8"):
            qid, text = line.rstrip("\n").split("\t", 1)
            qs.append({"qid": qid, "q": text, "split": split})
    rel = collections.defaultdict(set)
    for split in ("train", "dev", "test"):
        for line in open(os.path.join(DATA, "qrels", f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold"), encoding="utf-8"):
            p = line.split()
            if len(p) >= 4 and p[2] != "-1" and int(p[3]) > 0:
                rel[p[0]].add(p[2])
    qs = [q for q in qs if rel.get(q["qid"])]
    return passages, qs, rel


class BM25:
    def __init__(self, docs):
        self.docs = [collections.Counter(d) for d in docs]
        self.n = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = self.dl.mean()
        df = collections.Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q):
        s = np.zeros(self.n)
        for t in set(q):
            if t not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(t)
                if tf:
                    s[i] += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.dl[i] / self.avgdl))
        return s


def order(scores):
    return list(np.argsort(-scores, kind="stable"))


def rrf(lists, N):
    acc = {}
    for lst in lists:
        for r, i in enumerate(lst[:N]):
            acc[i] = acc.get(i, 0.0) + 1.0 / (60 + r + 1)
    return [i for i, _ in sorted(acc.items(), key=lambda x: -x[1])]


def first_rank(lst, gold):
    return next((r for r, i in enumerate(lst, 1) if i in gold), None)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def encode_all(passages, qs):
    from transformers import AutoModel, AutoTokenizer
    out = {}
    for name, path, pooling, maxlen, prefix in (("arctic", X.ARCTIC, "cls", 512, "query: "), ("minilm", X.MINILM, "mean", 128, "")):
        tok = AutoTokenizer.from_pretrained(path)
        m = X.gpu_model(AutoModel, path)
        X.thermal_guard(hot=78, cool=70)
        P = X.encode(tok, m, [X._DIAC.sub("", t) for _, t in passages], pooling, maxlen, batch=16)
        X.thermal_guard(hot=78, cool=70)
        Q = X.encode(tok, m, [X._DIAC.sub("", q["q"]) for q in qs], pooling, maxlen, prefix=prefix, batch=16)
        out[name] = (P, Q)
        if name == "arctic":
            unk = [[tok.cls_token_id] + [tok.unk_token_id] * len(("query: " + q["q"]).split()) + [tok.sep_token_id] for q in qs]
            out["unk"] = (P, X.encode(tok, m, None, pooling, maxlen, input_ids=unk))
        del m
        import torch
        torch.cuda.empty_cache()
    for s in (1, 2, 3):
        r = np.random.default_rng(1000 + s).standard_normal((len(qs), out["arctic"][0].shape[1])).astype(np.float32)
        out[f"rand{s}"] = (out["arctic"][0], r / np.linalg.norm(r, axis=1, keepdims=True))
    return out


def main():
    passages, qs, rel = load()
    pid_index = {pid: i for i, (pid, _) in enumerate(passages)}
    gold = [{pid_index[p] for p in rel[q["qid"]] if p in pid_index} for q in qs]
    bm = BM25([norm_words(t) for _, t in passages])
    bm_lists = [order(bm.scores(norm_words(q["q"]))) for q in qs]
    enc = encode_all(passages, qs)
    conds = ["on", "minilm", "unk", "rand1", "rand2", "rand3", "off"]
    src = {"on": "arctic", "minilm": "minilm", "unk": "unk", "rand1": "rand1", "rand2": "rand2", "rand3": "rand3"}
    rows = []
    for qi, q in enumerate(qs):
        g = gold[qi]
        row = {"qid": q["qid"], "split": q["split"], "n_gold": len(g), "bm25_rank": first_rank(bm_lists[qi], g)}
        for c in conds:
            if c == "off":
                dl = []
            else:
                P, Q = enc[src[c]]
                dl = order(P @ Q[qi])
                row[f"{c}|dense_rank"] = first_rank(dl, g)
                row[f"{c}|overlap48"] = len(set(dl[:48]) & set(bm_lists[qi][:48]))
            for N in (48, 200):
                fused = rrf([dl, bm_lists[qi]] if dl else [bm_lists[qi]], N)
                row[f"{c}|N{N}|fused_rank"] = first_rank(fused, g)
        rows.append(row)
    json.dump(rows, open(os.path.join(HERE, "results", "x2_per_question.json"), "w"), indent=1)

    def hit(r, k):
        return r is not None and r <= k

    n = len(rows)
    res = {"n_questions": n, "n_passages": len(passages), "controls": {}, "recall": {}, "halving": {}, "gap": {},
           "detect_dense_R10": {}, "overlap48": {}}
    # C1: our BM25 against the organisers' BM25 run, on dev questions
    run = collections.defaultdict(list)
    for line in open(os.path.join(DATA, "runs", "bigIR_BM25.tsv"), encoding="utf-8"):
        p = line.split()
        run[p[0]].append((int(p[3]), pid_index.get(p[2])))
    dev = [qi for qi, q in enumerate(qs) if q["split"] == "dev"]
    for k in (10, 20):
        theirs = [any(i in gold[qi] for r, i in sorted(run.get(qs[qi]["qid"], []))[:k]) for qi in dev]
        ours = [hit(rows[qi]["bm25_rank"], k) for qi in dev]
        res["controls"][f"C1_dev_R@{k}"] = {"n": len(dev), "ours": round(float(np.mean(ours)), 4),
                                            "organisers_bm25_run": round(float(np.mean(theirs)), 4)}
    res["controls"]["C2_dense_R@10_on_vs_bm25_R@10"] = [round(float(np.mean([hit(r["on|dense_rank"], 10) for r in rows])), 4),
                                                       round(float(np.mean([hit(r["bm25_rank"], 10) for r in rows])), 4)]
    for c in conds:
        rec = {}
        if c != "off":
            rec["dense_R10"] = round(float(np.mean([hit(r[f"{c}|dense_rank"], 10) for r in rows])), 4)
        for N in (48, 200):
            for k in (10, 20):
                rec[f"N{N}_fused_R{k}"] = round(float(np.mean([hit(r[f"{c}|N{N}|fused_rank"], k) for r in rows])), 4)
        res["recall"][c] = rec
    for k in (5, 10, 20):
        res["recall"].setdefault("bm25_alone", {})[f"R{k}"] = round(float(np.mean([hit(r["bm25_rank"], k) for r in rows])), 4)
    # prediction 1-2: halving
    for c in ("minilm", "unk", "rand1", "rand2", "rand3"):
        for N in (48, 200):
            for k in (10, 20):
                obs = [hit(r[f"{c}|N{N}|fused_rank"], k) for r in rows]
                pred = [hit(r["bm25_rank"], k // 2) for r in rows]
                res["halving"][f"{c}|N{N}|R@{k}"] = {"fused_dead": round(float(np.mean(obs)), 4),
                                                    "bm25_half": round(float(np.mean(pred)), 4),
                                                    "agreement": round(float(np.mean([a == b for a, b in zip(obs, pred)])), 4)}
    # prediction 3: gap decomposition (N=48)
    for k in (10, 20):
        on = np.mean([hit(r["on|N48|fused_rank"], k) for r in rows])
        off = np.mean([hit(r["bm25_rank"], k) for r in rows])
        half = np.mean([hit(r["bm25_rank"], k // 2) for r in rows])
        for c in ("minilm", "unk", "rand1"):
            dead = np.mean([hit(r[f"{c}|N48|fused_rank"], k) for r in rows])
            res["gap"][f"{c}|R@{k}"] = {"observed_on_minus_dead": round(float(on - dead), 4),
                                        "net_gain_on_minus_bm25": round(float(on - off), 4),
                                        "halving_cost_bm25_k_minus_k2": round(float(off - half), 4),
                                        "predicted": round(float((on - off) + (off - half)), 4)}
    # prediction 4: per-arm labelled check at the dense stage, and detection at the fused stage
    rng = random.Random(SEED)
    for c in ("minilm", "unk", "rand1"):
        a = np.array([hit(r["on|dense_rank"], 10) for r in rows])
        b = np.array([hit(r[f"{c}|dense_rank"], 10) for r in rows])
        fa = np.array([hit(r["on|N48|fused_rank"], 20) for r in rows])
        fb = np.array([hit(r[f"{c}|N48|fused_rank"], 20) for r in rows])
        det = {}
        for size in (10, 20, 30, 60, 120, 200):
            if size > n:
                continue
            hd = hf = 0
            for _ in range(300):
                sel = rng.sample(range(n), size)
                b1 = int(sum(a[i] and not b[i] for i in sel)); c1 = int(sum(b[i] and not a[i] for i in sel))
                hd += mcnemar_p(b1, c1) < 0.05 and b1 > c1
                b2 = int(sum(fa[i] and not fb[i] for i in sel)); c2 = int(sum(fb[i] and not fa[i] for i in sel))
                hf += mcnemar_p(b2, c2) < 0.05 and b2 > c2
            det[str(size)] = {"dense_R10_check": round(hd / 300, 3), "fused_R20": round(hf / 300, 3)}
        res["detect_dense_R10"][c] = det
    for c in ("on", "minilm", "unk", "rand1"):
        ov = [r[f"{c}|overlap48"] for r in rows]
        res["overlap48"][c] = {"mean": round(float(np.mean(ov)), 3), "median": float(np.median(ov)),
                               "share_zero": round(float(np.mean([o == 0 for o in ov])), 4)}
    json.dump(res, open(os.path.join(HERE, "results", "x2_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
