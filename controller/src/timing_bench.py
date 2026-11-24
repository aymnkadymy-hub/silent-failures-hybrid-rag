"""Timing micro-benchmark of the real T2 (Qur'an QA) pipeline on this laptop's GPU (A15): wall time per batch of 20
queries for each stage, with and without the probe bundle (2 probe questions co-batched through the normalizer and
encoder, the stored-vector search, BM25 top-5 through both normalizers, one reranker reference pair, a text hash).
Output: runs/v2/timing.json (ms per batch: p50, p95). BM25 is our pure-Python implementation, so its absolute time is
not representative of a production engine; the probe overhead on it is reported relative to its own cost."""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_build as T   # noqa: E402

X = T.X


def main():
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    texts, qs, lang = T.load("quran")
    bm = T.BM25([T.lex_tokens(t, lang) for t in texts])
    tok = AutoTokenizer.from_pretrained(X.ARCTIC)
    enc = X.gpu_model(AutoModel, X.ARCTIC)
    P = torch.from_numpy(X.encode(tok, enc, [T.dense_text(t, lang) for t in texts], "cls", 512, batch=16)).to("cuda", torch.float16)
    rtok = AutoTokenizer.from_pretrained(X.MMARCO)
    rr = X.gpu_model(AutoModelForSequenceClassification, X.MMARCO)
    rng = np.random.default_rng(0)
    probes = [0, 1]
    ref_vec = P[:2].clone()                      # stand-in stored reference vectors (same dtype/shape as the real ones)
    res = {"with": [], "without": []}
    with torch.no_grad():
        for it in range(60):
            X.thermal_guard()
            batch = [int(i) for i in rng.choice(len(qs), 20, replace=False)]
            for mode in ("without", "with"):
                tt = {}
                t0 = time.perf_counter()
                toks = [T.lex_tokens(qs[i]["q"], lang) for i in batch]
                dtexts = [T.dense_text(qs[i]["q"], lang) for i in batch]
                if mode == "with":
                    dtexts += [T.dense_text(qs[p]["q"], lang) for p in probes]
                tt["normalize"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                L = [bm.top(tk, 48) for tk in toks]
                if mode == "with":
                    for p in probes:
                        bm.top(T.lex_tokens(qs[p]["q"], lang), 5)
                        bm.top(T.lex_tokens(qs[p]["q"], lang, "raw"), 5)
                tt["bm25"] = time.perf_counter() - t0
                torch.cuda.synchronize(); t0 = time.perf_counter()
                e = tok(["query: " + t for t in dtexts], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                V = torch.nn.functional.normalize(enc(**e).last_hidden_state[:, 0].float(), dim=-1).half()
                torch.cuda.synchronize(); tt["encode"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                S = V @ P.T
                if mode == "with":
                    S2 = ref_vec @ P.T
                    S2.argmax(1)
                D = torch.topk(S.float(), 48, dim=1).indices.cpu().numpy()
                torch.cuda.synchronize(); tt["search"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                heads = []
                for j in range(20):
                    acc = {}
                    for lst in (D[j].tolist(), L[j]):
                        for r_, i in enumerate(lst):
                            acc[i] = acc.get(i, 0) + 1 / (61 + r_)
                    heads.append([i for i, _ in sorted(acc.items(), key=lambda x: -x[1])][:48])
                tt["fuse"] = time.perf_counter() - t0
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pairs = [(batch[j], i) for j in range(20) for i in heads[j]]
                if mode == "with":
                    pairs.append((probes[0], heads[0][0]))
                for s in range(0, len(pairs), 64):
                    part = pairs[s:s + 64]
                    x = rtok([qs[a]["q"] for a, b in part], [texts[b] for a, b in part], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to("cuda")
                    rr(**x)
                torch.cuda.synchronize(); tt["rerank"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                if mode == "with":
                    for p in probes:
                        hashlib.sha256(texts[heads[0][0]].encode()).hexdigest()
                tt["hash"] = time.perf_counter() - t0
                tt["total"] = sum(tt.values())
                if it >= 10:                      # warm-up batches dropped
                    res[mode].append({k: v * 1000 for k, v in tt.items()})
            time.sleep(0.3)
    summ = {}
    for mode, rows in res.items():
        summ[mode] = {k: {"p50": float(np.percentile([r[k] for r in rows], 50)), "p95": float(np.percentile([r[k] for r in rows], 95))}
                      for k in rows[0]}
    summ["overhead_total_p50_pct"] = (summ["with"]["total"]["p50"] / summ["without"]["total"]["p50"] - 1) * 100
    summ["overhead_total_p95_pct"] = (summ["with"]["total"]["p95"] / summ["without"]["total"]["p95"] - 1) * 100
    summ["n_batches"] = len(res["with"])
    os.makedirs(os.path.join(T.RUNS, "v2"), exist_ok=True)
    json.dump(summ, open(os.path.join(T.RUNS, "v2", "timing.json"), "w"), indent=1)
    X.log("timing", json.dumps({m: {k: round(v["p50"], 1) for k, v in summ[m].items()} for m in ("with", "without")}),
          "overhead p50 %.1f%% p95 %.1f%%" % (summ["overhead_total_p50_pct"], summ["overhead_total_p95_pct"]))


if __name__ == "__main__":
    main()
