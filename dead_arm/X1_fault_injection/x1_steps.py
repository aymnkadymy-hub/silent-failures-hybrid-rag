#!/usr/bin/env python3
"""I01 / X1: controlled dense-arm fault injection, split into low-memory steps (one model per process).

Same experiment as x1_gpu.py (see its docstring for the conditions and pool definitions); this version keeps host RAM
small on a shared laptop: weights load straight to the GPU (device_map="cuda", fp16, no CPU copy kept), passage
matrices are memory-mapped and moved to the GPU in chunks, and each step runs in its own process.

Steps (run in order; each skips work already saved under OUT):
  enc_arctic  arctic-embed-l-v2.0 query vectors (diacritics stripped, "query: " prompt), all-<unk> query vectors,
              and a parity check: 50 stored passage vectors re-encoded must match (cosine >= 0.99)
  enc_minilm  MiniLM query vectors + the same parity check
  dense       exact top-200 per condition (on, unk, minilm, rand1-3) and exact rank of the best gold chunk
  rerank      mmarco scores for every (question, chunk) in any pool head (bake and prod pools, all conditions)
  table       per-question long table + dense/BM25 top-48 overlap (label-free probe material)
Usage: x1_steps.py STEP COPIES_DIR OUT
"""
import gc
import json
import os
import random
import re
import sys
import time

import numpy as np

# accelerate (needed by transformers for device_map loading straight to the GPU) lives in a private overlay directory,
# so the borrowed exp-surya environment itself is not modified

ARCTIC = os.environ.get("ARCTIC_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0")
MINILM = os.environ.get("MINILM_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MMARCO = os.environ.get("MMARCO_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
_DIAC = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")   # same class as embed_bakeoff.py, written with escapes: a typed RTL class was silently reordered (see commands.log)
NCORPUS = 134837
CONDS = ["on", "unk", "minilm", "rand1", "rand2", "rand3"]


def log(*a):
    import resource
    print(time.strftime("%H:%M:%S"), f"[maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB]", *a, flush=True)


TEMP_FILE = "/sys/class/hwmon/hwmon5/temp1_input"     # coretemp "Package id 0" on this laptop (checked 2026-10-28)
RERANK_CONDS = {"bake": ["on", "minilm", "unk", "rand1", "off"], "prod": ["on", "minilm", "unk", "off"]}
RERANK_SETS = {("n5", "dev"), ("books_v1", "dev"), ("rgold", "dev")}


def thermal_guard(hot=78, cool=70):
    """Pause while the CPU package is at or above `hot` degrees C, until it falls to `cool` (shared laptop, live service)."""
    try:
        t = int(open(TEMP_FILE).read()) / 1000
    except OSError:
        return
    if t < hot:
        return
    log(f"thermal pause at {t:.0f} C")
    while True:
        time.sleep(5)
        t = int(open(TEMP_FILE).read()) / 1000
        if t <= cool:
            log(f"thermal resume at {t:.0f} C")
            return


def gpu_model(cls, path):
    """Load straight to the GPU. With device_map, transformers initialises the model on the meta device, and buffers that
    are not in the checkpoint (position_ids, token_type_ids of XLM-R/BERT embeddings) can be left uninitialised; the
    round-3 bake-off code hit the same problem (fix_remote_buffers). They are rebuilt here exactly as the model's
    __init__ builds them, and the passage parity check below verifies the result."""
    import torch
    m = cls.from_pretrained(path, dtype=torch.float16, device_map="cuda")
    fixed = []
    for name, mod in m.named_modules():
        bufs = dict(mod.named_buffers(recurse=False))
        pid = bufs.get("position_ids")
        if pid is not None and not pid.is_floating_point():
            n = pid.shape[-1]
            mod.position_ids = torch.arange(n, device=pid.device, dtype=pid.dtype).expand_as(pid).clone()
            fixed.append(name + ".position_ids")
        tt = bufs.get("token_type_ids")
        if tt is not None and not tt.is_floating_point():
            mod.token_type_ids = torch.zeros_like(tt)
            fixed.append(name + ".token_type_ids")
    log("rebuilt buffers:", fixed)
    return m.eval()


def encode(tok, m, texts, pooling, maxlen, prefix="", batch=64, input_ids=None):
    import torch
    out = []
    n = len(texts) if input_ids is None else len(input_ids)
    with torch.no_grad():
        for s in range(0, n, batch):
            if input_ids is None:
                enc = tok([prefix + t for t in texts[s:s + batch]], padding=True, truncation=True, max_length=maxlen,
                          return_tensors="pt").to("cuda")
            else:
                ids = input_ids[s:s + batch]
                L = max(len(x) for x in ids)
                arr = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in ids], device="cuda")
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device="cuda")
                enc = {"input_ids": arr, "attention_mask": att}
            h = m(**enc).last_hidden_state.float()
            if pooling == "cls":
                v = h[:, 0]
            else:
                a = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * a).sum(1) / a.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def probe_texts(copies):
    need = set(random.Random(1).sample(range(NCORPUS), 50))
    texts = {}
    with open(os.path.join(copies, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                texts[i] = json.loads(line)["t"]
    return sorted(need), texts


def step_enc(copies, out, which):
    from transformers import AutoModel, AutoTokenizer
    qs = json.load(open(os.path.join(copies, "questions.json"), encoding="utf-8"))
    fn = os.path.join(out, f"Q_{which}.npy")
    if os.path.exists(fn):
        log(which, "already encoded")
        return
    probe, texts = probe_texts(copies)
    path, pooling, maxlen, mat = ((ARCTIC, "cls", 512, "arctic-l2.nodiac.f16.npy") if which == "arctic"
                                  else (MINILM, "mean", 128, "minilm.plain.f16.npy"))
    tok = AutoTokenizer.from_pretrained(path)
    m = gpu_model(AutoModel, path)
    log(which, "loaded on GPU")
    E = np.load(os.path.join(copies, mat), mmap_mode="r")
    ptexts = [(_DIAC.sub("", texts[i]) if which == "arctic" else texts[i]) for i in probe]
    P = encode(tok, m, ptexts, pooling, maxlen)
    cos = [float(np.dot(P[k], np.asarray(E[i], dtype=np.float32))) for k, i in enumerate(probe)]
    t0 = time.time()
    if which == "arctic":
        Q = encode(tok, m, [_DIAC.sub("", q["q"]) for q in qs], pooling, maxlen, prefix="query: ")
        unk = [[tok.cls_token_id] + [tok.unk_token_id] * len(("query: " + q["q"]).split()) + [tok.sep_token_id] for q in qs]
        Qu = encode(tok, m, None, pooling, maxlen, input_ids=unk)
    else:
        Q = encode(tok, m, [q["q"] for q in qs], pooling, maxlen)
    secs = time.time() - t0
    parity = {"model": path, "probe_ids": probe, "cos": cos, "min": min(cos), "median": float(np.median(cos)),
              "query_encode_secs": round(secs, 1)}
    json.dump(parity, open(os.path.join(out, f"parity_{which}.json"), "w"), indent=1)
    log(which, "parity min", round(min(cos), 4), "median", round(float(np.median(cos)), 4))
    if min(cos) < 0.99:
        raise SystemExit(f"parity check failed for {which}")
    np.save(fn, Q)
    if which == "arctic":
        np.save(os.path.join(out, "Q_unk.npy"), Qu)
    del m
    gc.collect()


def step_dense(copies, out):
    import torch
    qs = json.load(open(os.path.join(copies, "questions.json"), encoding="utf-8"))
    nq = len(qs)
    Qs = {"on": np.load(os.path.join(out, "Q_arctic.npy")), "unk": np.load(os.path.join(out, "Q_unk.npy")),
          "minilm": np.load(os.path.join(out, "Q_minilm.npy"))}
    for s in (1, 2, 3):
        r = np.random.default_rng(1000 + s).standard_normal((nq, 1024)).astype(np.float32)
        Qs[f"rand{s}"] = r / np.linalg.norm(r, axis=1, keepdims=True)
    for mat_name, conds in (("arctic-l2.nodiac.f16.npy", ["on", "unk", "rand1", "rand2", "rand3"]),
                            ("minilm.plain.f16.npy", ["minilm"])):
        todo = [c for c in conds if not os.path.exists(os.path.join(out, f"dense_{c}.npz"))]
        if not todo:
            continue
        Emm = np.load(os.path.join(copies, mat_name), mmap_mode="r")
        E = torch.empty(Emm.shape, dtype=torch.float16, device="cuda")
        for s in range(0, Emm.shape[0], 20000):                     # chunked host->GPU copy
            E[s:s + 20000] = torch.from_numpy(np.ascontiguousarray(Emm[s:s + 20000]))
        del Emm
        gc.collect()
        log("matrix on GPU", mat_name)
        for c in todo:
            Q = Qs[c]
            ids, scs, gold_rank = [], [], []
            for s in range(0, nq, 128):
                S = torch.from_numpy(Q[s:s + 128]).to("cuda", dtype=torch.float16) @ E.T
                tv, ti = torch.topk(S.float(), 200, dim=1)
                ids.append(ti.cpu().numpy().astype(np.int32))
                scs.append(tv.cpu().numpy().astype(np.float16))
                for j in range(S.shape[0]):
                    rel = qs[s + j]["rel"]
                    if not rel:
                        gold_rank.append(-1)
                        continue
                    best = S[j, torch.tensor(rel, device="cuda")].max()
                    gold_rank.append(int((S[j] > best).sum().item()) + 1)
                del S
            np.savez(os.path.join(out, f"dense_{c}.npz"), ids=np.concatenate(ids), scores=np.concatenate(scs),
                     gold_rank=np.array(gold_rank, dtype=np.int64))
            log("dense", c, "done")
        del E
        torch.cuda.empty_cache()


def pools(copies, out):
    qs = json.load(open(os.path.join(copies, "questions.json"), encoding="utf-8"))
    res = {}
    for c in CONDS + ["off"]:
        d = None if c == "off" else np.load(os.path.join(out, f"dense_{c}.npz"))
        for variant in ("bake", "prod"):
            heads = []
            for n, q in enumerate(qs):
                b = q["bm25"] if variant == "bake" else q["bm25"][:48]
                lists = [b]
                if d is not None:
                    ids = d["ids"][n].tolist()
                    if variant == "prod":
                        sc = d["scores"][n].astype(np.float32)
                        ids = [i for i, x in zip(ids[:48], sc[:48]) if x >= 0.30]
                    lists = [ids, b]
                acc = {}
                for lst in lists:
                    for r, i in enumerate(lst):
                        acc[i] = acc.get(i, 0.0) + 1.0 / (60 + r + 1)
                heads.append([i for i, _ in sorted(acc.items(), key=lambda x: -x[1])][:48])
            res[(c, variant)] = heads
    return qs, res


def step_rerank(copies, out):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    qs, P = pools(copies, out)
    cache_fn = os.path.join(out, "mmarco_scores.json")
    cache = json.load(open(cache_fn)) if os.path.exists(cache_fn) else {}
    torch.set_num_threads(1)
    scope = {(c, v) for v, cs in RERANK_CONDS.items() for c in cs}
    want = sorted({f"{n}:{i}" for (c, v), heads in P.items() if (c, v) in scope for n, h in enumerate(heads)
                   if qs[n]["rel"] and (qs[n]["set"], qs[n]["split"]) in RERANK_SETS for i in h} - set(cache),
                  key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))
    log("pairs to score", len(want), "cached", len(cache))
    if not want:
        return
    need = {int(k.split(":")[1]) for k in want}
    text = {}
    with open(os.path.join(copies, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in need:
                text[i] = json.loads(line)["t"]
    log("texts loaded", len(text))
    tok = AutoTokenizer.from_pretrained(MMARCO)
    rr = gpu_model(AutoModelForSequenceClassification, MMARCO)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(want), 32):
            thermal_guard()                              # every batch: the GPU load heats the cooler
            tb = time.time()
            part = want[s:s + 32]
            enc = tok([qs[int(k.split(":")[0])]["q"] for k in part], [text[int(k.split(":")[1])] for k in part],
                      padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            cache.update(zip(part, rr(**enc).logits[:, 0].float().cpu().numpy().tolist()))
            time.sleep(max(0.05, time.time() - tb))     # duty cycle about 50%: idle as long as the batch took
            if (s // 32) % 600 == 0:
                log(f"rerank {s}/{len(want)}")
                json.dump(cache, open(cache_fn, "w"))
    json.dump(cache, open(cache_fn, "w"))
    log(f"scored {len(want)} pairs in {time.time() - t0:.0f}s")


def step_table(copies, out):
    qs, P = pools(copies, out)
    cache = json.load(open(os.path.join(out, "mmarco_scores.json")))
    dense = {c: np.load(os.path.join(out, f"dense_{c}.npz")) for c in CONDS}
    rows = []
    for (c, variant), heads in P.items():
        for n, q in enumerate(qs):
            if not q["rel"]:
                continue
            rel = set(q["rel"])
            h = heads[n]
            scored = all(f"{n}:{i}" in cache for i in h)
            order = sorted(h, key=lambda i: -cache[f"{n}:{i}"]) if scored else None
            rows.append({"cond": c, "pool": variant, "qid": q["id"], "set": q["set"], "split": q["split"], "lang": q["lang"],
                         "named": int(bool(q["named"])),
                         "dense_gold_rank": None if c == "off" else int(dense[c]["gold_rank"][n]),
                         "bm25_gold_rank": next((r for r, i in enumerate(q["bm25"], 1) if i in rel), None),
                         "pool_gold_rank": next((r for r, i in enumerate(h, 1) if i in rel), None),
                         "rerank_gold_rank": (next((r for r, i in enumerate(order, 1) if i in rel), None) if order is not None else "NA"),
                         "n_dense_in_prod_pool": None})
    json.dump(rows, open(os.path.join(out, "per_question.json"), "w"))
    ov = {c: [len(set(dense[c]["ids"][n][:48].tolist()) & set(q["bm25"][:48])) for n, q in enumerate(qs)] for c in CONDS}
    sc48 = {c: [float(dense[c]["scores"][n][47]) for n in range(len(qs))] for c in CONDS}
    sc1 = {c: [float(dense[c]["scores"][n][0]) for n in range(len(qs))] for c in CONDS}
    top1 = {c: [int(dense[c]["ids"][n][0]) for n in range(len(qs))] for c in CONDS}
    json.dump({"overlap48": ov, "score_rank48": sc48, "score_rank1": sc1, "top1_id": top1},
              open(os.path.join(out, "probe_material.json"), "w"))
    log("table rows", len(rows))


if __name__ == "__main__":
    step, copies, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    {"enc_arctic": lambda: step_enc(copies, out, "arctic"), "enc_minilm": lambda: step_enc(copies, out, "minilm"),
     "dense": lambda: step_dense(copies, out), "rerank": lambda: step_rerank(copies, out),
     "table": lambda: step_table(copies, out)}[step]()
