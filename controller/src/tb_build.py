"""Build the recomputable testbeds T2 (Qur'an QA 2023, Arabic) and T3 (BEIR SciFact, NFCorpus, English).

Real faults are injected into a real pipeline (own BM25, Arctic-Embed-L 2.0 dense arm, mmarco cross-encoder):
  query processing  ok | strip (a character class that deletes every letter, the class of bug met in the I01 study)
                    | raw (normaliser bypassed: the governor's Q fallback)
  lexical analyser  ok | nofold (query side not folded: Arabic alef/yaa/taa/hamza and diacritics; English case)
  dense arm         on | unk (collapsed tokenizer) | minilm (weak encoder, own passage vectors) | rand | mix25/50/75
                    (query vector moved toward the collapsed vector: gradual decay) | noise (Gaussian noise on the query)
  reranker          ok | trunc (input truncated to 24 tokens), plus score-level faults applied later
Usage: tb_build.py quran|scifact|nfcorpus   ->  runs/tb_<name>/build.npz, meta.json, rr.json
"""
import collections
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

X1SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/X1_fault_injection")
P1SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/src")
sys.path.insert(0, X1SRC)
sys.path.insert(0, P1SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x1_steps as X  # noqa: E402
from e1_common import norm_words  # noqa: E402

RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
QQA = os.environ.get("QURAN_QA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/quran_qa_2023"))
BEIR = os.environ.get("BEIR_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/beir"))
K1, BB = 1.2, 0.75
TOP = 200
STRIP_AR = re.compile("[ؐ-ً]")       # the reordered class from the I01 incident: it spans every Arabic letter
STRIP_EN = re.compile("[A-Za-z]")
_DIAC = X._DIAC
_NOFOLD_NON_AR = re.compile(r"[^؀-ۿ]+")


# ---------------------------------------------------------------------------------------------------------------
def load(name):
    if name == "quran":
        passages = []
        for line in open(os.path.join(QQA, "Thematic_QPC", "QQA23_TaskA_QPC_v1.1.tsv"), encoding="utf-8"):
            pid, text = line.rstrip("\n").split("\t", 1)
            passages.append((pid, text))
        rel = collections.defaultdict(set)
        qs = []
        for split in ("train", "dev", "test"):
            for line in open(os.path.join(QQA, f"QQA23_TaskA_ayatec_v1.2_{split}.tsv"), encoding="utf-8"):
                qid, text = line.rstrip("\n").split("\t", 1)
                qs.append({"id": qid, "q": text, "split": split})
            for line in open(os.path.join(QQA, "qrels", f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold"), encoding="utf-8"):
                p = line.split()
                if len(p) >= 4 and p[2] != "-1" and int(p[3]) > 0:
                    rel[p[0]].add(p[2])
        lang = "ar"
    else:
        c = [json.loads(l) for l in open(os.path.join(BEIR, name, "corpus.jsonl"), encoding="utf-8")]
        passages = [(d["_id"], ((d.get("title") or "") + " " + (d.get("text") or "")).strip()) for d in c]
        qtext = {d["_id"]: d["text"] for d in (json.loads(l) for l in open(os.path.join(BEIR, name, "queries.jsonl"), encoding="utf-8"))}
        rel = collections.defaultdict(set)
        qs, seen = [], set()
        for split in ("train", "dev", "test"):
            fn = os.path.join(BEIR, name, f"qrels_{split}.tsv")
            if not os.path.exists(fn):
                continue
            for line in list(open(fn, encoding="utf-8"))[1:]:
                qid, did, sc = line.rstrip("\n").split("\t")[:3]
                if int(float(sc)) > 0:
                    rel[qid].add(did)
                    if qid not in seen and qid in qtext:
                        seen.add(qid)
                        qs.append({"id": qid, "q": qtext[qid], "split": split})
        lang = "en"
    if name == "arguana":                  # only a test split exists: fixed 50/50 hash split into calibration / test
        import hashlib
        for q in qs:
            q["split"] = "calib" if int(hashlib.sha256(q["id"].encode()).hexdigest(), 16) % 2 == 0 else "test"
    pidx = {pid: i for i, (pid, _) in enumerate(passages)}
    for q in qs:
        q["rel"] = sorted(pidx[p] for p in rel.get(q["id"], ()) if p in pidx)
    qs = [q for q in qs if q["rel"]]
    return [t for _, t in passages], qs, lang


_MIN_FOLD = str.maketrans({"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0671": "\u0627",
                           "\u0649": "\u064a", "\u0629": "\u0647"})


def minimal_norm(text):
    """Independent fallback normaliser (N-version design): no regular expressions, only unicodedata categories, so a
    broken character class in the main normaliser cannot break it too. NFKC, drop combining marks and tatweel, fold the
    common alef/yaa/taa forms, keep letters and digits."""
    t = unicodedata.normalize("NFKC", text or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn" and ch != "\u0640").translate(_MIN_FOLD)
    t = "".join(ch if (unicodedata.category(ch)[0] in "LN") else " " for ch in t)
    return t.casefold()


def lex_tokens(text, lang, mode="ok"):
    if lang == "ar":
        if mode == "strip":
            text = STRIP_AR.sub("", text)
        if mode == "nofold":            # analyser fault on the query side: no NFKC, no diacritic strip, no letter folding
            return _NOFOLD_NON_AR.sub(" ", text).split()
        if mode == "raw":               # main normaliser bypassed: independent minimal normaliser
            return minimal_norm(text).split()
        return norm_words(text)
    if mode == "strip":
        text = STRIP_EN.sub("", text)
    if mode == "nofold":
        return re.findall(r"[A-Za-z0-9]+", text)
    if mode == "raw":
        return minimal_norm(text).split()
    return re.findall(r"[a-z0-9]+", text.lower())


def dense_text(text, lang, mode="ok"):
    if lang == "ar":
        if mode == "strip":
            text = STRIP_AR.sub("", text)
        if mode == "raw":
            return minimal_norm(text)
        return _DIAC.sub("", text)
    if mode == "strip":
        text = STRIP_EN.sub("", text)
    return text


class BM25:
    def __init__(self, docs):
        self.n = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = self.dl.mean()
        self.post = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for t, tf in collections.Counter(d).items():
                self.post[t].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def top(self, q, k=TOP):
        s = np.zeros(self.n)
        for t in set(q):
            for i, tf in self.post.get(t, ()):
                s[i] += self.idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - BB + BB * self.dl[i] / self.avgdl))
        nz = np.flatnonzero(s > 0)
        o = nz[np.argsort(-s[nz], kind="stable")][:k]
        return [int(i) for i in o]


def topk(P, Q, k=TOP):
    S = Q @ P.T
    ids = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return ids.astype(np.int32), np.take_along_axis(S, ids, 1).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
def build(name):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    od = os.path.join(RUNS, f"tb_{name}")
    os.makedirs(od, exist_ok=True)
    texts, qs, lang = load(name)
    X.log(name, "passages", len(texts), "questions", len(qs), collections.Counter(q["split"] for q in qs))
    out = {}
    bm = BM25([lex_tokens(t, lang) for t in texts])
    for mode in ("ok", "strip", "raw", "nofold"):
        L = [bm.top(lex_tokens(q["q"], lang, mode)) for q in qs]
        arr = np.full((len(qs), TOP), -1, np.int32)
        for n, l in enumerate(L):
            arr[n, :len(l)] = l
        out[f"L_{mode}"] = arr
    X.log("bm25 done")
    # dense
    for enc_name, path, pooling, maxlen, prefix in (("arctic", X.ARCTIC, "cls", 512, "query: "),
                                                    ("minilm", X.MINILM, "mean", 128, "")):
        tok = AutoTokenizer.from_pretrained(path)
        m = X.gpu_model(AutoModel, path)
        X.thermal_guard()
        P = X.encode(tok, m, [dense_text(t, lang) if enc_name == "arctic" else t for t in texts], pooling, maxlen, batch=16)
        X.thermal_guard()
        if enc_name == "arctic":
            Q = {mode: X.encode(tok, m, [dense_text(q["q"], lang, mode) for q in qs], pooling, maxlen, prefix=prefix, batch=16)
                 for mode in ("ok", "strip", "raw")}
            unk = [[tok.cls_token_id] + [tok.unk_token_id] * len((prefix + q["q"]).split()) + [tok.sep_token_id] for q in qs]
            Q["unk"] = X.encode(tok, m, None, pooling, maxlen, input_ids=unk)
            rng = np.random.default_rng(1001)
            r = rng.standard_normal(Q["ok"].shape).astype(np.float32)
            Q["rand"] = r / np.linalg.norm(r, axis=1, keepdims=True)
            for a in (25, 50, 75):
                v = (1 - a / 100) * Q["ok"] + (a / 100) * Q["unk"]
                Q[f"mix{a}"] = v / np.linalg.norm(v, axis=1, keepdims=True)
            v = Q["ok"] + rng.standard_normal(Q["ok"].shape).astype(np.float32) * (1.0 / math.sqrt(Q["ok"].shape[1]))
            Q["noise"] = v / np.linalg.norm(v, axis=1, keepdims=True)
            for k, v in Q.items():
                ids, sc = topk(P, v)
                out[f"D_{k}_ids"], out[f"D_{k}_sc"], out[f"Q_{k}"] = ids, sc, v.astype(np.float16)
        else:
            Qm = X.encode(tok, m, [q["q"] for q in qs], pooling, maxlen, batch=16)
            ids, sc = topk(P, Qm)
            out["D_minilm_ids"], out["D_minilm_sc"], out["Q_minilm"] = ids, sc, Qm.astype(np.float16)
        del m
        torch.cuda.empty_cache()
        X.log(enc_name, "done")
    np.savez_compressed(os.path.join(od, "build.npz"), **out)
    json.dump({"name": name, "lang": lang, "n_passages": len(texts),
               "qs": [{"id": q["id"], "split": q["split"], "rel": q["rel"]} for q in qs]},
              open(os.path.join(od, "meta.json"), "w"))
    # reranker pairs needed by the generic precompute (all worlds x all actions), then score them
    import tb_data
    if name != "quran":                    # T3 is pool-level only (reranker faults are tested on T2; GPU time is
        json.dump({}, open(os.path.join(od, "rr.json"), "w"))     # thermally limited)
        X.log("build complete (pool-level only)", name)
        return
    tb = tb_data.TB(name, with_rr=False)
    need = tb.needed_pairs()
    X.log("rerank pairs", len(need))
    tok = AutoTokenizer.from_pretrained(X.MMARCO)
    rr = X.gpu_model(AutoModelForSequenceClassification, X.MMARCO)
    res = {}
    with torch.no_grad():
        for tag, ml, pairs in (("ok", 512, need), ("trunc", 24, tb.healthy_pairs())):
            t0 = time.time()
            X.log("rerank", tag, len(pairs))
            for s in range(0, len(pairs), 32):
                X.thermal_guard()
                tb_ = time.time()
                part = pairs[s:s + 32]
                enc = tok([qs[n]["q"] for n, i in part], [texts[i] for n, i in part], padding=True, truncation=True,
                          max_length=ml, return_tensors="pt").to("cuda")
                res.update({f"{tag}|{n}:{i}": v for (n, i), v in
                            zip(part, rr(**enc).logits[:, 0].float().cpu().numpy().tolist())})
                time.sleep(max(0.05, time.time() - tb_))
            X.log(f"rerank {tag} {len(pairs)} pairs {time.time() - t0:.0f}s")
    json.dump(res, open(os.path.join(od, "rr.json"), "w"))
    X.log("build complete", name)


if __name__ == "__main__":
    build(sys.argv[1])
