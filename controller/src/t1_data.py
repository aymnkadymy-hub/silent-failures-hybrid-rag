"""T1: the frozen classical-Arabic retrieval package (read-only copy of the I01 study data), as fault worlds.

A world maps every pipeline component to a state. For each question it yields the lexical list L, the dense list D (+
cosines), and a reranker scoring function. Faults are injected on the query side or in the result plumbing; stored
passage vectors and BM25 index stay intact, as in the I01 study.
"""
import json
import math
import os
import random

import numpy as np

import frg_core as C

COPIES = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1"))
X1 = os.environ.get("X1_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/results"))
RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
NCORPUS = 134837
FLOOR = 0.30
DEPTH = 48

DENSE_STORED = ["on", "minilm", "unk", "rand1", "rand2", "rand3"]


class T1:
    def __init__(self, extra_dense=()):
        self.qs = json.load(open(os.path.join(COPIES, "questions.json"), encoding="utf-8"))
        self.dense = {}
        for c in list(DENSE_STORED) + list(extra_dense):
            fn = os.path.join(X1, f"dense_{c}.npz") if c in DENSE_STORED else os.path.join(RUNS, f"dense_{c}.npz")
            z = np.load(fn)
            self.dense[c] = (z["ids"], z["scores"].astype(np.float32))
        self.rr = json.load(open(os.path.join(X1, "mmarco_scores.json")))
        extra = os.path.join(RUNS, "mmarco_fill.json")
        if os.path.exists(extra):
            self.rr.update(json.load(open(extra)))
        self.qvec = {"on": np.load(os.path.join(X1, "Q_arctic.npy")), "unk": np.load(os.path.join(X1, "Q_unk.npy")),
                     "minilm": np.load(os.path.join(X1, "Q_minilm.npy"))}
        rng = np.random.default_rng(11)
        r = rng.standard_normal(self.qvec["on"].shape).astype(np.float32)
        self.qvec["rand"] = r / np.linalg.norm(r, axis=1, keepdims=True)
        perm = list(range(len(self.qs)))
        random.Random(5).shuffle(perm)
        self.cache_perm = perm               # off-by-one result cache: question n receives the lists of question perm[n]
        self.idx = {q["id"]: n for n, q in enumerate(self.qs)}

    # ---- component outputs under a world ------------------------------------------------------------------
    def L(self, n, w):
        st = w.get("L", "ok")
        b = self.qs[n]["bm25"]
        if w.get("Q") == "strip":            # normaliser deletes every Arabic letter: empty lexical query
            return []
        if st == "ok":
            out = b[:DEPTH]
        elif st == "empty":
            out = []
        elif st == "shard":                  # 40% of passage ids (hash shard) missing from the lexical index
            out = [i for i in b if (i * 2654435761 % 1000) >= 400][:DEPTH]
        elif st == "cache":
            out = self.qs[self.cache_perm[n]]["bm25"][:DEPTH]
        else:
            raise ValueError(st)
        return self._store(out, w)

    def D_raw(self, n, w):
        """Dense top-48 ids and cosines before the floor."""
        st = w.get("D", "ok")
        if w.get("Q") == "strip":
            st = "unk"
        if st == "ok":
            st = "on"
        if st == "adv":                      # adversarial dense list: BM25 items just below depth 20, never a gold passage,
            rel = set(self.qs[n]["rel"])     # so that under RRF they score from both lists and push BM25's top out
            b = self.qs[n]["bm25"]
            out = [i for i in b[20:] if i not in rel][:DEPTH]
            return self._store(out, w), np.full(len(out), 0.9, np.float32)
        if st in ("ann", "corrupt"):         # held-out index faults (A9): exact top-200 of the working arm
            ids, sc = self.dense["on"]
            ids, sc = [int(i) for i in ids[n]], sc[n]
            if st == "ann":                  # approximate search: each exact top-48 item kept with probability 0.5
                keep = [j for j in range(48) if (n * 7919 + ids[j] * 104729) % 100 < 50]
                fill = list(range(48, len(ids)))
                pos = sorted(keep + fill[: 48 - len(keep)])
            else:                            # 30% of passage vectors corrupted: those passages are never retrieved
                pos = [j for j in range(len(ids)) if (ids[j] * 2654435761 % 1000) >= 300][:48]
            return self._store([ids[j] for j in pos], w), np.array([sc[j] for j in pos], np.float32)
        if st == "offset":                   # vector row i is served as passage i+1 (row/id misalignment)
            ids, sc = self.dense["on"]
            out = [(int(i) + 1) % NCORPUS for i in ids[n][:DEPTH]]
            return self._store(out, w), sc[n][:DEPTH]
        ids, sc = self.dense[st]
        return self._store([int(i) for i in ids[n][:DEPTH]], w), sc[n][:DEPTH]

    def D(self, n, w, floor=FLOOR):
        ids, sc = self.D_raw(n, w)
        if floor is None:
            return ids
        return [i for i, x in zip(ids, sc) if x >= floor]

    def _store(self, ids, w):
        if w.get("S") == "shift":            # passage store / id map off by one: every component serves the wrong text
            return [(i + 1) % NCORPUS for i in ids]
        return ids

    def qvector(self, n, w):
        st = w.get("D", "ok")
        if w.get("Q") == "strip":
            st = "unk"
        key = {"ok": "on", "on": "on", "offset": "on", "adv": "on", "ann": "on", "corrupt": "on", "unk": "unk", "minilm": "minilm"}.get(st)
        if key is None and st.startswith("rand"):
            key = "rand"
        if key is None:                      # gradual mixes: a stored query matrix is kept beside the lists
            key = st
        return self.qvec[key][n]

    def rscore(self, n, i, w):
        """Reranker score of passage i for question n, or None if never scored."""
        st = w.get("R", "ok")
        if w.get("S") == "shift":            # ids were shifted at the store; the reranker reads the retrieved text,
            i = (i - 1) % NCORPUS            # so its score is that of the original id (the damage shows at serving)
        v = self.rr.get(f"{n}:{i}")
        if v is None:
            return None
        if st == "ok":
            return v
        if st == "random":
            return random.Random(n * 1000003 + i).gauss(0, 3)
        if st == "inverted":
            return -v
        if st == "constant":
            return 0.0
        raise ValueError(st)

    def gold(self, n):
        return set(self.qs[n]["rel"])


def check_reproduction():
    """Reproduce the stored per-question pool ranks of the I01 study (prod pool) before anything is built on T1."""
    t = T1()
    rows = json.load(open(os.path.join(X1, "per_question.json")))
    want = {(r["cond"], r["qid"]): r["pool_gold_rank"] for r in rows if r["pool"] == "prod"}
    bad = tot = 0
    for c in ["on", "minilm", "unk", "rand1", "off"]:
        for n, q in enumerate(t.qs):
            if not q["rel"] or (c, q["id"]) not in want:
                continue
            w = {} if c in ("on", "off") else {"D": c}
            L = t.L(n, w)
            lists = [L] if c == "off" else [t.D(n, w), L]
            h = C.rrf(lists)
            got = C.first_hit(h, t.gold(n))
            tot += 1
            bad += got != want[(c, q["id"])]
    return bad, tot


if __name__ == "__main__":
    print("prod-pool reproduction mismatches:", check_reproduction())
