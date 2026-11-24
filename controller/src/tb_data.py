"""Adapter that exposes a recomputed testbed (T2/T3, built by tb_build.py) through the same interface as t1_data.T1,
so that t1_precompute.run_world and the governor run unchanged. Adds the query-normaliser bypass (Qmode = raw)."""
import json
import os
import random

import numpy as np

import frg_core as C

RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
FLOOR = 0.30
DEPTH = 48

WORLDS = {
    "healthy": {},
    "D_minilm": {"D": "minilm"}, "D_unk": {"D": "unk"}, "D_rand": {"D": "rand"}, "D_offset": {"D": "offset"},
    "D_adv": {"D": "adv"}, "D_mix25": {"D": "mix25"}, "D_mix50": {"D": "mix50"}, "D_mix75": {"D": "mix75"},
    "D_noise": {"D": "noise"}, "D_ann": {"D": "ann"}, "D_corrupt": {"D": "corrupt"},
    "L_empty": {"L": "empty"}, "L_shard": {"L": "shard"}, "L_cache": {"L": "cache"}, "L_nofold": {"L": "nofold"},
    "R_random": {"R": "random"}, "R_inverted": {"R": "inverted"}, "R_constant": {"R": "constant"}, "R_trunc": {"R": "trunc"},
    "Q_strip": {"Q": "strip"}, "S_shift": {"S": "shift"},
}


SPLITS = {"quran": (("train",), ("dev", "test")), "scifact": (("train",), ("test",)), "nfcorpus": (("dev",), ("test",)),
          "arguana": (("calib",), ("test",))}


RERANK_WORLDS = {"healthy", "D_unk", "D_minilm", "L_nofold", "L_shard", "Q_strip"}   # reranked metrics; others pool only


class TB:
    has_bypass = True

    def __init__(self, name, with_rr=True):
        od = os.path.join(RUNS, f"tb_{name}")
        self.name = name
        self.z = dict(np.load(os.path.join(od, "build.npz")))
        meta = json.load(open(os.path.join(od, "meta.json")))
        self.lang = meta["lang"]
        self.N = meta["n_passages"]
        self.qs = meta["qs"]
        self.rr = json.load(open(os.path.join(od, "rr.json"))) if with_rr else {}
        perm = list(range(len(self.qs)))
        random.Random(5).shuffle(perm)
        self.cache_perm = perm
        self.qvec = {k[2:]: v.astype(np.float32) for k, v in self.z.items() if k.startswith("Q_")}

    def _lst(self, key, n):
        a = self.z[key][n]
        return [int(i) for i in a if i >= 0]

    def _qmode(self, w):
        if w.get("Qmode") == "raw":
            return "raw"
        if w.get("Q") == "strip":
            return "strip"
        return "ok"

    def L(self, n, w):
        qm = self._qmode(w)
        st = w.get("L", "ok")
        if st == "nofold" and qm == "ok":
            b = self._lst("L_nofold", n)
        else:
            b = self._lst(f"L_{qm}", n)
        if st == "empty":
            out = []
        elif st == "shard":
            out = [i for i in b if (i * 2654435761 % 1000) >= 400][:DEPTH]
        elif st == "cache":
            out = self._lst(f"L_{qm}", self.cache_perm[n])[:DEPTH]
        else:
            out = b[:DEPTH]
        return self._store(out, w)

    def D_raw(self, n, w):
        qm = self._qmode(w)
        st = w.get("D", "ok")
        if qm == "strip":
            key = "strip"
        elif qm == "raw":
            key = "raw" if st in ("ok", "offset", "adv") else st
        else:
            key = {"ok": "ok", "offset": "ok", "adv": "ok", "ann": "ok", "corrupt": "ok"}.get(st, st)
        if st in ("ann", "corrupt"):
            k2 = "raw" if qm == "raw" else "ok"
            ids = [int(i) for i in self.z[f"D_{k2}_ids"][n]]
            sc = self.z[f"D_{k2}_sc"][n]
            if st == "ann":
                keep = [j for j in range(48) if (n * 7919 + ids[j] * 104729) % 100 < 50]
                pos = sorted(keep + list(range(48, len(ids)))[: 48 - len(keep)])
            else:
                pos = [j for j in range(len(ids)) if (ids[j] * 2654435761 % 1000) >= 300][:48]
            return self._store([ids[j] for j in pos], w), np.array([sc[j] for j in pos], np.float32)
        if st == "adv":
            rel = set(self.qs[n]["rel"])
            b = self._lst("L_ok", n)
            out = [i for i in b[20:] if i not in rel][:DEPTH]
            return self._store(out, w), np.full(len(out), 0.9, np.float32)
        ids = self.z[f"D_{key}_ids"][n][:DEPTH]
        sc = self.z[f"D_{key}_sc"][n][:DEPTH]
        out = [int(i) for i in ids]
        if st == "offset":
            out = [(i + 1) % self.N for i in out]
        return self._store(out, w), sc

    def D(self, n, w, floor=FLOOR):
        ids, sc = self.D_raw(n, w)
        return [i for i, x in zip(ids, sc) if x >= floor]

    def _store(self, ids, w):
        if w.get("S") == "shift":
            return [(i + 1) % self.N for i in ids]
        return ids

    def qvector(self, n, w):
        qm = self._qmode(w)
        st = w.get("D", "ok")
        if qm == "strip":
            return self.qvec["strip"][n]
        key = {"ok": "ok", "offset": "ok", "adv": "ok", "ann": "ok", "corrupt": "ok"}.get(st, st)
        if qm == "raw" and key == "ok":
            key = "raw"
        return self.qvec[key][n]

    def rscore(self, n, i, w):
        st = w.get("R", "ok")
        if w.get("S") == "shift":
            i = (i - 1) % self.N
        tag = "trunc" if st == "trunc" else "ok"
        v = self.rr.get(f"{tag}|{n}:{i}")
        if v is None:
            return None
        if st in ("ok", "trunc"):
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

    def healthy_pairs(self):
        """Pairs of the healthy normal head and the BM25 head (all that the truncated-reranker world needs)."""
        need = set()
        for n, q in enumerate(self.qs):
            L = self.L(n, {})
            need.update((n, i) for h in (C.rrf([self.D(n, {}), L]), L) for i in h)
        return sorted(need)

    def needed_pairs(self):
        import t1_precompute as P
        need = set()
        calib, test = SPLITS[self.name]
        for n, q in enumerate(self.qs):        # calibration / canary questions: healthy heads only
            if q["split"] in calib:
                L = self.L(n, {})
                D = self.D(n, {})
                for h in (C.rrf([D, L]), L):
                    need.update((n, i) for i in h)
        for wn, w in WORLDS.items():
            if w.get("R") or w.get("S") or wn not in RERANK_WORLDS:
                continue
            for n, q in enumerate(self.qs):
                if q["split"] not in test:
                    continue
                for ww in (w, dict(w, Qmode="raw")):
                    L = self.L(n, ww)
                    Draw, dsc = self.D_raw(n, ww)
                    D = [i for i, x in zip(Draw, dsc) if x >= FLOOR]
                    for a in P.LIST_ACTIONS:
                        for i in P.served(self, n, ww, a, L, D):
                            need.add((n, i))
                for i in self.L(n, {}):
                    need.add((n, i))
        return sorted(need)
