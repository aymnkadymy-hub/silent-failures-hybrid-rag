"""Precompute, for every T1 fault world, every governor action and every question: the served head's first-gold rank,
the reranked top-8 first-gold rank (where reranker scores exist), the label-free signals and the canary probes.

Output: runs/t1_tables/<world>.npz  (one file per world; arrays indexed by question number)
"""
import json
import os
import sys

import numpy as np

import frg_core as C
import t1_data as T

BETAS = [0.5, 0.625, 0.75, 0.875]
RBETAS = [0.5, 0.75]
LIST_ACTIONS = ["N"] + [f"Dg{b}" for b in BETAS] + ["Doff"] + [f"Lg{b}" for b in BETAS] + ["Loff"]
R_MODES = ["on", "off"] + [f"g{b}" for b in RBETAS]

WORLDS = {
    "healthy": {},
    "D_minilm": {"D": "minilm"}, "D_unk": {"D": "unk"}, "D_rand1": {"D": "rand1"}, "D_offset": {"D": "offset"}, "D_adv": {"D": "adv"}, "D_ann": {"D": "ann"}, "D_corrupt": {"D": "corrupt"},
    "L_empty": {"L": "empty"}, "L_shard": {"L": "shard"}, "L_cache": {"L": "cache"},
    "R_random": {"R": "random"}, "R_inverted": {"R": "inverted"}, "R_constant": {"R": "constant"},
    "Q_strip": {"Q": "strip"}, "S_shift": {"S": "shift"},
}


def served(t, n, w, act, L, D):
    if act == "Qbyp":                         # main normaliser bypassed: independent minimal normaliser on both arms
        w2 = dict(w, Qmode="raw")
        return C.rrf([t.D(n, w2), t.L(n, w2)])
    F = C.rrf([D, L])
    if act == "N":
        return F
    if act == "Doff":
        return list(L)
    if act == "Loff":
        return list(D)
    b = float(act[2:])
    if act.startswith("Dg"):
        return C.constrained(L, F, b)
    if act.startswith("Lg"):
        return C.constrained(D, F, b)
    raise ValueError(act)


def rerank(t, n, w, head):
    sc = [t.rscore(n, i, w) for i in head]
    if any(s is None for s in sc):
        return None, None
    order = [i for _, i in sorted(zip(sc, head), key=lambda x: -x[0])]   # stable on ties (constant scores keep head order)
    return order, sc


def final(t, n, w, head, mode, rorder):
    if mode == "off":
        return head[:8]
    if rorder is None:
        return None
    if mode == "on":
        return rorder[:8]
    return C.constrained(head, rorder, float(mode[1:]), depth=8)


def run_world(t, wname, extra=None):
    w = WORLDS[wname] if extra is None else extra
    nq = len(t.qs)
    acts = LIST_ACTIONS + (["Qbyp"] if getattr(t, "has_bypass", False) else [])
    pool = {a: np.zeros(nq, np.int16) for a in acts}
    fin = {(a, m): np.full(nq, -1, np.int16) for a in acts for m in R_MODES}
    sig = {k: np.zeros(nq, np.float32) for k in ["a_LD", "a_RL", "a_RD", "a_RF", "nL", "nD", "d1", "rstd", "has_r", "disp20"]}
    can = {k: np.zeros(nq, np.int8) for k in ["D10", "L10", "R8", "E8", "has_r", "L10byp"]}
    kat = {k: np.zeros(nq, np.int8) for k in ["D", "L", "R", "S", "ok_r", "Dvec", "Dsrch", "Lraw"]}
    for n, q in enumerate(t.qs):
        rel = t.gold(n)
        if not rel:
            continue
        L = t.L(n, w)
        Draw, dsc = t.D_raw(n, w)
        D = [i for i, x in zip(Draw, dsc) if x >= T.FLOOR]
        F = None
        for a in acts:
            h = served(t, n, w, a, L, D)
            if a == "N":
                F = h
            r = C.first_hit(h, rel)
            pool[a][n] = r or 0
            ro, _ = rerank(t, n, w, h)
            for m in R_MODES:
                f = final(t, n, w, h, m, ro)
                if f is not None:
                    fin[(a, m)][n] = C.first_hit(f, rel) or 0
        # label-free signals on the normal pipeline (shadow)
        ro, sc = rerank(t, n, w, F)
        sig["a_LD"][n] = C.overlap(L, Draw)
        sig["nL"][n] = len(L)
        sig["disp20"][n] = len(set(L[:20]) - set(F[:20]))      # BM25 top-20 items that fusion pushed out of the top 20
        sig["nD"][n] = len(D)
        sig["d1"][n] = float(dsc[0]) if len(dsc) else 0.0
        if ro is not None:
            R8 = set(ro[:8])
            sig["a_RL"][n] = len(R8 & set(L))
            sig["a_RD"][n] = len(R8 & set(Draw))
            sig["a_RF"][n] = len(R8 & set(F[:16]))
            sig["rstd"][n] = float(np.std(sc))
            sig["has_r"][n] = 1
        # canary probes: each component alone
        can["D10"][n] = int(C.first_hit(Draw[:10], rel) is not None)
        can["L10"][n] = int(C.first_hit(L[:10], rel) is not None)
        if getattr(t, "has_bypass", False):
            can["L10byp"][n] = int(C.first_hit(t.L(n, dict(w, Qmode="raw"))[:10], rel) is not None)
        Bh = t.L(n, {k: v for k, v in w.items() if k in ("S",)})   # reranker probe input: the healthy BM25 head
        rob, _ = rerank(t, n, w, Bh)
        if rob is not None:
            can["R8"][n] = int(C.first_hit(rob[:8], rel) is not None)
            can["has_r"][n] = 1
        if ro is not None:
            can["E8"][n] = int(C.first_hit(ro[:8], rel) is not None)
        # known-answer attestation of question n used as a probe: compare with its healthy references
        h_v = t.qvector(n, {})
        w_v = t.qvector(n, w)
        cosv = float(np.dot(h_v, w_v[: len(h_v)]) / (np.linalg.norm(h_v) * np.linalg.norm(w_v) + 1e-12)) if len(w_v) == len(h_v) else 0.0
        Dh, _ = t.D_raw(n, {})
        kat["D"][n] = int(cosv < 0.99 or (len(Draw) == 0) or (len(Dh) > 0 and Draw[0] != Dh[0]))
        Lh = t.L(n, {})
        kat["L"][n] = int(L[:5] != Lh[:5])
        if Lh:
            s0 = t.rscore(n, Lh[0], {})
            s1 = t.rscore(n, Lh[0], w)
            if s0 is not None and s1 is not None:
                kat["R"][n] = int(abs(s1 - s0) > 0.05)
                kat["ok_r"][n] = 1
        kat["S"][n] = int(w.get("S") == "shift")
        kat["Dvec"][n] = int(cosv < 0.99)
        # index searched with the STORED healthy reference vector: only index/id-map/store faults change its top-1
        w_idx = {k: v for k, v in w.items() if k == "S" or (k == "D" and v in ("offset", "ann", "corrupt", "adv"))}
        Ds, _ = t.D_raw(n, w_idx)
        kat["Dsrch"][n] = int(len(Ds) == 0 or (len(Dh) > 0 and Ds[0] != Dh[0]))
        # lexical top-5 through the independent normaliser (bypasses the main normaliser)
        w_raw = {k: v for k, v in w.items() if k != "Q"}
        if getattr(t, "has_bypass", False):
            w_raw["Qmode"] = "raw"
            ref_raw = t.L(n, {"Qmode": "raw"})
        else:
            ref_raw = Lh
        kat["Lraw"][n] = int(t.L(n, w_raw)[:5] != ref_raw[:5])
    out = {f"pool|{a}": v for a, v in pool.items()}
    out.update({f"kat|{k}": v for k, v in kat.items()})
    out.update({f"fin|{a}|{m}": v for (a, m), v in fin.items()})
    out.update({f"sig|{k}": v for k, v in sig.items()})
    out.update({f"can|{k}": v for k, v in can.items()})
    return out


def main():
    worlds = sys.argv[1:] or list(WORLDS)
    extra_dense = [w[2:] for w in worlds if w.startswith("D_mix") or w.startswith("D_noise")]
    t = T.T1(extra_dense=extra_dense)
    od = os.path.join(T.RUNS, "t1_tables")
    os.makedirs(od, exist_ok=True)
    for wn in worlds:
        ex = None
        if wn not in WORLDS:
            ex = {"D": wn[2:]}
            t.qvec[wn[2:]] = np.load(os.path.join(T.RUNS, f"Q_{wn[2:]}.npy"))
        out = run_world(t, wn, ex)
        np.savez_compressed(os.path.join(od, f"{wn}.npz"), **out)
        dev = [n for n, q in enumerate(t.qs) if q["rel"] and q["set"] == "n5" and q["split"] == "dev"]
        h20 = np.mean([0 < out["pool|N"][n] <= 20 for n in dev]) * 100
        print(wn, f"N5dev pool Hit@20 normal = {h20:.1f}", flush=True)


if __name__ == "__main__":
    main()
