"""T2/T3 experiments on a recomputed testbed: precompute tables, certify, test with the governor constants frozen on T1.

  python3 tb_run.py NAME precompute
  python3 tb_run.py NAME test [rerank]     (SCEN=..., SEEDS=...)
No re-tuning: delta, h and lat come from runs/frozen_params.json (T1 dev). Each testbed calibrates only its healthy
signal baseline, its canary hit rates and its certified beta on its own calibration split (the deployment's own
traffic), as the plan prescribes.
"""
import json
import os
import sys

import numpy as np

import governor as G
import t1_precompute as P
import t1_run as R
import tb_data as TD

RUNS = TD.RUNS


def tabdir(name):
    return os.path.join(RUNS, f"tb_{name}", "tables")


def precompute(name):
    t = TD.TB(name)
    os.makedirs(tabdir(name), exist_ok=True)
    calib, test = TD.SPLITS[name]
    te = [n for n, q in enumerate(t.qs) if q["split"] in test]
    for wn, w in TD.WORLDS.items():
        out = P.run_world(t, wn, extra=w)
        np.savez_compressed(os.path.join(tabdir(name), f"{wn}.npz"), **out)
        h = np.mean([0 < out["pool|N"][n] <= 20 for n in te]) * 100
        f = out["fin|N|on"][te]
        f8 = np.mean((f[f >= 0] > 0) & (f[f >= 0] <= 8)) * 100 if (f >= 0).any() else float("nan")
        print(name, wn, f"test pool Hit@20 {h:.1f}  final Hit@8 {f8:.1f} (cov {(f >= 0).mean():.2f})", flush=True)


def load(name):
    t = TD.TB(name, with_rr=False)
    tab = {f[:-4]: dict(np.load(os.path.join(tabdir(name), f))) for f in os.listdir(tabdir(name)) if f.endswith(".npz")}
    V = {}
    for wn, w in TD.WORLDS.items():
        M = np.stack([t.qvector(n, w) for n in range(len(t.qs))]).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        if M.shape[1] < 1024:
            M = np.pad(M, ((0, 0), (0, 1024 - M.shape[1])))
        V[wn] = M

    def qcol(worlds_b, idx_b):
        X = np.stack([V[w][i] for w, i in zip(worlds_b, idx_b)])
        s = X.sum(0)
        n = len(X)
        return float((s @ s - n) / (n * (n - 1)))

    def pair(ws, ps):
        v = [V[w][p] for w, p in zip(ws, ps)]
        r = [V["healthy"][p] for p in ps]
        return bool(float(v[0] @ v[1]) > 0.95 and float(r[0] @ r[1]) < 0.8)
    qcol.pair = pair
    deg = V["D_unk"].mean(0)
    deg = deg / (np.linalg.norm(deg) + 1e-12)          # degenerate vector: mean of all-unknown-token encodings
    qcol.deg = deg
    qcol.V = V
    return t.qs, tab, qcol


def banks(name, qs):
    calib, test = TD.SPLITS[name]
    cal = np.array([n for n, q in enumerate(qs) if q["split"] in calib])
    ncan = 30 if name == "quran" else 60
    can = np.sort(np.random.default_rng(7).choice(cal, ncan, replace=False))
    cal = np.setdiff1d(cal, can)
    te = np.array([n for n, q in enumerate(qs) if q["split"] in test])
    # benign shift for testbeds without a dialect split: short queries first, then long ones (median split)
    # (query length is only a proxy for a change in what users ask; it uses no relevance labels)
    import tb_build
    _, qfull, _ = tb_build.load(name)
    qlen = {q["id"]: len(q["q"].split()) for q in qfull}
    srt = np.array(sorted(te, key=lambda n: (qlen[qs[n]["id"]], n)))
    half = len(srt) // 2
    return {"dev": cal, "ho": te, "ho_f": srt[:half], "ho_i": srt[half:], "books": te}, can


def scenarios(name):
    base = "ho"
    S = {"S1_healthy": [(2000, base, "healthy")],
         "S2_shift_split": [(1000, "ho_f", "healthy"), (1000, "ho_i", "healthy")]}
    for f in ["D_minilm", "D_unk", "D_rand", "D_offset", "D_adv", "D_ann", "D_corrupt", "D_noise", "L_empty", "L_shard", "L_cache", "L_nofold",
              "R_random", "R_inverted", "R_constant", "R_trunc", "Q_strip", "S_shift"]:
        S[f"S3_{f}"] = [(500, base, "healthy"), (1000, base, f), (1000, base, "healthy")]
    S["S4_decay_D_mix"] = [(500, base, "healthy"), (250, base, "D_mix25"), (250, base, "D_mix50"), (250, base, "D_mix75"),
                           (250, base, "D_unk"), (1000, base, "healthy")]
    S["S5_partial30_D_unk"] = [(500, base, "healthy"), (1000, base, R.frac("D_unk", 0.3)), (1000, base, "healthy")]
    for f in ["D_unk", "L_nofold"]:
        for dur in (300, 3000):
            S[f"S3dur{dur}_{f}"] = [(500, base, "healthy"), (dur, base, f), (1000, base, "healthy")]
    return S


def main():
    name, mode = sys.argv[1], sys.argv[2]
    if mode == "precompute":
        precompute(name)
        return
    qs, tab, qcol = load(name)
    bk, can = banks(name, qs)
    cert = R.certify(tab, {"dev": bk["dev"], "ho": bk["ho"]})
    json.dump(cert, open(os.path.join(RUNS, f"tb_{name}", "certify.json"), "w"), indent=1)
    fr = json.load(open(os.path.join(RUNS, "frozen_params.json")))
    fr = dict(fr, beta_eps=float(cert["beta_eps"][0.02]))
    use_r = len(sys.argv) > 3 and sys.argv[3] == "rerank"
    seeds = int(os.environ.get("SEEDS", "100"))
    scen = scenarios(name)
    only = os.environ.get("SCEN")
    if only:
        scen = {k: v for k, v in scen.items() if k in only.split(",")}
    if not use_r:
        scen = {k: v for k, v in scen.items() if "R_" not in k}
    for k in list(scen):
        segs = scen[k]
        scen[k] = segs
    R.DEPLOY.update({"D_offset": "D", "L_nofold": "L", "R_trunc": "R"})
    pols = os.environ.get("POLS")
    R.QCOLLAPSE = (lambda w, i: float(qcol.V[w][i] @ qcol.deg)) if hasattr(qcol, "deg") else None
    res, meta = R.run_suite(qs, tab, qcol, bk, can, fr, scen, seeds, use_r, fr["beta_eps"],
                            pols=pols.split(",") if pols else None)
    tag = ("rerank" if use_r else "pool") + (("_" + os.environ["TAG"]) if os.environ.get("TAG") else "")
    od = os.path.join(RUNS, "v2", f"tb_{name}") if os.environ.get("RUNVER") == "v2" else os.path.join(RUNS, f"tb_{name}")
    os.makedirs(od, exist_ok=True)
    json.dump({"meta": meta, "params": fr, "cert": cert, "agg": {s: {p: R.agg(v) for p, v in d.items()} for s, d in res.items()}},
              open(os.path.join(od, f"test_{tag}.json"), "w"), indent=1, default=str)
    json.dump(res, open(os.path.join(od, f"test_{tag}_rows.json"), "w"), default=str)


if __name__ == "__main__":
    main()
