"""T1 experiments: certification of beta_eps, tuning of the governor on N5 dev streams, test scenarios.

  python3 t1_run.py certify   -> runs/t1_certify.json
  python3 t1_run.py tune      -> runs/frozen_params.json (dev streams only)
  python3 t1_run.py test      -> runs/t1_test.json  (needs frozen_params.json)
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist

import frg_core as C
import governor as G
import t1_data as T
import t1_precompute as P

RUNS = T.RUNS
TAB = os.path.join(RUNS, "t1_tables")
BETAS = [0.5, 0.625, 0.75, 0.875, 1.0]
K = 20


def load():
    qs = json.load(open(os.path.join(T.COPIES, "questions.json"), encoding="utf-8"))
    tab = {f[:-4]: dict(np.load(os.path.join(TAB, f))) for f in os.listdir(TAB) if f.endswith(".npz")}
    X1 = T.X1
    qv = {"on": np.load(os.path.join(X1, "Q_arctic.npy")), "unk": np.load(os.path.join(X1, "Q_unk.npy")),
          "minilm": np.load(os.path.join(X1, "Q_minilm.npy"))}
    r = np.random.default_rng(11).standard_normal(qv["on"].shape).astype(np.float32)
    qv["rand"] = r / np.linalg.norm(r, axis=1, keepdims=True)
    for k in list(qv):
        v = qv[k] / np.linalg.norm(qv[k], axis=1, keepdims=True)
        if v.shape[1] < 1024:                # mixed batches (ramps, partial faults): pad the 384-d MiniLM vectors
            v = np.pad(v, ((0, 0), (0, 1024 - v.shape[1])))
        qv[k] = v.astype(np.float32)
    key = {"D_unk": "unk", "Q_strip": "unk", "D_minilm": "minilm", "D_rand1": "rand"}

    def qcol(worlds_b, idx_b):
        V = np.stack([qv[key.get(w, "on")][i] for w, i in zip(worlds_b, idx_b)])
        s = V.sum(0)
        n = len(V)
        return float((s @ s - n) / (n * (n - 1)))

    def pair(ws, ps):
        v = [qv[key.get(w, "on")][p] for w, p in zip(ws, ps)]
        r = [qv["on"][p] for p in ps]
        return bool(float(v[0] @ v[1]) > 0.95 and float(r[0] @ r[1]) < 0.8)
    qcol.pair = pair
    return qs, tab, qcol


def banks(qs):
    def sel(f):
        return np.array([n for n, q in enumerate(qs) if q["rel"] and f(q)], dtype=np.int64)
    dev = sel(lambda q: q["set"] == "n5" and q["split"] == "dev")
    rng = np.random.default_rng(7)
    canary = np.sort(rng.choice(dev, 60, replace=False))
    dev = np.setdiff1d(dev, canary)
    b = {
        "dev": dev,
        "dev_f": np.array([n for n in dev if qs[n]["lang"] == "fusha"]),
        "dev_i": np.array([n for n in dev if qs[n]["lang"] == "iraqi"]),
        "ho": sel(lambda q: q["set"] == "n5" and q["split"] == "heldout"),
        "ho_f": sel(lambda q: q["set"] == "n5" and q["split"] == "heldout" and q["lang"] == "fusha"),
        "ho_i": sel(lambda q: q["set"] == "n5" and q["split"] == "heldout" and q["lang"] == "iraqi"),
        "books": sel(lambda q: q["set"] == "books_v1"),
    }
    return b, canary


# ---------------------------------------------------------------------------------------------------------------
def cp_upper(x, n, delta):
    return 1.0 if x >= n else float(beta_dist.ppf(1 - delta, x + 1, n - x))


def certify(tab, bk, eps_levels=(0.01, 0.02, 0.05), delta=0.05):
    """Worst-case (any dense list) loss vs BM25@K at reservation beta: 1{m(beta,K) < r* <= K}."""
    r = tab["healthy"]["pool|Doff"]        # BM25 alone in BM25 order: first gold rank
    res = {"m": {b: math.ceil(b * K - 1e-9) for b in BETAS}, "banks": {}}     # ceil(beta k): the theorem's band edge
    assert all(res["m"][b] == C.m_reserved(b, K) for b in BETAS if b >= 0.5)  # equals the dead-arm count on this grid
    for name, idx in bk.items():
        rr = r[idx]
        row = {}
        for b in BETAS:
            m = res["m"][b]
            x = int(((rr > m) & (rr <= K)).sum())
            row[b] = {"loss": x, "n": len(idx), "rate": x / len(idx),
                      "ucb": cp_upper(x, len(idx), delta / len(BETAS))}
        res["banks"][name] = row
    cal = res["banks"]["dev"]
    res["beta_eps"] = {e: min([b for b in BETAS if cal[b]["ucb"] <= e] or [1.0]) for e in eps_levels}
    return res


# ---------------------------------------------------------------------------------------------------------------
def make_stream(bk, rng, segs):
    """segs: list of (length, bank, world or callable(t_in_seg, rng)->world, partial_filter)"""
    idx, worlds = [], []
    for length, bank, world in segs:
        ii = rng.choice(bk[bank], length)
        for j, i in enumerate(ii):
            w = world(j, rng, i) if callable(world) else world
            idx.append(int(i))
            worlds.append(w)
    return np.array(idx), worlds


def ramp(fault, length):
    return lambda j, rng, i: fault if rng.random() < min(1.0, j / length) else "healthy"


def only_lang(fault, lang, qs):
    f = lambda j, rng, i: fault if qs[i]["lang"] == lang else "healthy"
    f.probe_pred = lambda i: qs[i]["lang"] == lang       # the same content rule applies to probe questions
    return f


def frac(fault, p):
    return lambda j, rng, i: fault if rng.random() < p else "healthy"


def scenarios(qs, base="ho", shift_to="ho_i", shift_from="ho_f", books="books"):
    S = {
        "S1_healthy": [(2000, base, "healthy")],
        "S2_shift_dialect": [(1000, shift_from, "healthy"), (1000, shift_to, "healthy")],
        "S2_shift_books": [(1000, base, "healthy"), (1000, books, "healthy")],
    }
    for f in ["D_minilm", "D_unk", "D_rand1", "D_offset", "D_adv", "D_ann", "D_corrupt", "L_empty", "L_shard", "L_cache",
              "R_random", "R_inverted", "R_constant", "Q_strip", "S_shift"]:
        S[f"S3_{f}"] = [(500, base, "healthy"), (1000, base, f), (1000, base, "healthy")]
    for f in ["D_unk", "D_minilm"]:
        S[f"S4_ramp_{f}"] = [(500, base, "healthy"), (1000, base, ramp(f, 1000)), (500, base, f), (500, base, "healthy")]
    S["S5_partial_iraqi_D_unk"] = [(500, base, "healthy"), (1000, base, only_lang("D_unk", "iraqi", qs)), (1000, base, "healthy")]
    S["S5_partial30_D_unk"] = [(500, base, "healthy"), (1000, base, frac("D_unk", 0.3)), (1000, base, "healthy")]
    for f in ["D_unk", "L_shard"]:
        for dur in (300, 3000):
            S[f"S3dur{dur}_{f}"] = [(500, base, "healthy"), (dur, base, f), (1000, base, "healthy")]
    S["S9_shift_then_D_unk"] = [(1000, shift_from, "healthy"), (200, shift_to, "healthy"), (800, shift_to, "D_unk"),
                                (500, shift_to, "healthy")]
    return S


DEPLOY = {"D_minilm": "D", "D_unk": "D", "D_offset": "D", "L_shard": "L", "L_nofold": "L", "Q_strip": "Q",
          "R_random": "R", "R_inverted": "R", "R_constant": "R", "R_trunc": "R", "S_shift": "S"}


def stream_events(worlds, sd):
    """Fingerprint events: a deployment fault changes its component's fingerprint at onset and at repair (the fix is a
    deployment too). Streams with no deployment fault get two benign deployments at random times."""
    ev = []
    prev = "healthy"
    for t, w in enumerate(worlds):
        if w != prev:
            for x in (w, prev):
                if x in DEPLOY and (prev == "healthy" or w == "healthy"):
                    ev.append((t, DEPLOY[x]))
                    break
        prev = w
    if not ev:
        r = np.random.default_rng(5000 + sd)
        ev = [(int(t), str(c)) for t, c in zip(r.integers(100, len(worlds) - 100, 2), r.choice(["D", "L"], 2))]
    return sorted(ev)


def CF_CAUSE(w):
    import cafi as CF
    return CF.FAMILY_CAUSE.get(w, CF.HELD_OUT.get(w, "UNK"))


def v2metrics(out, worlds, idx, env, qacts):
    """damage per 1,000 fault queries (gold in healthy RRF top 20 but not served), availability, regret (points)."""
    hr = env.tab["healthy"]["pool|N"][np.asarray(idx)]
    hh = ((hr > 0) & (hr <= 20)).astype(float)
    sv = out["h20"].astype(float)
    f = np.array([w != "healthy" for w in worlds])
    res = {"availability": 1.0 - (np.mean([a[0] == "ABST" for a in qacts]) if qacts else 0.0)}
    res["damage"] = float(((hh == 1) & (sv == 0))[f].mean() * 1000) if f.any() else None
    res["regret"] = float((hh - sv)[~f].mean() * 100) if (~f).any() else None
    return res


def exposed(acts, worlds, qacts=None):
    if qacts is not None:
        n = 0
        for (la, rm), w in zip(qacts, worlds):
            if w == "healthy":
                continue
            n += (rm == "on") if w.startswith("R_") else (la == "N")
        return n
    return exposed_batch(acts, worlds)


def exposed_batch(acts, worlds):
    """Fault-window queries served WITHOUT containing the faulty component: a dense/lexical/query/store fault served by
    plain fusion, or a reranker fault served with the reranker on ("affected before detection")."""
    act_at = {}
    for s, la, rm in acts:
        act_at[s] = (la, rm)
    starts = sorted(act_at)
    n = 0
    cur = None
    j = 0
    for t, w in enumerate(worlds):
        while j < len(starts) and starts[j] <= t:
            cur = act_at[starts[j]]
            j += 1
        if w == "healthy" or cur is None:
            continue
        la, rm = cur
        if w.startswith("R_"):
            n += rm == "on"
        else:
            n += la == "N"
    return n


def fault_window(worlds):
    f = np.array([w != "healthy" for w in worlds])
    return f


# ---------------------------------------------------------------------------------------------------------------
def build_env(qs, tab, qcol, canary, use_r):
    p0 = {"D": float(tab["healthy"]["can|D10"][canary].mean()), "L": float(tab["healthy"]["can|L10"][canary].mean()),
          "R": float(tab["healthy"]["can|R8"][canary].mean())}
    return G.Env(tab, qcol, canary, p0, use_r)


def best_actions(tab, idx):
    """Oracle: per world, the (list action, rerank mode) with the best mean pool Hit@20 on the given bank; reranked
    Hit@8 breaks ties and decides the rerank mode where scores exist."""
    best = {}
    for w, t in tab.items():
        cands = []
        for la in P.LIST_ACTIONS + (["Qbyp"] if "pool|Qbyp" in t else []):
            h20 = float(np.mean((t[f"pool|{la}"][idx] > 0) & (t[f"pool|{la}"][idx] <= 20)))
            for rm in P.R_MODES:
                f = t[f"fin|{la}|{rm}"][idx]
                ok = f >= 0
                f8 = float(np.mean((f[ok] > 0) & (f[ok] <= 8))) if ok.mean() > 0.95 else -1
                cands.append((h20, f8, la, rm))
        h20, f8, la, rm = max(cands)
        best[w] = (la, rm)
    return best


def summarize(out, valid, worlds, pol):
    f = fault_window(worlds)
    T_ = len(worlds)
    res = {"h20": float(out["h20"].mean()), "h48": float(out["h48"].mean()),
           "h20_fault": float(out["h20"][f].mean()) if f.any() else None,
           "h20_healthy": float(out["h20"][~f].mean()) if (~f).any() else None,
           "f8": float(out["f8"][valid].mean()) if valid.mean() > 0.95 else None,
           "f8_cov": float(valid.mean()), "canary": pol.canary_calls, "kat_calls": getattr(pol, "kat_calls", 0)}
    ev = pol.log
    on = int(np.argmax(f)) if f.any() else None
    conf = [t for t, e, c in ev if e == "confirmed"]
    sus = [t for t, e, c in ev if e in ("suspect", "alarm", "confirmed")]
    res["false_confirm"] = sum(1 for t, e, c in ev if e == "confirmed" and (on is None or t < on))
    res["false_suspect"] = sum(1 for t, e, c in ev if e in ("suspect", "alarm") and (on is None or t < on))
    if on is not None:
        res["delay_suspect"] = next((t - on for t in sus if t >= on), None)
        res["delay_confirm"] = next((t - on for t in conf if t >= on), None)
        res["confirm_comp"] = next((c for t, e, c in ev if e == "confirmed" and t >= on), None)
        off = T_ - int(np.argmax(f[::-1]))  # first healthy index after the last fault
        rec = [t for t, e, c in ev if e in ("recovered", "back_on") and t > off]
        res["delay_recover"] = (rec[0] - off) if rec else None
    return res


def policies(env, cal, theta20, qthr, beta_eps, prm, best, use_r):
    kw = dict(delta=prm["delta"], h=prm["h"], lat=prm["lat"], use_r=use_r)
    cafi = {}
    if prm.get("cafi_model") is not None:
        import cafi as CF
        for rho in (1.0, 0.5, 0.2, 0.1):
            cafi[f"CIFC_r{int(rho * 100)}"] = (lambda rho=rho: CF.CAFI(env, cal, beta_eps, prm["cafi_model"], rho=rho,
                                                                         seed=prm.get("_seed", 0), **kw))
        cafi["CIFC_noint"] = lambda: CF.CAFI(env, cal, beta_eps, prm["cafi_model"], rho=0.2,
                                             exclude=("t_srch", "t_lexraw", "t_col"), **kw)
        cafi["CIFC_grade"] = lambda: CF.CAFI(env, cal, beta_eps, prm["cafi_model"], rho=0.2, graded=True, **kw)
        cafi["CIFC_gradec"] = lambda: CF.CAFI(env, cal, beta_eps, prm["cafi_model"], rho=0.2, graded=True,
                                              grade_cause=True, **kw)
        if prm.get("amodel") is not None:
            cafi["CIFC_Q"] = lambda: CF.CIFCQ2(env, cal, beta_eps, prm["cafi_model"], prm["amodel"], rho=0.2, **kw)
        cafi["CIFC_rand"] = lambda: CF.CAFI(env, cal, beta_eps, prm["cafi_model"], rho=0.2, random_select=True, **kw)
    return {**cafi, **{
        "P0_rrf": lambda: G.Static("N"),
        "P1_bm25": lambda: G.Static("Doff"),
        "P2_dense": lambda: G.Static("Loff"),
        "P3_deadarm": lambda: G.DeadArmAlg1(env, theta20, lat=prm["lat"]),
        "P3r_deadarm_back": lambda: G.DeadArmAlg1(env, theta20, lat=prm["lat"], back_after=1000),
        "P4_guard_0.75": lambda: G.Static("Dg0.75"),
        "P6_healthcheck": lambda: G.HealthCheck(env, qthr),
        "P7_farg_static": lambda: G.Static(f"Dg{beta_eps}" if beta_eps < 1 else "Doff"),
        "G": lambda: G.Governor(env, cal, beta_eps, **kw),
        "G_reg": lambda: G.Governor(env, cal, beta_eps, legacy=True, **kw),
        "G_eps05": lambda: G.Governor(env, cal, prm.get("beta_eps05", 0.75), **kw),
        "G_dt": lambda: G.Governor(env, cal, beta_eps, dt_curves=prm.get("dt"), **kw),
        "G_disp": lambda: G.Governor(env, cal, beta_eps, extra=("disp20",), **kw),
        "G_vgate": lambda: G.Governor(env, cal, beta_eps, vgate=True, **kw),
        "G_fastrec": lambda: G.Governor(env, cal, beta_eps, vperiod=40, probation=40, **kw),
        "G_kat": lambda: G.Governor(env, cal, beta_eps, kat=True, **kw),
        "G_noshift": lambda: G.Governor(env, cal, beta_eps, shift_rule=False, **kw),
        "G_fixedb_hard": lambda: G.Governor(env, cal, beta_eps, fixed_beta=1.0, **kw),
        "G_norecov": lambda: G.Governor(env, cal, beta_eps, recover=False, **kw),
        "O_oracle": lambda: G.Oracle(best),
        "CB": lambda: G.CircuitBreaker(env),
    }}


def collapse_rate(env, tab, f, pool, pairs=200):
    r = np.random.default_rng(17)
    return float(np.mean([env.collapse([f, f], list(r.choice(pool, 2, replace=False))) for _ in range(pairs)]))


QCOLLAPSE = None


def run_suite(qs, tab, qcol, bk, canary, prm, scen, seeds, use_r, beta_eps, cal_bank="dev", pols=None):
    env = build_env(qs, tab, qcol, canary, use_r)
    if QCOLLAPSE is not None:
        env.qcollapse = QCOLLAPSE
    env.collapse = lambda ws, ps: qcol.pair(ws, ps) if hasattr(qcol, "pair") else False
    rng = np.random.default_rng(1)
    cal, vals = G.calibrate(env, bk[cal_bank], 3000, rng, use_r, extra=("disp20",))
    rngc = np.random.default_rng(2)
    m = [float(np.mean(tab["healthy"]["sig|a_LD"][rngc.choice(bk[cal_bank], 20)])) for _ in range(10000)]
    theta20 = float(np.percentile(m, 1))
    qthr = float(np.percentile(vals["qcol"], 99.9)) + 5 * cal.sd["qcol"]
    best = best_actions(tab, bk[cal_bank])
    prm = dict(prm)
    ci = bk[cal_bank]
    r = tab["healthy"]["pool|Doff"][ci]
    prm["dt"] = {b: (float(np.mean((r > 0) & (r <= C.m_reserved(b, K)))),
                     float(np.mean((lambda a: (a > 0) & (a <= 20))(tab["healthy"][f"pool|{('Dg' + str(b)) if b < 1 else 'Doff'}"][ci]))))
                 for b in BETAS}
    if os.environ.get("CAFI", "1") == "1" and "kat|Dvec" in tab["healthy"]:
        import cafi as CF
        acts = ["N", "Dg0.75", "Doff", "Lg0.75", "Loff", "ABST"] + (["Qbyp"] if "pool|Qbyp" in tab["healthy"] else [])
        noise = prm.get("probe_noise")
        prm["cafi_model"] = CF.Model(tab, bk[cal_bank], canary, env.p0, lambda f: collapse_rate(env, tab, f, canary),
                                     use_r, acts, noise=noise)
    if hasattr(env, "qcollapse") and "kat|Dvec" in tab["healthy"]:
        import cafi as CF
        prm["amodel"] = CF.AnomalyModel(tab, bk[cal_bank], env.qcollapse)
        prm["qmodel"] = None
        ci = bk[cal_bank]
        hv = lambda w, a: float(np.mean((tab[w][f"pool|{a}"][ci] > 0) & (tab[w][f"pool|{a}"][ci] <= 20)))
        dead = [w for w in ("D_unk", "D_minilm", "D_rand", "D_rand1", "D_noise", "D_offset") if w in tab]
        gh = hv("healthy", "N") - hv("healthy", "Doff")
        gd = float(np.mean([hv(w, "Doff") - hv(w, "N") for w in dead]))
        prm["qtau"] = gh / (gh + gd) if (gh + gd) > 0 and gd > 0 else 1.01
    P_ = policies(env, cal, theta20, qthr, beta_eps, prm, best, use_r)
    if os.environ.get("POLSET") == "v2":
        keep = {"P0_rrf", "P1_bm25", "P3_deadarm", "G", "G_kat", "CB", "CIFC_r20", "CIFC_noint", "CIFC_grade",
                "CIFC_gradec", "CIFC_Q", "O_oracle"}
        P_ = {k: v for k, v in P_.items() if k in keep}
    if pols:
        P_ = {k: v for k, v in P_.items() if k in pols}
    res = {}
    for sn, segs in scen.items():
        res[sn] = {}
        env.probe_pred = next((getattr(w, "probe_pred") for _, _, w in segs if hasattr(w, "probe_pred")), None)
        for pn, mk in P_.items():
            rows = []
            for sd in range(seeds):
                idx, worlds = make_stream(bk, np.random.default_rng(1000 + sd), segs)
                pol = mk()
                out, valid, acts = G.run_stream(env, pol, idx, worlds, events=stream_events(worlds, sd))
                r_ = summarize(out, valid, worlds, pol)
                r_["exposed"] = exposed(acts, worlds, getattr(pol, "qacts", None))
                r_.update(v2metrics(out, worlds, idx, env, getattr(pol, "qacts", None)))
                r_["cost_units"] = float(pol.cost_units if hasattr(pol, "cost_units") else
                                         pol.canary_calls + getattr(pol, "kat_calls", 0) * (4.1 if use_r else 3.1))
                fw = [w for w in worlds if w != "healthy"]
                r_["true_cause"] = (CF_CAUSE(fw[0]) if fw else "NOFAULT")
                on = next((t for t, w in enumerate(worlds) if w != "healthy"), None)
                r_["cause"] = next((c for t, e, c in pol.log if e == "cause" and (on is None or t >= on) and c != "NOFAULT"), None)
                rows.append(r_)
            res[sn][pn] = rows
        print(sn, {p: round(float(np.mean([r["h20"] for r in v])) * 100, 1) for p, v in res[sn].items()}, flush=True)
    return res, {"theta20": theta20, "qthr": qthr, "calib_mu": cal.mu, "calib_sd": cal.sd, "best": best,
                 "p0": env.p0, "qtau": prm.get("qtau")}


def agg(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows if isinstance(r[k], (int, float)) and r[k] is not None]
        out[k] = float(np.mean(v)) if v else None
        if k.startswith("delay"):
            out[k + "_n"] = len(v)
            out[k + "_median"] = float(np.median(v)) if v else None
    out["confirm_comp"] = sorted({str(r.get("confirm_comp")) for r in rows})
    tc = [r for r in rows if r.get("true_cause") not in (None, "NOFAULT")]
    if tc and any(r.get("cause") is not None for r in rows):
        out["cause_acc"] = float(np.mean([r.get("cause") == r["true_cause"] for r in tc]))
        out["cause_acc_or_unk"] = float(np.mean([r.get("cause") in (r["true_cause"], "UNK") for r in tc]))
    out["frac_false_confirm"] = float(np.mean([r["false_confirm"] > 0 for r in rows]))
    return out


def main():
    mode = sys.argv[1]
    qs, tab, qcol = load()
    bk, canary = banks(qs)
    if mode == "certify":
        res = certify(tab, bk)
        json.dump(res, open(os.path.join(RUNS, "t1_certify.json"), "w"), indent=1)
        print(json.dumps(res["beta_eps"]), res["m"])
        for n, row in res["banks"].items():
            print(n, {b: (v["loss"], round(v["rate"] * 100, 2), round(v["ucb"] * 100, 2)) for b, v in row.items()})
    elif mode == "tune":
        cert = json.load(open(os.path.join(RUNS, "t1_certify.json")))
        be = float(cert["beta_eps"]["0.02"])
        scen = scenarios(qs, base="dev", shift_to="dev_i", shift_from="dev_f", books="dev")
        keep = ["S1_healthy", "S2_shift_dialect", "S3_D_unk", "S3_D_minilm", "S3_L_shard", "S3_L_cache", "S3_D_offset",
                "S4_ramp_D_minilm", "S5_partial30_D_unk"]
        scen = {k: v for k, v in scen.items() if k in keep}
        grid = []
        for delta in (0.25, 0.5, 1.0):
            for h in (4.0, 6.0, 8.0, 12.0):
                prm = {"delta": delta, "h": h, "lat": 40}
                res, meta = run_suite(qs, tab, qcol, bk, canary, prm, scen, 30, False, be, pols=["G"])
                fa = np.mean([r["false_suspect"] for s in ("S1_healthy", "S2_shift_dialect") for r in res[s]["G"]])
                fc = np.mean([r["false_confirm"] > 0 for s in res for r in res[s]["G"]])
                dl = np.mean([agg(res[s]["G"])["delay_suspect"] or 9999 for s in res if s.startswith(("S3", "S4", "S5"))])
                h20 = np.mean([agg(res[s]["G"])["h20"] for s in res])
                grid.append({"delta": delta, "h": h, "false_suspect_per_stream": float(fa), "frac_false_confirm": float(fc),
                             "mean_delay_suspect": float(dl), "mean_h20": float(h20)})
                print(grid[-1], flush=True)
        ok = [g for g in grid if g["frac_false_confirm"] == 0 and g["false_suspect_per_stream"] <= 4]
        pick = max(ok or grid, key=lambda g: g["mean_h20"])
        frozen = {"delta": pick["delta"], "h": pick["h"], "lat": 40, "beta_eps": be, "grid": grid,
                  "rule": "max mean dev Hit@20 over tuning scenarios among settings with no false confirmation and <= 4 "
                          "false suspicions per 2,000-query healthy/shift stream"}
        json.dump(frozen, open(os.path.join(RUNS, "frozen_params.json"), "w"), indent=1)
        print("FROZEN", frozen["delta"], frozen["h"], be)
    elif mode == "test":
        fr = json.load(open(os.path.join(RUNS, "frozen_params.json")))
        use_r = len(sys.argv) > 2 and sys.argv[2] == "rerank"
        seeds = int(os.environ.get("SEEDS", "100"))
        scen = scenarios(qs)
        only = os.environ.get("SCEN")
        if only:
            scen = {k: v for k, v in scen.items() if k in only.split(",")}
        if not use_r:
            scen = {k: v for k, v in scen.items() if "R_" not in k}
        res, meta = run_suite(qs, tab, qcol, bk, canary, fr, scen, seeds, use_r, fr["beta_eps"])
        tag = ("rerank" if use_r else "pool") + (("_" + os.environ["TAG"]) if os.environ.get("TAG") else "")
        json.dump({"meta": meta, "params": fr, "agg": {s: {p: agg(v) for p, v in d.items()} for s, d in res.items()}},
                  open(os.path.join(RUNS, f"t1_test_{tag}.json"), "w"), indent=1)
        json.dump(res, open(os.path.join(RUNS, f"t1_test_{tag}_rows.json"), "w"))


if __name__ == "__main__":
    main()
