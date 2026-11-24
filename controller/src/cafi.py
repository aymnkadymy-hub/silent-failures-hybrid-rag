"""CAFI: causal-aware active fault isolation for hybrid retrieval (design: the registered plan addendum A9).

A Bayesian belief over the cause of a silent fault, sequential test selection by expected information gain per unit
cost, containment during diagnosis by the action with the highest expected Hit@20 under the current belief, and
route-around/recovery through the governor's Fallback/Validation machinery.
"""
import math

import numpy as np

import governor as G

CAUSES = ["NOFAULT", "Q", "L", "Dtok", "Denc", "Didx", "R", "S", "UNK"]
FAMILY_CAUSE = {"D_unk": "Dtok", "D_minilm": "Denc", "D_rand1": "Denc", "D_rand": "Denc", "D_noise": "Denc",
                "D_mix25": "Denc", "D_mix50": "Denc", "D_mix75": "Denc", "D_offset": "Didx",
                "L_empty": "L", "L_shard": "L", "L_nofold": "L", "Q_strip": "Q",
                "R_random": "R", "R_inverted": "R", "R_constant": "R", "S_shift": "S"}
HELD_OUT = {"D_ann": "Didx", "D_corrupt": "Didx", "L_cache": "L", "R_trunc": "R", "D_adv": "Didx"}
CAUSE_COMP = {"Q": "Q", "L": "L", "Dtok": "D", "Denc": "D", "Didx": "D", "R": "R", "S": "S"}
# probe tests: name -> (table key, cost units). Two probes per call; a call flags if either probe flags.
PROBE_TESTS = {"t_vec": ("kat|Dvec", 2.0), "t_col": ("COL", 0.0), "t_srch": ("kat|Dsrch", 2.0),
               "t_lex": ("kat|L", 2.0), "t_lexraw": ("kat|Lraw", 2.0), "t_rr": ("kat|R", 2.0),
               "t_store": ("kat|S", 0.2)}
CANARY_TESTS = {"t_canD": ("D", 30.0), "t_canL": ("L", 30.0), "t_canR": ("R", 30.0)}
SPARSE_BUNDLE = ["t_vec", "t_srch", "t_lex", "t_rr", "t_store"]
EPS = 0.01


def entropy(p):
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class Model:
    """Per-cause test likelihoods and per-cause action values, estimated on calibration (dev) questions from SEEN
    fault families only. UNK has flat likelihoods (0.5) and the worst action values over causes."""

    def __init__(self, tab, dev_idx, probe_pool, canary_p, collapse_rate, use_r, actions, noise=None):
        self.use_r = use_r
        self.lik = {}             # (cause, test) -> P(single probe flags)
        noise = noise or {}
        fams = {}
        for f, c in FAMILY_CAUSE.items():
            if f in tab:
                fams.setdefault(c, []).append(f)
        fams["NOFAULT"] = ["healthy"]
        for c in CAUSES:
            for tn, (key, _) in PROBE_TESTS.items():
                if c == "UNK":
                    p = 0.5
                elif key == "COL":
                    p = float(np.mean([collapse_rate(f) for f in fams.get(c, [])])) if c in fams else EPS
                else:
                    if key not in tab["healthy"] or (tn == "t_rr" and not use_r):
                        continue
                    if c not in fams:
                        p = EPS
                    else:
                        if tn == "t_rr":
                            ok = tab["healthy"]["kat|ok_r"][probe_pool] > 0
                            pp = probe_pool[ok] if ok.any() else probe_pool
                        else:
                            pp = probe_pool
                        p = float(np.mean([tab[f][key][pp].mean() for f in fams[c]]))
                if c == "NOFAULT":
                    p = max(p, noise.get(tn, EPS))
                self.lik[(c, tn)] = min(max(p, EPS), 1 - EPS)
        # canary hit rates per cause (component alone on the labelled canary set)
        self.can = {}
        for c in CAUSES:
            for tn, (comp, _) in CANARY_TESTS.items():
                if tn == "t_canR" and not use_r:
                    continue
                key = f"can|{G.CANARY_KEY[comp]}"
                if c == "UNK":
                    p = canary_p[comp] / 2
                elif c in fams:
                    p = float(np.mean([tab[f][key][probe_pool].mean() for f in fams[c]]))
                else:
                    p = canary_p[comp]
                self.can[(c, tn)] = min(max(p, 0.02), 0.98)
        # action values: mean pool Hit@20 on dev under each cause
        self.val = {}
        for c in CAUSES:
            for a in actions:
                if c == "UNK":
                    continue
                if a == "ABST":
                    v = 0.0
                else:
                    v = float(np.mean([np.mean((tab[f][f"pool|{a}"][dev_idx] > 0) & (tab[f][f"pool|{a}"][dev_idx] <= 20))
                                       for f in fams.get(c, ["healthy"])]))
                self.val[(c, a)] = v
        for a in actions:
            self.val[("UNK", a)] = min(self.val[(c, a)] for c in CAUSES if c != "UNK")
        self.actions = actions

    def tests(self):
        out = [t for t in PROBE_TESTS if ("NOFAULT", t) in self.lik]
        out += [t for t in CANARY_TESTS if ("NOFAULT", t) in self.can]
        return out

    def p_flag(self, c, tn):
        """P(test call outcome = flag | cause). Probe tests use two probes (flag if either flags)."""
        if tn in PROBE_TESTS:
            p = self.lik[(c, tn)]
            return 1 - (1 - p) ** 2
        return None

    def canary_loglik(self, c, tn, hits, n):
        p = self.can[(c, tn)]
        return hits * math.log(p) + (n - hits) * math.log(1 - p)


class CAFI(G.Governor):
    def __init__(self, env, calib, beta_eps, model, rho=0.2, budget=100.0, seed=0, exclude=(), random_select=False,
                 graded=False, grade_cause=False, **kw):
        kw.pop("kat", None)
        self.exclude = set(exclude)
        self.graded = graded
        self.grade_cause = grade_cause
        self.graded_action = None
        self.accepted_until = -1
        self.random_select = random_select
        super().__init__(env, calib, beta_eps, **kw)
        self.M = model
        self.rho = rho
        self.budget = budget
        self.rng = np.random.default_rng(10_000 + seed)
        self.post = None             # current belief (dict) during an episode
        self.spent = 0.0
        self.cost_units = 0.0
        self.used = {}
        self.episode_comp = None
        self.cause_log = []

    # ---------------------------------------------------------------- probes in the current batch
    def _probe_outcome(self, tn, worlds_b):
        """Run one probe test call (two probes) under the batch's runtime state. Returns True if it flags."""
        key, cost = PROBE_TESTS[tn]
        self.cost_units += cost
        self.kat_calls += 1
        pool = self.env.canary_idx
        pred = getattr(self.env, "probe_pred", None)
        faulty = [w for w in worlds_b if w != "healthy"]
        ps, ws = [], []
        for j in range(2):
            self.batch_no += 1
            p = pool[(self.batch_no * 2 + j) % len(pool)]
            if pred is None:
                wp = worlds_b[j % len(worlds_b)]
            else:
                wp = faulty[0] if (faulty and pred(p)) else "healthy"
            ps.append(p)
            ws.append(wp)
        if key == "COL":
            return bool(self.env.collapse(ws, ps))
        flag = False
        for p, wp in zip(ps, ws):
            tb = self.env.tab[wp]
            if tn == "t_rr" and not tb["kat|ok_r"][p]:
                continue
            flag |= bool(tb[key][p])
        return flag

    def _update(self, tn, flag=None, hits=None, n=None):
        lp = {}
        for c in CAUSES:
            if tn in PROBE_TESTS:
                q = self.M.p_flag(c, tn)
                lp[c] = math.log(q if flag else 1 - q)
            else:
                lp[c] = self.M.canary_loglik(c, tn, hits, n)
        m = max(math.log(max(self.post[c], 1e-300)) + lp[c] for c in CAUSES)
        z = {c: math.exp(math.log(max(self.post[c], 1e-300)) + lp[c] - m) for c in CAUSES}
        s = sum(z.values())
        self.post = {c: v / s for c, v in z.items()}

    def _eig(self, tn):
        H0 = entropy(list(self.post.values()))
        if tn in PROBE_TESTS:
            pf = {c: self.M.p_flag(c, tn) for c in CAUSES}
            out = 0.0
            for flag in (True, False):
                py = sum(self.post[c] * (pf[c] if flag else 1 - pf[c]) for c in CAUSES)
                if py <= 0:
                    continue
                post = [self.post[c] * (pf[c] if flag else 1 - pf[c]) / py for c in CAUSES]
                out += py * entropy(post)
            return H0 - out
        # canary: summarise outcome as pass/fail at the healthy-vs-half threshold
        n = 30
        thr = int(round(n * 0.75 * self.M.can[("NOFAULT", tn)]))
        pf = {}
        for c in CAUSES:
            p = self.M.can[(c, tn)]
            pf[c] = float(sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(0, thr)))
        out = 0.0
        for fail in (True, False):
            py = sum(self.post[c] * (pf[c] if fail else 1 - pf[c]) for c in CAUSES)
            if py <= 0:
                continue
            post = [self.post[c] * (pf[c] if fail else 1 - pf[c]) / py for c in CAUSES]
            out += py * entropy(post)
        return H0 - out

    def _start(self, t, prior, why):
        if self.post is None:
            self.post = dict(prior)
            self.spent = 0.0
            self.used = {}
            self.log.append((t, "suspect", why))
        else:                         # merge evidence: average the priors' fault mass into the running belief
            self.post = {c: 0.5 * self.post[c] + 0.5 * prior[c] for c in CAUSES}

    def _prior(self, kind, comps=()):
        if kind == "sparse":
            base = {c: 0.02 / 7 for c in CAUSES}
            base["NOFAULT"] = 0.98
            base["UNK"] = 0.0
            s = sum(base.values())
            return {c: v / s for c, v in base.items()}
        w = {}
        for c in CAUSES:
            if c in ("NOFAULT", "UNK"):
                continue
            w[c] = 1.0 if CAUSE_COMP[c] in comps else 0.1
        s = sum(w.values())
        pr = {c: 0.45 * v / s for c, v in w.items()}
        pr["NOFAULT"] = 0.5
        pr["UNK"] = 0.05
        return pr

    def _diagnose(self, t, worlds_b):
        """Run immediate probe tests by EIG/cost until stop; launch at most one canary if only canaries help."""
        while self.post is not None:
            best_c, best = max(self.post.items(), key=lambda kv: kv[1])
            if best >= 0.95 or self.spent >= self.budget:
                self._conclude(t, best_c if best >= 0.95 else max(
                    ((c, v) for c, v in self.post.items() if c != "NOFAULT"), key=lambda kv: kv[1])[0])
                return
            cands = []
            for tn in self.M.tests():
                if tn in self.exclude:
                    continue
                if self.used.get(tn, 0) >= 3:
                    continue
                if tn in CANARY_TESTS and any(p[1] == CANARY_TESTS[tn][0] and p[2] == "cafi" for p in self.pending):
                    continue
                cost = PROBE_TESTS[tn][1] if tn in PROBE_TESTS else CANARY_TESTS[tn][1]
                gain = self._eig(tn)
                if gain > 1e-4:
                    cands.append((gain / max(cost, 0.1), tn))
            if not cands:
                if any(p[2] == "cafi" for p in self.pending):
                    return                   # a labelled canary is in flight: wait for its evidence
                self._conclude(t, best_c if best >= 0.5 else "UNK")
                return
            _, tn = (cands[int(self.rng.integers(len(cands)))] if self.random_select else max(cands))
            self.used[tn] = self.used.get(tn, 0) + 1
            if tn in PROBE_TESTS:
                self.spent += PROBE_TESTS[tn][1]
                self._update(tn, flag=self._probe_outcome(tn, worlds_b))
            else:
                comp, cost = CANARY_TESTS[tn]
                self.spent += cost
                self.cost_units += cost
                self.pending.append((t + self.lat, comp, "cafi"))
                return

    def _conclude(self, t, cause):
        self.cause_log.append((t, cause))
        self.log.append((t, "cause", cause))
        self.post = None
        if cause == "NOFAULT":
            for k in self.alarm_sigs:
                if self.shift_rule and self.recent:
                    self.mu[k] = float(np.mean([r[k] for r in self.recent if k in r]))
                self.cs[k] = [0.0, 0.0]
            self.alarm_sigs = set()
            self.log.append((t, "cleared", "ALL"))
            return
        comp = CAUSE_COMP.get(cause, self.episode_comp or "D")
        targets = {"Q": ("D", "L"), "S": ("D", "L")}.get(comp, (comp,))
        if cause == "UNK":
            targets = (self.episode_comp or "D",) if (self.episode_comp or "D") in ("D", "L", "R") else ("D",)
        if self.graded and targets == ("D",):
            self.pending.append((t + self.lat, "D", "grade"))
        for c in targets:
            if c == "R" and not self.use_r:
                continue
            if self.state[c] != "F":
                self.state[c] = "F"
                self.vllr[c] = 0.0
                self.next_v[c] = t + self.vperiod
                self.log.append((t, "confirmed", c))

    # ---------------------------------------------------------------- per batch
    def precommit(self, t, worlds_b, idx_b):
        # recovery probes for failed stages (cheap): a clean stage probe starts fast validation
        for c in ("D", "L", "R"):
            if self.state[c] == "F" and not self.fast[c]:
                tn = {"D": "t_srch" if (self._last_cause_is("Didx") and "t_srch" not in self.exclude) else "t_vec",
                      "L": "t_lex", "R": "t_rr"}[c]
                if c == "R" and not self.use_r:
                    continue
                if not self._probe_outcome(tn, worlds_b):
                    self.fast[c] = True
                    self._launch(t, c, "valid")
        if self.graded and self.accepted_until >= 0 and t >= self.accepted_until:
            self.accepted_until = -1
            self.graded_action = None
            self.pending.append((t + self.lat, "D", "grade"))
        # sparse random probing
        if self.post is None and self.rng.random() < self.rho:
            flags = {tn: self._probe_outcome(tn, worlds_b) for tn in SPARSE_BUNDLE
                     if ("NOFAULT", tn) in self.M.lik and tn not in self.exclude
                     and all(self.state[x] != "F" for x in ("D", "L"))}
            if self.accepted_until > t:     # a graded, accepted dense degradation: ignore dense-only probe flags
                for tn in ("t_vec", "t_srch"):
                    flags.pop(tn, None)
            if any(flags.values()):
                self._start(t, self._prior("sparse"), "probe")
                for tn, f in flags.items():
                    self._update(tn, flag=f)
        if self.post is not None:
            self._diagnose(t, worlds_b)

    def _last_cause_is(self, c):
        return bool(self.cause_log) and self.cause_log[-1][1] == c

    def on_fingerprint(self, t, comp):
        comps = ("D", "L") if comp in ("Q", "S") else (comp,)
        for c in comps:
            if self.state.get(c) == "F":
                self.next_v[c] = t
                self.fast[c] = True
                self._launch(t, c, "valid")
                return
        self.episode_comp = comps[0]
        self._start(t, self._prior("fp", comps), "fingerprint")

    def step(self, t, worlds_b, idx_b, world_now):
        sig = G.signals_of(self.env, worlds_b, idx_b, self.use_r, self.extra)
        alarmed = self.observe(t, sig)
        if alarmed:
            explained = {k for k in alarmed if any(self.state.get(c) in ("F", "V") for c in G.SIG_COMPONENTS[k])}
            for k in explained:
                self.cs[k] = [0.0, 0.0]
            alarmed -= explained
        if alarmed and self.accepted_until > t:
            alarmed = {k for k in alarmed if "D" not in G.SIG_COMPONENTS[k]}
        if alarmed:
            comps = set()
            for k in alarmed:
                comps |= set(G.SIG_COMPONENTS[k])
            if not self.use_r:
                comps.discard("R")
            self.alarm_sigs |= alarmed
            for k in alarmed:
                self.cs[k] = [0.0, 0.0]
            self.episode_comp = sorted(comps - {"Q"})[0] if comps - {"Q"} else "D"
            self._start(t, self._prior("alarm", comps), "alarm")
        # scheduled canaries (store faults that probes and signals cannot see are covered by t_store in the bundle)
        if self.recover:
            for c in ("D", "L", "R"):
                if self.state[c] == "F" and self.next_v[c] is not None and t >= self.next_v[c]:
                    self._launch(t, c, "valid")
                    self.next_v[c] = t + self.vperiod
        done = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for due, c, purpose in done:
            hits, n, _ = self.env.canary(world_now, c, self.run_no)
            self.run_no += 1
            self.canary_calls += n
            if purpose == "grade":
                self.cost_units += n
                q = min(1.0, (hits / n) / max(self.env.p0["D"], 1e-6))
                last = self.cause_log[-1][1] if self.cause_log else None
                dead = (last,) if (self.grade_cause and last in ("Dtok", "Denc", "Didx")) else ("Dtok", "Denc", "Didx")
                def h(a):
                    hd = float(np.mean([self.M.val[(c, a)] for c in dead]))
                    return hd + q * (self.M.val[("NOFAULT", a)] - hd)
                a = max(("N", "Dg0.75", "Doff"), key=h)
                self.log.append((t, "grade", f"{q:.2f}:{a}"))
                if a != "Doff":
                    self.state["D"] = "N"
                    self.graded_action = None if a == "N" else a
                    self.accepted_until = t + 500
                continue
            if purpose == "cafi":
                if self.post is not None:
                    tn = {"D": "t_canD", "L": "t_canL", "R": "t_canR"}[c]
                    self._update(tn, hits=hits, n=n)
                    self._diagnose(t, worlds_b)
            else:
                self.cost_units += n
                self.on_canary(t, c, purpose, hits, n)
        for c in ("D", "L", "R"):
            if self.state[c] == "V" and t >= self.prob_until[c]:
                self.state[c] = "N"
                self.log.append((t, "recovered", c))
        both = self.state["D"] == "F" and self.state["L"] == "F"
        can_byp = "can|L10byp" in self.env.tab.get("healthy", {})
        if both and can_byp and self.qbyp is None:
            self.qbyp = "testing"
            self.pending.append((t + self.lat, "Q", "bypass"))
        if not both and self.qbyp is not None:
            self.qbyp = None
        self.abstain = both and self.qbyp != "on"

    def action(self):
        if self.qbyp == "on":
            return "Qbyp", ("off" if self.state["R"] == "F" else "on")
        if self.abstain:
            return "ABST", "off"
        if self.graded_action and self.state["D"] == "N" and self.post is None:
            return self.graded_action, ("off" if self.state["R"] == "F" else "on")
        if self.post is not None and all(self.state[c] == "N" for c in ("D", "L")):
            a = max(self.M.actions, key=lambda a: sum(self.post[c] * self.M.val[(c, a)] for c in CAUSES))
            return a, "on"
        return super().action()


class QueryModel:
    """Per-query evidence that this query's dense contribution is broken (logistic regression on label-free features,
    fitted on calibration questions: healthy against SEEN dense fault families; held-out families excluded)."""
    FEATS = ("a_LD", "d1", "nD", "disp20")

    def __init__(self, tab, dev_idx, collapse_fn, seen=("D_unk", "D_minilm", "D_rand", "D_rand1", "D_noise", "D_offset")):
        self.collapse = collapse_fn
        X, y = [], []
        fams = [f for f in seen if f in tab]
        for w, lab in [("healthy", 0)] + [(f, 1) for f in fams]:
            X.append(self._feats(tab, w, dev_idx))
            y.append(np.full(len(dev_idx), lab))
        X = np.vstack(X)
        y = np.concatenate(y)
        wts = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))   # balance classes
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        b = np.zeros(Z.shape[1])
        for _ in range(500):                                   # plain gradient descent, L2 0.01
            p = 1 / (1 + np.exp(-Z @ b))
            grad = Z.T @ (wts * (p - y)) / len(y) + 0.01 * np.r_[b[:-1], 0]
            b -= 0.5 * grad
        self.b = b
        self.tab = tab

    def _feats(self, tab, w, idx):
        cols = [tab[w][f"sig|{k}"][idx].astype(float) for k in self.FEATS]
        cols.append(np.array([self.collapse(w, i) for i in idx], float))
        return np.vstack(cols).T

    def prob(self, w, i):
        x = np.array([float(self.tab[w][f"sig|{k}"][i]) for k in self.FEATS] + [self.collapse(w, i)])
        z = np.r_[(x - self.mu) / self.sd, 1.0]
        return float(1 / (1 + np.exp(-z @ self.b)))


class CIFCQ(CAFI):
    """CIFC with query-scoped containment: per query, serve BM25 only when the per-query evidence says this query's
    dense contribution is broken (P > tau*). tau* is the decision threshold from calibration values."""

    def __init__(self, env, calib, beta_eps, model, qmodel, tau, **kw):
        super().__init__(env, calib, beta_eps, model, **kw)
        self.Q, self.tau = qmodel, tau
        self.q_overrides = 0

    def per_query(self, la, w, i):
        if la in ("N", "Dg0.75") and self.Q.prob(w, i) > self.tau:
            self.q_overrides += 1
            return "Doff"
        return la


class AnomalyModel:
    """Two-sided per-query anomaly score: Mahalanobis distance of label-free features from healthy calibration."""
    FEATS = ("a_LD", "d1", "nD", "disp20")

    def __init__(self, tab, dev_idx, collapse_fn, q=99.0):
        self.tab, self.collapse = tab, collapse_fn
        X = self._feats("healthy", dev_idx)
        self.mu = X.mean(0)
        self.Ci = np.linalg.inv(np.cov(X.T) + 1e-6 * np.eye(X.shape[1]))
        d = self._d(X)
        self.thr = float(np.percentile(d, q))
        self.p_flag = float(np.mean(d > self.thr))

    def _feats(self, w, idx):
        return np.array([[float(self.tab[w][f"sig|{k}"][i]) for k in self.FEATS] + [self.collapse(w, i)] for i in idx])

    def _d(self, X):
        Z = X - self.mu
        return np.einsum("ij,jk,ik->i", Z, self.Ci, Z)

    def flag(self, w, i):
        return bool(self._d(self._feats(w, [i]))[0] > self.thr)


class CIFCQ2(CAFI):
    """CIFC-Q (A16): dense containment is query-scoped. While the dense alarm state is active, flagged queries get BM25
    only and unflagged queries plain RRF; the dense arm is never switched off globally."""

    def __init__(self, env, calib, beta_eps, model, amodel, window=100, alpha=1e-3, **kw):
        super().__init__(env, calib, beta_eps, model, **kw)
        self.A, self.window, self.alpha = amodel, window, alpha
        self.recent_flags = []
        self.q_overrides = 0
        self.dense_confirmed = False

    def _dense_active(self):
        if self.post is not None:
            best = max(self.post.items(), key=lambda kv: kv[1])[0]
            if best in ("Dtok", "Denc", "Didx"):
                return True
        if self.cause_log and self.cause_log[-1][1] in ("Dtok", "Denc", "Didx") and self.state["D"] in ("F", "V"):
            return True
        w = self.recent_flags[-self.window:]
        if len(w) >= 20:
            from scipy.stats import binom
            if binom.sf(sum(w) - 1, len(w), self.A.p_flag) < self.alpha:
                return True
        return False

    def action(self):
        la, rm = super().action()
        if la in ("Doff", "Dg0.75", "Dg0.5", "Dg0.625", "Dg0.875") and self.state["L"] != "F":
            la = "N"                      # dense containment is handled per query
        self._active = self._dense_active()
        return la, rm

    def per_query(self, la, w, i):
        f = self.A.flag(w, i)
        self.recent_flags.append(int(f))
        if la == "N" and getattr(self, "_active", False) and f:
            self.q_overrides += 1
            return "Doff"
        return la
