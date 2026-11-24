"""The Failure-Aware Retrieval Governor (G) and the comparison policies, simulated over query streams.

A stream is a sequence of (question index, world name). Every B = 20 queries the policy sees the batch's label-free
signals (computed in shadow on the normal pipeline) and decides the configuration served to the next batch.
Canary probes (fixed labelled questions, one component alone) return their result after `lat` queries.

The governor is a deterministic agent loop:
  sense      batch means of the signals, standardised against healthy calibration, two-sided CUSUM per signal
  diagnose   alarmed signals -> suspect components (agreement graph); canary SPRT per suspect gives pi (fault belief)
  act        per component state N (normal) / G (guarded, certified beta) / F (fallback) / V (shadow validation)
  verify     in F, shadow canaries every `vperiod` queries; two clears -> probation in G -> N
  re-baseline when every suspect of an alarm is cleared, the alarm is read as traffic shift: signal means move to
             the recent window and the CUSUM restarts
"""
import math

import numpy as np

B = 20
SIG_COMPONENTS = {           # which components a signal implicates (agreement graph)
    "a_LD": ("L", "D"), "a_RL": ("L", "R"), "a_RD": ("D", "R"), "a_RF": ("R",),
    "nL": ("L", "Q"), "nD": ("D", "Q"), "disp20": ("D", "L"), "d1": ("D", "Q"), "qcol": ("D", "Q"), "rstd": ("R",),
}
CANARY_KEY = {"D": "D10", "L": "L10", "R": "R8", "Q": "L10byp"}


class Calib:
    """Healthy batch-mean statistics of each signal (mean, sd), from calibration streams."""

    def __init__(self, means, sds):
        self.mu = dict(means)
        self.sd = {k: max(v, 1e-6) for k, v in sds.items()}


def sprt_llr(hits, n, p0):
    p1 = p0 / 2
    p0 = min(max(p0, 1e-3), 1 - 1e-3)
    p1 = min(max(p1, 1e-3), 1 - 1e-3)
    return hits * math.log(p1 / p0) + (n - hits) * math.log((1 - p1) / (1 - p0))


def sigmoid(x):
    return 1 / (1 + math.exp(-max(min(x, 50), -50)))


class Env:
    """What a policy can observe and what the stream serves. `tab[world]` are the precomputed arrays."""

    def __init__(self, tab, qcol_fn, canary_idx, canary_p0, has_rerank):
        self.tab = tab
        self.qcol = qcol_fn                  # (world, idx array) -> mean pairwise cosine of the batch's query vectors
        self.canary_idx = canary_idx         # array of canary question indices (pool of 60)
        self.p0 = canary_p0                  # healthy canary hit rate per component
        self.has_rerank = has_rerank

    def canary(self, world, comp, run_no, size=30):
        pool = self.canary_idx
        start = (run_no * size) % len(pool)
        sel = np.take(pool, range(start, start + size), mode="wrap")
        key = CANARY_KEY[comp]
        return int(self.tab[world][f"can|{key}"][sel].sum()), size, sel


# ----------------------------------------------------------------------------------------------------------------
def signals_of(env, worlds_b, idx_b, use_r, extra=()):
    """Batch means of each signal. worlds_b: list of world names per query; idx_b: question indices."""
    out = {}
    keys = ["a_LD", "nL", "nD", "d1"] + (["a_RL", "a_RD", "a_RF", "rstd"] if use_r else []) + list(extra)
    for k in keys:
        out[k] = float(np.mean([env.tab[w][f"sig|{k}"][i] for w, i in zip(worlds_b, idx_b)]))
    out["qcol"] = env.qcol(worlds_b, idx_b)
    return out


def calibrate(env, idx_pool, n_batches, rng, use_r, extra=()):
    vals = {}
    for _ in range(n_batches):
        idx = rng.choice(idx_pool, B)
        s = signals_of(env, ["healthy"] * B, idx, use_r, extra)
        for k, v in s.items():
            vals.setdefault(k, []).append(v)
    return Calib({k: float(np.mean(v)) for k, v in vals.items()}, {k: float(np.std(v)) for k, v in vals.items()}), vals


# ----------------------------------------------------------------------------------------------------------------
class Governor:
    def __init__(self, env, calib, beta_eps, delta=0.5, h=8.0, lat=40, vperiod=200, sched=500, probation=100,
                 rbeta=0.75, shift_rule=True, fixed_beta=None, recover=True, use_r=True, dt_curves=None, extra=(),
                 vgate=False, kat=False, kat_m=2, legacy=False):
        self.env, self.cal = env, calib
        self.vgate = vgate
        self.legacy = legacy                 # True = registered rules (per-run validation, no explaining away)
        self.vllr = {"D": 0.0, "L": 0.0, "R": 0.0}
        self.kat, self.kat_m, self.kat_calls, self.batch_no = kat, kat_m, 0, 0
        self.fast = {"D": False, "L": False, "R": False}
        self.extra = tuple(extra)
        self.mu = {k: v for k, v in calib.mu.items() if k not in ("disp20",) or k in self.extra}
        self.beta_eps = beta_eps
        self.delta, self.h, self.lat, self.vperiod, self.sched, self.prob = delta, h, lat, vperiod, sched, probation
        self.rbeta = rbeta
        self.shift_rule, self.fixed_beta, self.recover, self.use_r = shift_rule, fixed_beta, recover, use_r
        self.dt = dt_curves                  # optional decision-theoretic beta*(pi): dict beta -> (h_dead, h_healthy)
        self.cs = {k: [0.0, 0.0] for k in self.mu}
        self.state = {"D": "N", "L": "N", "R": "N"}
        self.pi = {"D": 0.0, "L": 0.0, "R": 0.0}
        self.llr = {"D": 0.0, "L": 0.0, "R": 0.0}
        self.pending = []                    # canaries in flight: (due_t, comp, purpose)
        self.vclears = {"D": 0, "L": 0, "R": 0}
        self.prob_until = {"D": -1, "L": -1, "R": -1}
        self.next_v = {"D": None, "L": None, "R": None}
        self.recent = []
        self.run_no = 0
        self.log = []                        # (t, event, comp)
        self.canary_calls = 0
        self.abstain = False
        self.qbyp = None                     # None | "testing" | "on": independent-normaliser bypass when L and D fail
        self.last_sched = 0
        self.alarm_sigs = set()

    # -- sense
    def observe(self, t, sig):
        self.recent.append(sig)
        self.recent = self.recent[-5:]
        alarmed = set()
        for k, v in sig.items():
            if k not in self.mu:
                continue
            z = (v - self.mu[k]) / self.cal.sd[k]
            c = self.cs[k]
            c[0] = max(0.0, c[0] + z - self.delta)
            c[1] = max(0.0, c[1] - z - self.delta)
            if max(c) > self.h:
                alarmed.add(k)
        return alarmed

    def _launch(self, t, comp, purpose):
        if any(p[1] == comp for p in self.pending):
            return
        self.pending.append((t + self.lat, comp, purpose))

    def step(self, t, worlds_b, idx_b, world_now):
        """Called at the end of a batch ending at time t (queries served so far). Returns nothing; updates state."""
        sig = signals_of(self.env, worlds_b, idx_b, self.use_r, self.extra)
        alarmed = self.observe(t, sig)
        # diagnose: alarmed signals -> suspects
        if alarmed and not self.legacy:
            explained = {k for k in alarmed if any(self.state.get(c) in ("F", "V") for c in SIG_COMPONENTS[k])}
            for k in explained:
                self.cs[k] = [0.0, 0.0]
            alarmed = alarmed - explained
        if alarmed:
            sus = set()
            for k in alarmed:
                sus |= set(SIG_COMPONENTS[k])
            sus.discard("Q")
            if not self.use_r:
                sus.discard("R")
            for c in sus:
                if self.state[c] == "N":
                    self.state[c] = "G"
                    self.pi[c] = 0.5
                    self.llr[c] = 0.0
                    self.log.append((t, "suspect", c))
                if self.state[c] in ("N", "G"):
                    self._launch(t, c, "diag")
            self.alarm_sigs |= alarmed
        # scheduled canaries (catch faults the label-free signals cannot see, e.g. a store shift)
        if t - self.last_sched >= self.sched:
            self.last_sched = t
            for c in ("D", "L") + (("R",) if self.use_r else ()):
                if self.state[c] == "N":
                    self._launch(t, c, "sched")
        # validation canaries for components in fallback
        if self.recover:
            for c in ("D", "L", "R"):
                if self.state[c] == "F" and self.next_v[c] is not None and t >= self.next_v[c]:
                    self._launch(t, c, "valid")
                    self.next_v[c] = t + self.vperiod
        # canary results that are due (they see the world at the time they run)
        done = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for due, c, purpose in done:
            hits, n, _ = self.env.canary(world_now, c, self.run_no)
            self.run_no += 1
            self.canary_calls += n
            self.on_canary(t, c, purpose, hits, n)
        # probation
        for c in ("D", "L", "R"):
            if self.state[c] == "V" and t >= self.prob_until[c]:
                self.state[c] = "N"
                self.pi[c] = 0.0
                self.log.append((t, "recovered", c))
        # shift rule: an alarm whose suspects are all cleared is read as traffic shift
        if self.alarm_sigs and not self.pending and all(self.state[c] in ("N",) for c in "DLR"):
            if self.shift_rule:
                for k in self.alarm_sigs:
                    self.mu[k] = float(np.mean([r[k] for r in self.recent if k in r]))
                    self.cs[k] = [0.0, 0.0]
                self.log.append((t, "rebaseline", ",".join(sorted(self.alarm_sigs))))
            else:
                for k in self.alarm_sigs:
                    self.cs[k] = [0.0, 0.0]
            self.alarm_sigs = set()
        both = self.state["D"] == "F" and self.state["L"] == "F"
        can_byp = "can|L10byp" in self.env.tab.get("healthy", {})
        if both and can_byp and self.qbyp is None:
            self.qbyp = "testing"
            self.pending.append((t + self.lat, "Q", "bypass"))
        if not both and self.qbyp is not None:
            self.qbyp = None
            self.log.append((t, "bypass_off", "Q"))
        self.abstain = both and self.qbyp != "on"

    def precommit(self, t, worlds_b, idx_b):
        """Known-answer attestation run with the batch, before fusion commits: probes take the batch's runtime state."""
        if not self.kat:
            return
        pool = self.env.canary_idx
        self.batch_no += 1
        flags = {"D": False, "L": False, "R": False, "S": False}
        pred = getattr(self.env, "probe_pred", None)
        faulty = [w for w in worlds_b if w != "healthy"]
        for j in range(min(self.kat_m, len(worlds_b))):
            p = pool[(self.batch_no * self.kat_m + j) % len(pool)]
            if pred is None:
                wp = worlds_b[j]                 # random partial faults: a probe is hit at the live rate
            else:                                # content-dependent faults: the probe is hit only if its content matches
                wp = faulty[0] if (faulty and pred(p)) else "healthy"
            tb = self.env.tab[wp]
            for c in flags:
                if c == "R" and (not self.use_r or not tb["kat|ok_r"][p]):
                    continue
                flags[c] |= bool(tb[f"kat|{c}"][p])
            self.kat_calls += 1
        if flags["S"]:
            flags["D"] = flags["L"] = True
        for c in ("D", "L", "R"):
            if c == "R" and not self.use_r:
                continue
            if flags[c] and self.state[c] != "F":
                self.state[c] = "F"
                self.next_v[c] = t + self.vperiod
                self.log.append((t, "confirmed", c))
            elif not flags[c] and self.state[c] == "F" and not self.fast[c]:
                self.fast[c] = True
                self.vclears[c] = 0
                self._launch(t, c, "valid")
        if flags["D"] and flags["L"]:
            self.abstain = True

    def on_fingerprint(self, t, comp):
        """A deployment changed component `comp` (model, tokenizer, normaliser or index fingerprint)."""
        if not self.vgate:
            return
        comps = ("D", "L") if comp == "Q" else (("D", "L") if comp == "S" else (comp,))
        for c in comps:
            if c == "R" and not self.use_r:
                continue
            self.log.append((t, "fingerprint", c))
            if self.state[c] in ("N", "V"):
                self.state[c] = "G"
                self.pi[c] = 0.5
                self.llr[c] = 0.0
                self._launch(t, c, "diag")
            elif self.state[c] == "F":
                self.next_v[c] = t
                self.fast[c] = True
                self._launch(t, c, "valid")

    def on_canary(self, t, c, purpose, hits, n):
        if purpose == "bypass":
            if self.qbyp == "testing":
                if sprt_llr(hits, n, self.env.p0["L"]) < math.log(0.05 / 0.95):
                    self.qbyp = "on"
                    self.log.append((t, "bypass_on", "Q"))
                else:
                    self.qbyp = "failed"
                    self.log.append((t, "bypass_failed", "Q"))
            return
        p0 = self.env.p0[c]
        if purpose == "valid" and not self.legacy:
            l = sprt_llr(hits, n, p0)
            if l > math.log(0.95 / 0.05):
                self.vllr[c] = 0.0
            else:
                self.vllr[c] += l
            if self.vllr[c] <= 2 * math.log(0.05 / 0.95):
                self.state[c] = "V"
                self.prob_until[c] = t + self.prob
                self.vllr[c] = 0.0
                self.fast[c] = False
                self.log.append((t, "validated", c))
            elif self.fast[c]:
                self._launch(t, c, "valid")
            return
        if purpose == "valid":
            if sprt_llr(hits, n, p0) < math.log(0.05 / 0.95):      # this run alone looks healthy
                self.vclears[c] += 1
                if self.fast[c] and self.vclears[c] < 2:
                    self._launch(t, c, "valid")
                if self.vclears[c] >= 2:
                    self.fast[c] = False
                    self.state[c] = "V"
                    self.prob_until[c] = t + self.prob
                    self.vclears[c] = 0
                    self.log.append((t, "validated", c))
            else:
                self.vclears[c] = 0
            return
        if purpose == "sched" and self.state[c] == "N":
            l = sprt_llr(hits, n, p0)
            if l > math.log(0.95 / 0.05):
                self.state[c] = "F"
                self.next_v[c] = t + self.vperiod
                self.log.append((t, "confirmed", c))
            elif l > 0:
                self.state[c] = "G"
                self.pi[c] = sigmoid(l)
                self.llr[c] = l
                self._launch(t, c, "diag")
            return
        if self.state[c] not in ("G",):
            return
        self.llr[c] += sprt_llr(hits, n, p0)
        self.pi[c] = sigmoid(self.llr[c])
        if self.pi[c] >= 0.95:
            self.vllr[c] = 0.0
            self.state[c] = "F"
            self.next_v[c] = t + self.vperiod
            self.log.append((t, "confirmed", c))
        elif self.pi[c] <= 0.05:
            self.state[c] = "N"
            self.pi[c] = 0.0
            self.log.append((t, "cleared", c))
        else:
            self._launch(t, c, "diag")                              # inconclusive: keep testing, stay guarded

    # -- act
    def beta(self, c):
        if self.fixed_beta is not None:
            return self.fixed_beta
        if self.dt is not None:
            p = self.pi[c]
            return max(self.dt, key=lambda b: p * self.dt[b][0] + (1 - p) * self.dt[b][1])
        return self.beta_eps

    def action(self):
        """(list action, rerank mode) to serve the next batch."""
        if self.qbyp == "on":
            return "Qbyp", ("off" if self.state["R"] == "F" else "on")
        if self.abstain:
            return "ABST", "off"
        d, l, r = self.state["D"], self.state["L"], self.state["R"]
        if d == "F":
            la = "Doff"
        elif l == "F":
            la = "Loff"
        elif d in ("G", "V"):
            la = f"Dg{self.beta('D')}"
        elif l in ("G", "V"):
            la = f"Lg{self.beta('L')}"
        else:
            la = "N"
        la = la.replace("Dg1.0", "Doff").replace("Lg1.0", "Loff").replace("Dg0.0", "N").replace("Lg0.0", "N")
        rm = "off" if r == "F" else (f"g{self.rbeta}" if r in ("G", "V") else "on")
        return la, rm


# ----------------------------------------------------------------------------------------------------------------
class Static:
    def __init__(self, la, rm="on"):
        self.la, self.rm = la, rm
        self.log, self.canary_calls, self.abstain = [], 0, False

    def step(self, *a):
        pass

    def action(self):
        return self.la, self.rm


class DeadArmAlg1:
    """Algorithm 1 of the Dead-Arm paper: overlap monitor (mean |D48 & L48| over a batch of 20 < theta20, the 1st
    percentile of healthy calibration batches) -> labelled check A on 20 questions, dense alone, exact McNemar against
    the healthy arm's per-question hits (p < 0.05 and lower) -> switch the dense arm off. P3 never switches it back;
    P3r switches it back after `back_after` queries."""

    def __init__(self, env, theta, lat=40, back_after=None):
        self.env, self.theta, self.lat, self.back = env, theta, lat, back_after
        self.off = False
        self.off_at = None
        self.pending = None
        self.log, self.canary_calls, self.abstain = [], 0, False
        self.run_no = 0

    def step(self, t, worlds_b, idx_b, world_now):
        if self.off and self.back is not None and t - self.off_at >= self.back:
            self.off = False
            self.log.append((t, "back_on", "D"))
        m = float(np.mean([self.env.tab[w]["sig|a_LD"][i] for w, i in zip(worlds_b, idx_b)]))
        if not self.off and self.pending is None and m < self.theta:
            self.pending = t + self.lat
            self.log.append((t, "alarm", "D"))
        if self.pending is not None and t >= self.pending:
            self.pending = None
            hits_now, n, sel = self.env.canary(world_now, "D", self.run_no, size=20)
            self.run_no += 1
            self.canary_calls += n
            base = self.env.tab["healthy"]["can|D10"][sel]
            now = self.env.tab[world_now]["can|D10"][sel]
            b_ = int(((base == 1) & (now == 0)).sum())
            c_ = int(((base == 0) & (now == 1)).sum())
            p = mcnemar_exact(b_, c_)
            if p < 0.05 and now.sum() < base.sum():
                self.off = True
                self.off_at = t
                self.log.append((t, "confirmed", "D"))

    def action(self):
        return ("Doff" if self.off else "N"), "on"


class HealthCheck:
    """P6: the load-time style health check deployed after the incident: query-vector collapse (mean pairwise cosine
    of a batch) far above healthy -> dense arm off. It sees nothing else."""

    def __init__(self, env, thr):
        self.env, self.thr = env, thr
        self.off = False
        self.log, self.canary_calls, self.abstain = [], 0, False

    def step(self, t, worlds_b, idx_b, world_now):
        if not self.off and self.env.qcol(worlds_b, idx_b) > self.thr:
            self.off = True
            self.log.append((t, "confirmed", "D"))

    def action(self):
        return ("Doff" if self.off else "N"), "on"


class Oracle:
    """Knows the world of the next batch and serves the action that is best for that world on the tuning bank."""

    def __init__(self, best):
        self.best = best
        self.log, self.canary_calls, self.abstain = [], 0, False
        self.nextw = "healthy"

    def step(self, *a):
        pass

    def action(self):
        return self.best[self.nextw]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)


# ----------------------------------------------------------------------------------------------------------------
def run_stream(env, policy, idx, worlds, metric_keys=("h20", "h48", "f8"), events=()):
    """Serve the stream batch by batch. Returns per-query outcome arrays and the served actions."""
    T = len(idx)
    out = {k: np.zeros(T, np.float32) for k in metric_keys}
    f8_valid = np.zeros(T, bool)
    acts = []
    qacts = []
    for s in range(0, T, B):
        e = min(T, s + B)
        if isinstance(policy, Oracle):
            policy.nextw = worlds[s]
        if hasattr(policy, "precommit"):
            policy.precommit(s, list(worlds[s:e]), idx[s:e])
        la_b, rm = policy.action()
        acts.append((s, la_b, rm))
        for j in range(s, e):
            w, i = worlds[j], idx[j]
            tb = env.tab[w]
            la = policy.per_query(la_b, w, i) if hasattr(policy, "per_query") else la_b
            qacts.append((la, rm))
            if la == "ABST":
                out["h20"][j] = out["h48"][j] = out["f8"][j] = 0
                f8_valid[j] = True
                continue
            r = tb[f"pool|{la}"][i]
            out["h20"][j] = 0 < r <= 20
            out["h48"][j] = 0 < r <= 48
            fr = tb[f"fin|{la}|{rm}"][i]
            if fr >= 0:
                out["f8"][j] = 0 < fr <= 8
                f8_valid[j] = True
        policy.step(e, list(worlds[s:e]), idx[s:e], worlds[e - 1])
        if hasattr(policy, "on_fingerprint"):
            for (te, comp) in events:
                if s < te <= e:
                    policy.on_fingerprint(e, comp)
    policy.qacts = qacts
    return out, f8_valid, acts


class CircuitBreaker:
    """CB: the strong engineering baseline. Every `period` queries, 30 labelled canary questions go through the full
    pipeline (pool Hit@20). A one-sided exact binomial test (p < alpha) against the healthy canary rate opens the
    circuit: serve BM25 only. Half-open after `half_open` queries: one canary run in shadow closes it if it passes.
    No localisation."""

    def __init__(self, env, period=200, n=30, alpha=0.01, half_open=200):
        self.env, self.period, self.n, self.alpha, self.half = env, period, n, alpha, half_open
        self.open = False
        self.opened_at = None
        self.last = 0
        self.run_no = 0
        self.log, self.canary_calls, self.abstain = [], 0, False
        pool = env.canary_idx
        r = env.tab["healthy"]["pool|N"][pool]
        self.p0 = float(np.mean((r > 0) & (r <= 20)))

    def _canary(self, world):
        pool = self.env.canary_idx
        start = (self.run_no * self.n) % len(pool)
        sel = np.take(pool, range(start, start + self.n), mode="wrap")
        self.run_no += 1
        self.canary_calls += self.n
        r = self.env.tab[world]["pool|N"][sel]
        hits = int(((r > 0) & (r <= 20)).sum())
        from scipy.stats import binom
        return binom.cdf(hits, self.n, self.p0) < self.alpha     # True = failed

    def step(self, t, worlds_b, idx_b, world_now):
        if not self.open and t - self.last >= self.period:
            self.last = t
            if self._canary(world_now):
                self.open, self.opened_at = True, t
                self.log.append((t, "confirmed", "D"))
        elif self.open and t - self.opened_at >= self.half:
            if self._canary(world_now):
                self.opened_at = t
            else:
                self.open = False
                self.last = t
                self.log.append((t, "recovered", "D"))

    def action(self):
        return ("Doff" if self.open else "N"), "on"
