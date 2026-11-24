"""Evaluate the registered hypotheses from the stored results (the registered plan sections 7 and addenda A5, A6, A9, A11 to A13).
Writes runs/hypotheses.json and runs/hypotheses.md. Every verdict is computed, never typed."""
import glob
import json
import os

import numpy as np

RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
TBS = {"T1": "t1", "T2": "quran", "T3a": "scifact", "T3b": "nfcorpus"}
DEPLOY = ["S3_D_unk", "S3_D_minilm", "S3_D_offset", "S3_L_shard", "S3_L_nofold", "S3_Q_strip", "S3_S_shift"]
DENSE = ["S3_D_unk", "S3_D_minilm", "S3_D_rand", "S3_D_rand1", "S3_D_offset", "S3_D_noise", "S3_D_ann", "S3_D_corrupt",
         "S3_D_adv"]
MILD = ["S3_D_ann", "S3_D_corrupt", "S3_D_noise"]
SEEN_D = {"S3_D_unk": "Dtok", "S3_D_minilm": "Denc", "S3_D_rand": "Denc", "S3_D_rand1": "Denc", "S3_D_noise": "Denc",
          "S3_D_offset": "Didx", "S3_L_empty": "L", "S3_L_shard": "L", "S3_L_nofold": "L", "S3_Q_strip": "Q",
          "S3_S_shift": "S", "S3_R_random": "R", "S3_R_inverted": "R", "S3_R_constant": "R"}
UNSEEN = ["S3_D_ann", "S3_D_corrupt", "S3_D_adv", "S3_L_cache", "S3_R_trunc"]


def load(tb, mode="pool"):
    pat = (os.path.join(RUNS, f"t1_test_{mode}_*.json") if tb == "t1"
           else os.path.join(RUNS, f"tb_{tb}", f"test_{mode}_*.json"))
    out, rows = {}, {}
    for fn in glob.glob(pat):
        if fn.endswith("_rows.json"):
            for sc, pv in json.load(open(fn)).items():
                rows.setdefault(sc, {}).update(pv)
        else:
            for sc, pv in json.load(open(fn))["agg"].items():
                out.setdefault(sc, {}).update(pv)
    return out, rows


def verdict(ok):
    return "supported" if ok else "not supported"


def main():
    H = []
    for lab, tb in TBS.items():
        d, rows = load(tb)
        if not d:
            continue
        conf = "exploratory" if lab == "T1" else "confirmatory"
        shift = [s for s in d if s.startswith("S2")]
        healthy = ["S1_healthy"] + shift
        # H1: healthy/shift Hit@20 within 0.5 of P0 (G and CIFC_r20)
        for p in ("G", "CIFC_r20"):
            gaps = [(d[s]["P0_rrf"]["h20"] - d[s][p]["h20"]) * 100 for s in healthy if p in d[s]]
            if gaps:
                H.append((lab, conf, "H1", p, f"max Hit@20 loss vs RRF in healthy/shift = {max(gaps):.2f} pts",
                          verdict(max(gaps) <= 0.5)))
        # H2: no false fallback in shift streams
        for p in ("G", "CIFC_r20", "P3_deadarm"):
            fc = [d[s][p]["frac_false_confirm"] for s in healthy if p in d[s]]
            if fc:
                H.append((lab, conf, "H2/H14", p, f"share of healthy/shift streams with a false fallback = {max(fc):.2f}",
                          verdict(max(fc) <= 0.05)))
        # H3: dense faults, G losses vs oracle <= P3's and <= 25% of P0's
        for p in ("G", "CIFC_r20"):
            ok, txt = [], []
            for s in [x for x in ("S3_D_unk", "S3_D_minilm", "S3_D_rand", "S3_D_rand1", "S3_D_offset") if x in d]:
                lo = lambda q: (d[s]["O_oracle"]["h20"] - d[s][q]["h20"]) * 1000
                ok.append(lo(p) <= lo("P3_deadarm") and lo(p) <= 0.25 * max(lo("P0_rrf"), 1e-9))
                txt.append(f"{s[3:]}: {lo(p):.0f} vs P3 {lo('P3_deadarm'):.0f}, P0 {lo('P0_rrf'):.0f}")
            if ok:
                H.append((lab, conf, "H3", p, "; ".join(txt), f"{sum(ok)}/{len(ok)} faults"))
        # H4: beyond the dense arm: losses <= 25% of P0's; localisation
        for p in ("G", "CIFC_r20"):
            ok, txt = [], []
            for s in [x for x in ("S3_L_empty", "S3_L_shard", "S3_L_cache", "S3_L_nofold", "S3_Q_strip", "S3_S_shift") if x in d]:
                lo = lambda q: (d[s]["O_oracle"]["h20"] - d[s][q]["h20"]) * 1000
                ok.append(lo(p) <= 0.25 * max(lo("P0_rrf"), 1e-9))
                txt.append(f"{s[3:]}: {lo(p):.0f} vs P0 {lo('P0_rrf'):.0f}")
            if ok:
                H.append((lab, conf, "H4", p, "; ".join(txt), f"{sum(ok)}/{len(ok)} faults"))
        # H8/H6: recovery within 200 (G_vgate) / 600 (G) queries in >= 90% of seeds
        for p, lim, hn in (("G", 600, "H6"), ("G_vgate", 200, "H8"), ("CIFC_r20", 200, "H8-CIFC")):
            ok, txt = [], []
            for s in [x for x in DEPLOY if x in rows and p in rows[x]]:
                rr = [r.get("delay_recover") for r in rows[s][p]]
                frac = np.mean([x is not None and x <= lim for x in rr])
                ok.append(frac >= 0.9)
                txt.append(f"{s[3:]} {frac:.2f}")
            if ok:
                H.append((lab, conf, hn, p, "share of seeds recovered within " + str(lim) + ": " + ", ".join(txt),
                          f"{sum(ok)}/{len(ok)} faults"))
        # H9/H11: G_vgate and G_kat >= P0 and >= G on deployment faults
        for p, hn in (("G_vgate", "H9"), ("G_kat", "H11")):
            ok, txt = [], []
            for s in [x for x in DEPLOY if x in d and p in d[x]]:
                ok.append(d[s][p]["h20"] >= d[s]["P0_rrf"]["h20"] and d[s][p]["h20"] >= d[s]["G"]["h20"])
                txt.append(f"{s[3:]} {d[s][p]['h20']*100:.1f}/{d[s]['P0_rrf']['h20']*100:.1f}/{d[s]['G']['h20']*100:.1f}")
            if ok:
                H.append((lab, conf, hn, p, "Hit@20 policy/P0/G: " + "; ".join(txt), f"{sum(ok)}/{len(ok)} faults"))
        # H10: G_kat detection within one batch on deployment faults
        ok = []
        for s in [x for x in DEPLOY if x in rows and "G_kat" in rows[x]]:
            ok.append(np.mean([(r.get("delay_confirm") is not None and r["delay_confirm"] <= 20) for r in rows[s]["G_kat"]]) >= 0.9)
        if ok:
            H.append((lab, conf, "H10", "G_kat", "detect within 20 queries in >= 90% of seeds", f"{sum(ok)}/{len(ok)} faults"))
        # H12: localisation on seen families >= 90%; unseen reported
        for p in ("CIFC_r20", "CIFC_noint"):
            seen = [d[s][p].get("cause_acc") for s in SEEN_D if s in d and p in d[s] and d[s][p].get("cause_acc") is not None]
            unseen = {s[3:]: d[s][p].get("cause_acc_or_unk") for s in UNSEEN if s in d and p in d[s]}
            if seen:
                H.append((lab, conf, "H12", p, f"seen-family accuracy mean {np.mean(seen):.2f} (min {min(seen):.2f}); "
                          f"unseen correct-or-UNK {unseen}", verdict(min(seen) >= 0.9)))
        # H13: CIFC exposure <= G_kat's at <= cost on deployment faults
        ok, txt = [], []
        for s in [x for x in DEPLOY if x in d and "CIFC_r20" in d[x]]:
            c, k = d[s]["CIFC_r20"], d[s]["G_kat"]
            ok.append(c["exposed"] <= k["exposed"] + 1e-9 and c["cost_units"] <= k["cost_units"])
            txt.append(f"{s[3:]} exp {c['exposed']:.0f}/{k['exposed']:.0f} cost {c['cost_units']:.0f}/{k['cost_units']:.0f}")
        if ok:
            H.append((lab, conf, "H13", "CIFC_r20 vs G_kat", "; ".join(txt), f"{sum(ok)}/{len(ok)} faults"))
        # H16: interventions: localisation accuracy on Didx/Q-L-separating faults and cost ratio
        ok, txt = [], []
        for s in [x for x in ("S3_D_offset", "S3_D_unk", "S3_D_ann", "S3_Q_strip", "S3_L_shard") if x in d and "CIFC_noint" in d[x]]:
            a, b = d[s]["CIFC_r20"], d[s]["CIFC_noint"]
            ok.append((a.get("cause_acc") or 0) > (b.get("cause_acc") or 0) and a["cost_units"] <= 1.5 * b["cost_units"])
            txt.append(f"{s[3:]} acc {a.get('cause_acc')}/{b.get('cause_acc')} cost {a['cost_units']:.0f}/{b['cost_units']:.0f} "
                       f"exp {a['exposed']:.0f}/{b['exposed']:.0f}")
        if ok:
            H.append((lab, conf, "H16", "CIFC_r20 vs CIFC_noint", "; ".join(txt), f"{sum(ok)}/{len(ok)} faults"))
        # H17 / H18: graded containment
        for p, hn in (("CIFC_grade", "H17"), ("CIFC_gradec", "H18")):
            ok, txt = [], []
            for s in [x for x in DENSE if x in d and p in d[x]]:
                ok.append(d[s][p]["h20"] >= d[s]["CIFC_r20"]["h20"] - 1e-9)
                txt.append(f"{s[3:]} {d[s][p]['h20']*100:.1f}/{d[s]['CIFC_r20']['h20']*100:.1f}/{d[s]['P0_rrf']['h20']*100:.1f}")
            # second registered clause: H17 strictly higher than CIFC on D_ann and D_corrupt; H18 at least P0 on the mild faults
            if hn == "H17":
                c2 = [d[s][p]["h20"] > d[s]["CIFC_r20"]["h20"] for s in ("S3_D_ann", "S3_D_corrupt") if s in d and p in d[s]]
            else:
                c2 = [d[s][p]["h20"] >= d[s]["P0_rrf"]["h20"] for s in MILD if s in d and p in d[s]]
            if ok:
                ok = ok if all(c2) else ok + [False]
                txt.append(f"second clause {sum(c2)}/{len(c2)}")
                H.append((lab, "confirmatory" if (hn == "H18" and lab.startswith("T3")) or (hn == "H17" and lab != "T1")
                          else "exploratory", hn, p, "Hit@20 graded/CIFC/P0: " + "; ".join(txt),
                          "supported" if all(ok) else "not supported"))
    json.dump(H, open(os.path.join(RUNS, "hypotheses.json"), "w"), indent=1)
    lines = ["| Bed | Status | Hyp. | Policy | Evidence | Verdict |", "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {p} | {e} | {v} |" for a, b, c, p, e, v in H]
    open(os.path.join(RUNS, "hypotheses.md"), "w").write("\n".join(lines))
    for h in H:
        print(" | ".join(map(str, h))[:260])


if __name__ == "__main__":
    main()
