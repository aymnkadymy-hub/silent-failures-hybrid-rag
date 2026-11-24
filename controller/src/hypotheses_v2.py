"""Verdicts of the v2 hypotheses (PLAN A15/A16) from runs/v2/tb_*/test_pool_*.json. ArguAna (T4) is confirmatory;
T2/T3 are exploratory. Writes runs/v2/hypotheses_v2.json and .md"""
import glob
import json
import os

import numpy as np

R = os.path.join(os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results")), "v2")
BEDS = [("T4", "arguana", "confirmatory"), ("T2", "quran", "exploratory"), ("T3a", "scifact", "exploratory"),
        ("T3b", "nfcorpus", "exploratory")]


def load(tb):
    d = {}
    for fn in glob.glob(os.path.join(R, f"tb_{tb}", "test_pool_*.json")):
        if fn.endswith("_rows.json"):
            continue
        for s, pv in json.load(open(fn))["agg"].items():
            d.setdefault(s, {}).update(pv)
    return d


def main():
    H = []
    for lab, tb, st in BEDS:
        d = load(tb)
        if not d:
            continue
        g = lambda s, p, k="h20": d[s][p][k]
        if "S5_partial30_D_unk" in d:
            s = "S5_partial30_D_unk"
            ok = g(s, "CIFC_Q") >= g(s, "P0_rrf") and g(s, "CIFC_Q") > g(s, "CIFC_r20")
            H.append((lab, st, "H19", f"partial 30%: CIFC-Q {g(s,'CIFC_Q')*100:.1f}, RRF {g(s,'P0_rrf')*100:.1f}, "
                      f"CIFC {g(s,'CIFC_r20')*100:.1f}", ok))
        fs = [s for s in d if s.startswith("S3_") and not s.startswith("S3dur")]
        if fs:
            dq = float(np.mean([g(s, "CIFC_Q", "damage") for s in fs]))
            dc = float(np.mean([g(s, "CB", "damage") for s in fs]))
            H.append((lab, st, "H20", f"mean damage per 1,000 fault queries over {len(fs)} faults: CIFC-Q {dq:.1f}, CB {dc:.1f}",
                      dq < dc))
            av = {s: g(s, "CIFC_Q", "availability") for s in d if s != "S3_S_shift"}
            H.append((lab, st, "H21", f"min availability outside the store fault: {min(av.values())*100:.2f}%",
                      min(av.values()) >= 0.99))
        hs = [s for s in ("S1_healthy", "S2_shift_split") if s in d]
        if hs:
            rg = max(g(s, "CIFC_Q", "regret") for s in hs)
            H.append((lab, st, "H22", f"max regret with no fault or shift: {rg:.2f} points", rg <= 0.5))
        if "S3_D_adv" in d:
            s = "S3_D_adv"
            H.append((lab, st, "H23", f"adversarial: CIFC-Q {g(s,'CIFC_Q')*100:.1f}, CIFC {g(s,'CIFC_r20')*100:.1f}",
                      g(s, "CIFC_Q") >= g(s, "CIFC_r20") - 0.01))
        sc = [s for s in ("S3_D_adv", "S3_D_unk", "S5_partial30_D_unk") if s in d]
        if sc:
            txt = "; ".join(f"{s[3:] if s.startswith('S3') else 'partial'} {g(s,'CIFC_Q','damage'):.0f}/{g(s,'CB','damage'):.0f}"
                            for s in sc)
            H.append((lab, st, "H24", "damage CIFC-Q/CB: " + txt, all(g(s, "CIFC_Q", "damage") <= g(s, "CB", "damage") for s in sc)))
    json.dump(H, open(os.path.join(R, "hypotheses_v2.json"), "w"), indent=1)
    lines = ["| Bed | Status | Hyp. | Evidence | Verdict |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {e} | {'supported' if v else 'not supported'} |" for a, b, c, e, v in H]
    open(os.path.join(R, "hypotheses_v2.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
