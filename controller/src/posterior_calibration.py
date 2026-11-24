"""Accuracy of confirmed causes (CIFC stops at posterior >= 0.95): share of confirmations that name the true cause,
split by full-traffic faults vs the 30% partial fault, per testbed (v1 runs). Output: runs/v2/posterior_calibration.json"""
import glob, json, os
R = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
out = {}
for tb in ("quran", "scifact", "nfcorpus"):
    rows = {}
    for fn in glob.glob(os.path.join(R, f"tb_{tb}", "test_pool_*_rows.json")):
        for s, pv in json.load(open(fn)).items():
            rows.setdefault(s, {}).update(pv)
    for kind, sel in (("full", lambda s: s.startswith("S3_")), ("partial", lambda s: s.startswith("S5_"))):
        conf = ok = 0
        for s, pv in rows.items():
            if not sel(s) or "CIFC_r20" not in pv:
                continue
            for r in pv["CIFC_r20"]:
                if r.get("cause") is not None:
                    conf += 1
                    ok += r["cause"] == r["true_cause"]
        out[f"{tb}|{kind}"] = {"confirmed": conf, "correct": ok, "accuracy": ok / conf if conf else None}
json.dump(out, open(os.path.join(R, "v2", "posterior_calibration.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v)
