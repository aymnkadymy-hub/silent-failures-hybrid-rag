"""False-flag rate of the reranker known-answer probe (|score - reference| > 0.05) under batch-composition changes,
from the stored runs of the INT8 batch-dependence study (read-only). Reference = the 'live' arrangement; probe = any
other arrangement of the same model. Output: runs/v2/int8_probe_noise.json"""
import glob, json, os
import numpy as np
D = os.environ.get("INT8_RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/int8_runs"))
out = {}
for model in ("int8", "fp32", "frozen"):
    for data in ("C", "P"):
        ref_fn = os.path.join(D, f"{model}__{data}__live.json")
        if not os.path.exists(ref_fn):
            continue
        ref = {r["id"]: np.array(r["scores"], float) for r in json.load(open(ref_fn))["rows"]}
        for fn in sorted(glob.glob(os.path.join(D, f"{model}__{data}__*.json"))):
            arm = os.path.basename(fn).split("__")[2][:-5]
            if arm == "live":
                continue
            diffs = []
            for r in json.load(open(fn))["rows"]:
                if r["id"] in ref and len(r["scores"]) == len(ref[r["id"]]):
                    diffs.append(np.abs(np.array(r["scores"], float) - ref[r["id"]]))
            if not diffs:
                continue
            d = np.concatenate(diffs)
            top = np.array([x[0] for x in diffs])      # one reference pair per question (its first candidate)
            out[f"{model}|{data}|{arm}"] = {"pairs": int(d.size), "max": float(d.max()), "p99": float(np.percentile(d, 99)),
                                            "flag_rate_005_all": float((d > 0.05).mean()),
                                            "flag_rate_005_one_pair": float((top > 0.05).mean()), "questions": len(top)}
json.dump(out, open(os.path.join(os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results")), "v2/int8_probe_noise.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:22s} pairs {v['pairs']:6d} max {v['max']:.4f} p99 {v['p99']:.4f} flag>0.05 all {v['flag_rate_005_all']*100:5.1f}% one-pair {v['flag_rate_005_one_pair']*100:5.1f}%")
