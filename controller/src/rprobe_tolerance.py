"""Detection power of the reranker known-answer probe at a tolerance calibrated to dynamic-INT8 batch noise
(tau_R = 2 x max |diff| from runs/v2/int8_probe_noise.json). T2 reference pairs = each calibration question with its
healthy BM25 top-1 passage. Score-level faults as in the replay (random: N(0, 3); inverted: -s; constant: 0) and the real
truncated-input scores. Output: runs/v2/rprobe_tolerance.json"""
import os
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_data
R = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
noise = json.load(open(os.path.join(R, "v2", "int8_probe_noise.json")))
mx = max(v["max"] for k, v in noise.items() if k.startswith("int8"))
t = tb_data.TB("quran")
calib = [n for n, q in enumerate(t.qs) if q["split"] == "train"]
out = {"tau_int8": 2 * mx, "tau_registered": 0.05}
for tau_name in ("tau_int8", "tau_registered"):
    tau = out[tau_name]
    for fault in ("random", "inverted", "constant", "trunc"):
        det = []
        for n in calib:
            L = t.L(n, {})
            if not L:
                continue
            s0 = t.rscore(n, L[0], {})
            s1 = t.rscore(n, L[0], {"R": fault})
            if s0 is None or s1 is None:
                continue
            det.append(abs(s1 - s0) > tau)
        # a probe call uses two reference pairs; it flags if either does
        p = float(np.mean(det))
        out[f"{tau_name}|{fault}"] = {"pairs": len(det), "detect_one_pair": p, "detect_two_pairs": 1 - (1 - p) ** 2}
json.dump(out, open(os.path.join(R, "v2", "rprobe_tolerance.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
