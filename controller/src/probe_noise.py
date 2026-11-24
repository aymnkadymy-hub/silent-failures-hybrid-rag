"""Measured false-flag rate of the known-answer probes on T2 (Qur'an QA), healthy pipeline (A9).

The stored references (query vector, dense top-1, reranker score) were produced in batches of 16. Here each probe
question is re-encoded inside batches of size 1, 8, 32 and 64, filled with random other questions (3 fillings each),
in fp16 on the GPU, and the reranker reference pair is re-scored inside batches of other pairs. We record the cosine
to the reference vector, whether the dense top-1 id changes, and |score - reference|.
Output: runs/tb_quran/probe_noise.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_build as T   # noqa: E402  (puts the X1 helpers and the accelerate overlay on sys.path)

X = T.X
RUNS = T.RUNS


def main():
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    od = os.path.join(RUNS, "tb_quran")
    z = np.load(os.path.join(od, "build.npz"))
    rr = json.load(open(os.path.join(od, "rr.json")))
    texts, qs, lang = T.load("quran")
    meta = json.load(open(os.path.join(od, "meta.json")))
    assert [q["id"] for q in qs] == [q["id"] for q in meta["qs"]]
    rng = np.random.default_rng(3)
    probes = list(range(len(qs)))
    tok = AutoTokenizer.from_pretrained(X.ARCTIC)
    m = X.gpu_model(AutoModel, X.ARCTIC)
    P = X.encode(tok, m, [T.dense_text(t, lang) for t in texts], "cls", 512, batch=16)
    ref = z["Q_ok"].astype(np.float32)
    ref_top1 = z["D_ok_ids"][:, 0]
    out = {"vec": []}
    with torch.no_grad():
        for bs in (1, 8, 32, 64):
            for rep in range(3):
                X.thermal_guard()
                for n in probes:
                    others = [int(i) for i in rng.choice(len(qs), bs - 1, replace=False) if i != n][: bs - 1]
                    batch = [n] + others
                    rng.shuffle(batch)
                    V = X.encode(tok, m, [T.dense_text(qs[i]["q"], lang) for i in batch], "cls", 512, prefix="query: ",
                                 batch=len(batch))
                    v = V[batch.index(n)]
                    cos = float(v @ ref[n] / (np.linalg.norm(v) * np.linalg.norm(ref[n])))
                    top1 = int(np.argmax(P @ v))
                    out["vec"].append({"n": n, "bs": bs, "rep": rep, "cos": cos, "top1_same": top1 == int(ref_top1[n])})
            X.log("vec bs", bs, "done")
        del m
        torch.cuda.empty_cache()
        rtok = AutoTokenizer.from_pretrained(X.MMARCO)
        rm = X.gpu_model(AutoModelForSequenceClassification, X.MMARCO)
        out["rr"] = []
        pairs = [(n, int(z["L_ok"][n][0])) for n in probes if z["L_ok"][n][0] >= 0 and f"ok|{n}:{int(z['L_ok'][n][0])}" in rr]
        keys = list(rr)
        for bs in (1, 8, 32):
            X.thermal_guard()
            for n, i in pairs:
                fill = [tuple(map(int, keys[j].split("|")[1].split(":"))) for j in rng.choice(len(keys), bs - 1)]
                batch = [(n, i)] + fill
                enc = rtok([qs[a]["q"] for a, b in batch], [texts[b] for a, b in batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to("cuda")
                s = float(rm(**enc).logits[0, 0].float().cpu())
                out["rr"].append({"n": n, "bs": bs, "diff": abs(s - rr[f"ok|{n}:{i}"])})
            X.log("rr bs", bs, "done")
    cos = np.array([r["cos"] for r in out["vec"]])
    same = np.array([r["top1_same"] for r in out["vec"]])
    d = np.array([r["diff"] for r in out["rr"]])
    out["summary"] = {"n_vec": int(len(cos)), "cos_min": float(cos.min()), "cos_p01": float(np.percentile(cos, 1)),
                      "flag_rate_cos099": float((cos < 0.99).mean()), "top1_change_rate": float((~same).mean()),
                      "n_rr": int(len(d)), "rr_diff_max": float(d.max()) if len(d) else None,
                      "rr_flag_rate_005": float((d > 0.05).mean()) if len(d) else None}
    json.dump(out, open(os.path.join(od, "probe_noise.json"), "w"))
    X.log("summary", out["summary"])


if __name__ == "__main__":
    main()
