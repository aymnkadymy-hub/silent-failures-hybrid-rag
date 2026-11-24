"""GPU fill of mmarco cross-encoder scores for the (question, passage) pairs in runs/rerank_need.json.
Same model, inputs (question text q, passage text t, max_length 512) and loading code as the I01 X1 study
(x1_steps.step_rerank), with its per-batch thermal guard (pause >= 78 C, resume <= 70 C) and ~50% duty cycle.
Resumable: saves runs/mmarco_fill.json every ~20k pairs."""
import os
import json, os, sys, time
X1SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "dead_arm/X1_fault_injection")
sys.path.insert(0, X1SRC)
import x1_steps as X   # noqa: E402  (puts the accelerate overlay on sys.path before transformers)
RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "controller/results"))
COPIES = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data/t1"))


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    need = json.load(open(os.path.join(RUNS, "rerank_need.json")))
    out_fn = os.path.join(RUNS, "mmarco_fill.json")
    done = json.load(open(out_fn)) if os.path.exists(out_fn) else {}
    want = [k for k in need if k not in done]
    X.log("pairs", len(want), "already", len(done))
    qs = json.load(open(os.path.join(COPIES, "questions.json"), encoding="utf-8"))
    ids = {int(k.split(":")[1]) for k in want}
    text = {}
    with open(os.path.join(COPIES, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in ids:
                text[i] = json.loads(line)["t"]
    X.log("texts", len(text))
    # parity: re-score 64 stored pairs from the I01 cache before filling
    ref = json.load(open(os.path.join(X1SRC, "results", "mmarco_scores.json")))
    probe = list(ref)[:64]
    ptext = {}
    pid = {int(k.split(":")[1]) for k in probe} - set(text)
    with open(os.path.join(COPIES, "corpus.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in pid:
                ptext[i] = json.loads(line)["t"]
    ptext.update(text)
    tok = AutoTokenizer.from_pretrained(X.MMARCO)
    rr = X.gpu_model(AutoModelForSequenceClassification, X.MMARCO)
    torch.set_num_threads(1)
    with torch.no_grad():
        enc = tok([qs[int(k.split(":")[0])]["q"] for k in probe], [ptext[int(k.split(":")[1])] for k in probe],
                  padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        got = rr(**enc).logits[:, 0].float().cpu().numpy().tolist()
    dev = max(abs(g - ref[k]) for g, k in zip(got, probe))
    X.log(f"parity max |diff| over 64 stored pairs = {dev:.4f}")
    if dev > 0.05:
        sys.exit("parity failed")
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(want), 32):
            X.thermal_guard()
            tb = time.time()
            part = want[s:s + 32]
            enc = tok([qs[int(k.split(":")[0])]["q"] for k in part], [text[int(k.split(":")[1])] for k in part],
                      padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            done.update(zip(part, rr(**enc).logits[:, 0].float().cpu().numpy().tolist()))
            time.sleep(max(0.05, time.time() - tb))
            if (s // 32) % 600 == 0:
                X.log(f"fill {s}/{len(want)} ({time.time() - t0:.0f}s)")
                json.dump(done, open(out_fn, "w"))
    json.dump(done, open(out_fn, "w"))
    X.log(f"filled {len(want)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
