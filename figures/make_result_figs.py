"""Result figures. Every value is read from the run outputs (runs/*.json); nothing is typed in.
  fig2_certificate   worst-case loss band vs beta, per bank, with Clopper-Pearson bounds (T1, T2, T3)
  fig3_cost_exposure verification cost vs exposed queries on deployment faults (T2), with the spot-check expectation
  fig4_delta_rrf     Hit@20 minus RRF per fault family and policy (T2 and T3)
"""
import glob
import json
import os

import numpy as np

from figstyle import *  # noqa: F401,F403

RUNS = os.environ.get("RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller/results"))


def load_tb(tb):
    out = {}
    pat = os.path.join(RUNS, "t1_test_pool_*.json") if tb == "t1" else os.path.join(RUNS, f"tb_{tb}", "test_pool_*.json")
    for fn in glob.glob(pat):
        if fn.endswith("_rows.json"):
            continue
        for sc, pv in json.load(open(fn))["agg"].items():
            out.setdefault(sc, {}).update(pv)
    return out


# ---------------------------------------------------------------- fig 2: certificate
def fig2():
    # Readable in black and white: collection = marker shape, split = line style and marker fill.
    # Colours are kept for the colour version. Values are read from the same run files as before.
    MK = {"Classical Arabic": "o", "Qur'an QA": "s", "SciFact": "^", "NFCorpus": "D"}
    banks = []
    c1 = json.load(open(os.path.join(RUNS, "t1_certify.json")))
    banks += [("Classical Arabic, calibration", c1["banks"]["dev"], BLUE, "-"),
              ("Classical Arabic, held-out", c1["banks"]["ho"], BLUE, "--"),
              ("Classical Arabic, quote bank", c1["banks"]["books"], BLUE, ":")]
    for tb, col, nm in (("quran", ORANGE, "Qur'an QA"), ("scifact", AQUA, "SciFact"), ("nfcorpus", DBLUE, "NFCorpus")):
        fn = os.path.join(RUNS, f"tb_{tb}", "certify.json")
        if os.path.exists(fn):
            c = json.load(open(fn))
            banks += [(f"{nm}, calibration", c["banks"]["dev"], col, "-"), (f"{nm}, test", c["banks"]["ho"], col, "--")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7))
    for nm, b, col, ls in banks:
        betas = sorted(float(x) for x in b)
        r = [b[str(x) if str(x) in b else x]["rate"] * 100 for x in betas]
        u = [b[str(x) if str(x) in b else x]["ucb"] * 100 for x in betas]
        mk = MK[nm.split(",")[0]]
        filled = "calibration" in nm
        style = dict(ls=ls, color=col, marker=mk, ms=4.2 if mk != "D" else 3.6, lw=1.1,
                     markerfacecolor=col if filled else "white", markeredgecolor=col, markeredgewidth=0.9)
        axs[0].plot(betas, r, label=nm, **style)
        if filled:
            axs[1].plot(betas, u, label=nm.replace(", calibration", ""), **style)
    for e in (1, 2, 5):
        axs[1].axhline(e, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        axs[1].text(0.515, e - 0.12, f"$\\varepsilon$ = {e}%", fontsize=7, color=MUTED, ha="left", va="top")
    axs[0].set_title("(a) Measured worst-case loss band", loc="left")
    axs[1].set_title("(b) Upper bound on calibration (95%)", loc="left")
    for a in axs:
        a.set_xlabel(r"Reserved share $\beta$")
        a.grid(True)
    axs[0].set_ylabel("Questions in band (%)")
    axs[0].legend(fontsize=6.4, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, handlelength=3.2)
    axs[1].legend(fontsize=6.4, frameon=False, handlelength=3.2)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.9, wspace=0.25)
    save(fig, "fig2_certificate")


# ---------------------------------------------------------------- fig 3: cost vs exposure
DEPLOY_FAULTS = ["S3_D_unk", "S3_D_minilm", "S3_D_offset", "S3_L_shard", "S3_L_nofold", "S3_Q_strip", "S3_S_shift"]
SILENT_FAULTS = ["S3_D_rand", "S3_D_noise", "S3_D_ann", "S3_D_corrupt", "S3_D_adv", "S3_L_cache", "S3_L_empty"]


def fig3(tb="quran"):
    d = load_tb(tb)
    pols = [("P3_deadarm", "Dead-Arm audit", MUTED, "s"), ("G", "G (canaries)", INK2, "D"),
            ("G_kat", "Full probes, every batch", ORANGE, "^"), ("CIFC_noint", "Active probing, no interventions", DBLUE, "v"),
            ("CIFC_rand", "CIFC, random test order", AQUA, "x")]
    cifc = [("CIFC_r100", 1.0), ("CIFC_r50", 0.5), ("CIFC_r20", 0.2), ("CIFC_r10", 0.1)]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
    for ax, fam, title in ((axs[0], DEPLOY_FAULTS, "(a) Deployment faults"), (axs[1], SILENT_FAULTS, "(b) Silent faults")):
        fam = [s for s in fam if s in d]

        def mean(p, k):
            return float(np.mean([d[s][p].get(k) or 0 for s in fam]))
        for p, nm, col, mk in pols:
            ax.scatter(mean(p, "cost_units"), mean(p, "exposed"), color=col, marker=mk, s=18, label=nm, zorder=3)
        xs = [mean(p, "cost_units") for p, _ in cifc]
        ys = [mean(p, "exposed") for p, _ in cifc]
        ax.plot(xs, ys, color=BLUE, marker="o", ms=3.5, lw=1.0, label="CIFC, probing 100/50/20/10%", zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Verification cost per stream (stage runs)")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(True)
    axs[0].set_ylabel("Exposed queries per fault (log)")
    axs[1].legend(fontsize=6.2, frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.9, wspace=0.22)
    save(fig, "fig3_cost_exposure")


# ---------------------------------------------------------------- fig 4: delta vs RRF
def fig4():
    fams = [("S3_D_unk", "Tokenizer"), ("S3_D_minilm", "Weak\nencoder"), ("S3_D_offset", "Index\noffset"),
            ("S3_D_ann", "ANN\nloss*"), ("S3_D_adv", "Adver-\nsarial*"), ("S3_L_shard", "Shard\nlost"),
            ("S3_L_cache", "Cache\nbug*"), ("S3_Q_strip", "Normal-\niser"), ("S3_S_shift", "Store\nids"),
            ("S2_shift_split", "Shift,\nno fault")]
    pols = [("P3_deadarm", "Dead-Arm audit", MUTED), ("G", "G", INK2), ("CIFC_r20", "CIFC", BLUE),
            ("CIFC_gradec", "CIFC, graded by cause (exploratory on Qur'an QA)", AQUA), ("O_oracle", "Oracle", ORANGE)]
    tbs = [t for t in ("quran", "scifact", "nfcorpus") if glob.glob(os.path.join(RUNS, f"tb_{t}", "test_pool_*.json"))]
    fig, axs = plt.subplots(len(tbs), 1, figsize=(W, 2.0 * len(tbs) + 0.6), squeeze=False)
    for ax, tb in zip(axs[:, 0], tbs):
        d = load_tb(tb)
        fl = [(s, n) for s, n in fams if s in d or (s == "S2_shift_split" and s in d)]
        x = np.arange(len(fl))
        w = 0.8 / len(pols)
        for j, (p, nm, col) in enumerate(pols):
            y = [(d[s][p]["h20"] - d[s]["P0_rrf"]["h20"]) * 100 if p in d[s] else np.nan for s, _ in fl]
            ax.bar(x + (j - len(pols) / 2 + 0.5) * w, y, width=w, color=col, label=nm)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in fl], fontsize=6.6, rotation=0, linespacing=1.0)
        ax.set_ylabel("Hit@20 minus RRF (points)", fontsize=7.5)
        ax.set_title({"quran": "Qur'an QA (Arabic)", "scifact": "SciFact (English)", "nfcorpus": "NFCorpus (English)"}[tb],
                     loc="left")
        ax.grid(True, axis="y")
    hnd, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(hnd, lab, fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.07, top=0.93, hspace=0.62)
    save(fig, "fig4_delta_rrf")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["fig2", "fig3", "fig4"]:
        globals()[f]()
        print(f, "ok")
