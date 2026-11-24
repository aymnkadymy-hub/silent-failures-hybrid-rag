"""Figure 1: the retrieval chain and the governor (sense, detect, diagnose, act, verify)."""
from figstyle import *  # noqa: F401,F403

fig = plt.figure(figsize=(W, 3.55))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 60)
ax.set_ylim(0, 35.5)
ax.axis('off')
LB, LO, LG, LGR = '#e8f1fc', '#fdeee7', '#e3f6ee', '#f2f1ee'

# retrieval chain (top)
ax.text(1, 34.2, 'Retrieval chain (served path)', fontsize=8.5, fontweight='bold', va='top')
box(ax, 1, 25.5, 6.2, 5.6, 'Query\nnormaliser\n(Q)', LGR, INK2)
box(ax, 9.6, 28.9, 9.4, 4.0, 'BM25 index\nand analyser (L)', LB, BLUE, fs=7.4)
box(ax, 9.6, 22.6, 9.4, 5.4, 'Tokenizer,\nencoder, vector\nindex (D)', LO, ORANGE, fs=7.4)
box(ax, 21.6, 25.5, 9.8, 5.6, 'Merge (F)\nRRF, or prefix-\nconstrained fusion', LG, AQUA)
box(ax, 34.0, 25.5, 7.4, 5.6, 'Cross-encoder\nreranker (R)', LGR, INK2)
box(ax, 44.0, 25.5, 6.6, 5.6, 'Passage\nstore and\nid map (S)', LGR, INK2)
box(ax, 53.0, 25.5, 6.2, 5.6, 'Served\npassages', 'white', INK2)
arrow(ax, (7.2, 29.6), (9.6, 30.6))
arrow(ax, (7.2, 27.0), (9.6, 25.8))
arrow(ax, (19.0, 30.6), (21.6, 29.4))
arrow(ax, (19.0, 25.6), (21.6, 27.0))
arrow(ax, (31.4, 28.3), (34.0, 28.3))
arrow(ax, (41.4, 28.3), (44.0, 28.3))
arrow(ax, (50.6, 28.3), (53.0, 28.3))

# governor (bottom)
ax.text(1, 19.2, 'Governor (deterministic agent, one loop per batch of 20 queries)', fontsize=8.5, fontweight='bold',
        va='top')
box(ax, 1, 8.6, 10.6, 8.6, 'Sense\nagreement signals,\nempty lists, vector\ncollapse, displacement', 'white', INK2,
    fs=7.4)
box(ax, 13.6, 8.6, 9.0, 8.6, 'Detect\ntwo-sided CUSUM\nper signal', 'white', INK2, fs=7.4)
box(ax, 24.6, 8.6, 11.6, 8.6, 'Diagnose\nsuspects from the\nagreement graph;\ncanary test per part;\nfault or traffic shift', 'white',
    INK2, fs=7.4)
box(ax, 38.2, 8.6, 10.6, 8.6, 'Act per part\nNormal, Guarded,\nFallback,\nValidation', 'white', INK2, fs=7.4)
box(ax, 50.8, 8.6, 8.4, 8.6, 'Verify\nshadow canaries,\nprobation,\nthen Normal', 'white', INK2, fs=7.4)
for a, b in ((11.6, 13.6), (22.6, 24.6), (36.2, 38.2), (48.8, 50.8)):
    arrow(ax, (a, 12.9), (b, 12.9))
box(ax, 1, 0.6, 27.5, 6.6, 'Known-answer probes travel with each batch\nthrough Q, L, D, R and S; a mismatch removes\nthat part before the merge commits', LG, AQUA, fs=7.4)
box(ax, 30.5, 0.6, 28.7, 6.6, 'Version fingerprints (weights, tokenizer,\nnormaliser, index): a deployment triggers\nvalidation before the part is trusted', LB, BLUE, fs=7.4)
# signals up, actions down
arrow(ax, (6.0, 23.4), (6.0, 17.2), color=MUTED, lw=0.7)
ax.text(6.6, 20.6, 'signals', fontsize=7.2, color=MUTED)
arrow(ax, (43.5, 17.2), (27.0, 25.5), color=AQUA, lw=0.9)
ax.text(36.6, 21.4, 'beta, route-around', fontsize=7.2, color=AQUA)
check_fit(fig)
save(fig, 'fig1_architecture')
print('ok')
