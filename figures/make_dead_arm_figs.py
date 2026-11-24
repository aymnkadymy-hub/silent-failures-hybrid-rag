"""Figures for the dead-arm paper. Same style as the first IJS paper: Times New Roman, 6.0 in wide,
exact size (no tight bbox), 600 dpi, one palette. All values are read from the experiment outputs."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

X1 = sys.argv[1]; X2 = sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else '.'
for f in ['/usr/share/fonts/TTF/Times.TTF', '/usr/share/fonts/TTF/Timesbd.TTF',
          '/usr/share/fonts/TTF/Timesi.TTF', '/usr/share/fonts/TTF/Timesbi.TTF']:
    if os.path.exists(f):            # Times New Roman, if installed at this path
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.3,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold', 'mathtext.sf': 'Times New Roman', 'mathtext.fallback': 'stix',
    'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b', 'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
    'savefig.dpi': 600, 'figure.dpi': 150})
W = 6.0
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8c8a85'
BLUE, ORANGE, AQUA, DBLUE = '#2a78d6', '#eb6834', '#1baf7a', '#104281'

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=600, facecolor='white')
    fig.savefig(f'{OUT}/{name}.tif', dpi=600, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

BOXES = []
def box(ax, x, y, w, h, text, fc, ec, fs=8.0, weight='normal', lw=0.9):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.06', fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight=weight, color=INK,
                zorder=3, linespacing=1.15)
    if text:
        BOXES.append((p, t))

def arrow(ax, p, q, color=INK2, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=lw, color=color, zorder=4,
                                 shrinkA=0, shrinkB=0))

def check_fit(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for p, t in BOXES:
        pb = p.get_window_extent(r); tb = t.get_window_extent(r)
        m = (tb.x0 - pb.x0, pb.x1 - tb.x1, tb.y0 - pb.y0, pb.y1 - tb.y1)
        if min(m) < 3:
            bad.append(t.get_text().replace('\n', ' | ') + ' ' + ','.join(f'{v:.1f}' for v in m))
    if bad:
        raise SystemExit('text overflows its box: ' + '; '.join(bad))

def fig_pipeline():
    T_FC, T_EC = '#e7f0fc', BLUE
    F_FC, F_EC = '#fdebe3', ORANGE
    S_FC = '#ffffff'
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.set_position([0.004, 0.004, 0.992, 0.992]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis('off')
    box(ax, 0.1, 1.75, 1.15, 0.8, 'Query', S_FC, INK2, fs=8.3)
    box(ax, 1.65, 2.75, 2.15, 0.85, 'Lexical arm\nBM25, top N', T_FC, T_EC, fs=8.0)
    box(ax, 1.65, 0.7, 2.15, 1.05, 'Dense arm\nencoder + vector index\ntop N, cosine floor', T_FC, T_EC, fs=8.0)
    box(ax, 4.25, 1.75, 1.5, 0.8, 'RRF fusion\n$\\kappa$ = 60, head 48', S_FC, INK2, fs=8.0)
    box(ax, 6.15, 1.75, 1.55, 0.8, 'Cross-encoder\nreranker', S_FC, INK2, fs=8.0)
    box(ax, 8.1, 1.75, 1.8, 0.8, 'Top passages\nto the LLM', S_FC, INK2, fs=8.0)
    arrow(ax, (1.25, 2.3), (1.65, 3.05)); arrow(ax, (1.25, 2.0), (1.65, 1.3))
    arrow(ax, (3.8, 3.17), (4.25, 2.45)); arrow(ax, (3.8, 1.2), (4.25, 1.85))
    arrow(ax, (5.75, 2.15), (6.15, 2.15)); arrow(ax, (7.7, 2.15), (8.1, 2.15))
    # injected faults
    box(ax, 1.4, 0.05, 2.65, 0.45, 'Faults: near-dead encoder, collapsed\ntokenizer, random vectors; arm off', F_FC, F_EC, fs=7.6)
    arrow(ax, (2.72, 0.5), (2.72, 0.7), color=F_EC)
    # measurement points
    for x, lab in [(2.72, 'S1'), (5.0, 'S2'), (6.92, 'S3')]:
        pass
    ax.text(2.72, 1.92, 'S1: dense Hit@10', ha='center', va='bottom', fontsize=7.8, color=INK2)
    ax.text(5.0, 2.62, 'S2: pool Hit@20', ha='center', va='bottom', fontsize=7.8, color=INK2)
    ax.text(6.92, 2.62, 'S3: reranked Hit@8', ha='center', va='bottom', fontsize=7.8, color=INK2)
    # checks
    box(ax, 4.3, 0.12, 2.3, 0.88, 'Check A: dense Hit@10 on\n20 to 30 labelled questions', '#eaf7f1', AQUA, fs=7.8)
    box(ax, 7.1, 0.12, 2.8, 0.88, 'Check B: overlap of dense and\nBM25 top-48 over 20 queries', '#eaf7f1', AQUA, fs=7.8)
    arrow(ax, (3.8, 0.95), (4.3, 0.56), color=AQUA)
    arrow(ax, (5.6, 1.75), (7.3, 1.0), color=AQUA)
    check_fit(fig)
    save(fig, 'fig1_pipeline')

def fig_detection():
    d = json.load(open(f'{X1}/x1_detect.json')); e = json.load(open(f'{X1}/x1_extra.json'))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3))
    ax = axes[0]; ax.grid(True, axis='y', zorder=0)
    sizes = ['30', '60', '120', '200', '500']; xs = [int(s) for s in sizes]
    series = [('unk|bake|n5dev_all|pool_R20', 'Generated, pool Hit@20', BLUE, 'o'),
              ('unk|bake|n5dev_all|rerank_R8', 'Generated, reranked Hit@8', ORANGE, 'D'),
              ('unk|bake|books_v1_dev|rerank_R8', 'Quote-anchored, any stage', INK2, 's')]
    for key, lab, c, mk in series:
        dp = d[key]['detect_prob_by_n']; xx = [int(s) for s in sizes if s in dp]; yy = [100 * dp[s] for s in sizes if s in dp]
        ax.plot(xx, yy, color=c, lw=1.6, marker=mk, ms=5, mec='white', mew=0.7, label=lab, zorder=3)
    ax.set_xscale('log'); ax.set_xticks(xs); ax.set_xticklabels(sizes); ax.minorticks_off()
    ax.set_ylim(-3, 105); ax.set_xlabel('Questions in the evaluation bank'); ax.set_ylabel('Detection probability (%)')
    ax.set_title('(a)', loc='left', fontweight='bold'); ax.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.0, 0.03), fontsize=8)
    ax = axes[1]; ax.grid(True, axis='y', zorder=0)
    ls = ['10', '20', '30', '50', '100']; lx = [int(s) for s in ls]
    for bank, lab, c, mk in [('n5dev_all', 'Generated questions', BLUE, 'o'), ('books_v1_dev', 'Quote-anchored questions', INK2, 's')]:
        dp = e['labelled'][f'unk|{bank}|dense_R10']['detect_prob_by_n']
        ax.plot(lx, [100 * dp[s] for s in ls], color=c, lw=1.6, marker=mk, ms=5, mec='white', mew=0.7, label=lab, zorder=3)
    ax.set_xscale('log'); ax.set_xticks(lx); ax.set_xticklabels(ls); ax.minorticks_off()
    ax.set_ylim(-3, 105); ax.set_xlabel('Labelled questions (dense arm alone)'); ax.set_ylabel('Detection probability (%)')
    ax.set_title('(b)', loc='left', fontweight='bold'); ax.legend(loc='lower right', frameon=False)
    fig.tight_layout(pad=0.3, w_pad=2.0)
    save(fig, 'fig2_detection')

def fig_halving():
    e = json.load(open(f'{X1}/x1_extra.json'))['halving']; x2 = json.load(open(f'{X2}/x2_results.json'))['halving']
    fig, ax = plt.subplots(figsize=(W, 2.35))
    ax.grid(True, zorder=0)
    lo, hi = 25, 75
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, zorder=1)
    style = {'minilm': (DBLUE, 'D', False), 'unk': (BLUE, 'o', True), 'rand1': (AQUA, 's', True)}
    seen = set()
    for k, v in e.items():
        c, pool, bank, rk = k.split('|')
        if pool != 'bake' or bank not in ('n5dev_all', 'n5heldout', 'books_v1_dev') or c not in style:
            continue
        col, mk, filled = style[c]
        ax.plot(100 * v['bm25_at_half_depth'], 100 * v['fused_dead'], ls='none', marker=mk, ms=6, color=col,
                mfc=col if filled else 'white', mec=col if not filled else 'white', mew=1.0 if not filled else 0.6, zorder=3)
    for k, v in x2.items():
        c, n, rk = k.split('|')
        if n != 'N48' or c not in ('unk', 'rand1', 'minilm'):
            continue
        col, mk, filled = style[c]
        ax.plot(100 * v['bm25_half'], 100 * v['fused_dead'], ls='none', marker=mk, ms=7.5, color=ORANGE,
                mfc=ORANGE if filled else 'white', mec=ORANGE if not filled else 'white', mew=1.0 if not filled else 0.6, zorder=4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='none', marker='o', color=BLUE, mec='white', ms=6, label='Collapsed tokenizer'),
         Line2D([], [], ls='none', marker='s', color=AQUA, mec='white', ms=6, label='Random vectors'),
         Line2D([], [], ls='none', marker='D', color=DBLUE, mfc='white', ms=6, label='Near-dead encoder (MiniLM)'),
         Line2D([], [], ls='none', marker='o', color=ORANGE, mec='white', ms=7, label='Qur’an QA 2023, same three faults'),
         Line2D([], [], color=MUTED, lw=0.9, label='Prediction: fused Hit@k = BM25 Hit@k/2')]
    ax.legend(handles=h, loc='upper left', frameon=False, fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('BM25 hit rate at half depth, Hit@k/2 (%)'); ax.set_ylabel('Fused hit rate, dead arm, Hit@k (%)')
    ax.annotate('MiniLM on Qur’an QA:\nweak, not dead', xy=(42.0, 52.2), xytext=(50.5, 35.5), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    fig.tight_layout(pad=0.3)
    save(fig, 'fig3_halving')

if __name__ == '__main__':
    fig_pipeline(); fig_detection(); fig_halving(); print('ok')
