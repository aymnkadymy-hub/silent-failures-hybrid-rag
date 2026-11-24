"""Figures for the dead-arm paper. Same style as the first IJS paper: Times New Roman, 6.0 in wide,
exact size (no tight bbox), 600 dpi, one palette. All values are read from the experiment outputs."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '.'
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

