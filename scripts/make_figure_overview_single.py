from __future__ import annotations
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO

def _resolve(rel, extra=()):
    for p in (REPO / rel, REPO / rel.replace('outputs/', 'results/')):
        if p.exists():
            return p
    raise FileNotFoundError(rel)
SPEECH_CSV = _resolve('outputs/metrics/speech_convergence.csv')
FIG_DIR = REPO / 'outputs' / 'figures'
OUT_PDF = FIG_DIR / 'figure_convergence_overview_single.pdf'
OUT_PNG = FIG_DIR / 'figure_convergence_overview_single.png'
CONDITIONS = ['name', 'named-speech', 'anon-speech']
COND_LABEL = {'name': 'Name+\ncontext', 'named-speech': 'Named\nspeech', 'anon-speech': 'Anon.\nspeech'}
FRAMING = 'O'
EMPH = {'gpt-5.5': ('GPT-5.5', '#0072B2', 'o', 1.9), 'pooled': ('Open ensemble', '#E69F00', 'D', 1.7)}
CONTEXT = ['gemma-27b', 'gemma-12b', 'qwen-32b', 'qwen-8b']
CONTEXT_MARKER = {'gemma-27b': 's', 'gemma-12b': '^', 'qwen-32b': 'v', 'qwen-8b': 'P'}
C_GREY = '#9E9E9E'
C_CEIL = '#3A3A3A'

def _apply_rcparams() -> None:
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.4, 'figure.dpi': 150})

def main() -> None:
    _apply_rcparams()
    d = pd.read_csv(SPEECH_CSV)
    o = d[(d.trait == 'overall_mean') & (d.framing == FRAMING)].copy()
    order = o.groupby('model').pearson_r.mean().sort_values(ascending=False).index.tolist()
    ceiling = float(o.ceiling.dropna().iloc[0])
    xpos = {c: i for i, c in enumerate(CONDITIONS)}
    dodge = {m: (i - (len(order) - 1) / 2) * 0.065 for i, m in enumerate(order)}
    fig, ax = plt.subplots(figsize=(3.1, 2.75))
    y_lo, y_hi = (0.1, 0.92)
    ax.axhspan(ceiling, y_hi, facecolor='#000000', alpha=0.045, zorder=0)
    ax.axhline(ceiling, color=C_CEIL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(-0.38, ceiling + 0.008, 'Reliability ceiling', ha='left', va='bottom', fontsize=7.5, color=C_CEIL)

    def xy(model):
        g = o[o.model == model].set_index('condition').reindex(CONDITIONS)
        xs = np.array([xpos[c] + dodge[model] for c in CONDITIONS], dtype=float)
        ys = g.pearson_r.to_numpy(float)
        return (xs, ys, g.ci_low.to_numpy(float), g.ci_high.to_numpy(float))
    for model in CONTEXT:
        xs, ys, _, _ = xy(model)
        ok = ~np.isnan(ys)
        ax.plot(xs[ok], ys[ok], color=C_GREY, lw=0.9, marker=CONTEXT_MARKER[model], ms=2.8, mec='white', mew=0.3, alpha=0.85, zorder=3)
    for model, (_, col, mk, lw) in EMPH.items():
        xs, ys, lo, hi = xy(model)
        ok = ~np.isnan(ys)
        ax.errorbar(xs[ok], ys[ok], yerr=[ys[ok] - lo[ok], hi[ok] - ys[ok]], color=col, lw=lw, marker=mk, ms=4.8, mec='white', mew=0.5, ecolor=to_rgba(col, 0.5), elinewidth=0.8, capsize=1.8, capthick=0.7, zorder=5)
    ax.set_xlim(-0.42, 2.3)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax.set_yticks(np.arange(0.2, 0.81, 0.2))
    ax.set_ylabel('Convergence  (Pearson $r$)')
    ax.yaxis.grid(True, color='0.90', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5)
    handles = [Line2D([], [], color=EMPH['gpt-5.5'][1], marker='o', lw=1.9, ms=5.0, mec='white', mew=0.5, label=EMPH['gpt-5.5'][0]), Line2D([], [], color=EMPH['pooled'][1], marker='D', lw=1.7, ms=5.0, mec='white', mew=0.5, label=EMPH['pooled'][0]), Line2D([], [], color=C_GREY, marker='o', lw=0.9, ms=2.8, label='Individual open-weight models')]
    ax.legend(handles=handles, loc='lower left', frameon=False, handletextpad=0.5, labelspacing=0.4, borderaxespad=0.2)
    fig.subplots_adjust(left=0.145, right=0.99, top=0.975, bottom=0.17)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print('overall r (Ø) by model x condition:')
    print(o.pivot_table(index='model', columns='condition', values='pearson_r').reindex(order)[CONDITIONS].round(3))
    print(f'\nceiling={ceiling:.2f}   wrote {OUT_PDF}\n      {OUT_PNG}')
if __name__ == '__main__':
    main()
