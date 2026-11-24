from __future__ import annotations
import json
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO

def _resolve(rel, extra=()):
    for p in (REPO / rel, REPO / rel.replace('outputs/', 'results/')):
        if p.exists():
            return p
    raise FileNotFoundError(rel)
CONV_CSV = _resolve('outputs/metrics/convergence.csv')
BIAS_CSV = _resolve('outputs/metrics/directional_bias.csv')
CEIL_JSON = _resolve('outputs/expert_ceiling.json')
FIG_DIR = REPO / 'outputs' / 'figures'
OUT_PDF = FIG_DIR / 'figure_convergence_bias.pdf'
OUT_PNG = FIG_DIR / 'figure_convergence_bias.png'
STRATUM = 'c1s'
FRAMING = 'O'
ANCHOR = 'gpt-5.5'
ENSEMBLE = 'pooled'
CEILING_RHO = '0.3'
C_ANCHOR = '#0072B2'
C_ENSEMBLE = '#E69F00'
C_CEIL = '#3A3A3A'
C_CEIL_FILL = '#000000'
C_ZERO = '#3A3A3A'
PRETTY = {'extraversion': 'Extraversion', 'agreeableness': 'Agreeableness', 'conscientiousness': 'Conscientiousness', 'emotional_stability': 'Emotional stability', 'openness': 'Openness', 'narcissism': 'Narcissism', 'psychopathy': 'Psychopathy', 'machiavellianism': 'Machiavellianism'}
BIG_FIVE = ['extraversion', 'agreeableness', 'conscientiousness', 'emotional_stability', 'openness']
DARK_TRIAD = ['narcissism', 'psychopathy', 'machiavellianism']

def _apply_rcparams() -> None:
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 9.0, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.4, 'figure.dpi': 150})

def load() -> pd.DataFrame:
    conv = pd.read_csv(CONV_CSV)
    bias = pd.read_csv(BIAS_CSV)
    ceil = json.load(open(CEIL_JSON))
    cell = conv[(conv.stratum == STRATUM) & (conv.framing == FRAMING)]
    bcell = bias[(bias.stratum == STRATUM) & (bias.framing == FRAMING)]

    def cseries(model: str) -> pd.DataFrame:
        s = cell[cell.model == model].set_index('trait')
        return s[['pearson_r', 'pearson_ci_low', 'pearson_ci_high']]

    def bseries(model: str) -> pd.Series:
        return bcell[bcell.model == model].set_index('trait')['bias_sd_units_mean']
    a, e = (cseries(ANCHOR), cseries(ENSEMBLE))
    ba, be = (bseries(ANCHOR), bseries(ENSEMBLE))
    rows = []
    for t in BIG_FIVE + DARK_TRIAD:
        rows.append({'trait': t, 'family': 'Big Five' if t in BIG_FIVE else 'Dark Triad', 'a_r': a.loc[t, 'pearson_r'], 'a_lo': a.loc[t, 'pearson_ci_low'], 'a_hi': a.loc[t, 'pearson_ci_high'], 'e_r': e.loc[t, 'pearson_r'], 'e_lo': e.loc[t, 'pearson_ci_low'], 'e_hi': e.loc[t, 'pearson_ci_high'], 'ceiling': ceil['traits'][t]['by_rho'][CEILING_RHO]['ceiling'], 'a_bias': ba.loc[t], 'e_bias': be.loc[t]})
    df = pd.DataFrame(rows)
    df['mean_r'] = (df.a_r + df.e_r) / 2
    df = pd.concat([df[df.family == fam].sort_values('mean_r', ascending=True) for fam in ('Dark Triad', 'Big Five')], ignore_index=True)
    return df

def main() -> None:
    _apply_rcparams()
    df = load()
    ys, fam_spans = ({}, {})
    cur = 0.0
    GAP = 0.7
    y_mean = cur
    cur += 1.0 + GAP
    for fam in ('Dark Triad', 'Big Five'):
        block = df[df.family == fam]
        lo = cur
        for _, r in block.iterrows():
            ys[r.trait] = cur
            cur += 1.0
        hi = cur - 1.0
        fam_spans[fam] = (lo, hi)
        cur += GAP
    top = cur - GAP
    fig_h = 3.5
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.3, fig_h), sharey=True, gridspec_kw=dict(width_ratios=[1.32, 1.0], wspace=0.06))
    off = 0.19
    xL_lo, xL_hi = (0.3, 0.94)
    for _, r in df.iterrows():
        yy = ys[r.trait]
        axL.add_patch(Rectangle((r.ceiling, yy - 0.5), xL_hi - r.ceiling, 1.0, facecolor=C_CEIL_FILL, alpha=0.055, edgecolor='none', zorder=0))
        axL.plot([r.ceiling, r.ceiling], [yy - 0.42, yy + 0.42], color=C_CEIL, lw=1.2, solid_capstyle='butt', zorder=3)

    def whisker(ax, lo, hi, r, yy, color, marker, ms):
        ax.errorbar(r, yy, xerr=[[r - lo], [hi - r]], fmt='none', ecolor=color, elinewidth=1.2, capsize=2.0, capthick=0.9, zorder=4)
        ax.plot(r, yy, marker=marker, ms=ms, color=color, mec='white', mew=0.5, zorder=6)
    for _, r in df.iterrows():
        yy = ys[r.trait]
        whisker(axL, r.a_lo, r.a_hi, r.a_r, yy + off, C_ANCHOR, 'o', 4.8)
        whisker(axL, r.e_lo, r.e_hi, r.e_r, yy - off, C_ENSEMBLE, 's', 4.6)
    a_mr, e_mr, c_mr = (df.a_r.mean(), df.e_r.mean(), df.ceiling.mean())
    axL.add_patch(Rectangle((c_mr, y_mean - 0.5), xL_hi - c_mr, 1.0, facecolor=C_CEIL_FILL, alpha=0.055, edgecolor='none', zorder=0))
    axL.plot([c_mr, c_mr], [y_mean - 0.42, y_mean + 0.42], color=C_CEIL, lw=1.2, zorder=3)
    axL.plot(a_mr, y_mean + off, marker='D', ms=5.2, color=C_ANCHOR, mec='white', mew=0.5, zorder=6)
    axL.plot(e_mr, y_mean - off, marker='D', ms=5.2, color=C_ENSEMBLE, mec='white', mew=0.5, zorder=6)
    axL.set_xlim(xL_lo, xL_hi)
    axL.set_xticks(np.arange(0.3, 0.91, 0.1))
    axL.set_xlabel('Convergence')
    axL.xaxis.grid(True, color='0.90', lw=0.5, zorder=0)
    xR_lo, xR_hi = (-2.2, 2.2)
    axR.axvline(0.0, color=C_ZERO, lw=1.2, zorder=3)

    def lolli(ax, val, yy, color, marker, ms):
        ax.plot([0.0, val], [yy, yy], color=color, lw=1.2, alpha=0.55, zorder=4)
        ax.plot(val, yy, marker=marker, ms=ms, color=color, mec='white', mew=0.5, zorder=6)
    for _, r in df.iterrows():
        yy = ys[r.trait]
        lolli(axR, r.a_bias, yy + off, C_ANCHOR, 'o', 4.8)
        lolli(axR, r.e_bias, yy - off, C_ENSEMBLE, 's', 4.6)
    axR.set_xlim(xR_lo, xR_hi)
    axR.set_xticks(np.arange(-2.0, 2.001, 1.0))
    axR.set_xlabel('Bias')
    axR.xaxis.grid(True, color='0.90', lw=0.5, zorder=0)
    yticks = [y_mean] + [ys[t] for t in df.trait]
    ylabels = ['Mean (8 traits)'] + [PRETTY[t] for t in df.trait]
    axL.set_yticks(yticks)
    axL.set_yticklabels(ylabels)
    for lab, tk in zip(axL.get_yticklabels(), yticks):
        if tk == y_mean:
            lab.set_fontweight('bold')
    for ax in (axL, axR):
        ax.set_ylim(y_mean - 0.6, top + 0.5)
        ax.set_axisbelow(True)
        ax.axhline(y_mean + 1.0 + GAP / 2 - 0.45, color='0.80', lw=0.5, zorder=1)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=2.5)
    for fam, (lo, hi) in fam_spans.items():
        axR.annotate('', xy=(1.012, hi + 0.42), xytext=(1.012, lo - 0.42), xycoords=('axes fraction', 'data'), arrowprops=dict(arrowstyle='-', color='0.45', lw=0.9))
        axR.text(1.03, (lo + hi) / 2, fam, rotation=270, va='center', ha='left', fontsize=8.0, color='0.30', transform=axR.get_yaxis_transform())
    handles = [Line2D([], [], color=C_ANCHOR, marker='o', ms=4.8, lw=1.2, mec='white', mew=0.5, label='GPT-5.5'), Line2D([], [], color=C_ENSEMBLE, marker='s', ms=4.6, lw=1.2, mec='white', mew=0.5, label='Open ensemble'), Line2D([], [], color=C_CEIL, lw=0, marker='|', ms=10, mew=1.4, label='Expert ceiling'), Patch(facecolor=C_CEIL_FILL, alpha=0.12, edgecolor='none', label='Unattainable')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.895), ncol=4, frameon=False, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0)
    fig.subplots_adjust(left=0.225, right=0.915, top=0.875, bottom=0.16)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f'anchor mean r={a_mr:.3f} ({a_mr / c_mr:.3f} of ceiling); ensemble mean r={e_mr:.3f} ({e_mr / c_mr:.3f}); mean ceiling={c_mr:.3f}')
    print(f'anchor bias range [{df.a_bias.min():+.2f}, {df.a_bias.max():+.2f}] SD; ensemble [{df.e_bias.min():+.2f}, {df.e_bias.max():+.2f}] SD')
    print(f'wrote {OUT_PDF}\n      {OUT_PNG}')
if __name__ == '__main__':
    main()
