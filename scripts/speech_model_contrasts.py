from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO = SCRIPTS_DIR.parent
for p in (str(REPO), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import name_sweep_metrics as nsm
import speech_metrics as sm
from pinf import config, metrics
SEED = nsm.SEED
N_BOOT = nsm.N_BOOT
CI_LEVEL = nsm.CI_LEVEL
TRAITS = config.TRAITS
GROUPS = sm.GROUPS
LO, HI = (sm.LO, sm.HI)
OUT_DIR = sm.OUT_DIR
MODEL_ORDER: tuple[str, ...] = (*sm.ANCHORS, *nsm.OPEN_MODELS, nsm.POOLED)
MODEL_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.combinations(MODEL_ORDER, 2))
CONDITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (('name', ('G', 'O', 'K')), ('named-speech', ('G', 'O', 'K')), ('anon-speech', ('G', 'O')))
OUT_COLS = ['condition', 'framing', 'model_a', 'model_b', 'trait', 'n', 'r_a', 'r_b', 'diff', 'ci_low', 'ci_high', 'frac_a', 'frac_b', 'diff_frac', 'ci_low_frac', 'ci_high_frac', 'p_two_sided', 'n_neg_traits']

def _aggregate_paired_models(Xe: np.ndarray, Xa: np.ndarray, Xb: np.ndarray, cvec: np.ndarray) -> dict:
    n = Xe.shape[1]
    ra_t, rb_t = (sm._rowwise_r(Xe, Xa), sm._rowwise_r(Xe, Xb))
    out = {'n': n, 'r_a': float(np.nanmean(ra_t)), 'r_b': float(np.nanmean(rb_t)), 'frac_a': float(np.nanmean(ra_t / cvec)), 'frac_b': float(np.nanmean(rb_t / cvec)), 'n_neg_traits': int(np.sum(ra_t - rb_t < 0))}
    out['diff'] = out['r_a'] - out['r_b']
    out['diff_frac'] = out['frac_a'] - out['frac_b']
    if n < 3:
        out.update(ci_low=float('nan'), ci_high=float('nan'), ci_low_frac=float('nan'), ci_high_frac=float('nan'), p_two_sided=float('nan'))
        return out
    rng = np.random.default_rng(SEED)
    # Keep the original per-world RNG calls, then batch only the arithmetic.
    indices = np.stack([rng.integers(0, n, n) for _ in range(N_BOOT)])
    def boot_r(X, Y):
        shape = (N_BOOT * X.shape[0], n)
        xs = X[:, indices].transpose(1, 0, 2).reshape(shape)
        ys = Y[:, indices].transpose(1, 0, 2).reshape(shape)
        return sm._rowwise_r(xs, ys).reshape(N_BOOT, X.shape[0])
    ra, rb = boot_r(Xe, Xa), boot_r(Xe, Xb)
    boot_raw = np.nanmean(ra, axis=1) - np.nanmean(rb, axis=1)
    boot_frac = np.nanmean(ra / cvec, axis=1) - np.nanmean(rb / cvec, axis=1)
    p = 2.0 * min(float(np.mean(boot_frac <= 0)), float(np.mean(boot_frac >= 0)))
    out.update(ci_low=float(np.nanpercentile(boot_raw, LO)), ci_high=float(np.nanpercentile(boot_raw, HI)), ci_low_frac=float(np.nanpercentile(boot_frac, LO)), ci_high_frac=float(np.nanpercentile(boot_frac, HI)), p_two_sided=min(p, 1.0))
    return out

def _paired_trait(criterion, a, b, *, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED):
    """The executed paired bootstrap with vectorized arithmetic, unchanged RNG draws."""
    c,a,b=(np.asarray(x,float) for x in (criterion,a,b))
    mask=np.isfinite(c)&np.isfinite(a)&np.isfinite(b)
    c,a,b=c[mask],a[mask],b[mask]
    n=len(c)
    if n<3 or np.ptp(c)==0:
        return metrics.CorrelationDiff(float('nan'),None,None,float('nan'),float('nan'),n)
    ra=metrics._safe_stat(metrics.stats.pearsonr,c,a)
    rb=metrics._safe_stat(metrics.stats.pearsonr,c,b)
    rng=np.random.default_rng(seed)
    ix=np.stack([rng.integers(0,n,n) for _ in range(n_boot)])
    boot=sm._rowwise_r(c[ix],a[ix])-sm._rowwise_r(c[ix],b[ix])
    lo,hi=np.nanpercentile(boot,[(1-ci_level)/2*100,(1+ci_level)/2*100])
    return metrics.CorrelationDiff(float(ra-rb),float(lo),float(hi),float(ra),float(rb),n)


def build_speech_model_contrasts(namep, speechp, crit, ceilings) -> pd.DataFrame:
    rows = []
    for condition, framings in CONDITIONS:
        for framing in framings:
            ebl = {}
            for model in MODEL_ORDER:
                if condition == 'name':
                    ebl[model] = sm.est_dict(namep, model, framing)
                else:
                    ebl[model] = sm.est_dict(speechp, model, framing, condition)
            for model_a, model_b in MODEL_PAIRS:
                a_ebl, b_ebl = (ebl[model_a], ebl[model_b])
                if not a_ebl or not b_ebl:
                    continue
                base = {'condition': condition, 'framing': framing, 'model_a': model_a, 'model_b': model_b}
                leaders = sorted(set(a_ebl) & set(b_ebl))
                for t in TRAITS:
                    exp = np.array([crit.get(l, {}).get(f'expert_{t}_mean', np.nan) for l in leaders], float)
                    av = np.array([a_ebl[l].get(t, np.nan) for l in leaders], float)
                    bv = np.array([b_ebl[l].get(t, np.nan) for l in leaders], float)
                    cd = _paired_trait(exp, av, bv, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                    ceil = ceilings[t]
                    rows.append({**base, 'trait': t, 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high, 'frac_a': cd.r_a / ceil, 'frac_b': cd.r_b / ceil, 'diff_frac': cd.diff / ceil, 'ci_low_frac': cd.ci_low / ceil, 'ci_high_frac': cd.ci_high / ceil, 'p_two_sided': '', 'n_neg_traits': ''})
                for gname, gtraits in GROUPS.items():
                    E, A, B, _ = sm._triplet(a_ebl, b_ebl, crit, gtraits)
                    Xe = np.array([E[t] for t in gtraits], float)
                    Xa = np.array([A[t] for t in gtraits], float)
                    Xb = np.array([B[t] for t in gtraits], float)
                    cvec = np.array([ceilings[t] for t in gtraits], float)
                    rows.append({**base, 'trait': gname, **_aggregate_paired_models(Xe, Xa, Xb, cvec)})
    return pd.DataFrame(rows, columns=OUT_COLS)

def crosscheck(mc: pd.DataFrame) -> None:
    conv = pd.read_csv(OUT_DIR / 'speech_convergence.csv')
    agg = mc[mc['trait'].isin(GROUPS)]
    worst = 0.0
    for _, row in agg.iterrows():
        for side in ('a', 'b'):
            ref = conv[(conv['condition'] == row['condition']) & (conv['framing'] == row['framing']) & (conv['model'] == row[f'model_{side}']) & (conv['trait'] == row['trait'])]
            if len(ref) == 1:
                worst = max(worst, abs(float(ref.iloc[0]['pearson_r']) - row[f'r_{side}']))
    print(f"crosscheck vs speech_convergence.csv: max |r| gap = {worst:.4f} ({('OK' if worst < 0.005 else 'MISMATCH')})")

def summarize(mc: pd.DataFrame) -> None:
    sel = mc[(mc['trait'] == 'overall_mean') & (mc['framing'] == 'O') & (mc['model_a'] == sm.ANCHOR)]
    for condition in ('name', 'named-speech', 'anon-speech'):
        sub = sel[sel['condition'] == condition]
        if sub.empty:
            continue
        print(f'\n--- {condition} (framing O, overall_mean): GPT-5.5 minus each model ---')
        for _, r in sub.iterrows():
            star = '*' if r['ci_low'] > 0 or r['ci_high'] < 0 else ' '
            print(f" {star} vs {r['model_b']:<11} diff={r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  diff_frac={100 * r['diff_frac']:+5.1f} pp [{100 * r['ci_low_frac']:+.1f}, {100 * r['ci_high_frac']:+.1f}]  p={r['p_two_sided']:.3f}  n={r['n']}")
