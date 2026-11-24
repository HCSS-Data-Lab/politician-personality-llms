from __future__ import annotations
import argparse
import glob
import itertools
import json
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
from pinf import config, metrics
SEED = nsm.SEED
N_BOOT = nsm.N_BOOT
CI_LEVEL = nsm.CI_LEVEL
FRAMINGS = nsm.FRAMINGS
OPEN_MODELS = nsm.OPEN_MODELS
ANCHOR = nsm.ANCHOR
POOLED = nsm.POOLED
CONTRAST_MODELS = (*OPEN_MODELS, POOLED, ANCHOR)
STRATA = ('c1s',)
FRAMING_CONTRASTS: tuple[tuple[str, str], ...] = (('K', 'G'), ('O', 'G'), ('K', 'O'))
OUT_DIR = nsm.OUT_DIR
SPEECH_GLOB = str(nsm.SWEEP_REPO / 'outputs' / 'results_*speech*.jsonl')
OUT_COLS = ['stratum', 'model', 'trait', 'contrast', 'n', 'r_a', 'r_b', 'diff', 'ci_low', 'ci_high']
FRAMING_OUT_COLS = [*OUT_COLS, 'p_two_sided', 'n_neg_traits']

def _leader_framing_wide(points: pd.DataFrame, model: str, trait: str) -> pd.DataFrame:
    pe = points[(points['model'] == model) & (points['trait'] == trait)]
    return pe.pivot_table(index='leader_id', columns='framing', values='llm_mean').reset_index()

def framing_contrasts_table(points: pd.DataFrame, crit: pd.DataFrame, stratum: str) -> pd.DataFrame:
    sub_crit = nsm._stratum_leaders(crit, stratum)
    leaders = sorted(sub_crit['leader_id'])
    exp_wide = sub_crit.set_index('leader_id').reindex(leaders)
    Xe_all = np.array([exp_wide[f'expert_{t}_mean'].to_numpy(float) for t in config.TRAITS])
    rows = []
    for model in CONTRAST_MODELS:
        for trait in config.TRAITS:
            wide = _leader_framing_wide(points, model, trait)
            exp = sub_crit[['leader_id', f'expert_{trait}_mean']].rename(columns={f'expert_{trait}_mean': 'expert_mean'})
            m = wide.merge(exp, on='leader_id', how='inner')
            for fa, fb in FRAMING_CONTRASTS:
                cd = metrics.correlation_difference(m['expert_mean'].to_numpy(float), m[fa].to_numpy(float), m[fb].to_numpy(float), n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                rows.append({'stratum': stratum, 'model': model, 'trait': trait, 'contrast': f'{fa}-{fb}', 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high, 'p_two_sided': '', 'n_neg_traits': ''})
        mats = {f: _model_trait_matrix(points, leaders, model, f, config.TRAITS) for f in FRAMINGS}
        for fa, fb in FRAMING_CONTRASTS:
            for gname, gtraits in GROUPS.items():
                gi = [config.TRAITS.index(t) for t in gtraits]
                Xe, Xa, Xb = (Xe_all[gi], mats[fa][gi], mats[fb][gi])
                ones = np.ones(len(gtraits), float)
                ag = _aggregate_paired_models(Xe, Xa, Xb, ones)
                rows.append({'stratum': stratum, 'model': model, 'trait': gname, 'contrast': f'{fa}-{fb}', 'n': ag['n'], 'r_a': ag['r_a'], 'r_b': ag['r_b'], 'diff': ag['diff'], 'ci_low': ag['ci_low'], 'ci_high': ag['ci_high'], 'p_two_sided': ag['p_two_sided'], 'n_neg_traits': ag['n_neg_traits']})
    return pd.DataFrame(rows, columns=FRAMING_OUT_COLS)

def build_framing_contrasts(points: pd.DataFrame, crit: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([framing_contrasts_table(points, crit, s) for s in STRATA], ignore_index=True)

def crosscheck_ra_rb(contrasts: pd.DataFrame, points: pd.DataFrame, crit: pd.DataFrame, n_checks: int=6) -> None:
    per_trait = contrasts[contrasts['trait'].isin(config.TRAITS)]
    sample = per_trait.sample(n=min(n_checks, len(per_trait)), random_state=0)
    for _, r in sample.iterrows():
        sub_crit = nsm._stratum_leaders(crit, r['stratum'])
        fa, fb = r['contrast'].split('-')
        wide = _leader_framing_wide(points, r['model'], r['trait'])
        exp = sub_crit[['leader_id', f"expert_{r['trait']}_mean"]].rename(columns={f"expert_{r['trait']}_mean": 'expert_mean'})
        m = wide.merge(exp, on='leader_id', how='inner')
        for framing, expected in ((fa, r['r_a']), (fb, r['r_b'])):
            pr = metrics.pearson(m['expert_mean'].to_numpy(float), m[framing].to_numpy(float))
            assert np.isclose(pr.statistic, expected, atol=1e-09, equal_nan=True), f"r mismatch {r['stratum']}/{r['model']}/{r['trait']} framing {framing}: contrast={expected} pearson={pr.statistic}"
    print(f"  crosscheck OK: {len(sample)} contrast rows' r_a/r_b match a direct metrics.pearson")

def validate_self_contrast(points: pd.DataFrame, crit: pd.DataFrame) -> None:
    sub_crit = nsm._stratum_leaders(crit, 'c1s')
    wide = _leader_framing_wide(points, POOLED, 'extraversion')
    exp = sub_crit[['leader_id', 'expert_extraversion_mean']].rename(columns={'expert_extraversion_mean': 'expert_mean'})
    m = wide.merge(exp, on='leader_id', how='inner').dropna(subset=['O', 'expert_mean'])
    expert = m['expert_mean'].to_numpy(float)
    series = m['O'].to_numpy(float)
    cd = metrics.correlation_difference(expert, series, series, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
    assert cd.diff == 0.0, f'self-contrast diff != 0: {cd.diff}'
    assert cd.ci_low is not None and cd.ci_high is not None
    assert cd.ci_low <= 0.0 <= cd.ci_high, f'self-contrast CI does not bracket 0: {cd}'
    assert cd.ci_low == 0.0 and cd.ci_high == 0.0, f'self-contrast CI not degenerate at 0: {cd}'
    assert np.isclose(cd.r_a, cd.r_b), 'self-contrast r_a != r_b'
    print(f'  self-contrast [name]-[name] OK: diff={cd.diff:.1f}, CI=[{cd.ci_low:.1f}, {cd.ci_high:.1f}], n={cd.n}')
    shuffled = np.random.default_rng(SEED).permutation(series)
    cds = metrics.correlation_difference(expert, series, shuffled, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
    assert np.isfinite(cds.diff) and np.isfinite(cds.ci_low) and np.isfinite(cds.ci_high), cds
    assert not np.isclose(cds.r_a, cds.r_b), "shuffled series produced identical r's (unexpected)"
    print(f'  shuffled-vs-itself OK: r_a={cds.r_a:.3f} r_b={cds.r_b:.3f} diff={cds.diff:.3f} CI=[{cds.ci_low:.3f}, {cds.ci_high:.3f}]')
MODEL_ORDER: tuple[str, ...] = (ANCHOR, *OPEN_MODELS, POOLED)
MODEL_PAIRS: tuple[tuple[str, str], ...] = tuple(itertools.combinations(MODEL_ORDER, 2))
GROUPS: dict[str, tuple[str, ...]] = {'big_five_mean': config.TRAITS_BIG_FIVE, 'dark_triad_mean': config.TRAITS_DARK_TRIAD, 'overall_mean': config.TRAITS}
MODEL_OUT_COLS = ['stratum', 'framing', 'model_a', 'model_b', 'trait', 'n', 'r_a', 'r_b', 'diff', 'ci_low', 'ci_high', 'frac_a', 'frac_b', 'diff_frac', 'ci_low_frac', 'ci_high_frac', 'p_two_sided', 'n_neg_traits']

def _rowwise_r(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(X) & np.isfinite(Y)
    with np.errstate(invalid='ignore', divide='ignore'):
        cnt = valid.sum(axis=1, keepdims=True)
        Xm = np.where(valid, X, 0.0).sum(axis=1, keepdims=True) / cnt
        Ym = np.where(valid, Y, 0.0).sum(axis=1, keepdims=True) / cnt
        Xc = np.where(valid, X - Xm, 0.0)
        Yc = np.where(valid, Y - Ym, 0.0)
        num = (Xc * Yc).sum(axis=1)
        den = np.sqrt((Xc ** 2).sum(axis=1) * (Yc ** 2).sum(axis=1))
        return np.where(den > 0, num / den, np.nan)

def _model_trait_matrix(points: pd.DataFrame, leaders: list[str], model: str, framing: str, traits: tuple[str, ...]) -> np.ndarray:
    sub = points[(points['model'] == model) & (points['framing'] == framing)]
    wide = sub.pivot_table(index='leader_id', columns='trait', values='llm_mean')
    wide = wide.reindex(index=leaders, columns=list(traits))
    return wide.to_numpy(float).T

def _aggregate_paired_models(Xe: np.ndarray, Xa: np.ndarray, Xb: np.ndarray, cvec: np.ndarray) -> dict:
    n = Xe.shape[1]
    ra_t, rb_t = (_rowwise_r(Xe, Xa), _rowwise_r(Xe, Xb))
    out = {'n': n, 'r_a': float(np.nanmean(ra_t)), 'r_b': float(np.nanmean(rb_t)), 'frac_a': float(np.nanmean(ra_t / cvec)), 'frac_b': float(np.nanmean(rb_t / cvec)), 'n_neg_traits': int(np.sum(ra_t - rb_t < 0))}
    out['diff'] = out['r_a'] - out['r_b']
    out['diff_frac'] = out['frac_a'] - out['frac_b']
    if n < 3:
        out.update(ci_low=float('nan'), ci_high=float('nan'), ci_low_frac=float('nan'), ci_high_frac=float('nan'), p_two_sided=float('nan'))
        return out
    rng = np.random.default_rng(SEED)
    boot_raw, boot_frac = (np.empty(N_BOOT), np.empty(N_BOOT))
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ra, rb = (_rowwise_r(Xe[:, idx], Xa[:, idx]), _rowwise_r(Xe[:, idx], Xb[:, idx]))
        boot_raw[i] = np.nanmean(ra) - np.nanmean(rb)
        boot_frac[i] = np.nanmean(ra / cvec) - np.nanmean(rb / cvec)
    lo_q, hi_q = ((1.0 - CI_LEVEL) / 2.0 * 100.0, (1.0 + CI_LEVEL) / 2.0 * 100.0)
    p = 2.0 * min(float(np.mean(boot_frac <= 0)), float(np.mean(boot_frac >= 0)))
    out.update(ci_low=float(np.nanpercentile(boot_raw, lo_q)), ci_high=float(np.nanpercentile(boot_raw, hi_q)), ci_low_frac=float(np.nanpercentile(boot_frac, lo_q)), ci_high_frac=float(np.nanpercentile(boot_frac, hi_q)), p_two_sided=min(p, 1.0))
    return out

def model_contrasts_table(points: pd.DataFrame, crit: pd.DataFrame, ceilings: dict[str, float], stratum: str) -> pd.DataFrame:
    sub_crit = nsm._stratum_leaders(crit, stratum)
    leaders = sorted(sub_crit['leader_id'])
    exp_wide = sub_crit.set_index('leader_id').reindex(leaders)
    Xe_all = np.array([exp_wide[f'expert_{t}_mean'].to_numpy(float) for t in config.TRAITS])
    rows = []
    for framing in FRAMINGS:
        mats = {m: _model_trait_matrix(points, leaders, m, framing, config.TRAITS) for m in MODEL_ORDER}
        for model_a, model_b in MODEL_PAIRS:
            Xa_all, Xb_all = (mats[model_a], mats[model_b])
            for ti, trait in enumerate(config.TRAITS):
                cd = metrics.correlation_difference(Xe_all[ti], Xa_all[ti], Xb_all[ti], n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                ceil = ceilings[trait]
                rows.append({'stratum': stratum, 'framing': framing, 'model_a': model_a, 'model_b': model_b, 'trait': trait, 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high, 'frac_a': cd.r_a / ceil if cd.r_a is not None else float('nan'), 'frac_b': cd.r_b / ceil if cd.r_b is not None else float('nan'), 'diff_frac': cd.diff / ceil, 'ci_low_frac': cd.ci_low / ceil if cd.ci_low is not None else float('nan'), 'ci_high_frac': cd.ci_high / ceil if cd.ci_high is not None else float('nan'), 'p_two_sided': '', 'n_neg_traits': ''})
            for gname, gtraits in GROUPS.items():
                gi = [config.TRAITS.index(t) for t in gtraits]
                Xe, Xa, Xb = (Xe_all[gi], Xa_all[gi], Xb_all[gi])
                cvec = np.array([ceilings[t] for t in gtraits], float)
                ag = _aggregate_paired_models(Xe, Xa, Xb, cvec)
                rows.append({'stratum': stratum, 'framing': framing, 'model_a': model_a, 'model_b': model_b, 'trait': gname, **ag})
    return pd.DataFrame(rows, columns=MODEL_OUT_COLS)

def build_model_contrasts(points: pd.DataFrame, crit: pd.DataFrame, ceilings: dict[str, float]) -> pd.DataFrame:
    return pd.concat([model_contrasts_table(points, crit, ceilings, s) for s in STRATA], ignore_index=True)

def validate_model_contrasts(mc: pd.DataFrame, conv_csv: Path) -> None:
    agg = mc[mc['trait'] == 'overall_mean']
    row = agg.iloc[0]
    Xe = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    self_ag = _aggregate_paired_models(Xe, Xe * 0.5 + 1, Xe * 0.5 + 1, np.array([0.85]))
    assert self_ag['diff'] == 0.0 and self_ag['ci_low'] == 0.0 and (self_ag['ci_high'] == 0.0), self_ag
    print(f"  self-contrast (model vs itself) OK: diff={self_ag['diff']:.1f}, CI=[{self_ag['ci_low']:.1f}, {self_ag['ci_high']:.1f}], p={self_ag['p_two_sided']:.2f}")
    d = (agg['frac_a'] - agg['frac_b'] - agg['diff_frac']).abs().max()
    assert d < 1e-09, f'diff_frac != frac_a - frac_b (max delta {d})'
    print(f'  aggregate internal consistency OK: max|diff_frac-(frac_a-frac_b)| = {d:.2e}')
    conv = pd.read_csv(conv_csv)
    pt = mc[mc['trait'].isin(config.TRAITS)]
    j = pt.merge(conv[['stratum', 'model', 'framing', 'trait', 'pearson_r', 'r_frac_ceiling']], left_on=['stratum', 'model_a', 'framing', 'trait'], right_on=['stratum', 'model', 'framing', 'trait'], how='inner')
    dr = (j['r_a'] - j['pearson_r']).abs().max()
    print(f'  per-trait r_a vs convergence.csv: {len(j)} rows matched, max|delta| = {dr:.2e}')
    assert dr < 1e-09, f'per-trait r_a disagrees with convergence.csv (max {dr})'
    gaps = []
    for (stratum, framing, model), g in conv.groupby(['stratum', 'framing', 'model']):
        a = agg[(agg['stratum'] == stratum) & (agg['framing'] == framing) & (agg['model_a'] == model)]
        if len(a):
            gaps.append(abs(a.iloc[0]['frac_a'] - g['r_frac_ceiling'].mean()))
    print(f'  complete-case aggregate vs pairwise-deleted mean-of-ceiling: max gap = {max(gaps):.4f} (expected: tiny, machiavellianism drops 1 leader)')

def summarize_model_contrasts(mc: pd.DataFrame) -> None:
    a = mc[(mc['stratum'] == 'c1s') & (mc['framing'] == 'O') & (mc['trait'] == 'overall_mean')]
    print('\n=== model contrasts, c1s / Ø, mean fraction-of-ceiling over 8 traits ===')
    for _, r in a.sort_values('diff_frac', ascending=False).iterrows():
        sig = 'SIG' if r['ci_low_frac'] > 0 or r['ci_high_frac'] < 0 else 'n.s.'
        print(f'  {r.model_a:10s} - {r.model_b:10s}: {r.frac_a:.3f} vs {r.frac_b:.3f}  diff={r.diff_frac:+.4f}  CI=[{r.ci_low_frac:+.4f}, {r.ci_high_frac:+.4f}]  P={r.p_two_sided:.3f}  {sig}')
LEVEL_OUT_COLS = ['stratum', 'model', 'framing', 'trait_group', 'n', 'mean_r', 'ci_low', 'ci_high', 'mean_frac_ceiling', 'ci_low_frac', 'ci_high_frac']

def _aggregate_level(Xe: np.ndarray, Xa: np.ndarray, cvec: np.ndarray) -> dict:
    n = Xe.shape[1]
    ra_t = _rowwise_r(Xe, Xa)
    out = {'n': n, 'mean_r': float(np.nanmean(ra_t)), 'mean_frac_ceiling': float(np.nanmean(ra_t / cvec))}
    if n < 3:
        out.update(ci_low=float('nan'), ci_high=float('nan'), ci_low_frac=float('nan'), ci_high_frac=float('nan'))
        return out
    rng = np.random.default_rng(SEED)
    boot_raw, boot_frac = (np.empty(N_BOOT), np.empty(N_BOOT))
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        ra = _rowwise_r(Xe[:, idx], Xa[:, idx])
        boot_raw[i] = np.nanmean(ra)
        boot_frac[i] = np.nanmean(ra / cvec)
    lo_q, hi_q = ((1.0 - CI_LEVEL) / 2.0 * 100.0, (1.0 + CI_LEVEL) / 2.0 * 100.0)
    out.update(ci_low=float(np.nanpercentile(boot_raw, lo_q)), ci_high=float(np.nanpercentile(boot_raw, hi_q)), ci_low_frac=float(np.nanpercentile(boot_frac, lo_q)), ci_high_frac=float(np.nanpercentile(boot_frac, hi_q)))
    return out

def convergence_aggregates_table(points: pd.DataFrame, crit: pd.DataFrame, ceilings: dict[str, float], stratum: str) -> pd.DataFrame:
    sub_crit = nsm._stratum_leaders(crit, stratum)
    leaders = sorted(sub_crit['leader_id'])
    exp_wide = sub_crit.set_index('leader_id').reindex(leaders)
    Xe_all = np.array([exp_wide[f'expert_{t}_mean'].to_numpy(float) for t in config.TRAITS])
    rows = []
    for framing in FRAMINGS:
        for model in MODEL_ORDER:
            Xa_all = _model_trait_matrix(points, leaders, model, framing, config.TRAITS)
            for gname, gtraits in GROUPS.items():
                gi = [config.TRAITS.index(t) for t in gtraits]
                Xe, Xa = (Xe_all[gi], Xa_all[gi])
                cvec = np.array([ceilings[t] for t in gtraits], float)
                ag = _aggregate_level(Xe, Xa, cvec)
                rows.append({'stratum': stratum, 'model': model, 'framing': framing, 'trait_group': gname, **ag})
    return pd.DataFrame(rows, columns=LEVEL_OUT_COLS)

def build_convergence_aggregates(points: pd.DataFrame, crit: pd.DataFrame, ceilings: dict[str, float]) -> pd.DataFrame:
    return pd.concat([convergence_aggregates_table(points, crit, ceilings, s) for s in STRATA], ignore_index=True)

def validate_convergence_aggregates(agg: pd.DataFrame, mc: pd.DataFrame, framing: pd.DataFrame) -> None:
    lv = agg.set_index(['stratum', 'framing', 'model', 'trait_group'])
    deltas = []
    for _, r in mc[mc['trait'].isin(GROUPS)].iterrows():
        la = lv.loc[r['stratum'], r['framing'], r['model_a'], r['trait']]
        lb = lv.loc[r['stratum'], r['framing'], r['model_b'], r['trait']]
        deltas.append(max(abs(la['mean_r'] - r['r_a']), abs(lb['mean_r'] - r['r_b']), abs(la['mean_frac_ceiling'] - r['frac_a']), abs(lb['mean_frac_ceiling'] - r['frac_b'])))
    print(f'  levels vs model_contrasts aggregates: {len(deltas)} rows, max|delta| = {max(deltas):.2e}')
    assert max(deltas) < 1e-12, 'levels disagree with model_contrasts aggregate r_a/r_b'
    deltas = []
    for _, r in framing[framing['trait'].isin(GROUPS)].iterrows():
        fa, fb = r['contrast'].split('-')
        la = lv.loc[r['stratum'], fa, r['model'], r['trait']]
        lb = lv.loc[r['stratum'], fb, r['model'], r['trait']]
        deltas.append(max(abs(la['mean_r'] - r['r_a']), abs(lb['mean_r'] - r['r_b'])))
    print(f'  levels vs framing aggregates: {len(deltas)} rows, max|delta| = {max(deltas):.2e}')
    assert max(deltas) < 1e-12, 'levels disagree with framing aggregate r_a/r_b'
    ok = (agg['ci_low'] <= agg['mean_r']) & (agg['mean_r'] <= agg['ci_high']) & (agg['ci_low_frac'] <= agg['mean_frac_ceiling']) & (agg['mean_frac_ceiling'] <= agg['ci_high_frac'])
    assert ok.all(), f'{int((~ok).sum())} aggregate rows have a CI not bracketing the estimate'
    print(f'  CI brackets point estimate on both scales: {int(ok.sum())}/{len(agg)} rows')

def summarize_convergence_aggregates(agg: pd.DataFrame) -> None:
    a = agg[(agg['stratum'] == 'c1s') & (agg['framing'] == 'O') & (agg['trait_group'] == 'overall_mean')]
    print('\n=== convergence aggregate levels, c1s / Ø, mean over 8 traits ===')
    for _, r in a.sort_values('mean_frac_ceiling', ascending=False).iterrows():
        print(f'  {r.model:10s} r = {r.mean_r:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]   /ceiling = {r.mean_frac_ceiling:.3f} [{r.ci_low_frac:.3f}, {r.ci_high_frac:.3f}]  (n={int(r.n)})')

def resolve_speech_path(arg: str | None) -> Path | None:
    if arg is None:
        return None
    pattern = SPEECH_GLOB if arg == 'auto' else str(Path(arg).expanduser())
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])

def load_speech_scored(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get('parse_ok'):
                continue
            na_id = r.get('name_arm_id')
            if not na_id:
                continue
            for trait, score in nsm.score_ratings(r['ratings']).items():
                rows.append((na_id, r['condition'], r['model'], r['framing'], trait, score))
    df = pd.DataFrame(rows, columns=['leader_id', 'condition', 'model', 'framing', 'trait', 'score'])
    if df.empty:
        return df
    est = df.groupby(['leader_id', 'condition', 'model', 'framing', 'trait'], as_index=False)['score'].mean().rename(columns={'score': 'value'})
    pooled = est[est['model'].isin(OPEN_MODELS)].groupby(['leader_id', 'condition', 'framing', 'trait'], as_index=False)['value'].mean()
    pooled['model'] = POOLED
    return pd.concat([est, pooled], ignore_index=True)

def _condition_wide(speech_est: pd.DataFrame, name_points: pd.DataFrame, model: str, framing: str, trait: str) -> pd.DataFrame:

    def cond(c):
        s = speech_est[(speech_est['condition'] == c) & (speech_est['model'] == model) & (speech_est['framing'] == framing) & (speech_est['trait'] == trait)]
        return s[['leader_id', 'value']].rename(columns={'value': c})
    named = cond('named-speech')
    anon = cond('anon-speech')
    name = name_points[(name_points['model'] == model) & (name_points['framing'] == framing) & (name_points['trait'] == trait)][['leader_id', 'llm_mean']].rename(columns={'llm_mean': 'name'})
    out = named.merge(anon, on='leader_id', how='outer').merge(name, on='leader_id', how='left')
    return out

def evidence_contrasts_table(speech_est: pd.DataFrame, name_points: pd.DataFrame, crit: pd.DataFrame) -> pd.DataFrame:
    models = [m for m in (*OPEN_MODELS, POOLED) if m in set(speech_est['model'])]
    rows = []
    for framing in FRAMINGS:
        for model in models:
            for trait in config.TRAITS:
                wide = _condition_wide(speech_est, name_points, model, framing, trait)
                exp = crit[['leader_id', f'expert_{trait}_mean']].rename(columns={f'expert_{trait}_mean': 'expert_mean'})
                m = wide.merge(exp, on='leader_id', how='inner')
                for label, a, b in (('named-speech-name', 'named-speech', 'name'), ('named-speech-anon-speech', 'named-speech', 'anon-speech')):
                    if a not in m or b not in m:
                        continue
                    cd = metrics.correlation_difference(m['expert_mean'].to_numpy(float), m[a].to_numpy(float), m[b].to_numpy(float), n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                    rows.append({'stratum': f'bridge:{framing}', 'model': model, 'trait': trait, 'contrast': label, 'n': cd.n, 'r_a': cd.r_a, 'r_b': cd.r_b, 'diff': cd.diff, 'ci_low': cd.ci_low, 'ci_high': cd.ci_high})
    return pd.DataFrame(rows, columns=OUT_COLS)

def _excludes_zero(row) -> bool:
    lo, hi = (row['ci_low'], row['ci_high'])
    return pd.notna(lo) and pd.notna(hi) and (lo > 0 or hi < 0)

def summarize_framing(contrasts: pd.DataFrame) -> None:
    print('\n=== FRAMING CONTRASTS: mean diff over the 8 traits (K,Ø=O,G) ===')
    print("    label 'X-Y' = r(expert,X) - r(expert,Y); +ve K-G => knowledge-licensing improves convergence")
    for stratum in STRATA:
        n = int(contrasts[contrasts['stratum'] == stratum]['n'].iloc[0])
        print(f'\n  stratum {stratum} (n={n} leaders):')
        for model in (POOLED, ANCHOR):
            print(f'    {model}:')
            for fa, fb in FRAMING_CONTRASTS:
                sub = contrasts[(contrasts['stratum'] == stratum) & (contrasts['model'] == model) & (contrasts['contrast'] == f'{fa}-{fb}')]
                pt = sub[sub['trait'].isin(config.TRAITS)]
                ag = sub[sub['trait'] == 'overall_mean'].iloc[0]
                n_excl = int(pt.apply(_excludes_zero, axis=1).sum())
                n_pos_excl = int(pt[pt['diff'] > 0].apply(_excludes_zero, axis=1).sum())
                print(f"      {fa}-{fb}:  aggregate diff = {ag['diff']:+.3f} CI=[{ag['ci_low']:+.3f}, {ag['ci_high']:+.3f}] P={ag['p_two_sided']:.3f}  (per-trait mean {pt['diff'].mean():+.3f}; CI excl 0 in {n_excl}/8, {n_pos_excl} +ve)")

def summarize_evidence(ev: pd.DataFrame) -> None:
    print('\n=== EVIDENCE-AXIS CONTRASTS: mean diff over the 8 traits, per framing ===')
    for stratum in sorted(ev['stratum'].unique()):
        print(f'\n  {stratum}:')
        for model in [POOLED] + [m for m in OPEN_MODELS if m in set(ev['model'])]:
            for label in ('named-speech-name', 'named-speech-anon-speech'):
                sub = ev[(ev['stratum'] == stratum) & (ev['model'] == model) & (ev['contrast'] == label)]
                if sub.empty:
                    continue
                n_excl = int(sub.apply(_excludes_zero, axis=1).sum())
                nn = int(sub['n'].iloc[0])
                print(f"    {model:8s} {label:26s} mean diff = {sub['diff'].mean():+.3f}  (n={nn}; CI excl 0 in {n_excl}/{len(sub)})")
