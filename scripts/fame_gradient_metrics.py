from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pinf import config
from scripts.name_sweep_metrics import score_ratings
SWEEP_REPO = REPO
RESULTS_FILES = (SWEEP_REPO / 'outputs' / 'results_20260705T223912.jsonl', SWEEP_REPO / 'outputs' / 'results_openrouter_gpt-5.5_20260706.jsonl')
CRITERION = SWEEP_REPO / 'data' / 'criterion.csv'
OUT_DIR = REPO / 'outputs' / 'fame_gradient'
PROMINENCE = REPO / 'measurements' / 'prominence_2026-07-06.csv'
SEED, N_BOOT, CI_LEVEL = (20260706, 2000, 0.95)
FRAMINGS = ('G', 'O', 'K')
OPEN_MODELS = ('qwen-8b', 'qwen-32b', 'gemma-12b', 'gemma-27b')
ANCHOR = 'gpt-5.5'
MODEL_GROUPS = ('gpt-5.5', 'open_ensemble')
PROXIES = {'pageviews': 'pageviews_election_year', 'sitelinks': 'sitelinks'}
STRATA = ('c1s',)
BIG_FIVE = list(config.TRAITS_BIG_FIVE)
DARK_TRIAD = list(config.TRAITS_DARK_TRIAD)
TRAITS = list(config.TRAITS)
MID = (config.SCALE_MIN + config.SCALE_MAX) / 2

def load_responses() -> pd.DataFrame:
    rows = []
    for path in RESULTS_FILES:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get('parse_ok'):
                    raise ValueError(f"parse_ok=False: {r.get('key')}")
                ratings = r['ratings']
                vals = [ratings[f's{i}'] for i in range(1, 17)]
                sc = score_ratings(ratings)
                rows.append({'leader_id': r['leader_id'], 'model': r['model'], 'framing': r['framing'], 'repeat': r['repeat'], 'all2s': int(all((v == MID for v in vals))), 'near_parked': int(sum((v == MID for v in vals)) >= 14), **sc})
    return pd.DataFrame(rows)

def point_estimates(resp: pd.DataFrame) -> pd.DataFrame:
    long = resp.melt(id_vars=['leader_id', 'model', 'framing', 'repeat'], value_vars=TRAITS, var_name='trait', value_name='score')
    per_model = long.groupby(['leader_id', 'model', 'framing', 'trait'], as_index=False)['score'].mean().rename(columns={'score': 'est'})
    anchor = per_model[per_model['model'] == ANCHOR].copy()
    anchor['model_group'] = ANCHOR
    ens = per_model[per_model['model'].isin(OPEN_MODELS)].groupby(['leader_id', 'framing', 'trait'], as_index=False)['est'].mean()
    ens['model_group'] = 'open_ensemble'
    return pd.concat([anchor[['leader_id', 'framing', 'trait', 'est', 'model_group']], ens], ignore_index=True)

def abstention_rates(resp: pd.DataFrame) -> pd.DataFrame:
    out = []
    for grp, models in (('gpt-5.5', [ANCHOR]), ('open_ensemble', list(OPEN_MODELS))):
        sub = resp[resp['model'].isin(models)]
        g = sub.groupby(['leader_id', 'framing'], as_index=False).agg(all2s_rate=('all2s', 'mean'), near_parked_rate=('near_parked', 'mean'), n_resp=('all2s', 'size'))
        g['model_group'] = grp
        out.append(g)
    return pd.concat(out, ignore_index=True)

def _per_trait_r(llm: np.ndarray, exp: np.ndarray) -> np.ndarray:
    out = np.full(llm.shape[1], np.nan)
    for j in range(llm.shape[1]):
        m = ~(np.isnan(llm[:, j]) | np.isnan(exp[:, j]))
        if m.sum() >= 3 and np.std(llm[m, j]) > 0 and (np.std(exp[m, j]) > 0):
            out[j] = np.corrcoef(llm[m, j], exp[m, j])[0, 1]
    return out

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return np.nan
    return float(stats.spearmanr(x[m], y[m]).statistic)

def _boot_ci(fn, n: int, seed: int=SEED) -> tuple:
    if n < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([fn(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    a = (1 - CI_LEVEL) / 2
    return (float(np.nanquantile(boots, a)), float(np.nanquantile(boots, 1 - a)))

def _mean_r(ids, est_wide, exp_wide, cols):
    ids = sorted(ids)
    llm = est_wide.reindex(ids)[cols].to_numpy(float)
    exp = exp_wide.reindex(ids)[cols].to_numpy(float)
    return (llm, exp)

def assign_tertiles(prom: pd.Series) -> tuple[pd.Series, float, float]:
    q1, q2 = prom.quantile([1 / 3, 2 / 3]).tolist()
    t = pd.Series(np.where(prom <= q1, 'low', np.where(prom <= q2, 'mid', 'high')), index=prom.index)
    return (t, q1, q2)

def _overall_r(ids, ew, exp_wide) -> float:
    return float(np.nanmean(_per_trait_r(*_mean_r(ids, ew, exp_wide, TRAITS))))

def tertile_gap_boot(low_ids, high_ids, ew, exp_wide) -> dict:
    llm_lo, exp_lo = _mean_r(low_ids, ew, exp_wide, TRAITS)
    llm_hi, exp_hi = _mean_r(high_ids, ew, exp_wide, TRAITS)
    r_lo = float(np.nanmean(_per_trait_r(llm_lo, exp_lo)))
    r_hi = float(np.nanmean(_per_trait_r(llm_hi, exp_hi)))
    rng = np.random.default_rng(SEED)
    nlo, nhi = (len(low_ids), len(high_ids))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ilo, ihi = (rng.integers(0, nlo, nlo), rng.integers(0, nhi, nhi))
        boots[b] = np.nanmean(_per_trait_r(llm_hi[ihi], exp_hi[ihi])) - np.nanmean(_per_trait_r(llm_lo[ilo], exp_lo[ilo]))
    a = (1 - CI_LEVEL) / 2
    return {'r_low': r_lo, 'r_high': r_hi, 'gap': r_hi - r_lo, 'gap_ci_low': float(np.nanquantile(boots, a)), 'gap_ci_high': float(np.nanquantile(boots, 1 - a)), 'p_gap_le_0': float(np.mean(boots <= 0)), 'n_low': nlo, 'n_high': nhi}
