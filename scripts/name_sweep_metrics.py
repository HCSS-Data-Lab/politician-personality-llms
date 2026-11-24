from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pinf import config, metrics
SWEEP_REPO = REPO
RESULTS_FILES = (SWEEP_REPO / 'outputs' / 'results_20260705T223912.jsonl', SWEEP_REPO / 'outputs' / 'results_openrouter_gpt-5.5_20260706.jsonl')
CRITERION = SWEEP_REPO / 'data' / 'criterion.csv'
CEILING_JSON = REPO / 'results' / 'expert_ceiling.json'
OUT_DIR = SWEEP_REPO / 'outputs' / 'metrics'
SEED = 20260706
N_BOOT = 2000
CI_LEVEL = 0.95
CEILING_RHO = '0.3'
FRAMINGS = ('G', 'O', 'K')
OPEN_MODELS = ('qwen-8b', 'qwen-32b', 'gemma-12b', 'gemma-27b')
ANCHOR = 'gpt-5.5'
MODELS = (*OPEN_MODELS, ANCHOR)
POOLED = 'pooled'
ITEM_TO_S = {item.id: f's{i + 1}' for i, item in enumerate(config.ALL_ITEMS)}
S_TO_ITEM = {v: k for k, v in ITEM_TO_S.items()}

def score_ratings(ratings: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for trait, (id_a, id_b) in config.TRAIT_AGG.items():
        vals = []
        for item_id in (id_a, id_b):
            item = config.ITEMS_BY_ID[item_id]
            raw = ratings[ITEM_TO_S[item_id]]
            vals.append(raw if item.keyed == +1 else config.SCALE_MAX - raw)
        out[trait] = float(np.mean(vals))
    return out

def icc_1_1(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2 or (not np.isfinite(m).all()):
        return float('nan')
    n, k = m.shape
    grand = m.mean()
    row_means = m.mean(axis=1)
    ss_between = k * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((m - row_means[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom != 0 else float('nan')

def load_scored() -> pd.DataFrame:
    rows = []
    for path in RESULTS_FILES:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get('parse_ok'):
                    raise ValueError(f"unexpected parse_ok=False row: {r.get('key')}")
                scores = score_ratings(r['ratings'])
                for trait, score in scores.items():
                    rows.append((r['leader_id'], r['model'], r['framing'], r['repeat'], trait, score))
    df = pd.DataFrame(rows, columns=['leader_id', 'model', 'framing', 'repeat', 'trait', 'score'])
    assert df['score'].between(config.SCALE_MIN, config.SCALE_MAX).all(), 'trait score out of [0,4]'
    return df

def load_criterion() -> pd.DataFrame:
    df = pd.read_csv(CRITERION)
    df['stratum'] = df['leader_id'].str.split('__').str[0]
    return df

def load_ceilings() -> dict[str, float]:
    j = json.load(open(CEILING_JSON))
    return {t: j['traits'][t]['by_rho'][CEILING_RHO]['ceiling'] for t in config.TRAITS}

def point_estimates(scored: pd.DataFrame) -> pd.DataFrame:
    per_model = scored.groupby(['leader_id', 'model', 'framing', 'trait'], as_index=False)['score'].mean().rename(columns={'score': 'llm_mean'})
    pooled = per_model[per_model['model'].isin(OPEN_MODELS)].groupby(['leader_id', 'framing', 'trait'], as_index=False)['llm_mean'].mean()
    pooled['model'] = POOLED
    return pd.concat([per_model, pooled], ignore_index=True)

def _stratum_leaders(crit: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum == 'all':
        return crit
    return crit[crit['stratum'] == stratum]

def convergence_table(points: pd.DataFrame, crit: pd.DataFrame, ceilings: dict[str, float], stratum: str) -> pd.DataFrame:
    sub_crit = _stratum_leaders(crit, stratum)
    rows = []
    for model in (*MODELS, POOLED):
        for framing in FRAMINGS:
            for trait in config.TRAITS:
                pe = points[(points['model'] == model) & (points['framing'] == framing) & (points['trait'] == trait)][['leader_id', 'llm_mean']]
                exp = sub_crit[['leader_id', f'expert_{trait}_mean']].rename(columns={f'expert_{trait}_mean': 'expert_mean'})
                m = pe.merge(exp, on='leader_id', how='inner').dropna(subset=['llm_mean', 'expert_mean'])
                x = m['expert_mean'].to_numpy(float)
                y = m['llm_mean'].to_numpy(float)
                pr = metrics.pearson(x, y, ci=True, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=SEED)
                sr = metrics.spearman(x, y)
                ceil = ceilings[trait]
                rows.append({'stratum': stratum, 'model': model, 'framing': framing, 'trait': trait, 'n': pr.n, 'pearson_r': pr.statistic, 'pearson_p': pr.pvalue, 'pearson_ci_low': pr.ci_low, 'pearson_ci_high': pr.ci_high, 'spearman_rho': sr.statistic, 'spearman_p': sr.pvalue, 'ceiling': ceil, 'r_frac_ceiling': pr.statistic / ceil if ceil else float('nan'), 'ci_low_frac_ceiling': pr.ci_low / ceil if ceil and pr.ci_low is not None else float('nan'), 'ci_high_frac_ceiling': pr.ci_high / ceil if ceil and pr.ci_high is not None else float('nan')})
    return pd.DataFrame(rows)

def bias_table(points: pd.DataFrame, crit: pd.DataFrame, stratum: str) -> pd.DataFrame:
    sub_crit = _stratum_leaders(crit, stratum)
    rows = []
    for model in (*MODELS, POOLED):
        for framing in FRAMINGS:
            for trait in config.TRAITS:
                pe = points[(points['model'] == model) & (points['framing'] == framing) & (points['trait'] == trait)][['leader_id', 'llm_mean']]
                cols = {f'expert_{trait}_mean': 'expert_mean', f'expert_{trait}_sd': 'expert_sd'}
                exp = sub_crit[['leader_id', *cols]].rename(columns=cols)
                m = pe.merge(exp, on='leader_id', how='inner').dropna(subset=['llm_mean', 'expert_mean'])
                raw_gaps, sd_gaps = ([], [])
                for _, row in m.iterrows():
                    b = metrics.directional_bias(row['llm_mean'], row['expert_mean'], None if pd.isna(row['expert_sd']) else row['expert_sd'])
                    raw_gaps.append(b.raw_gap)
                    if b.sd_units is not None:
                        sd_gaps.append(b.sd_units)
                rows.append({'stratum': stratum, 'model': model, 'framing': framing, 'trait': trait, 'n_raw': len(raw_gaps), 'n_sd': len(sd_gaps), 'bias_raw_mean': float(np.mean(raw_gaps)) if raw_gaps else float('nan'), 'bias_sd_units_mean': float(np.mean(sd_gaps)) if sd_gaps else float('nan')})
    return pd.DataFrame(rows)

def reliability_table(scored: pd.DataFrame, crit: pd.DataFrame, stratum: str) -> pd.DataFrame:
    keep = set(_stratum_leaders(crit, stratum)['leader_id'])
    sub = scored[scored['leader_id'].isin(keep)]
    rows = []
    for model in MODELS:
        for framing in FRAMINGS:
            for trait in config.TRAITS:
                cell = sub[(sub['model'] == model) & (sub['framing'] == framing) & (sub['trait'] == trait)]
                mat = cell.pivot(index='leader_id', columns='repeat', values='score')
                mat = mat.dropna(axis=0)
                arr = mat.to_numpy(float)
                icc11 = icc_1_1(arr)
                icc21 = metrics.icc(arr).icc1 if arr.shape[0] >= 2 and arr.shape[1] >= 2 else float('nan')
                rows.append({'stratum': stratum, 'model': model, 'framing': framing, 'trait': trait, 'n_subjects': arr.shape[0], 'n_repeats': arr.shape[1], 'icc_1_1': icc11, 'icc_2_1_crosscheck': icc21})
    return pd.DataFrame(rows)

def reliability_rollup(rel: pd.DataFrame) -> pd.DataFrame:
    return rel.groupby(['stratum', 'model', 'trait'], as_index=False).agg(icc_1_1_median=('icc_1_1', 'median'), icc_1_1_min=('icc_1_1', 'min'), icc_1_1_max=('icc_1_1', 'max'))
