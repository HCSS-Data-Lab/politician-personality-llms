"""Ideology robustness analyses using locally obtained NEGex 2.0 aggregates.

Statistical functions preserve the September analysis, including shared bootstrap
resamples and sequential RNG consumption across cells. Only aggregate outputs
are written. The original NEGex-supplied adjusted scores are read from the source
release; no attempt is made to reconstruct its unpublished residualization step.
"""
from __future__ import annotations
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]


def load_points(path: Path, criterion_ids: set[str]):
    """Read public, associated item responses and return the September point schema.

    The export retains one row per politician even where inference requests were
    shared. Averaging these associated records preserves the original weighting.
    """
    rows, seen = [], set()
    allowed = {
        'name-context': {'E', 'neutral', 'P'},
        'name-only': {'neutral'},
        'context-only': {'neutral'},
        'named-speech': {'E', 'neutral', 'P'},
        'anon-speech': {'E', 'neutral'},
    }
    frames = {'E': 'G', 'neutral': 'O', 'P': 'K'}
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            leader = record['leader_id']
            condition, framing = record['condition'], record['framing']
            if leader not in criterion_ids:
                raise ValueError(f'Record {number}: politician outside the criterion')
            if record['model'] not in MODELS[:-1]:
                raise ValueError(f'Record {number}: unknown model')
            if condition not in allowed or framing not in allowed[condition]:
                raise ValueError(f'Record {number}: unsupported experimental cell')
            repeat = record['repeat']
            if type(repeat) is not int or repeat not in (0, 1, 2):
                raise ValueError(f'Record {number}: invalid repeat')
            speech = record.get('speech_id')
            if bool(speech) != (condition in ('named-speech', 'anon-speech')):
                raise ValueError(f'Record {number}: speech identifier inconsistent with condition')
            key = (leader, record['model'], condition, framing, repeat, speech)
            if key in seen:
                raise ValueError(f'Record {number}: duplicate experimental cell')
            seen.add(key)
            ratings = record['ratings']
            if set(ratings) != {f's{i}' for i in range(1, 17)} or any(
                    type(v) is not int or not 0 <= v <= 4 for v in ratings.values()):
                raise ValueError(f'Record {number}: invalid item ratings')
            for trait, value in score_ratings(ratings).items():
                rows.append((leader, record['model'], condition,
                             frames[framing], trait, value))
    if not rows:
        raise ValueError('No model responses found')
    raw = pd.DataFrame(rows, columns=['leader_id', 'model', 'condition',
                                      'framing', 'trait', 'score'])
    points = (raw.groupby(['leader_id', 'model', 'condition', 'framing', 'trait'],
                          as_index=False)['score'].mean()
              .rename(columns={'score': 'llm_mean'}))
    points = add_pooled(points, ['leader_id', 'condition', 'framing', 'trait'])
    name = points[points.condition == 'name-context'].copy()
    speech = points[points.condition.isin(['named-speech', 'anon-speech'])].copy()
    role = points[points.condition == 'context-only'].copy()
    return name, speech, role


def context_inventory(criterion: pd.DataFrame) -> pd.DataFrame:
    """Count unique authored context strings without writing individual metadata."""
    contexts = []
    for record in criterion.to_dict('records'):
        prefix = record['full_name'] + ', '
        if not record['identity'].startswith(prefix):
            raise ValueError('Identity must begin with the canonical politician name')
        contexts.append(record['identity'][len(prefix):])
    counts = pd.Series(contexts).value_counts()
    unique = int((counts == 1).sum())
    return pd.DataFrame([{
        'n_leaders': len(contexts), 'n_distinct_contexts': len(counts),
        'n_unique_leader_contexts': unique,
        'share_unique_leader_contexts': unique / len(contexts),
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criterion', type=Path, default=ROOT / 'data/criterion.csv')
    parser.add_argument('--negex-dir', type=Path, default=ROOT / 'data/NEGex_2.0')
    parser.add_argument('--records', type=Path, default=ROOT / 'records/ratings.jsonl')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/ideology')
    parser.add_argument('--prominence', type=Path, default=ROOT / 'measurements/prominence_2026-07-06.csv',
                        help='Historical Wikimedia snapshot; pass a fresh local CSV explicitly')
    parser.add_argument('--metadata', action='store_true',
                        help='Also run ideology gradients and metadata ridge baselines; requires prominence')
    args = parser.parse_args()
    criterion = pd.read_csv(args.criterion).fillna({'office': ''})
    if len(criterion) != 203 or criterion.leader_id.nunique() != 203:
        raise ValueError('Expected the fixed 203-politician criterion')
    if pd.to_numeric(criterion['is_bridge']).sum() != 50:
        raise ValueError('Expected the fixed 50-politician speech bridge')
    candidates = pd.read_stata(args.negex_dir / 'NEGex_2.0_candidates.dta',
                               convert_categoricals=False)
    elections = pd.read_stata(args.negex_dir / 'NEGex_2.0_elections.dta',
                              convert_categoricals=False)
    cards = map_candidate_cards(candidates, elections, criterion)
    adjusted = build_adjusted_criterion(criterion, cards)
    name, speech, role = load_points(args.records, set(criterion.leader_id))
    aggregates, traits = build_convergence_tables(name, speech, role, adjusted)
    outputs = {'adjusted_convergence_aggregate.csv': aggregates,
               'adjusted_convergence_trait.csv': traits,
               'criterion_adjustment_diagnostics.csv': adjustment_diagnostics(candidates, adjusted),
               'context_inventory.csv': context_inventory(criterion)}
    if args.metadata:
        prominence = pd.read_csv(args.prominence)
        if prominence.leader_id.duplicated().any() or set(prominence.leader_id) != set(criterion.leader_id):
            raise ValueError('Prominence measurements must cover the fixed sample exactly')
        metadata = latest_metadata(criterion, cards, prominence)
        outputs['ideology_gradients.csv'] = ideology_gradients(adjusted, metadata, name, role)
        baseline, residual = metadata_baselines(adjusted, metadata, name)
        outputs['metadata_baseline.csv'] = baseline
        outputs['metadata_residual_convergence.csv'] = residual
        outputs['metadata_identity_uniqueness.csv'] = uniqueness_table(metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(args.out_dir / filename, index=False)
    print(f'Wrote {len(outputs)} aggregate tables (bootstrap draws={N_BOOT}, seed={SEED})')
SEED = 20260919

N_BOOT = 5000

TRAIT_ITEMS = {'extraversion': (('s1', +1), ('s6', -1)), 'agreeableness': (('s7', +1), ('s2', -1)), 'conscientiousness': (('s3', +1), ('s8', -1)), 'emotional_stability': (('s9', +1), ('s4', -1)), 'openness': (('s5', +1), ('s10', -1)), 'narcissism': (('s11', +1), ('s14', +1)), 'psychopathy': (('s12', +1), ('s15', +1)), 'machiavellianism': (('s13', +1), ('s16', +1))}

TRAITS = tuple(TRAIT_ITEMS)

OPEN_MODELS = ('qwen-8b', 'qwen-32b', 'gemma-12b', 'gemma-27b')

MODELS = (*OPEN_MODELS, 'gpt-5.5', 'pooled')

TRAIT_DTA = {'extraversion': ('bfi_extrav', 'adj_bfi_extrav', 'numrespt_bfi'), 'agreeableness': ('bfi_agree', 'adj_bfi_agree', 'numrespt_bfi'), 'conscientiousness': ('bfi_consc', 'adj_bfi_consc', 'numrespt_bfi'), 'emotional_stability': ('bfi_emot', 'adj_bfi_emot', 'numrespt_bfi'), 'openness': ('bfi_open', 'adj_bfi_open', 'numrespt_bfi'), 'narcissism': ('triad_narciss', 'adj_triad_narciss', 'numrespt_triad'), 'psychopathy': ('triad_psycho', 'adj_triad_psycho', 'numrespt_triad'), 'machiavellianism': ('triad_machiav', 'adj_triad_machiav', 'numrespt_triad')}

def norm_actor(value: object) -> str:
    text = ''.join((c for c in unicodedata.normalize('NFKD', str(value).lower()) if not unicodedata.combining(c)))
    return re.sub('[^a-z0-9]+', ' ', text).strip()

def safe_r(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = (np.asarray(x, float), np.asarray(y, float))
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float('nan')
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

def bootstrap_r_vector(x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    a, b = (np.asarray(x, float)[indices], np.asarray(y, float)[indices])
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(axis=1)
    sa = np.where(ok, a, 0.0).sum(axis=1)
    sb = np.where(ok, b, 0.0).sum(axis=1)
    ma = np.divide(sa, n, out=np.full(len(n), np.nan), where=n > 0)
    mb = np.divide(sb, n, out=np.full(len(n), np.nan), where=n > 0)
    da = np.where(ok, a - ma[:, None], 0.0)
    db = np.where(ok, b - mb[:, None], 0.0)
    denom = np.sqrt((da * da).sum(axis=1) * (db * db).sum(axis=1))
    return np.divide((da * db).sum(axis=1), denom, out=np.full(len(n), np.nan), where=(n >= 3) & (denom > 0))

def score_ratings(ratings: dict[str, int]) -> dict[str, float]:
    scored: dict[str, float] = {}
    for trait, items in TRAIT_ITEMS.items():
        vals = [float(ratings[key]) if keyed == 1 else 4.0 - float(ratings[key]) for key, keyed in items]
        scored[trait] = float(np.mean(vals))
    return scored

def add_pooled(points: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    pooled = points[points['model'].isin(OPEN_MODELS)].groupby(keys, as_index=False)['llm_mean'].mean()
    pooled['model'] = 'pooled'
    return pd.concat([points, pooled], ignore_index=True)

def map_candidate_cards(candidates: pd.DataFrame, elections: pd.DataFrame, criterion: pd.DataFrame) -> pd.DataFrame:
    wide_rows: list[tuple] = []
    for row in elections.itertuples(index=False):
        for i in range(1, 11):
            value = getattr(row, f'candidate{i}')
            if isinstance(value, str) and value.strip():
                wide_rows.append((row.electID, i, value, norm_actor(value)))
    wide = pd.DataFrame(wide_rows, columns=['electID', 'candnum', 'wide_actor', 'actor_n'])
    cand = candidates.copy()
    cand['actor_n'] = cand['actor'].map(norm_actor)
    mapped = wide.merge(cand, on=['electID', 'actor_n'], how='left', validate='one_to_one')
    mapped['source_id'] = mapped['electID'] + '#cand' + mapped['candnum'].astype(str)
    ids = criterion.assign(source_id=criterion['negex_ids'].str.split(';')).explode('source_id')
    cards = ids.merge(mapped, on='source_id', how='left', suffixes=('_criterion', '_negex'), validate='many_to_one')
    if cards['actor'].isna().any():
        missing = cards.loc[cards['actor'].isna(), 'source_id'].tolist()
        raise AssertionError(f'Unmatched criterion source ids: {missing}')
    cards['card_year'] = pd.to_datetime(cards['election_date'], format='%d-%b-%y').dt.year
    return cards

def weighted_by_leader(cards: pd.DataFrame, value: str, weight: str) -> pd.Series:
    sub = cards.dropna(subset=[value]).copy()
    sub['_weight'] = pd.to_numeric(sub[weight], errors='coerce').fillna(0.0)
    totals = sub.groupby('leader_id')['_weight'].transform('sum')
    sub.loc[totals <= 0, '_weight'] = 1.0
    sub['_weighted'] = sub[value].astype(float) * sub['_weight']
    agg = sub.groupby('leader_id').agg(total=('_weighted', 'sum'), weight=('_weight', 'sum'))
    return agg['total'] / agg['weight']

def build_adjusted_criterion(criterion: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    out = criterion[['leader_id', 'full_name', 'country', 'office', 'arm', 'is_bridge', 'election_year', 'election_type']].copy()
    out = out.set_index('leader_id')
    for trait, (raw, adjusted, weight) in TRAIT_DTA.items():
        out[f'raw_{trait}'] = criterion.set_index('leader_id')[f'expert_{trait}_mean']
        out[f'adjusted_{trait}'] = weighted_by_leader(cards, adjusted, weight)
        rebuilt = weighted_by_leader(cards, raw, weight)
        check = pd.concat([out[f'raw_{trait}'], rebuilt.rename('rebuilt')], axis=1).dropna()
        max_diff = float((check.iloc[:, 0] - check.iloc[:, 1]).abs().max())
        if max_diff > 1e-05:
            raise AssertionError(f'Raw criterion reconstruction failed for {trait}: {max_diff}')
    return out.reset_index()

def latest_metadata(criterion: pd.DataFrame, cards: pd.DataFrame, prominence: pd.DataFrame) -> pd.DataFrame:
    latest = cards.sort_values(['leader_id', 'card_year', 'election_date']).groupby('leader_id', as_index=False).tail(1).copy()
    keep = ['leader_id', 'source_id', 'electID', 'card_year', 'c_party', 'c_position', 'c_positionr', 'c_incumbent', 'c_votesprc', 'c_rank', 'mediacover', 'lrscale', 'sdlrscale', 'domestic']
    meta = latest[keep].merge(criterion[['leader_id', 'country', 'office', 'arm', 'election_year', 'election_type']], on='leader_id', how='left', validate='one_to_one')
    prom_keep = [c for c in ['leader_id', 'sitelinks', 'pageviews_election_year', 'snapshot_date'] if c in prominence]
    meta = meta.merge(prominence[prom_keep], on='leader_id', how='left', validate='one_to_one')
    meta['candidate_position_13'] = pd.to_numeric(meta['c_position'], errors='coerce')
    meta['candidate_position_7'] = pd.to_numeric(meta['c_positionr'], errors='coerce')
    meta['incumbent'] = pd.to_numeric(meta['c_incumbent'], errors='coerce')
    meta['panel_lr_mean_10'] = pd.to_numeric(meta['lrscale'], errors='coerce')
    meta['panel_lr_sd'] = pd.to_numeric(meta['sdlrscale'], errors='coerce')
    meta['candidate_position_01'] = (meta['candidate_position_7'] - 1.0) / 6.0
    meta['panel_lr_mean_01'] = (meta['panel_lr_mean_10'] - 1.0) / 9.0
    meta['candidate_panel_lr_distance_01'] = (meta['candidate_position_01'] - meta['panel_lr_mean_01']).abs()
    meta['log_sitelinks'] = np.log1p(pd.to_numeric(meta['sitelinks'], errors='coerce'))
    meta['log_pageviews'] = np.log1p(pd.to_numeric(meta['pageviews_election_year'], errors='coerce'))
    return meta

def adjustment_diagnostics(candidates: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    full = candidates.copy()
    full['candidate_position_7'] = pd.to_numeric(full['c_positionr'], errors='coerce')
    full['candidate_panel_lr_distance_01'] = ((pd.to_numeric(full['c_positionr'], errors='coerce') - 1.0) / 6.0 - (pd.to_numeric(full['lrscale'], errors='coerce') - 1.0) / 9.0).abs()
    rows: list[dict] = []
    for scope, frame in (('all_NEGex_candidate_cards', full), ('main_203_leaders', adjusted)):
        for trait, (raw_dta, adj_dta, _) in TRAIT_DTA.items():
            if scope == 'all_NEGex_candidate_cards':
                cols = [raw_dta, adj_dta, 'lrscale', 'candidate_position_7', 'candidate_panel_lr_distance_01']
                sub = frame[cols].dropna().copy()
                raw, adj, lr = (sub[raw_dta], sub[adj_dta], sub['lrscale'])
                candidate_position = sub['candidate_position_7']
                distance = sub['candidate_panel_lr_distance_01']
            else:
                cols = [f'raw_{trait}', f'adjusted_{trait}']
                sub = frame[cols].dropna().copy()
                raw, adj = (sub[cols[0]], sub[cols[1]])
                lr = pd.Series(np.nan, index=sub.index)
                candidate_position = pd.Series(np.nan, index=sub.index)
                distance = pd.Series(np.nan, index=sub.index)
            rec = {'scope': scope, 'trait': trait, 'n': len(sub), 'raw_adjusted_r': safe_r(raw, adj), 'raw_panel_lr_r': safe_r(raw, lr), 'raw_panel_lr_r2': safe_r(raw, lr) ** 2 if np.isfinite(safe_r(raw, lr)) else np.nan, 'adjusted_panel_lr_r': safe_r(adj, lr), 'raw_candidate_position_r': safe_r(raw, candidate_position), 'adjusted_candidate_position_r': safe_r(adj, candidate_position), 'raw_candidate_panel_distance_r': safe_r(raw, distance), 'adjusted_candidate_panel_distance_r': safe_r(adj, distance), 'adjusted_mean': float(np.mean(adj)), 'adjusted_sd': float(np.std(adj, ddof=1))}
            if scope == 'all_NEGex_candidate_cards':
                for label, predictors in {'panel_lr': [lr], 'candidate_and_panel_lr': [candidate_position, lr], 'candidate_panel_distance': [distance]}.items():
                    X = np.column_stack([np.ones(len(sub)), *[np.asarray(x, float) for x in predictors]])
                    beta = np.linalg.lstsq(X, np.asarray(raw, float), rcond=None)[0]
                    aggregate_resid = np.asarray(raw, float) - X @ beta
                    diff = aggregate_resid - np.asarray(adj, float)
                    rec[f'candidate_level_{label}_residual_rmse'] = float(np.sqrt(np.mean(diff ** 2)))
                    rec[f'candidate_level_{label}_residual_max_abs_error'] = float(np.max(np.abs(diff)))
            rows.append(rec)
    return pd.DataFrame(rows)

def percentile(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (float('nan'), float('nan'))
    return tuple(np.percentile(finite, [2.5, 97.5]))

def convergence_pair(points: pd.DataFrame, adjusted: pd.DataFrame, leader_ids: Sequence[str], rng: np.random.Generator) -> tuple[dict, list[dict]]:
    criterion = adjusted.set_index('leader_id')
    leaders = np.asarray(sorted(set(leader_ids) & set(criterion.index)))
    by_trait: list[dict] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for trait in TRAITS:
        vals = points[points['trait'] == trait].set_index('leader_id')['llm_mean']
        x_raw = criterion.loc[leaders, f'raw_{trait}'].to_numpy(float)
        x_adj = criterion.loc[leaders, f'adjusted_{trait}'].to_numpy(float)
        y = vals.reindex(leaders).to_numpy(float)
        arrays[trait] = (x_raw, x_adj, y)
        common = np.isfinite(x_raw) & np.isfinite(x_adj) & np.isfinite(y)
        full = np.isfinite(x_raw) & np.isfinite(y)
        by_trait.append({'trait': trait, 'n_raw_full': int(full.sum()), 'n_matched': int(common.sum()), 'r_raw_full': safe_r(x_raw[full], y[full]), 'r_raw_matched': safe_r(x_raw[common], y[common]), 'r_adjusted': safe_r(x_adj[common], y[common])})
    for row in by_trait:
        row['delta_adjusted_minus_raw'] = row['r_adjusted'] - row['r_raw_matched']
    n = len(leaders)
    ix = rng.integers(0, n, size=(N_BOOT, n))
    raw_by_trait, adj_by_trait = ([], [])
    trait_boot: dict[str, np.ndarray] = {}
    for trait in TRAITS:
        x_raw, x_adj, y = arrays[trait]
        common = np.isfinite(x_raw) & np.isfinite(x_adj) & np.isfinite(y)
        xr = np.where(common, x_raw, np.nan)
        xa = np.where(common, x_adj, np.nan)
        yy = np.where(common, y, np.nan)
        rr = bootstrap_r_vector(xr, yy, ix)
        ra = bootstrap_r_vector(xa, yy, ix)
        raw_by_trait.append(rr)
        adj_by_trait.append(ra)
        trait_boot[trait] = ra - rr
    boot_raw = np.nanmean(np.vstack(raw_by_trait), axis=0)
    boot_adj = np.nanmean(np.vstack(adj_by_trait), axis=0)
    boot_delta = boot_adj - boot_raw
    for row in by_trait:
        lo, hi = percentile(trait_boot[row['trait']])
        row['delta_ci_low'], row['delta_ci_high'] = (lo, hi)
    raw_point = float(np.nanmean([r['r_raw_matched'] for r in by_trait]))
    adj_point = float(np.nanmean([r['r_adjusted'] for r in by_trait]))
    delta_point = adj_point - raw_point
    raw_lo, raw_hi = percentile(boot_raw)
    adj_lo, adj_hi = percentile(boot_adj)
    d_lo, d_hi = percentile(boot_delta)
    aggregate = {'n_leaders': len(leaders), 'n_min_trait_matched': min((r['n_matched'] for r in by_trait)), 'n_max_trait_matched': max((r['n_matched'] for r in by_trait)), 'mean_r_raw_matched': raw_point, 'raw_ci_low': raw_lo, 'raw_ci_high': raw_hi, 'mean_r_adjusted': adj_point, 'adjusted_ci_low': adj_lo, 'adjusted_ci_high': adj_hi, 'delta_adjusted_minus_raw': delta_point, 'delta_ci_low': d_lo, 'delta_ci_high': d_hi, 'bootstrap_share_delta_le_zero': float(np.mean(np.asarray(boot_delta) <= 0))}
    return (aggregate, by_trait)

def build_convergence_tables(name_points: pd.DataFrame, speech_points: pd.DataFrame, role_points: pd.DataFrame, adjusted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    agg_rows: list[dict] = []
    trait_rows: list[dict] = []
    all_ids = adjusted['leader_id'].tolist()
    bridge_ids = adjusted.loc[adjusted['is_bridge'] == 1, 'leader_id'].tolist()
    cells: list[tuple[str, str, str, str, pd.DataFrame, list[str]]] = []
    for model in MODELS:
        for framing in ('G', 'O', 'K'):
            sub = name_points[(name_points.model == model) & (name_points.framing == framing)]
            cells.append(('main_203', 'name', model, framing, sub, all_ids))
            cells.append(('bridge_50', 'name', model, framing, sub, bridge_ids))
    for condition in ('named-speech', 'anon-speech'):
        for model in MODELS:
            for framing in ('G', 'O', 'K'):
                sub = speech_points[(speech_points.condition == condition) & (speech_points.model == model) & (speech_points.framing == framing)]
                if len(sub):
                    cells.append(('bridge_50', condition, model, framing, sub, bridge_ids))
    # The original September table had no GPT-5.5 context-only cell. Append
    # that additional cell last, preserving the RNG worlds of all original cells.
    for model in (*OPEN_MODELS, 'pooled', 'gpt-5.5'):
        sub = role_points[(role_points.model == model) & (role_points.framing == 'O')]
        if len(sub):
            cells.append(('main_203', 'role-context', model, 'O', sub, all_ids))
    for scope, condition, model, framing, sub, ids in cells:
        aggregate, traits = convergence_pair(sub, adjusted, ids, rng)
        base = {'scope': scope, 'condition': condition, 'model': model, 'framing': framing}
        agg_rows.append({**base, **aggregate})
        trait_rows.extend(({**base, **row} for row in traits))
    return (pd.DataFrame(agg_rows), pd.DataFrame(trait_rows))

def fdr_bh(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors='coerce').to_numpy(float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return pd.Series(q, index=pvalues.index)
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return pd.Series(q, index=pvalues.index)

def fit_position_slope(frame: pd.DataFrame, outcome: str, specification: str) -> dict:
    needed = [outcome, 'candidate_position_13', 'country']
    if specification == 'full_controls':
        needed += ['incumbent', 'election_year', 'election_type', 'log_sitelinks', 'arm']
    sub = frame.dropna(subset=needed).copy()
    sub['position_centered'] = sub['candidate_position_13'] - 7.0
    sub['year_centered'] = sub['election_year'] - sub['election_year'].mean()
    if specification == 'between_leader':
        formula = f'{outcome} ~ position_centered'
    elif specification == 'country_fe':
        formula = f'{outcome} ~ position_centered + C(country)'
    elif specification == 'full_controls':
        formula = f'{outcome} ~ position_centered + incumbent + year_centered + C(election_type) + log_sitelinks + C(arm) + C(country)'
    else:
        raise ValueError(specification)
    fit = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['country']})
    coef = float(fit.params['position_centered'])
    se = float(fit.bse['position_centered'])
    p = float(fit.pvalues['position_centered'])
    return {'n': len(sub), 'n_countries': sub.country.nunique(), 'slope': coef, 'se': se, 'ci_low': coef - 1.96 * se, 'ci_high': coef + 1.96 * se, 'p_value': p, 'r_squared': float(fit.rsquared)}

def ideology_gradients(adjusted: pd.DataFrame, metadata: pd.DataFrame, name_points: pd.DataFrame, role_points: pd.DataFrame) -> pd.DataFrame:
    base = metadata.merge(adjusted, on=['leader_id', 'country', 'office', 'arm', 'election_year', 'election_type'], how='inner')
    rows: list[dict] = []
    for trait in TRAITS:
        frame = base.copy()
        frame['expert_raw'] = frame[f'raw_{trait}']
        frame['expert_adjusted'] = frame[f'adjusted_{trait}']
        for criterion_kind, outcome in (('expert_raw', 'expert_raw'), ('expert_adjusted', 'expert_adjusted')):
            for spec in ('between_leader', 'country_fe', 'full_controls'):
                rows.append({'condition': 'criterion', 'outcome_kind': criterion_kind, 'model': 'expert', 'trait': trait, 'specification': spec, **fit_position_slope(frame, outcome, spec)})
        for condition, points in (('name', name_points), ('role-context', role_points)):
            neutral = points[points.framing == 'O'].pivot_table(index='leader_id', columns=['model', 'trait'], values='llm_mean')
            for model in MODELS:
                if (model, trait) not in neutral:
                    continue
                frame['llm'] = frame['leader_id'].map(neutral[model, trait])
                frame['error'] = frame['llm'] - frame['expert_raw']
                for kind, outcome in (('llm_rating', 'llm'), ('llm_minus_expert_error', 'error')):
                    for spec in ('between_leader', 'country_fe', 'full_controls'):
                        rows.append({'condition': condition, 'outcome_kind': kind, 'model': model, 'trait': trait, 'specification': spec, **fit_position_slope(frame, outcome, spec)})
    out = pd.DataFrame(rows)
    for (condition, kind, spec), idx in out.groupby(['condition', 'outcome_kind', 'specification']).groups.items():
        out.loc[idx, 'q_value_bh'] = fdr_bh(out.loc[idx, 'p_value'])
    return out

def design_matrix(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    numeric = ['candidate_position_13', 'incumbent', 'election_year', 'log_sitelinks', 'log_pageviews']
    categorical = ['election_type', 'arm']
    if feature_set == 'context_plus_country_office':
        categorical += ['country', 'office']
    elif feature_set != 'structured_context':
        raise ValueError(feature_set)
    xtr = [np.ones(len(train))]
    xte = [np.ones(len(test))]
    for col in numeric:
        tr = pd.to_numeric(train[col], errors='coerce').to_numpy(float).copy()
        te = pd.to_numeric(test[col], errors='coerce').to_numpy(float).copy()
        med = float(np.nanmedian(tr)) if np.isfinite(tr).any() else 0.0
        tr_missing, te_missing = (~np.isfinite(tr), ~np.isfinite(te))
        tr[tr_missing], te[te_missing] = (med, med)
        sd = float(np.std(tr, ddof=0)) or 1.0
        mean = float(np.mean(tr))
        xtr.extend([(tr - mean) / sd, tr_missing.astype(float)])
        xte.extend([(te - mean) / sd, te_missing.astype(float)])
    for col in categorical:
        levels = sorted(set(train[col].fillna('<missing>').astype(str)))
        trv = train[col].fillna('<missing>').astype(str)
        tev = test[col].fillna('<missing>').astype(str)
        for level in levels[1:]:
            xtr.append((trv == level).to_numpy(float))
            xte.append((tev == level).to_numpy(float))
    return (np.column_stack(xtr), np.column_stack(xte))

def ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(X_train.T @ X_train + penalty) @ X_train.T @ y_train
    return X_test @ beta

def make_folds(frame: pd.DataFrame, scheme: str, n_splits: int=10) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    if scheme == 'leader_random_10fold':
        ids = rng.permutation(len(frame))
        return [np.asarray(x, int) for x in np.array_split(ids, n_splits)]
    if scheme == 'country_group_10fold':
        groups = frame.groupby('country').indices
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        folds: list[list[int]] = [[] for _ in range(n_splits)]
        for _, indices in ordered:
            smallest = min(range(n_splits), key=lambda j: len(folds[j]))
            folds[smallest].extend(indices)
        return [np.asarray(sorted(x), int) for x in folds]
    raise ValueError(scheme)

def crossfit(frame: pd.DataFrame, outcome: str, feature_set: str, scheme: str, alpha: float=10.0) -> np.ndarray:
    pred = np.full(len(frame), np.nan)
    valid = np.flatnonzero(np.isfinite(pd.to_numeric(frame[outcome], errors='coerce')))
    work = frame.iloc[valid].reset_index(drop=True)
    y = pd.to_numeric(work[outcome], errors='coerce').to_numpy(float)
    for test_ix in make_folds(work, scheme):
        train_ix = np.setdiff1d(np.arange(len(work)), test_ix)
        Xtr, Xte = design_matrix(work.iloc[train_ix], work.iloc[test_ix], feature_set)
        pred[valid[test_ix]] = ridge_predict(Xtr, y[train_ix], Xte, alpha)
    return pred

def bootstrap_mean_r(arrays: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(np.nanmean([safe_r(x, y) for x, y in arrays]))
    n = len(arrays[0][0])
    ix = rng.integers(0, n, size=(N_BOOT, n))
    draws = np.nanmean(np.vstack([bootstrap_r_vector(x, y, ix) for x, y in arrays]), axis=0)
    lo, hi = percentile(draws)
    return (point, lo, hi)

def metadata_baselines(adjusted: pd.DataFrame, metadata: pd.DataFrame, name_points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = metadata.merge(adjusted, on=['leader_id', 'country', 'office', 'arm', 'election_year', 'election_type'], how='inner')
    neutral = name_points[name_points.framing == 'O'].pivot_table(index='leader_id', columns=['model', 'trait'], values='llm_mean')
    baseline_rows: list[dict] = []
    residual_rows: list[dict] = []
    rng = np.random.default_rng(SEED + 1)
    for feature_set in ('structured_context', 'context_plus_country_office'):
        for scheme in ('leader_random_10fold', 'country_group_10fold'):
            expert_preds: dict[str, np.ndarray] = {}
            expert_arrays: list[tuple[np.ndarray, np.ndarray]] = []
            for trait in TRAITS:
                outcome = f'raw_{trait}'
                pred = crossfit(frame, outcome, feature_set, scheme)
                expert_preds[trait] = pred
                actual = frame[outcome].to_numpy(float)
                expert_arrays.append((actual, pred))
                baseline_rows.append({'feature_set': feature_set, 'cv_scheme': scheme, 'trait': trait, 'n': int((np.isfinite(actual) & np.isfinite(pred)).sum()), 'expert_prediction_r': safe_r(actual, pred), 'ridge_alpha': 10.0})
            point, lo, hi = bootstrap_mean_r(expert_arrays, rng)
            baseline_rows.append({'feature_set': feature_set, 'cv_scheme': scheme, 'trait': 'overall_mean', 'n': len(frame), 'expert_prediction_r': point, 'ci_low': lo, 'ci_high': hi, 'ridge_alpha': 10.0})
            for model in MODELS:
                original_arrays: list[tuple[np.ndarray, np.ndarray]] = []
                residual_arrays: list[tuple[np.ndarray, np.ndarray]] = []
                for trait in TRAITS:
                    if (model, trait) not in neutral:
                        continue
                    llm = frame['leader_id'].map(neutral[model, trait]).to_numpy(float)
                    expert = frame[f'raw_{trait}'].to_numpy(float)
                    temp = frame.copy()
                    temp['_llm'] = llm
                    llm_pred = crossfit(temp, '_llm', feature_set, scheme)
                    expert_pred = expert_preds[trait]
                    original_arrays.append((expert, llm))
                    residual_arrays.append((expert - expert_pred, llm - llm_pred))
                r0, r0lo, r0hi = bootstrap_mean_r(original_arrays, rng)
                rr, rrlo, rrhi = bootstrap_mean_r(residual_arrays, rng)
                residual_rows.append({'feature_set': feature_set, 'cv_scheme': scheme, 'model': model, 'framing': 'O', 'n': len(frame), 'mean_r_original': r0, 'original_ci_low': r0lo, 'original_ci_high': r0hi, 'mean_r_crossfit_residual': rr, 'residual_ci_low': rrlo, 'residual_ci_high': rrhi, 'change_residual_minus_original': rr - r0, 'ridge_alpha': 10.0})
    return (pd.DataFrame(baseline_rows), pd.DataFrame(residual_rows))

def uniqueness_table(metadata: pd.DataFrame) -> pd.DataFrame:
    combos = {'country + election year': ['country', 'election_year'], 'country + election year + election type': ['country', 'election_year', 'election_type'], 'country + party + election year': ['country', 'c_party', 'election_year'], 'country + office + election year': ['country', 'office', 'election_year'], 'country + election type + year + incumbency': ['country', 'election_type', 'election_year', 'incumbent'], 'country + office': ['country', 'office'], 'country + ideology(13) + incumbency + year': ['country', 'candidate_position_13', 'incumbent', 'election_year']}
    rows = []
    for name, cols in combos.items():
        complete = metadata.dropna(subset=cols)
        sizes = complete.groupby(cols, dropna=False)['leader_id'].transform('size')
        rows.append({'combination': name, 'variables': ';'.join(cols), 'n_complete': len(complete), 'n_unique_rows': int((sizes == 1).sum()), 'share_unique': float((sizes == 1).mean())})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
