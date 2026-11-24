"""Three identity-evidence baselines with the executed paired bootstrap.

Uses the original September per-politician response associations and shared
resamples for every level and contrast. Only aggregate statistics are written.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pinf import metrics
from scripts.ideology import TRAITS, OPEN_MODELS, score_ratings, load_points as validate_records

BOOTSTRAP_SEED, N_BOOT, CI_LEVEL = 20260706, 2000, 0.95
CONDITION = 'NAME_ONLY'
CONDITIONS = ('ROLE', CONDITION, 'NAME_CONTEXT')
MODELS = (*OPEN_MODELS, 'pooled', 'gpt-5.5')


def load_points(path: Path, ids: set[str]) -> pd.DataFrame:
    """Validate associated records, then aggregate neutral identity-only responses."""
    validate_records(path, ids)
    condition_map = {'name-context': 'NAME_CONTEXT', 'name-only': 'NAME_ONLY',
                     'context-only': 'ROLE'}
    rows, keys = [], set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec['framing'] != 'neutral' or rec['condition'] not in condition_map:
                continue
            keys.add((rec['leader_id'], rec['model'], rec['condition'], rec['repeat']))
            for trait, value in score_ratings(rec['ratings']).items():
                rows.append((rec['leader_id'], rec['model'], condition_map[rec['condition']],
                             trait, value))
    expected = {(leader, model, condition, repeat)
                for leader in ids for model in (*OPEN_MODELS, 'gpt-5.5')
                for condition in condition_map for repeat in range(3)}
    if keys != expected:
        raise ValueError(f'Incomplete baseline response grid: {len(expected - keys)} missing cells')
    raw = pd.DataFrame(rows, columns=['leader_id', 'model', 'condition', 'trait', 'score'])
    points = (raw.groupby(['leader_id', 'model', 'condition', 'trait'], as_index=False)
              ['score'].mean().rename(columns={'score': 'value'}))
    pooled = (points[points.model.isin(OPEN_MODELS)]
              .groupby(['leader_id', 'condition', 'trait'], as_index=False)['value'].mean())
    pooled['model'] = 'pooled'
    return pd.concat([points, pooled], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criterion', type=Path, default=ROOT / 'data/criterion.csv')
    parser.add_argument('--records', type=Path, default=ROOT / 'records/ratings.jsonl')
    parser.add_argument('--ceilings', type=Path, default=ROOT / 'outputs/expert_ceiling.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/identity_baselines')
    args = parser.parse_args()
    criterion = pd.read_csv(args.criterion)
    if len(criterion) != 203 or criterion.leader_id.nunique() != 203:
        raise ValueError('Expected the fixed 203-politician criterion')
    criterion['stratum'] = 'c1s'
    points = load_points(args.records, set(criterion.leader_id))
    source = json.loads(args.ceilings.read_text())
    ceilings = {trait: source['traits'][trait]['by_rho']['0.3']['ceiling'] for trait in TRAITS}
    aggregates, traits = comparison_tables(points, criterion, ceilings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregates.to_csv(args.out_dir / 'comparison_overall.csv', index=False)
    traits.to_csv(args.out_dir / 'comparison_by_trait.csv', index=False)
    print(f'Wrote {len(aggregates)} aggregate rows and {len(traits)} trait rows')
def _rowwise_r(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid='ignore', divide='ignore'):
        count = valid.sum(axis=1, keepdims=True)
        xm = np.where(valid, x, 0).sum(axis=1, keepdims=True) / count
        ym = np.where(valid, y, 0).sum(axis=1, keepdims=True) / count
        xc = np.where(valid, x - xm, 0)
        yc = np.where(valid, y - ym, 0)
        denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
        return np.where(denom > 0, (xc * yc).sum(axis=1) / denom, np.nan)

def _matrices(points: pd.DataFrame, criterion: pd.DataFrame, stratum: str, model: str):
    sub = criterion if stratum == 'all' else criterion[criterion['stratum'] == stratum]
    leaders = sorted(sub['leader_id'])
    exp = sub.set_index('leader_id').reindex(leaders)
    xe = np.array([pd.to_numeric(exp[f'expert_{t}_mean'], errors='coerce').to_numpy(float) for t in TRAITS])
    mats = {}
    for condition in CONDITIONS:
        frame = points[(points['condition'] == condition) & (points['model'] == model)]
        wide = frame.pivot(index='leader_id', columns='trait', values='value')
        mats[condition] = wide.reindex(index=leaders, columns=list(TRAITS)).to_numpy(float).T
    return (leaders, xe, mats)

def _aggregate_stats(xe: np.ndarray, mats: dict[str, np.ndarray], ceilings: np.ndarray) -> dict:
    n = xe.shape[1]
    rt = {condition: _rowwise_r(xe, mats[condition]) for condition in CONDITIONS}
    r = {condition: float(np.nanmean(rt[condition])) for condition in CONDITIONS}
    frac = {condition: float(np.nanmean(rt[condition] / ceilings)) for condition in CONDITIONS}
    out = {'n': n, 'role_r': r['ROLE'], 'name_only_r': r[CONDITION], 'name_context_r': r['NAME_CONTEXT'], 'role_frac_ceiling': frac['ROLE'], 'name_only_frac_ceiling': frac[CONDITION], 'name_context_frac_ceiling': frac['NAME_CONTEXT'], 'name_only_minus_role_r': r[CONDITION] - r['ROLE'], 'name_only_minus_name_context_r': r[CONDITION] - r['NAME_CONTEXT'], 'role_minus_name_context_r': r['ROLE'] - r['NAME_CONTEXT'], 'role_minus_name_context_frac_ceiling': frac['ROLE'] - frac['NAME_CONTEXT'], 'name_only_over_name_context': r[CONDITION] / r['NAME_CONTEXT'], 'role_over_name_context': r['ROLE'] / r['NAME_CONTEXT'], 'n_traits_role_below_name_context': int(np.sum(rt['ROLE'] - rt['NAME_CONTEXT'] < 0))}
    keys = ('role_r', 'name_only_r', 'name_context_r', 'role_frac_ceiling', 'name_only_frac_ceiling', 'name_context_frac_ceiling', 'name_only_minus_role_r', 'name_only_minus_name_context_r', 'role_minus_name_context_r', 'role_minus_name_context_frac_ceiling', 'name_only_over_name_context', 'role_over_name_context')
    boot = {key: np.empty(N_BOOT) for key in keys}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        brt = {condition: _rowwise_r(xe[:, idx], mats[condition][:, idx]) for condition in CONDITIONS}
        br = {condition: float(np.nanmean(brt[condition])) for condition in CONDITIONS}
        bf = {condition: float(np.nanmean(brt[condition] / ceilings)) for condition in CONDITIONS}
        boot['role_r'][i] = br['ROLE']
        boot['name_only_r'][i] = br[CONDITION]
        boot['name_context_r'][i] = br['NAME_CONTEXT']
        boot['role_frac_ceiling'][i] = bf['ROLE']
        boot['name_only_frac_ceiling'][i] = bf[CONDITION]
        boot['name_context_frac_ceiling'][i] = bf['NAME_CONTEXT']
        boot['name_only_minus_role_r'][i] = br[CONDITION] - br['ROLE']
        boot['name_only_minus_name_context_r'][i] = br[CONDITION] - br['NAME_CONTEXT']
        boot['role_minus_name_context_r'][i] = br['ROLE'] - br['NAME_CONTEXT']
        boot['role_minus_name_context_frac_ceiling'][i] = bf['ROLE'] - bf['NAME_CONTEXT']
        boot['name_only_over_name_context'][i] = br[CONDITION] / br['NAME_CONTEXT']
        boot['role_over_name_context'][i] = br['ROLE'] / br['NAME_CONTEXT']
    qlo, qhi = ((1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100)
    for key in keys:
        out[f'{key}_ci_low'] = float(np.nanpercentile(boot[key], qlo))
        out[f'{key}_ci_high'] = float(np.nanpercentile(boot[key], qhi))
    for key in ('name_only_minus_role_r', 'name_only_minus_name_context_r', 'role_minus_name_context_r'):
        out[f'{key}_p_two_sided'] = min(1.0, 2 * min(float(np.mean(boot[key] <= 0)), float(np.mean(boot[key] >= 0))))
    return out

def comparison_tables(points: pd.DataFrame, criterion: pd.DataFrame, ceilings: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows, trait_rows = ([], [])
    cvec = np.array([ceilings[t] for t in TRAITS], float)
    for stratum in ('c1s',):
        for model in MODELS:
            leaders, xe, mats = _matrices(points, criterion, stratum, model)
            aggregate_rows.append({'stratum': stratum, 'model': model, 'trait_group': 'overall_mean', **_aggregate_stats(xe, mats, cvec)})
            for ti, trait in enumerate(TRAITS):
                valid = np.isfinite(xe[ti])
                for condition in CONDITIONS:
                    valid &= np.isfinite(mats[condition][ti])
                expert = xe[ti, valid]
                levels = {condition: metrics.pearson(expert, mats[condition][ti, valid], ci=True, n_boot=N_BOOT, ci_level=CI_LEVEL, seed=BOOTSTRAP_SEED) for condition in CONDITIONS}
                d_nr = metrics.correlation_difference(expert, mats[CONDITION][ti, valid], mats['ROLE'][ti, valid], n_boot=N_BOOT, ci_level=CI_LEVEL, seed=BOOTSTRAP_SEED)
                d_nn = metrics.correlation_difference(expert, mats[CONDITION][ti, valid], mats['NAME_CONTEXT'][ti, valid], n_boot=N_BOOT, ci_level=CI_LEVEL, seed=BOOTSTRAP_SEED)
                d_rn = metrics.correlation_difference(expert, mats['ROLE'][ti, valid], mats['NAME_CONTEXT'][ti, valid], n_boot=N_BOOT, ci_level=CI_LEVEL, seed=BOOTSTRAP_SEED)
                ceiling = ceilings[trait]
                row = {'stratum': stratum, 'model': model, 'trait': trait, 'n': int(valid.sum()), 'ceiling': ceiling}
                for condition, prefix in (('ROLE', 'role'), (CONDITION, 'name_only'), ('NAME_CONTEXT', 'name_context')):
                    level = levels[condition]
                    row.update({f'{prefix}_r': level.statistic, f'{prefix}_r_ci_low': level.ci_low, f'{prefix}_r_ci_high': level.ci_high, f'{prefix}_frac_ceiling': level.statistic / ceiling})
                row.update({'name_only_minus_role_r': d_nr.diff, 'name_only_minus_role_r_ci_low': d_nr.ci_low, 'name_only_minus_role_r_ci_high': d_nr.ci_high, 'name_only_minus_name_context_r': d_nn.diff, 'name_only_minus_name_context_r_ci_low': d_nn.ci_low, 'name_only_minus_name_context_r_ci_high': d_nn.ci_high, 'role_minus_name_context_r': d_rn.diff, 'role_minus_name_context_r_ci_low': d_rn.ci_low, 'role_minus_name_context_r_ci_high': d_rn.ci_high, 'role_over_name_context': levels['ROLE'].statistic / levels['NAME_CONTEXT'].statistic, 'name_only_over_name_context': levels[CONDITION].statistic / levels['NAME_CONTEXT'].statistic})
                trait_rows.append(row)
    return (pd.DataFrame(aggregate_rows), pd.DataFrame(trait_rows))


if __name__ == '__main__':
    main()
