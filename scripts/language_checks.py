"""Reproduce the paper's country-proxy language-composition sensitivity checks.

These published tables group politicians using the original country-to-dominant-
language proxy. They are not a text-level language-identification evaluation.
Different input-language labels require a different, transcript-level analysis.

The family gap is the mean of two model-level correlations per family, not the
correlation of an ensemble prediction. Intervals use 2,000 shared politician
bootstrap draws with seed 20260706, matching the executed September analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import speech_metrics as sm
from scripts.ideology import load_points

QWEN = ('qwen-8b', 'qwen-32b')
GEMMA = ('gemma-12b', 'gemma-27b')
MODELS = (*QWEN, *GEMMA, 'gpt-5.5')
TRAITS = tuple(sm.TRAITS)
CONDITIONS = ('name', 'named-speech', 'anon-speech')
FRAMING = 'O'
SEED, N_BOOT = 20260706, 2000
LO, HI = 2.5, 97.5


def language_index(manifest: pd.DataFrame, bridge_ids: set[str]) -> dict[str, str]:
    """Read the original politician-level language proxies, not inferred text labels."""
    required = {'speech_id', 'leader_id', 'language'}
    if required - set(manifest):
        raise ValueError('Speech manifest must contain speech_id, leader_id and language')
    if manifest.speech_id.duplicated().any():
        raise ValueError('Duplicate speech identifiers')
    if set(manifest.leader_id) != bridge_ids:
        raise ValueError('Speech manifest does not match the fixed speech bridge')
    if manifest.language.isna().any() or manifest.language.astype(str).str.strip().eq('').any():
        raise ValueError('Missing country-proxy language')
    result = {}
    for leader, rows in manifest.groupby('leader_id'):
        languages = set(rows.language)
        if len(languages) != 1:
            raise ValueError('This published analysis requires one country-proxy language per politician')
        result[leader] = languages.pop()
    return result


def load_inputs(criterion: pd.DataFrame, records: Path, manifest: pd.DataFrame):
    if criterion.leader_id.duplicated().any():
        raise ValueError('Duplicate politician identifiers in criterion')
    bridge = set(criterion.loc[pd.to_numeric(criterion.is_bridge) == 1, 'leader_id'])
    languages = language_index(manifest, bridge)
    expected = {(r.leader_id, model, condition, repeat, r.speech_id)
                for r in manifest.itertuples(index=False) for model in MODELS
                for condition in ('named-speech', 'anon-speech') for repeat in range(3)}
    expected.update((leader, model, 'name-context', repeat, None)
                    for leader in bridge for model in MODELS for repeat in range(3))
    actual = set()
    with records.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record['framing'] == 'neutral' and record['condition'] in (
                    'name-context', 'named-speech', 'anon-speech') and record['leader_id'] in bridge:
                actual.add((record['leader_id'], record['model'], record['condition'],
                            record['repeat'], record.get('speech_id')))
    if actual != expected:
        raise ValueError('Incomplete or mismatched neutral response grid for language checks')
    name, speech, _ = load_points(records, set(criterion.leader_id))
    name = name[name.leader_id.isin(bridge)].rename(columns={'llm_mean': 'value'})
    speech = speech[speech.leader_id.isin(bridge)].rename(columns={'llm_mean': 'value'})
    estimates = {}
    for model in MODELS:
        estimates[model] = {'name': sm.est_dict(name, model, FRAMING)}
        for condition in ('named-speech', 'anon-speech'):
            estimates[model][condition] = sm.est_dict(speech, model, FRAMING, condition)
        for condition in CONDITIONS:
            if set(estimates[model][condition]) != bridge:
                raise ValueError(f'Incomplete neutral responses for {model}/{condition}')
            if any(set(values) != set(TRAITS) for values in estimates[model][condition].values()):
                raise ValueError(f'Incomplete trait estimates for {model}/{condition}')
    criteria = {r['leader_id']: r for r in criterion.to_dict('records')}
    return estimates, criteria, languages


def compute_tables(estimates, criterion, languages):
    # lolo uses point correlations only; no ceiling values enter its results.
    summary, _ = lolo(estimates, criterion, {}, languages)
    return {
        'family_convergence_gap.csv': family_gap(estimates, criterion, languages),
        'family_convergence_change_did.csv': family_convergence_change_did(estimates, criterion, languages),
        'leave_one_language_out_summary.csv': summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--criterion', type=Path, default=ROOT / 'data/criterion.csv')
    parser.add_argument('--records', type=Path, default=ROOT / 'records/ratings.jsonl')
    parser.add_argument('--speeches', type=Path, default=ROOT / 'manifests/speeches.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/language')
    args = parser.parse_args()
    criterion = pd.read_csv(args.criterion)
    if len(criterion) != 203 or pd.to_numeric(criterion.is_bridge).sum() != 50:
        raise ValueError('Expected the fixed 203-politician criterion and 50-politician bridge')
    manifest = pd.read_csv(args.speeches)
    if len(manifest) != 192:
        raise ValueError('Expected the fixed 192-speech manifest')
    estimates, criteria, languages = load_inputs(criterion, args.records, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in compute_tables(estimates, criteria, languages).items():
        frame.to_csv(args.out_dir / filename, index=False)
    print('Wrote three country-proxy language sensitivity tables')

def subset_est(ebl: dict, keep: set[str]) -> dict:
    return {leader: values for leader, values in ebl.items() if leader in keep}

def aggregate_point(ebl: dict, crit: dict) -> tuple[float, int]:
    aligned, leaders = sm._aligned(ebl, crit, TRAITS)
    x = np.array([aligned[t][0] for t in TRAITS], dtype=float)
    y = np.array([aligned[t][1] for t in TRAITS], dtype=float)
    return (float(np.nanmean(sm._rowwise_r(x, y))), len(leaders))

def lolo(estimates: dict, crit: dict, ceilings: dict, leader_language: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_leaders = set(leader_language)
    languages = sorted(set(leader_language.values()))
    detail_rows, summary_rows = ([], [])
    for model in MODELS:
        for condition in ('named-speech', 'anon-speech'):
            full, _ = aggregate_point(subset_est(estimates[model][condition], all_leaders), crit)
            vals = []
            for language in languages:
                omitted = {leader for leader, lang in leader_language.items() if lang == language}
                keep = all_leaders - omitted
                lolo_r, n_remaining = aggregate_point(subset_est(estimates[model][condition], keep), crit)
                row = {'model': model, 'condition': condition, 'omitted_language': language, 'n_omitted_leaders': len(omitted), 'n_remaining_leaders': n_remaining, 'full_r': full, 'lolo_r': lolo_r, 'change': lolo_r - full, 'abs_change': abs(lolo_r - full)}
                vals.append(row)
                detail_rows.append(row)
            v = pd.DataFrame(vals)
            rmin = v.loc[v['lolo_r'].idxmin()]
            rmax = v.loc[v['lolo_r'].idxmax()]
            rabs = v.loc[v['abs_change'].idxmax()]
            summary_rows.append({'model': model, 'condition': condition, 'full_r': full, 'min_lolo_r': rmin['lolo_r'], 'min_omitted_language': rmin['omitted_language'], 'max_lolo_r': rmax['lolo_r'], 'max_omitted_language': rmax['omitted_language'], 'largest_abs_change': rabs['abs_change'], 'largest_change_signed': rabs['change'], 'largest_change_omitted_language': rabs['omitted_language']})
    return (pd.DataFrame(summary_rows), pd.DataFrame(detail_rows))

def _condition_matrices(estimates: dict, condition: str, keep: set[str], crit: dict) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    leaders = sorted(keep)
    expert = np.array([[crit.get(leader, {}).get(f'expert_{trait}_mean', np.nan) for leader in leaders] for trait in TRAITS], dtype=float)
    pred = {model: np.array([[estimates[model][condition].get(leader, {}).get(trait, np.nan) for leader in leaders] for trait in TRAITS], dtype=float) for model in MODELS}
    return (leaders, expert, pred)

def family_gap(estimates: dict, crit: dict, leader_language: dict[str, str]) -> pd.DataFrame:
    all_leaders = set(leader_language)
    subsets = {'full': all_leaders, 'without_spanish': {l for l in all_leaders if leader_language[l] != 'Spanish'}, 'without_english': {l for l in all_leaders if leader_language[l] != 'English'}, 'spanish_only': {l for l in all_leaders if leader_language[l] == 'Spanish'}}
    rows = []
    for subset_name, keep in subsets.items():
        for condition in CONDITIONS:
            leaders, expert, pred = _condition_matrices(estimates, condition, keep, crit)

            def model_r(model: str, idx: np.ndarray | None=None) -> float:
                x = expert if idx is None else expert[:, idx]
                y = pred[model] if idx is None else pred[model][:, idx]
                return float(np.nanmean(sm._rowwise_r(x, y)))
            qwen_r = float(np.mean([model_r(m) for m in QWEN]))
            gemma_r = float(np.mean([model_r(m) for m in GEMMA]))
            gap = gemma_r - qwen_r
            rng = np.random.default_rng(SEED)
            boot = np.empty(N_BOOT)
            n = len(leaders)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                q = np.mean([model_r(m, idx) for m in QWEN])
                g = np.mean([model_r(m, idx) for m in GEMMA])
                boot[b] = g - q
            rows.append({'subset': subset_name, 'condition': condition, 'n_leaders': n, 'qwen_mean_model_r': qwen_r, 'gemma_mean_model_r': gemma_r, 'gemma_minus_qwen': gap, 'ci_low': float(np.nanpercentile(boot, LO)), 'ci_high': float(np.nanpercentile(boot, HI))})
    return pd.DataFrame(rows)

def family_convergence_change_did(estimates: dict, crit: dict, leader_language: dict[str, str]) -> pd.DataFrame:
    all_leaders = set(leader_language)
    subsets = {'full': all_leaders, 'without_spanish': {l for l in all_leaders if leader_language[l] != 'Spanish'}, 'without_english': {l for l in all_leaders if leader_language[l] != 'English'}, 'spanish_only': {l for l in all_leaders if leader_language[l] == 'Spanish'}}
    rows = []
    for subset_name, keep in subsets.items():
        leaders = sorted(keep)
        expert = np.array([[crit.get(leader, {}).get(f'expert_{trait}_mean', np.nan) for leader in leaders] for trait in TRAITS], dtype=float)
        pred = {condition: {model: np.array([[estimates[model][condition].get(leader, {}).get(trait, np.nan) for leader in leaders] for trait in TRAITS], dtype=float) for model in MODELS} for condition in CONDITIONS}

        def r(model: str, condition: str, idx: np.ndarray | None=None) -> float:
            x = expert if idx is None else expert[:, idx]
            y0 = pred[condition][model]
            y = y0 if idx is None else y0[:, idx]
            return float(np.nanmean(sm._rowwise_r(x, y)))
        for condition in ('named-speech', 'anon-speech'):
            q_change = float(np.mean([r(m, condition) - r(m, 'name') for m in QWEN]))
            g_change = float(np.mean([r(m, condition) - r(m, 'name') for m in GEMMA]))
            did = g_change - q_change
            rng = np.random.default_rng(SEED)
            boot = np.empty(N_BOOT)
            n = len(leaders)
            for b in range(N_BOOT):
                idx = rng.integers(0, n, n)
                qb = np.mean([r(m, condition, idx) - r(m, 'name', idx) for m in QWEN])
                gb = np.mean([r(m, condition, idx) - r(m, 'name', idx) for m in GEMMA])
                boot[b] = gb - qb
            rows.append({'subset': subset_name, 'condition': condition, 'n_leaders': n, 'qwen_mean_change_from_name': q_change, 'gemma_mean_change_from_name': g_change, 'gemma_minus_qwen_change': did, 'ci_low': float(np.nanpercentile(boot, LO)), 'ci_high': float(np.nanpercentile(boot, HI))})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
