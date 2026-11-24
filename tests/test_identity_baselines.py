"""Scientific invariants of the three-condition identity comparison."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('identity_baselines', ROOT / 'scripts/identity_baselines.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_paired_identity_comparison_has_exact_zero_self_contrast(monkeypatch):
    monkeypatch.setattr(baseline, 'N_BOOT', 100)
    expert = np.vstack([np.arange(10.) + i / 10 for i in range(8)])
    values = expert + np.array([0, 1, 0, 0, 1, 0, -1, 0, 1, 0])
    matrices = {condition: values.copy() for condition in baseline.CONDITIONS}
    out = baseline._aggregate_stats(expert, matrices, np.linspace(.7, .9, 8))
    for contrast in ['name_only_minus_role_r', 'name_only_minus_name_context_r',
                     'role_minus_name_context_r', 'role_minus_name_context_frac_ceiling']:
        assert out[contrast] == 0
        assert out[contrast + '_ci_low'] == 0
        assert out[contrast + '_ci_high'] == 0
    assert out['name_only_over_name_context'] == 1


def test_aggregate_ceiling_is_mean_of_trait_ratios(monkeypatch):
    monkeypatch.setattr(baseline, 'N_BOOT', 30)
    rng = np.random.default_rng(7)
    expert = rng.normal(size=(8, 25))
    values = expert + rng.normal(size=(8, 25)) * np.arange(1, 9)[:, None] / 4
    ceilings = np.linspace(.45, .95, 8)
    matrices = {condition: values for condition in baseline.CONDITIONS}
    out = baseline._aggregate_stats(expert, matrices, ceilings)
    expected = np.mean([np.corrcoef(x, y)[0, 1] / c
                        for x, y, c in zip(expert, values, ceilings)])
    assert out['name_context_frac_ceiling'] == pytest.approx(expected)
    assert not np.isclose(expected, out['name_context_r'] / ceilings.mean())


def test_matrices_sort_politicians_and_preserve_trait_missingness():
    criterion = pd.DataFrame({'leader_id': ['b', 'a', 'c'], 'stratum': ['c1s'] * 3,
                              **{f'expert_{t}_mean': [2., 1., 3.] for t in baseline.TRAITS}})
    criterion.loc[0, 'expert_machiavellianism_mean'] = np.nan
    rows = [(leader, 'qwen-8b', cond, trait, score)
            for leader, score in [('c', 3), ('a', 1), ('b', 2)]
            for cond in baseline.CONDITIONS for trait in baseline.TRAITS]
    points = pd.DataFrame(rows, columns=['leader_id', 'model', 'condition', 'trait', 'value'])
    leaders, experts, matrices = baseline._matrices(points, criterion, 'c1s', 'qwen-8b')
    assert leaders == ['a', 'b', 'c']
    np.testing.assert_array_equal(experts[0], [1, 2, 3])
    assert np.isnan(experts[-1, 1])
    np.testing.assert_array_equal(matrices['NAME_ONLY'][0], [1, 2, 3])


def test_incomplete_associated_response_grid_is_rejected(tmp_path):
    row = {'leader_id': 'a', 'model': 'qwen-8b', 'condition': 'name-only',
           'framing': 'neutral', 'repeat': 0, 'seed': 0, 'speech_id': None,
           'ratings': {f's{i}': 2 for i in range(1, 17)}}
    path = tmp_path / 'records.jsonl'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Incomplete baseline response grid'):
        baseline.load_points(path, {'a'})


def test_associated_repeats_and_open_ensemble_are_averaged(tmp_path):
    rows = []
    for model in (*baseline.OPEN_MODELS, 'gpt-5.5'):
        for condition in ['name-context', 'name-only', 'context-only']:
            for repeat in range(3):
                ratings = {f's{i}': 0 for i in range(1, 17)}
                ratings['s1'] = repeat if model == 'qwen-8b' else 4
                rows.append({'leader_id': 'a', 'model': model, 'condition': condition,
                             'framing': 'neutral', 'repeat': repeat, 'seed': repeat,
                             'speech_id': None, 'ratings': ratings})
    path = tmp_path / 'records.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    points = baseline.load_points(path, {'a'})
    sub = points[(points.trait == 'extraversion') & (points.condition == 'NAME_ONLY')]
    assert sub[sub.model == 'qwen-8b'].value.item() == 2.5
    assert sub[sub.model == 'pooled'].value.item() == (2.5 + 4 + 4 + 4) / 4


def test_frozen_identity_statistics_from_downloaded_negex():
    source = os.getenv('NEGEX_LOCAL_DIR')
    if not source:
        pytest.skip('Set NEGEX_LOCAL_DIR to the independently obtained NEGex 2.0 directory')
    builder_spec = importlib.util.spec_from_file_location('build_inputs', ROOT / 'scripts/build_inputs.py')
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    source = Path(source)
    criterion = builder.build_criterion(
        pd.read_csv(ROOT / 'manifests/sample.csv'),
        pd.read_stata(source / 'NEGex_2.0_candidates.dta', convert_categoricals=False),
        pd.read_stata(source / 'NEGex_2.0_elections.dta', convert_categoricals=False))
    criterion['stratum'] = 'c1s'
    points = baseline.load_points(ROOT / 'records/ratings.jsonl', set(criterion.leader_id))
    ceiling_data = json.loads((ROOT / 'results/expert_ceiling.json').read_text())
    ceilings = {t: ceiling_data['traits'][t]['by_rho']['0.3']['ceiling'] for t in baseline.TRAITS}
    aggregate, traits = baseline.comparison_tables(points, criterion, ceilings)
    for frame, stem, keys in [
            (aggregate, 'comparison_overall', ['stratum', 'model', 'trait_group']),
            (traits, 'comparison_by_trait', ['stratum', 'model', 'trait'])]:
        reference = pd.concat([pd.read_csv(ROOT / f'results/identity_baselines/{stem}_{suffix}.csv')
                               for suffix in ('open', 'gpt55')], ignore_index=True).set_index(keys)
        actual = frame.set_index(keys).loc[reference.index]
        columns = [c for c in reference if pd.api.types.is_numeric_dtype(reference[c])]
        np.testing.assert_allclose(actual[columns], reference[columns],
                                   atol=1e-12, rtol=1e-12, equal_nan=True)
