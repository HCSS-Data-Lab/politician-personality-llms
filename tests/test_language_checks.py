"""Language sensitivity tests, including family-vs-ensemble distinction."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('language_checks', ROOT / 'scripts/language_checks.py')
language = importlib.util.module_from_spec(spec)
spec.loader.exec_module(language)


def synthetic_inputs():
    rng = np.random.default_rng(9)
    ids = [f'p{i:02d}' for i in range(24)]
    expert = rng.normal(size=(8, len(ids)))
    criterion = {leader: {f'expert_{t}_mean': expert[j, i]
                          for j, t in enumerate(language.TRAITS)}
                 for i, leader in enumerate(ids)}
    labels = {leader: ('Spanish' if i < 8 else 'English' if i < 16 else 'German')
              for i, leader in enumerate(ids)}
    estimates = {}
    for m, model in enumerate(language.MODELS):
        predictions = expert + rng.normal(size=expert.shape) * (m + 1) / 2
        values = {leader: {t: predictions[j, i] for j, t in enumerate(language.TRAITS)}
                  for i, leader in enumerate(ids)}
        estimates[model] = {condition: values for condition in language.CONDITIONS}
    return estimates, criterion, labels


def test_family_gap_averages_correlations_not_predictions(monkeypatch):
    monkeypatch.setattr(language, 'N_BOOT', 20)
    estimates, criterion, labels = synthetic_inputs()
    result = language.family_gap(estimates, criterion, labels)
    full = result[(result.subset == 'full') & (result.condition == 'name')].iloc[0]
    ids = sorted(labels)
    per_model = {}
    for model in language.MODELS:
        correlations = []
        for trait in language.TRAITS:
            x = [criterion[i]['expert_' + trait + '_mean'] for i in ids]
            y = [estimates[model]['name'][i][trait] for i in ids]
            correlations.append(np.corrcoef(x, y)[0, 1])
        per_model[model] = np.mean(correlations)
    expected = np.mean([per_model[m] for m in language.GEMMA]) - np.mean(
        [per_model[m] for m in language.QWEN])
    assert full.gemma_minus_qwen == pytest.approx(expected)
    pooled = {}
    for family, models in [('gemma', language.GEMMA), ('qwen', language.QWEN)]:
        pooled[family] = np.mean([
            np.corrcoef([criterion[i]['expert_' + t + '_mean'] for i in ids],
                        [np.mean([estimates[m]['name'][i][t] for m in models]) for i in ids])[0, 1]
            for t in language.TRAITS])
    assert not np.isclose(full.gemma_minus_qwen, pooled['gemma'] - pooled['qwen'])


def test_identical_conditions_have_zero_paired_family_change(monkeypatch):
    monkeypatch.setattr(language, 'N_BOOT', 25)
    estimates, criterion, labels = synthetic_inputs()
    result = language.family_convergence_change_did(estimates, criterion, labels)
    np.testing.assert_array_equal(result.gemma_minus_qwen_change, 0)
    np.testing.assert_array_equal(result.ci_low, 0)
    np.testing.assert_array_equal(result.ci_high, 0)


def test_missing_expert_trait_is_dropped_only_from_that_correlation(monkeypatch):
    monkeypatch.setattr(language, 'N_BOOT', 20)
    estimates, criterion, labels = synthetic_inputs()
    criterion['p00']['expert_machiavellianism_mean'] = np.nan
    _, expert, pred = language._condition_matrices(estimates, 'name', set(labels), criterion)
    assert expert.shape == (8, 24)
    assert np.isnan(expert[-1, 0])
    assert np.isfinite(expert[:-1, 0]).all()
    actual = language.sm._rowwise_r(expert, pred['qwen-8b'])
    expected = np.corrcoef(expert[-1, 1:], pred['qwen-8b'][-1, 1:])[0, 1]
    assert actual[-1] == pytest.approx(expected)
    assert np.isfinite(actual).all()


def test_language_manifest_rejects_mixed_or_missing_country_proxies():
    manifest = pd.DataFrame({'speech_id': ['a.txt', 'b.txt'],
                              'leader_id': ['p', 'p'], 'language': ['Spanish', 'English']})
    with pytest.raises(ValueError, match='one country-proxy'):
        language.language_index(manifest, {'p'})
    manifest.loc[1, 'language'] = None
    with pytest.raises(ValueError, match='Missing country-proxy'):
        language.language_index(manifest, {'p'})
    manifest.loc[1, 'language'] = 'Spanish'
    assert language.language_index(manifest, {'p'}) == {'p': 'Spanish'}


def test_lolo_removes_entire_proxy_group():
    estimates, criterion, labels = synthetic_inputs()
    summary, detail = language.lolo(estimates, criterion, {}, labels)
    assert len(summary) == 10
    assert len(detail) == 30
    assert set(detail.n_omitted_leaders) == {8}
    assert set(detail.n_remaining_leaders) == {16}
    for _, row in summary.iterrows():
        assert row.min_lolo_r <= row.max_lolo_r
        assert row.largest_abs_change == abs(row.largest_change_signed)


def test_shipped_proxy_inventory_preserves_original_country_aliases():
    manifest = pd.read_csv(ROOT / 'manifests/speeches.csv')
    labels = language.language_index(manifest, set(manifest.leader_id))
    assert len(manifest) == 192
    assert len(labels) == 50
    assert manifest.language.nunique() == 30
    assert 'Unknown' not in set(manifest.language)
    assert (manifest.language == 'Spanish').sum() == 48
    assert (manifest.language == 'English').sum() == 20
    assert sum(value == 'English' for value in labels.values()) == 5
    assert labels['c1s__czechia__milos_zeman'] == 'Czech'
    assert labels['c1s__netherlands__mark_rutte'] == 'Dutch'
    assert labels['c1s__north_macedonia__nikola_gruevski'] == 'Macedonian'
    assert labels['c1s__united_states__donald_trump'] == 'English'


def test_frozen_language_tables_from_downloaded_negex():
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
    inputs = language.load_inputs(criterion, ROOT / 'records/ratings.jsonl',
                                  pd.read_csv(ROOT / 'manifests/speeches.csv'))
    keys = {'family_convergence_gap.csv': ['subset', 'condition'],
            'family_convergence_change_did.csv': ['subset', 'condition'],
            'leave_one_language_out_summary.csv': ['model', 'condition']}
    for name, frame in language.compute_tables(*inputs).items():
        reference = pd.read_csv(ROOT / 'results/language' / name).set_index(keys[name])
        actual = frame.set_index(keys[name]).loc[reference.index]
        for column in reference:
            if pd.api.types.is_numeric_dtype(reference[column]):
                np.testing.assert_allclose(actual[column], reference[column],
                                           atol=1e-12, rtol=1e-12, equal_nan=True)
            else:
                assert actual[column].tolist() == reference[column].tolist()
