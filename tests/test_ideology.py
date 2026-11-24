"""Checks for September bootstrap semantics and the cleaned response adapter."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ideology', ROOT / 'scripts/ideology.py')
ideology = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ideology)


def test_scorer_reverse_keys_and_dark_pairs():
    values = {f's{i}': 0 for i in range(1, 17)}
    values.update(s1=4, s6=0, s7=3, s2=1, s11=4, s14=2)
    scores = ideology.score_ratings(values)
    assert scores['extraversion'] == 4
    assert scores['agreeableness'] == 3
    assert scores['narcissism'] == 3
    assert scores['conscientiousness'] == 2


def test_bootstrap_pairwise_missing_values():
    x = np.array([0., 1., np.nan, 3., 4.])
    y = np.array([1., 4., 2., 2., 3.])
    indices = np.array([[0, 1, 2, 3, 4], [0, 0, 0, 0, 0], [4, 3, 2, 1, 0]])
    actual = ideology.bootstrap_r_vector(x, y, indices)
    expected = np.corrcoef(x[[0, 1, 3, 4]], y[[0, 1, 3, 4]])[0, 1]
    np.testing.assert_allclose(actual[[0, 2]], expected)
    assert np.isnan(actual[1])


def _synthetic_adjusted():
    adjusted = pd.DataFrame({'leader_id': list('abcdef'),
                             'is_bridge': [1, 1, 1, 0, 0, 0]})
    points = []
    for t, trait in enumerate(ideology.TRAITS):
        x = np.arange(6., dtype=float) + t / 10
        adjusted['raw_' + trait] = x
        adjusted['adjusted_' + trait] = x
        points.extend((leader, trait, x[i] + (i % 2) * .3)
                      for i, leader in enumerate(adjusted.leader_id))
    adjusted.loc[1, 'raw_machiavellianism'] = np.nan
    adjusted.loc[2, 'adjusted_machiavellianism'] = np.nan
    return adjusted, pd.DataFrame(points, columns=['leader_id', 'trait', 'llm_mean'])


def test_raw_adjusted_share_complete_cases_and_draws(monkeypatch):
    monkeypatch.setattr(ideology, 'N_BOOT', 75)
    adjusted, points = _synthetic_adjusted()
    agg, traits = ideology.convergence_pair(
        points, adjusted, adjusted.leader_id, np.random.default_rng(ideology.SEED))
    # Raw and adjusted are equal wherever both are available. The mismatched
    # missing cells must be dropped from BOTH sides, including bootstrap draws.
    assert agg['delta_adjusted_minus_raw'] == 0
    assert agg['delta_ci_low'] == 0
    assert agg['delta_ci_high'] == 0
    assert agg['n_min_trait_matched'] == 4
    assert agg['n_max_trait_matched'] == 6
    assert all(r['r_raw_matched'] == r['r_adjusted'] for r in traits)


def test_september_rng_call_order_is_preserved(monkeypatch):
    adjusted, points = _synthetic_adjusted()
    frames = []
    for model in ideology.MODELS:
        for framing in ('G', 'O', 'K'):
            frames.append(points.assign(model=model, framing=framing))
    name = pd.concat(frames, ignore_index=True)
    speech = name.assign(condition='named-speech')
    role = name[name.framing == 'O']
    calls = []

    def spy(sub, criterion, ids, rng):
        calls.append((sub.model.iloc[0], sub.framing.iloc[0], len(ids),
                      int(rng.integers(0, 2**31))))
        return {}, []

    monkeypatch.setattr(ideology, 'convergence_pair', spy)
    ideology.build_convergence_tables(name, speech, role, adjusted)
    assert [c[:3] for c in calls[:6]] == [
        ('qwen-8b', 'G', 6), ('qwen-8b', 'G', 3),
        ('qwen-8b', 'O', 6), ('qwen-8b', 'O', 3),
        ('qwen-8b', 'K', 6), ('qwen-8b', 'K', 3)]
    assert calls[6][:3] == ('qwen-32b', 'G', 6)
    expected = np.random.default_rng(20260919).integers(0, 2**31, size=len(calls))
    np.testing.assert_array_equal([c[3] for c in calls], expected)


def _record(leader='a', model='qwen-8b', **updates):
    record = {'leader_id': leader, 'model': model, 'condition': 'name-context',
              'framing': 'neutral', 'repeat': 0, 'seed': 0, 'speech_id': None,
              'ratings': {f's{i}': 0 for i in range(1, 17)}}
    record.update(updates)
    return record


def test_adapter_averages_open_models_before_correlating(tmp_path):
    a, b = _record(), _record(model='gemma-12b')
    b['ratings']['s1'] = 4
    source = tmp_path / 'records.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in [a, b]))
    name, speech, role = ideology.load_points(source, {'a'})
    ensemble = name[(name.model == 'pooled') & (name.trait == 'extraversion')]
    assert ensemble.llm_mean.item() == 3
    assert ensemble.framing.item() == 'O'
    assert speech.empty and role.empty


@pytest.mark.parametrize('records', [
    [_record(), _record()],
    [_record(leader='outside')],
    [_record(repeat=True)],
    [_record(condition='anon-speech', framing='P', speech_id='speech.txt')],
    [_record(speech_id='speech.txt')],
    [_record(ratings={f's{i}': True for i in range(1, 17)})],
])
def test_adapter_rejects_invalid_or_duplicate_cells(tmp_path, records):
    source = tmp_path / 'records.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in records))
    with pytest.raises(ValueError):
        ideology.load_points(source, {'a'})


def test_candidate_slot_join_uses_wide_names_not_rank():
    elections = pd.DataFrame({'electID': ['e1'],
                             **{f'candidate{i}': ['B' if i == 2 else '']
                                for i in range(1, 11)}})
    candidates = pd.DataFrame({'electID': ['e1'], 'actor': ['B'], 'c_rank': [7],
                               'election_date': ['01-Jan-20']})
    criterion = pd.DataFrame({'leader_id': ['a'], 'negex_ids': ['e1#cand2']})
    cards = ideology.map_candidate_cards(candidates, elections, criterion)
    assert cards.source_id.item() == 'e1#cand2'
    assert cards.c_rank.item() == 7


def test_weighting_ignores_missing_weights_unless_all_missing():
    cards = pd.DataFrame({'leader_id': ['a', 'a', 'b', 'b'],
                          'value': [1., 4., 2., 4.], 'weight': [2., np.nan, 0., np.nan]})
    weighted = ideology.weighted_by_leader(cards, 'value', 'weight')
    assert weighted['a'] == 1
    assert weighted['b'] == 3


def test_ridge_preprocessing_uses_training_statistics_only():
    train = pd.DataFrame({
        'candidate_position_13': [1., 3., np.nan], 'incumbent': [0., 1., 0.],
        'election_year': [2016., 2017., 2018.], 'log_sitelinks': [1., 2., 3.],
        'log_pageviews': [3., 2., 1.], 'election_type': ['a', 'b', 'a'],
        'arm': ['candidate', 'office-holder', 'candidate'],
    })
    test = train.iloc[[0]].copy()
    first, test_matrix = ideology.design_matrix(train, test, 'structured_context')
    changed_test = test.copy()
    changed_test['candidate_position_13'] = 1e9
    changed_test['election_type'] = 'unseen'
    unchanged, changed_matrix = ideology.design_matrix(train, changed_test, 'structured_context')
    np.testing.assert_array_equal(first, unchanged)
    assert changed_matrix[0, 1] > test_matrix[0, 1]
    assert changed_matrix.shape == test_matrix.shape


def test_country_group_crossvalidation_never_splits_a_country():
    frame = pd.DataFrame({'country': ['a'] * 4 + ['b'] * 3 + ['c'] * 2 + ['d']})
    folds = ideology.make_folds(frame, 'country_group_10fold', n_splits=3)
    assignment = {int(row): fold for fold, indices in enumerate(folds) for row in indices}
    assert sorted(assignment) == list(range(len(frame)))
    for country, group in frame.groupby('country'):
        assert len({assignment[i] for i in group.index}) == 1


def test_frozen_paper_table_from_downloaded_negex():
    """Opt-in full reproduction; no source dataset or criterion is bundled."""
    source = os.getenv('NEGEX_LOCAL_DIR')
    if not source:
        pytest.skip('Set NEGEX_LOCAL_DIR to the independently obtained NEGex 2.0 directory')
    source = Path(source)
    builder_spec = importlib.util.spec_from_file_location('build_inputs', ROOT / 'scripts/build_inputs.py')
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    candidates = pd.read_stata(source / 'NEGex_2.0_candidates.dta', convert_categoricals=False)
    elections = pd.read_stata(source / 'NEGex_2.0_elections.dta', convert_categoricals=False)
    criterion = builder.build_criterion(pd.read_csv(ROOT / 'manifests/sample.csv'), candidates, elections)
    criterion = criterion.fillna({'office': ''})
    cards = ideology.map_candidate_cards(candidates, elections, criterion)
    adjusted = ideology.build_adjusted_criterion(criterion, cards)
    name, speech, role = ideology.load_points(ROOT / 'records/ratings.jsonl', set(criterion.leader_id))
    aggregate, traits = ideology.build_convergence_tables(name, speech, role, adjusted)
    reference = pd.read_csv(ROOT / 'results/ideology/adjusted_convergence_aggregate.csv')
    keys = ['scope', 'condition', 'model', 'framing']
    actual = aggregate.set_index(keys)
    expected = reference.set_index(keys)
    # Compare all original 71 cells, not merely the six neutral rows displayed.
    columns = [c for c in expected if pd.api.types.is_numeric_dtype(expected[c])]
    np.testing.assert_allclose(actual.loc[expected.index, columns], expected[columns],
                               atol=1e-12, rtol=1e-12, equal_nan=True)
