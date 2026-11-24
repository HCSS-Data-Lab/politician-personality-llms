"""Integration gates compare recomputed statistics to frozen aggregate references."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))


def assert_numeric_equal(actual,expected,keys):
    assert not actual.duplicated(keys).any()
    assert not expected.duplicated(keys).any()
    actual=actual.set_index(keys).sort_index();expected=expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index,expected.index)
    for c in expected:
        if pd.api.types.is_numeric_dtype(expected[c]):
            np.testing.assert_allclose(actual[c].to_numpy(float),expected[c].to_numpy(float),rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=c)
        else:
            assert actual[c].fillna('').tolist()==expected[c].fillna('').tolist(),c


@pytest.mark.integration
@pytest.mark.parametrize('filename,keys',[
 ('convergence',['stratum','model','framing','trait']),
 ('directional_bias',['stratum','model','framing','trait']),
 ('reliability_icc',['stratum','model','framing','trait']),
 ('reliability_icc_rollup',['stratum','model','trait']),
 ('convergence_aggregates',['stratum','model','framing','trait_group']),
 ('framing_contrasts',['stratum','model','trait','contrast']),
 ('speech_convergence',['condition','framing','model','trait']),
 ('evidence_contrasts',['stratum','model','trait','contrast']),
 ('anon_by_recognition',['recog_def','framing','subset','model','trait_group']),
 ('speech_model_contrasts',['condition','framing','model_a','model_b','trait']),
 ('model_contrasts',['stratum','model_a','model_b','framing','trait']),
])
def test_full_recomputed_primary_statistics(filename,keys):
    if os.environ.get('CHECK_RECOMPUTED')!='1':
        pytest.skip('Run the documented analysis commands, then set CHECK_RECOMPUTED=1')
    target=ROOT/'outputs/metrics'/f'{filename}.csv'
    assert target.exists(),f'Missing recomputation: {target.name}'
    actual=pd.read_csv(target);expected=pd.read_csv(ROOT/'results/metrics'/f'{filename}.csv')
    assert_numeric_equal(actual,expected,keys)


def test_frozen_recognition_totals_and_corrected_rates():
    import json
    rows=[json.loads(l) for l in (ROOT/'records/recognition.jsonl').open()]
    gpt=[r for r in rows if r['model']=='gpt-5.5'];op=[r for r in rows if r['model']!='gpt-5.5']
    assert len(gpt)==192 and len(op)==768
    assert sum(r['label']=='correct' for r in gpt)==184
    assert sum(r['label']!='wrong' for r in gpt)==191
    assert sum(r['label']=='correct' for r in op)==211
    assert sum(r['label']!='wrong' for r in op)==428
    assert sum(r['auto_label']!=r['label'] for r in rows)==52
