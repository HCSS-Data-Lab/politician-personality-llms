"""Opt-in checks after recomputing supplementary analyses with the frozen inputs."""
import os
from pathlib import Path
import pandas as pd
import pytest
from test_reproduction import assert_numeric_equal
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize('group,name,keys',[
 ('metrics','ceiling_sensitivity',['stratum','model','framing','trait_group','rho_w']),
 ('fame_gradient','fame_gradient_by_tertile',['proxy','stratum','model_group','framing','tertile']),
 ('fame_gradient','fame_gradient_tertile_gap',['proxy','stratum','model_group','framing']),
 ('fame_gradient','desirability_by_prominence',['proxy','model','metric','framing']),
 ('ideology','metadata_baseline',['feature_set','cv_scheme','trait']),
])
def test_supplementary_reference_tables(group,name,keys):
    if os.environ.get('CHECK_SUPPLEMENTARY')!='1':
        pytest.skip('Recompute using the bundled historical prominence snapshot, then set CHECK_SUPPLEMENTARY=1')
    actual=ROOT/'outputs'/group/(name+'.csv')
    assert actual.exists(),f'Missing recomputation: {name}'
    assert_numeric_equal(pd.read_csv(actual),pd.read_csv(ROOT/'results'/group/(name+'.csv')),keys)


@pytest.mark.integration
def test_criterion_and_name_ideology_gradients():
    if os.environ.get('CHECK_SUPPLEMENTARY') != '1':
        pytest.skip('Run ideology.py --metadata, then set CHECK_SUPPLEMENTARY=1')
    actual = pd.read_csv(ROOT/'outputs/ideology/ideology_gradients.csv')
    expected = pd.read_csv(ROOT/'results/ideology/ideology_gradients.csv')
    # Historical context-only references omit GPT; current expanded BH families
    # differ. The criterion and name gradients retain their original scope.
    assert_numeric_equal(actual[actual.condition != 'role-context'],
                         expected[expected.condition != 'role-context'],
                         ['condition','outcome_kind','model','trait','specification'])


@pytest.mark.integration
def test_nonoffice_metadata_uniqueness():
    if os.environ.get('CHECK_SUPPLEMENTARY') != '1':
        pytest.skip('Run ideology.py --metadata, then set CHECK_SUPPLEMENTARY=1')
    actual = pd.read_csv(ROOT/'outputs/ideology/metadata_identity_uniqueness.csv')
    expected = pd.read_csv(ROOT/'results/ideology/metadata_identity_uniqueness.csv')
    # Blank office strings are currently counted as complete, whereas the
    # historical references omit them. This documented difference is not a pass.
    assert_numeric_equal(actual[~actual.variables.str.contains('office')],
                         expected[~expected.variables.str.contains('office')], ['combination'])
