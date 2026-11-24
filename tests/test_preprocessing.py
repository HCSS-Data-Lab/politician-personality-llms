import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_speeches import apply_steps,digest,prepare


def test_transform_copies_source_spans_and_applies_authored_substitution():
    source='HEADER Hello Ada, thank you.'
    steps=[{'copy':[7,13]},{'insert':'Person A'},{'copy':[16,len(source)]}]
    assert apply_steps(source,steps)=='Hello Person A, thank you.'


@pytest.mark.parametrize('steps',[[{'copy':[-1,4]}],[{'copy':[1,999]}],[{'copy':[True,2]}],[{'unknown':'text'}]])
def test_reject_invalid_recipe(steps):
    with pytest.raises((ValueError,TypeError)):apply_steps('source',steps)


def test_verifies_source_and_both_outputs_before_writing(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'sample.txt').write_text('header hello')
    r={'speech_id':'sample.txt','source_sha256':digest('header hello'),
       'clean_sha256':digest('hello'),'anon_sha256':digest('person'),
       'clean_steps':[{'copy':[7,12]}],'anon_steps':[{'insert':'person'}]}
    recipes=tmp_path/'recipes.json';recipes.write_text(json.dumps([r]))
    dest=tmp_path/'derived'
    assert prepare(raw,recipes,dest)==1
    assert (dest/'clean/sample.txt').read_text()=='hello'
    assert (dest/'anon/sample.txt').read_text()=='person'
    (raw/'sample.txt').write_text('changed')
    with pytest.raises(ValueError,match='source differs'):prepare(raw,recipes,tmp_path/'should_not_exist')
    assert not (tmp_path/'should_not_exist').exists()


def test_public_recipes_cover_exactly_the_selected_speeches():
    import csv
    ids={r['speech_id'] for r in csv.DictReader((ROOT/'manifests/speeches.csv').open())}
    recipes=json.loads((ROOT/'manifests/preprocessing.json').read_text())
    assert len(recipes)==len(ids)==192
    assert {r['speech_id'] for r in recipes}==ids
    for r in recipes:
        for key in ('clean_steps','anon_steps'):
            for step in r[key]:
                assert set(step) in ({'copy'},{'insert'})
                if 'insert' in step:
                    # No source transcript sentences are embedded as replacement literals.
                    assert not step['insert'].strip() or len(step['insert'])==1
